from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU, GlobalGQA
from .ssm import make_ssm, new_ssm_cache

@dataclass
class HybridCache:
    batch_size: int
    max_seq_len: int
    offset: int = 0
    attention: dict = field(default_factory=dict)
    ssm: object = None
    def __post_init__(self):
        self.ssm = new_ssm_cache(self.max_seq_len,self.batch_size)

class Block(nn.Module):
    def __init__(self,cfg,kind,idx):
        super().__init__()
        self.kind,self.idx = kind,idx
        self.norm1,self.norm2 = RMSNorm(cfg.hidden_size,cfg.norm_eps),RMSNorm(cfg.hidden_size,cfg.norm_eps)
        self.mixer = GlobalGQA(cfg) if kind=='A' else make_ssm(cfg,idx)
        self.ffn = SwiGLU(cfg.hidden_size,cfg.intermediate_size)

    def forward(self,x,cache=None):
        h = self.norm1(x)
        if self.kind == 'A':
            ac = cache.attention.setdefault(self.idx,{}) if cache is not None else None
            h = self.mixer(h,cache=ac,offset=cache.offset if cache is not None else 0)
        else:
            h = self.mixer(h,inference_params=cache.ssm if cache is not None else None)
        x = x+h
        return x+self.ffn(self.norm2(x))

class ScholarLM(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size,cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg,k,i) for i,k in enumerate(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden_size,cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size,cfg.vocab_size,bias=False)
        nn.init.normal_(self.embedding.weight,std=.02)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        else:
            nn.init.normal_(self.lm_head.weight,std=.02)
        # Do not call model.apply(init): that can overwrite upstream SSM initialization.
        for block in self.layers:
            for m in block.ffn.modules():
                if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=.02)
            if block.kind == 'A':
                for m in block.mixer.modules():
                    if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=.02)
            with torch.no_grad():
                block.ffn.down_proj.weight.div_(math.sqrt(2*cfg.num_hidden_layers))
                block.mixer.out_proj.weight.div_(math.sqrt(2*cfg.num_hidden_layers))

    def new_cache(self,batch_size=1):
        return HybridCache(batch_size,self.config.max_seq_len)

    def forward(self,input_ids,labels=None,cache=None,logits_to_keep=0):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError('Expected nonempty [batch, sequence] input_ids')
        offset = cache.offset if cache is not None else 0
        if offset+input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError('Context limit reached; no silent truncation')
        if cache is not None:
            if self.training or labels is not None or torch.is_grad_enabled():
                raise ValueError('Cache is inference-only; use eval() and torch.no_grad()')
            if input_ids.shape[0] != cache.batch_size:
                raise ValueError('Cache batch size mismatch')
            if offset and input_ids.shape[1] != 1 and 'M' in self.config.layers:
                raise ValueError('Hybrid cached decode accepts one token after prefill')
            cache.ssm.seqlen_offset = offset
        x = self.embedding(input_ids)
        for block in self.layers:
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(block,x,use_reentrant=False)
            else:
                x = block(x,cache)
        x = self.norm(x)
        if cache is not None:
            cache.offset += input_ids.shape[1]
            cache.ssm.seqlen_offset = cache.offset
        if labels is None:
            hidden = x[:,-logits_to_keep:] if logits_to_keep else x
            return {'logits':self.lm_head(hidden),'loss':None}
        if labels.shape != input_ids.shape:
            raise ValueError('Labels must be aligned, unshifted; use -100 for masked targets')
        h,target = x[:,:-1].reshape(-1,x.shape[-1]),labels[:,1:].reshape(-1)
        count = (target != -100).sum()
        if not count.item():
            raise ValueError('No supervised tokens in batch')
        total = x.new_zeros((),dtype=torch.float32)
        def chunk_ce(a,b):
            return F.cross_entropy(self.lm_head(a).float(),b,ignore_index=-100,reduction='sum')
        for start in range(0,len(target),self.config.loss_chunk_size):
            a,b = h[start:start+self.config.loss_chunk_size],target[start:start+self.config.loss_chunk_size]
            # Recompute each vocab projection in backward to avoid retaining B*T*V logits.
            loss = checkpoint(chunk_ce,a,b,use_reentrant=False) if self.training else chunk_ce(a,b)
            total = total+loss
        return {'logits':None,'loss':total/count,'loss_sum':total,'token_count':count}

    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens=128,temperature=0.0,stop_ids=()):
        if input_ids.shape[0] != 1:
            raise ValueError('Minimal generation supports an unpadded batch of size 1')
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError('Generation sizes/temperature must be nonnegative')
        self.eval()
        cache = self.new_cache(1)
        if input_ids.shape[1]+max_new_tokens > self.config.max_seq_len:
            raise ValueError('Prompt + generation budget exceeds configured context')
        if max_new_tokens == 0: return input_ids
        logits = self(input_ids,cache=cache,logits_to_keep=1)['logits'][:,-1]
        result = [input_ids]
        for i in range(max_new_tokens):
            token = (logits.argmax(-1,keepdim=True) if temperature==0 else
                     torch.multinomial(F.softmax(logits.float()/temperature,-1),1))
            result.append(token)
            if token.item() in stop_ids or i+1 == max_new_tokens: break
            logits = self(token,cache=cache,logits_to_keep=1)['logits'][:,-1]
        return torch.cat(result,dim=1)

def optimizer_groups(model,weight_decay):
    decay,no_decay = [],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        # Check names too: tensor metadata can be lost on conversion/loading.
        exempt = p.ndim<2 or getattr(p,'_no_weight_decay',False) or name.endswith(('A_log','D','dt_bias'))
        (no_decay if exempt else decay).append(p)
    return [{'params':decay,'weight_decay':weight_decay},
            {'params':no_decay,'weight_decay':0.0}]
