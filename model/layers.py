import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + self.eps)
                * self.weight.float()).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GlobalGQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = cfg.hidden_size // self.nh
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, self.nkv*self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.nkv*self.hd, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        inv = cfg.rope_theta ** (-torch.arange(0, self.hd, 2).float()/self.hd)
        self.register_buffer('inv_freq', inv, persistent=False)

    def rope(self, x, offset):
        pos = torch.arange(offset, offset+x.shape[-2], device=x.device, dtype=torch.float32)
        angle = torch.outer(pos, self.inv_freq.float())
        cos, sin = angle.cos()[None,None], angle.sin()[None,None]
        even, odd = x.float()[...,0::2], x.float()[...,1::2]
        return torch.stack((even*cos-odd*sin, even*sin+odd*cos), -1).flatten(-2).to(x.dtype)

    def forward(self, x, cache=None, offset=0):
        b,t,d = x.shape
        q = self.q_proj(x).view(b,t,self.nh,self.hd).transpose(1,2)
        k = self.k_proj(x).view(b,t,self.nkv,self.hd).transpose(1,2)
        v = self.v_proj(x).view(b,t,self.nkv,self.hd).transpose(1,2)
        q, k = self.rope(self.q_norm(q), offset), self.rope(self.k_norm(k), offset)
        if cache is not None:
            if 'k' in cache:
                if cache['k'].shape[-2] != offset:
                    raise ValueError('KV cache offset mismatch')
                k = torch.cat((cache['k'], k), dim=2)
                v = torch.cat((cache['v'], v), dim=2)
            elif offset:
                raise ValueError('Nonzero offset with empty KV cache')
            cache['k'], cache['v'] = k, v
        # Cache stores only nkv heads. SDPA may materialize expanded heads transiently.
        k = k.repeat_interleave(self.nh//self.nkv, dim=1)
        v = v.repeat_interleave(self.nh//self.nkv, dim=1)
        if offset == 0:
            y = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
        elif t == 1:
            # is_causal=True is WRONG for a single-token query with a longer KV cache.
            y = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
        else:
            mask = (torch.arange(k.shape[2],device=x.device)[None,:] <=
                    torch.arange(offset,offset+t,device=x.device)[:,None])
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        return self.out_proj(y.transpose(1,2).contiguous().view(b,t,d))
