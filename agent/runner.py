import json
import torch
from model.config import ModelConfig
from model.model_scholar import ScholarLM
from dataset.tokenizer import load_tokenizer,fingerprint
from dataset.protocol import encode_chat,parse_tool_calls,canonical_json
from trainer.train import trusted_load
from .tools import SCHEMAS,ToolExecutor
from .rag import citation_ids

def load_model(checkpoint,tokenizer_path,device='cpu',backend=None,precision='fp32'):
    state=trusted_load(checkpoint);cfg=dict(state['model_config'])
    if fingerprint(tokenizer_path)!=state['tokenizer_sha256']:raise ValueError('Tokenizer fingerprint mismatch')
    if backend:cfg['ssm_backend']=backend
    tok=load_tokenizer(tokenizer_path)
    model=ScholarLM(ModelConfig(**cfg))
    model.load_state_dict(state['model'],strict=True)
    dtype=torch.bfloat16 if precision=='bf16' else torch.float32
    if precision=='bf16' and not str(device).startswith('cuda'):raise ValueError('BF16 inference requires CUDA in this CLI')
    model.to(device=device,dtype=dtype).eval()
    return model,tok

@torch.inference_mode()
def complete(model,tok,messages,tools=None,max_new_tokens=256,temperature=0.):
    ids,_=encode_chat(tok,messages,tools,add_generation_prompt=True)
    device=next(model.parameters()).device
    inputs=torch.tensor([ids],device=device,dtype=torch.long)
    generated=model.generate(inputs,max_new_tokens=max_new_tokens,temperature=temperature,
                             stop_ids=(tok.token_id('<|im_end|>'),tok.token_id('<|eos|>')))
    output=generated[0,len(ids):].tolist()
    if output and output[-1] in (tok.token_id('<|im_end|>'),tok.token_id('<|eos|>')):output.pop()
    return tok.decode(output)

def run_agent(model,tok,question,rag_path=None,max_turns=5,max_new_tokens=256):
    messages=[{'role':'user','content':question}];executor=ToolExecutor(rag_path)
    observed=set();trace=[]
    for turn in range(max_turns):
        answer=complete(model,tok,messages,SCHEMAS,max_new_tokens)
        calls=parse_tool_calls(answer)
        if not calls:
            invalid=citation_ids(answer)-observed
            return {'answer':answer,'trace':trace,'invalid_citations':sorted(invalid),
                    'citation_check':'ID existence only, not entailment verification'}
        messages.append({'role':'assistant','content':'','tool_calls':calls})
        for call in calls:
            result=executor.execute(call)
            trace.append({'turn':turn,'call':call,'observation':result})
            if call['name']=='retrieve' and result.get('ok'):
                observed.update(x['chunk_id'] for x in result['result'])
            messages.append({'role':'tool','name':call['name'],'content':canonical_json(result)})
    return {'answer':None,'error':'Tool-turn limit reached','trace':trace}
