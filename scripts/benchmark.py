"""Measure random-weight kernel runtime, not model quality. GPU warmup excluded."""
import argparse,json,time
from dataclasses import replace
import torch
from model.config import ModelConfig
from model.model_scholar import ScholarLM

@torch.inference_mode()
def bench(cfg,length,decode_tokens,device,dtype):
    if length+decode_tokens>cfg.max_seq_len:raise ValueError('Benchmark exceeds model context')
    if cfg.ssm_backend=='reference' and 'M' in cfg.layers and length>256:
        raise ValueError('Reference recurrence is not a production benchmark; use --backend mamba2 on CUDA')
    model=ScholarLM(cfg).to(device=device,dtype=dtype).eval()
    sync=lambda:torch.cuda.synchronize(device) if str(device).startswith('cuda') else None
    ids=torch.randint(0,cfg.vocab_size,(1,length),device=device)
    warm=min(length,64)
    model(ids[:,:warm],logits_to_keep=1);sync()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    cache=model.new_cache();start=time.perf_counter()
    logits=model(ids,cache=cache,logits_to_keep=1)['logits'];sync();prefill=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(decode_tokens):
        token=logits[:,-1].argmax(-1,keepdim=True)
        logits=model(token,cache=cache,logits_to_keep=1)['logits']
    sync();elapsed=time.perf_counter()-start
    return {'length':length,'prefill_seconds':prefill,'prefill_tokens_per_second':length/prefill,
            'decode_tokens_per_second':decode_tokens/elapsed,
            'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30 if str(device).startswith('cuda') else None,
            'note':'Random weights; dynamic torch.cat KV cache; latency is implementation-specific'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--lengths',nargs='+',type=int,default=[2048,8000])
    p.add_argument('--decode-tokens',type=int,default=32);p.add_argument('--device',default='cuda')
    p.add_argument('--backend',choices=['reference','mamba2']);p.add_argument('--precision',choices=['fp32','bf16'],default='bf16')
    a=p.parse_args();cfg=ModelConfig.load(a.config)
    if a.backend:cfg=replace(cfg,ssm_backend=a.backend)
    dtype=torch.bfloat16 if a.precision=='bf16' else torch.float32
    print(json.dumps([bench(cfg,n,a.decode_tokens,a.device,dtype) for n in a.lengths],indent=2))
if __name__=='__main__':main()
