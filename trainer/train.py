"""Native PyTorch training: CPU/CUDA, DDP, BF16, resumable token-weighted loss."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.config import ModelConfig
from model.model_scholar import ScholarLM,optimizer_groups
from dataset.loader import BatchStream
from dataset.tokenizer import load_tokenizer,fingerprint


def lr_for(step,total,warmup,peak,min_ratio=.1):
    if step<warmup: return peak*(step+1)/max(warmup,1)
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return peak*(min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*progress)))


def trusted_load(path):
    # weights_only avoids arbitrary pickled Python objects. Only own checkpoints are supported.
    return torch.load(path,map_location='cpu',weights_only=True)

def compatible_configs(saved,current,allow_context_change=False):
    ignore={'ssm_backend','gradient_checkpointing','loss_chunk_size'}
    if allow_context_change: ignore.add('max_seq_len')
    if {k:v for k,v in saved.items() if k not in ignore}!={k:v for k,v in current.items() if k not in ignore}:
        raise ValueError('Checkpoint architecture does not match requested configuration')

@torch.inference_mode()
def validate(model,args,tok,device,amp_dtype,rank,world):
    stream=BatchStream(args.val_data,args.seq_len,args.batch_size,tok.token_id('<|pad|>'),seed=987,rank=rank)
    model.eval();sums=torch.zeros(2,device=device,dtype=torch.float64)
    for _ in range(args.val_batches):
        batch=stream.next()
        with torch.autocast(device.type,dtype=amp_dtype,enabled=amp_dtype is not None):
            result=model(batch['input_ids'].to(device),labels=batch['labels'].to(device))
        sums[0]+=result['loss_sum'];sums[1]+=result['token_count']
    if world>1:dist.all_reduce(sums)
    model.train()
    loss=(sums[0]/sums[1]).item()
    return {'val_loss':loss,'val_ppl':math.exp(min(loss,50)),'val_supervised_tokens':int(sums[1].item())}


def run(args):
    world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'))
    use_cuda=args.device!='cpu' and torch.cuda.is_available()
    if args.device=='cuda' and not use_cuda: raise RuntimeError('CUDA requested but unavailable')
    device=torch.device(f'cuda:{local}' if use_cuda else 'cpu')
    if use_cuda:torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if use_cuda else 'gloo')
    if args.cpu_threads:torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if use_cuda:torch.cuda.manual_seed_all(args.seed)
    cfg=ModelConfig.load(args.config)
    if args.backend: cfg=replace(cfg,ssm_backend=args.backend)
    if args.gradient_checkpointing:cfg=replace(cfg,gradient_checkpointing=True)
    if cfg.ssm_backend=='mamba2' and 'M' in cfg.layers and not use_cuda:
        raise RuntimeError('Official Mamba-2 requires CUDA here. Use configs/debug.yaml for CPU tests.')
    if args.seq_len>cfg.max_seq_len: raise ValueError('Training context exceeds model config')
    tok=load_tokenizer(args.tokenizer);toksha=fingerprint(args.tokenizer)
    if tok.vocab_size!=cfg.vocab_size:
        raise ValueError(f'Tokenizer vocab {tok.vocab_size} != config {cfg.vocab_size}. Use actual trained vocab size.')
    stream=BatchStream(args.data,args.seq_len,args.batch_size,tok.token_id('<|pad|>'),args.seed,rank)
    if any(ds.meta['tokenizer_sha256']!=toksha for ds in stream.datasets):raise ValueError('Prepared tokenizer fingerprint mismatch')
    raw=ScholarLM(cfg).to(device)
    # Parameters/master weights remain FP32; CUDA autocast uses BF16 activations.
    if args.precision=='bf16' and not use_cuda: raise ValueError('BF16 training in this runner requires CUDA')
    if args.precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('GPU has no BF16 support')
    amp_dtype=torch.bfloat16 if args.precision=='bf16' else None
    optimizer=torch.optim.AdamW(optimizer_groups(raw,args.weight_decay),lr=args.lr,betas=(.9,.95),eps=1e-8)
    step,tokens_seen,supervised_seen=0,0,0
    path=args.resume or args.init
    if path:
        state=trusted_load(path)
        compatible_configs(state['model_config'],cfg.to_dict(),allow_context_change=not bool(args.resume))
        if state['tokenizer_sha256']!=toksha:raise ValueError('Checkpoint tokenizer mismatch')
        raw.load_state_dict(state['model'],strict=True)
        if args.resume:
            if state['world_size']!=world:raise ValueError('Exact resume requires the same world size')
            if state['schedule']!={'max_steps':args.max_steps,'warmup_steps':args.warmup_steps,'lr':args.lr,
                                    'grad_accum':args.grad_accum,'precision':args.precision}:
                raise ValueError('Exact resume requires same LR schedule/accumulation/precision; use --init for a new stage')
            optimizer.load_state_dict(state['optimizer']);step=state['step']
            tokens_seen=state['tokens_seen'];supervised_seen=state['supervised_seen']
            local_state=state['ranks'][rank];stream.load_state_dict(local_state['stream'])
            torch.set_rng_state(local_state['torch_rng']);random.setstate(local_state['python_rng'])
            if use_cuda:torch.cuda.set_rng_state(local_state['cuda_rng'],device)
    model=DDP(raw,device_ids=[local] if use_cuda else None,broadcast_buffers=False) if world>1 else raw
    out=Path(args.out)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        (out/'run.json').write_text(json.dumps({'args':vars(args),'model':cfg.to_dict(),
                   'parameters':sum(p.numel() for p in raw.parameters()),'torch':str(torch.__version__),
                   'world_size':world,'tokenizer_sha256':toksha},indent=2),encoding='utf-8')
    if world>1:dist.barrier()
    model.train();start=time.perf_counter()
    def save():
        local_state={'stream':stream.state_dict(),'torch_rng':torch.get_rng_state(),
                     'python_rng':random.getstate(),
                     'cuda_rng':torch.cuda.get_rng_state(device) if use_cuda else None}
        states=[None]*world
        if world>1:dist.all_gather_object(states,local_state)
        else:states=[local_state]
        if rank==0:
            state={'format_version':1,'model':raw.state_dict(),'optimizer':optimizer.state_dict(),
                   'model_config':cfg.to_dict(),'tokenizer_sha256':toksha,'step':step,
                   'tokens_seen':tokens_seen,'supervised_seen':supervised_seen,
                   'world_size':world,'ranks':states,
                   'schedule':{'max_steps':args.max_steps,'warmup_steps':args.warmup_steps,'lr':args.lr,
                               'grad_accum':args.grad_accum,'precision':args.precision}}
            tmp=out/'last.pt.tmp';torch.save(state,tmp);tmp.replace(out/'last.pt')
        if world>1:dist.barrier()
    while step<args.max_steps and (not args.token_budget or supervised_seen<args.token_budget):
        batches=[stream.next() for _ in range(args.grad_accum)]
        # Global target count BEFORE backprop: variable-length SFT must be token weighted,
        # not an average of per-microbatch means or per-rank means.
        target_count=sum(int((b['labels'][:,1:]!=-100).sum()) for b in batches)
        counts=torch.tensor([target_count,sum(b['input_tokens'] for b in batches)],device=device,dtype=torch.int64)
        if world>1:dist.all_reduce(counts)
        if counts[0].item()==0:raise ValueError('Empty supervised global batch')
        lr=lr_for(step,args.max_steps,args.warmup_steps,args.lr)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum=torch.zeros((),device=device)
        for i,batch in enumerate(batches):
            sync=model.no_sync() if world>1 and i<len(batches)-1 else nullcontext()
            with sync:
                with torch.autocast(device.type,dtype=amp_dtype,enabled=amp_dtype is not None):
                    result=model(batch['input_ids'].to(device),labels=batch['labels'].to(device))
                    loss=result['loss_sum']*(world/counts[0])
                loss.backward();loss_sum+=result['loss_sum'].detach()
        grad_norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),args.grad_clip)
        if not torch.isfinite(grad_norm):raise FloatingPointError('Non-finite gradient norm; checkpoint not overwritten')
        optimizer.step();step+=1
        if world>1:dist.all_reduce(loss_sum)
        supervised_seen+=int(counts[0].item());tokens_seen+=int(counts[1].item())
        metrics={'step':step,'loss':(loss_sum/counts[0]).item(),'lr':lr,'grad_norm':float(grad_norm),
                 'supervised_tokens':supervised_seen,'input_tokens':tokens_seen,
                 'elapsed_seconds':time.perf_counter()-start}
        if args.val_data and step%args.eval_every==0:
            metrics.update(validate(raw,args,tok,device,amp_dtype,rank,world))
        if use_cuda:metrics['peak_allocated_gib']=torch.cuda.max_memory_allocated(device)/2**30
        if rank==0:
            print(json.dumps(metrics),flush=True)
            with (out/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(metrics)+'\n')
        if step%args.save_every==0:save()
        if args.stop_after and step>=args.stop_after:break
    save()
    if world>1:dist.destroy_process_group()
    return out/'last.pt'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--val-data');p.add_argument('--val-batches',type=int,default=8)
    p.add_argument('--seq-len',type=int,default=2048);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=16);p.add_argument('--max-steps',type=int,default=1000)
    p.add_argument('--token-budget',type=int,default=0,help='Optional predicted-token budget, up to one global batch overshoot')
    p.add_argument('--stop-after',type=int,default=0,help='Clean interruption without changing LR schedule')
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--warmup-steps',type=int,default=100)
    p.add_argument('--weight-decay',type=float,default=.1);p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--cpu-threads',type=int,default=0)
    p.add_argument('--backend',choices=['reference','mamba2']);p.add_argument('--gradient-checkpointing',action='store_true')
    g=p.add_mutually_exclusive_group();g.add_argument('--init');g.add_argument('--resume')
    return p

def main():
    p=parser();a=p.parse_args()
    if min(a.seq_len,a.batch_size,a.grad_accum,a.max_steps,a.save_every,a.eval_every,a.val_batches)<1:
        p.error('Lengths, steps and intervals must be positive')
    run(a)
if __name__=='__main__':main()
