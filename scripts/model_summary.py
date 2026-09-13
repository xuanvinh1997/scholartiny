import argparse,json
from dataclasses import replace
import torch
from model.config import ModelConfig
from model.model_scholar import ScholarLM

def summarize(path):
    cfg=ModelConfig.load(path)
    with torch.device('meta'):model=ScholarLM(replace(cfg,ssm_backend='reference'))
    n=sum(p.numel() for p in model.parameters())
    d=cfg.hidden_size;inner=d*cfg.ssm_expand
    att=cfg.layers.count('A');ssm=cfg.layers.count('M')
    cache=[]
    for length in (2048,8192,16384):
        kv=2*att*length*cfg.num_key_value_heads*(d//cfg.num_attention_heads)*2
        recurrent=ssm*(inner*cfg.state_size+(inner+2*cfg.state_size)*cfg.conv_kernel)*2
        cache.append({'length':length,'KV_MiB':kv/2**20,'SSM_MiB':recurrent/2**20,
                      'note':'theory, batch=1, 2-byte cache; excludes weights/activations/workspace'})
    return {'config':str(path),'parameters':n,'millions':n/1e6,'attention_layers':att,'ssm_layers':ssm,'cache':cache}

def main():
    p=argparse.ArgumentParser();p.add_argument('configs',nargs='+');a=p.parse_args()
    print(json.dumps([summarize(f) for f in a.configs],indent=2))
if __name__=='__main__':main()
