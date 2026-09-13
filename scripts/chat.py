import argparse,json
from agent.runner import load_model,run_agent,complete

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--prompt',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--backend',choices=['reference','mamba2']);p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--rag-db');p.add_argument('--agent',action='store_true');p.add_argument('--max-new-tokens',type=int,default=256)
    a=p.parse_args();model,tok=load_model(a.checkpoint,a.tokenizer,a.device,a.backend,a.precision)
    if a.agent:print(json.dumps(run_agent(model,tok,a.prompt,a.rag_db,max_new_tokens=a.max_new_tokens),ensure_ascii=False,indent=2))
    else:print(complete(model,tok,[{'role':'user','content':a.prompt}],max_new_tokens=a.max_new_tokens))
if __name__=='__main__':main()
