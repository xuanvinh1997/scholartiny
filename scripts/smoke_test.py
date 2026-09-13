"""Dependency-light, offline end-to-end smoke test. Never downloads public data."""
import argparse
import json
from pathlib import Path
import platform
import torch
from dataset.tokenizer import ByteTokenizer
from dataset.normalize import convert
from dataset.protocol import canonical_json
from dataset.prepare import prepare
from trainer.train import parser,run
from agent.tools import SCHEMAS,ToolExecutor
from agent.rag import build_index,search
from agent.runner import load_model,complete
from scripts.generate_academic import tool_dialogue


def smoke(out):
    torch.set_num_threads(1)
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new/empty --out directory')
    out.mkdir(parents=True,exist_ok=True)
    tok=out/'tokenizer.json';ByteTokenizer().save(tok)
    pretrain=out/'pretrain.jsonl';sft=out/'sft.jsonl'
    docs=[]
    for i in range(80):
        text=(f'In synthetic experiment {i}, the sample size is {i+10}. '
              f'The recorded value is {i+1}. This fictional record is for pipeline testing only. '
              'An equation relates quantities. A citation identifies the supporting passage.')
        row=convert({'text':text},{'adapter':'text','min_chars':1})
        docs.append(dict(row,split='train',source='original_smoke_fixture',provenance={'synthetic':True}))
    pretrain.write_text('\n'.join(canonical_json(x) for x in docs)+'\n',encoding='utf-8')
    conversations=[]
    for i in range(12):
        messages=[{'role':'system','content':'Use tools for exact calculations. Do not invent observations.'}]
        messages+=tool_dialogue(f'Calculate {i+2}+3.','math.calculate',{'expression':f'{i+2}+3'},str(i+5))
        conversations.append({'kind':'sft','messages':messages,'split':'train','tools':[SCHEMAS[0]]})
    sft.write_text('\n'.join(canonical_json(x) for x in conversations)+'\n',encoding='utf-8')
    pm=prepare(pretrain,tok,out/'packed_pretrain','pretrain',64)
    sm=prepare(sft,tok,out/'packed_sft','sft',1024)
    config=Path(__file__).parents[1]/'configs/debug.yaml'
    common=['--config',str(config),'--tokenizer',str(tok),'--device','cpu','--cpu-threads','1',
            '--max-steps','2','--warmup-steps','1','--save-every','2','--batch-size','1','--grad-accum','1']
    pre_ckpt=run(parser().parse_args(common+['--data',str(out/'packed_pretrain'),'--out',str(out/'pretrain_run'),'--seq-len','64']))
    sft_ckpt=run(parser().parse_args(common+['--data',str(out/'packed_sft'),'--out',str(out/'sft_run'),'--seq-len','1024','--init',str(pre_ckpt)]))
    index=build_index([pretrain],out/'rag.sqlite',limit=10)
    retrieved=search(out/'rag.sqlite','experiment sample size',top_k=2)
    tool=ToolExecutor().execute({'name':'math.calculate','arguments':{'expression':'1/3+1/6'}})
    assert tool['ok'] and tool['result']['exact']=='1/2'
    model,tok_obj=load_model(sft_ckpt,tok,'cpu','reference')
    generated=complete(model,tok_obj,[{'role':'system','content':'Answer the question.'},
                                    {'role':'user','content':'2+3?'}],max_new_tokens=4)
    report={'status':'passed','python':platform.python_version(),'torch':str(torch.__version__),
            'cuda':torch.cuda.is_available(),'backend':'reference','tokenizer':'debug bytes (264 tokens)',
            'pretrain_steps':2,'sft_steps':2,'pretrain_data':pm,'sft_data':sm,'retrieval_index':index,
            'retrieved_chunks':len(retrieved),'tool_result':tool,'generation_path_executed':True,
            'generated_untrained_text':generated,
            'limitations':['Not a quality benchmark','No public datasets downloaded',
                           'BPE dependency and official CUDA Mamba-2 not exercised by this smoke test']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='.work/smoke');a=p.parse_args();smoke(a.out)
if __name__=='__main__':main()
