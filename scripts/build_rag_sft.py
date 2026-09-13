"""Extractive, source-grounded RAG warmup from already split public documents.

This does not invent semantic question/answer pairs. It teaches passage selection,
exact quotation, citation IDs and insufficient-evidence behavior. The original
license and split are inherited; no blanket MIT relabeling of document text.
"""
import argparse
from collections import deque
import json
from pathlib import Path
from agent.tools import SCHEMAS
from dataset.normalize import sha
from dataset.protocol import canonical_json

def build(input_root,out,limit=10000):
    root,out=Path(input_root),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new/empty output directory')
    out.mkdir(parents=True,exist_ok=True);counts={}
    for split in ('train','val','test'):
        n=0;previous=deque(maxlen=16)
        with (root/(split+'.jsonl')).open(encoding='utf-8') as f, (out/(split+'.jsonl')).open('w',encoding='utf-8') as dest:
            for line in f:
                row=json.loads(line)
                if row.get('split')!=split:raise ValueError('Document split mismatch')
                text=row.get('text','').strip()
                if len(text)<80:continue
                doc_id='D'+row['id'][:12]
                excerpt=text[:700]
                quote=excerpt.split('\n')[0][:240]
                title=(row.get('provenance') or {}).get('title') or doc_id
                for missing in (False,True):
                    evidence=list(previous)[-2:]
                    if not missing:evidence.append({'chunk_id':doc_id,'text':excerpt})
                    question=(f'Trích nguyên văn phần mở đầu của tài liệu {title} trong các đoạn truy xuất. '
                              'Nêu mã trích dẫn; nếu không có tài liệu này thì nói chưa đủ bằng chứng.')
                    answer='Chưa đủ bằng chứng trong các đoạn được truy xuất.' if missing else f'“{quote}” [{doc_id}]'
                    messages=[{'role':'user','content':question},
                              {'role':'assistant','content':'','tool_calls':[{'name':'retrieve','arguments':{'query':str(title),'top_k':3}}]},
                              {'role':'tool','name':'retrieve','content':canonical_json({'ok':True,'result':evidence})},
                              {'role':'assistant','content':answer}]
                    payload={'id':sha(canonical_json(messages)),'group_id':row['group_id'],'parent_document_id':row['id'],
                             'source':'rag_grounded','split':split,'kind':'sft','language':'vi',
                             'provenance':row.get('provenance',{}),'tools':[s for s in SCHEMAS if s['name']=='retrieve'],
                             'messages':messages,'task':'extractive_rag_warmup','missing_evidence':missing}
                    dest.write(canonical_json(payload)+'\n');n+=1
                previous.append({'chunk_id':doc_id,'text':excerpt})
                if n>=limit:break
        counts[split]=n
    (out/'manifest.json').write_text(json.dumps({'source':str(root),'counts':counts,
                'split_policy':'inherit parent document split; distractors from same split only',
                'task':'extractive warmup, not research-level semantic QA'},indent=2),encoding='utf-8')
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',required=True)
    p.add_argument('--out',default='data/normalized/rag_grounded');p.add_argument('--limit',type=int,default=10000)
    a=p.parse_args();print(json.dumps(build(a.input_root,a.out,a.limit)))
if __name__=='__main__':main()
