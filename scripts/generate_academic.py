"""Generate verifiable bilingual toy tasks, not teacher-model reasoning distillation."""
import argparse
import json
from pathlib import Path
import random
from agent.tools import SCHEMAS,compute_math
from dataset.collect import RecordStore
from dataset.normalize import sha,convert
from dataset.protocol import canonical_json

def tool_dialogue(question,name,args,answer=None):
    result={'ok':True,'result':compute_math(name,args)}
    if answer is None:answer=canonical_json(result['result'])
    return [{'role':'user','content':question},
            {'role':'assistant','content':'','tool_calls':[{'name':name,'arguments':args}]},
            {'role':'tool','name':name,'content':canonical_json(result)},
            {'role':'assistant','content':answer}]

def generate_records(count,seed=123):
    rng=random.Random(seed)
    for i in range(count):
        vi=i%2==0;case=(i//2)%7;a,b,c=[rng.randint(2,999) for _ in range(3)]
        identity={'case':case,'a':a,'b':b,'c':c}
        if case==0:
            expr=f'({a}+{b})/{c}'
            messages=tool_dialogue(('Tính chính xác ' if vi else 'Calculate exactly ')+expr,
                                  'math.calculate',{'expression':expr})
        elif case==1:
            power=rng.randint(2,6);expr=f'{a}*x**{power}+{b}*x+{c}'
            messages=tool_dialogue(('Tính đạo hàm theo x của ' if vi else 'Differentiate with respect to x: ')+expr,
                                  'math.differentiate',{'expression':expr})
            identity['power']=power
        elif case==2:
            matrix=[[a,b],[0,c]]
            question=('Tìm trị riêng của ma trận ' if vi else 'Find the eigenvalues of ')+canonical_json(matrix)
            messages=tool_dialogue(question,'math.eigenvalues',{'matrix':matrix})
        elif case==3:
            # No-tool behavior on a self-contained definition, paired with a random task ID.
            text='Một ma trận Hermitian thỏa mãn A† = A.' if vi else 'A Hermitian matrix satisfies A† = A.'
            question=(f'Bài {a}: phát biểu định nghĩa ma trận Hermitian, không gọi công cụ.' if vi else
                      f'Exercise {a}: state the definition of a Hermitian matrix without tools.')
            messages=[{'role':'user','content':question},{'role':'assistant','content':text}]
            identity={'case':case} # Repeated definitions must not leak across splits.
        elif case in (4,5):
            docid='D'+sha(canonical_json(identity))[:12]
            distractor='D'+sha('distractor'+canonical_json(identity))[:12]
            relevant=f'In experiment {a}, the reported sample size is {b}.'
            question=(f'Thí nghiệm {a} có cỡ mẫu bao nhiêu? Trích dẫn nguồn.' if vi else
                      f'What sample size was reported in experiment {a}? Cite the source.')
            missing=case==5
            evidence=[{'chunk_id':distractor,'text':f'In a different experiment {c}, the instrument was calibrated.'}]
            if not missing:evidence.append({'chunk_id':docid,'text':relevant})
            rng.shuffle(evidence)
            final=(('Chưa đủ bằng chứng trong các đoạn được truy xuất.' if vi else 'The retrieved passages provide insufficient evidence.')
                   if missing else (f'Cỡ mẫu là {b} [{docid}].' if vi else f'The sample size is {b} [{docid}].'))
            messages=[{'role':'user','content':question},
                      {'role':'assistant','content':'','tool_calls':[{'name':'retrieve','arguments':{'query':f'experiment {a} sample size','top_k':3}}]},
                      {'role':'tool','name':'retrieve','content':canonical_json({'ok':True,'result':evidence})},
                      {'role':'assistant','content':final}]
        else:
            # A genuine second observation: not merely a single call-plan target.
            question=(f'Tính ({a}+{b}), sau đó nhân kết quả với {c} bằng hai lượt gọi công cụ.' if vi else
                      f'Calculate ({a}+{b}), then multiply the result by {c} using two tool calls.')
            messages=tool_dialogue(question,'math.calculate',{'expression':f'{a}+{b}'})[:-1]
            args={'expression':f'{a+b}*{c}'};result={'ok':True,'result':compute_math('math.calculate',args)}
            messages.extend([{'role':'assistant','content':'','tool_calls':[{'name':'math.calculate','arguments':args}]},
                             {'role':'tool','name':'math.calculate','content':canonical_json(result)},
                             {'role':'assistant','content':str((a+b)*c)}])
        row=convert({'messages':messages,'tools':SCHEMAS},{'adapter':'messages','min_chars':1,'language':'vi' if vi else 'en'})
        row['group_id']=sha(canonical_json(identity));row['task_family']=str(case)
        yield row

def generate(out,count=1000,seed=123,db_path=None):
    db_path=db_path or str(Path(out)/'synthetic.sqlite')
    store=RecordStore(db_path)
    try:
        for row in generate_records(count,seed):
            store.insert(row,'academic_synthetic',{'generator':'generate_academic.py:v1','seed':seed,
                         'license':'MIT','synthetic':True},seed=42)
        store.db.commit()
        result=store.export('academic_synthetic',out)
    finally:store.db.close()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='data/normalized')
    p.add_argument('--count',type=int,default=10000);p.add_argument('--seed',type=int,default=123)
    a=p.parse_args();print(json.dumps(generate(a.out,a.count,a.seed)))
if __name__=='__main__':main()
