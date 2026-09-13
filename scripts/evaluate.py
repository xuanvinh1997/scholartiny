"""Conditional-completion diagnostics on held-out normalized SFT conversations.

Tool plans use the gold history; final answers use gold tool observations. These
are NOT end-to-end agent success rates, semantic faithfulness, or MATH benchmark scores.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from agent.runner import load_model,complete
from agent.rag import citation_ids
from dataset.protocol import parse_tool_calls,canonical_json
from dataset.normalize import normalize_key

def evaluate(model,tok,rows,limit=100,max_new_tokens=256):
    metrics=Counter();predictions=[]
    for row in rows:
        if metrics['examples']>=limit:break
        if row.get('split') not in ('val','test'):raise ValueError('Evaluation accepts held-out val/test rows only')
        if row.get('kind')!='sft':continue
        metrics['examples']+=1
        messages=row['messages'];tools=row.get('tools',[])
        for index,m in enumerate(messages):
            if m['role']!='assistant':continue
            # Evaluate all call turns, and the last assistant answer.
            if not m.get('tool_calls') and index!=len(messages)-1:continue
            prediction=complete(model,tok,messages[:index],tools,max_new_tokens)
            output={'id':row['id'],'turn':index,'prediction':prediction}
            if m.get('tool_calls'):
                metrics['tool_turns']+=1;gold=[{'name':c['name'],'arguments':c['arguments']} for c in m['tool_calls']]
                try:calls=parse_tool_calls(prediction)
                except (ValueError,json.JSONDecodeError):calls=[]
                metrics['parseable_nonempty_tool_plans']+=bool(calls)
                metrics['tool_names_exact']+=bool(calls) and [c['name'] for c in calls]==[c['name'] for c in gold]
                metrics['tool_arguments_and_names_exact']+=bool(calls) and canonical_json(calls)==canonical_json(gold)
                output['gold_calls']=gold
            else:
                metrics['final_answers']+=1
                gold=m.get('content','');metrics['answer_string_exact']+=normalize_key(prediction)==normalize_key(gold)
                known=set()
                for history in messages[:index]:
                    if history['role']=='tool' and history.get('name')=='retrieve':
                        try:
                            value=json.loads(history['content'])
                            known.update(d['chunk_id'] for d in value.get('result',[]) if isinstance(d,dict) and 'chunk_id' in d)
                        except (ValueError,TypeError):pass
                cited=citation_ids(prediction);expected=citation_ids(gold)
                if known or expected:
                    metrics['rag_answers']+=1;metrics['citations_emitted']+=len(cited)
                    metrics['citation_ids_exist']+=len(cited&known)
                    metrics['gold_citations']+=len(expected);metrics['gold_citations_recalled']+=len(expected&cited)
                output['gold_answer']=gold
            predictions.append(output)
    result=dict(metrics)
    for name,num,den in [('tool_plan_validity','parseable_nonempty_tool_plans','tool_turns'),
                         ('tool_selection_exact','tool_names_exact','tool_turns'),
                         ('tool_call_exact','tool_arguments_and_names_exact','tool_turns'),
                         ('answer_exact_match','answer_string_exact','final_answers'),
                         ('citation_id_precision','citation_ids_exist','citations_emitted'),
                         ('citation_id_recall','gold_citations_recalled','gold_citations')]:
        result[name]=metrics[num]/metrics[den] if metrics[den] else None
    result['scope']='Gold-history conditional completion; not end-to-end success or semantic faithfulness'
    return result,predictions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','tokenizer','input','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--limit',type=int,default=100);p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--device',default='cpu');p.add_argument('--backend',choices=['reference','mamba2'])
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');a=p.parse_args()
    model,tok=load_model(a.checkpoint,a.tokenizer,a.device,a.backend,a.precision)
    with open(a.input,encoding='utf-8') as f:
        metrics,predictions=evaluate(model,tok,(json.loads(x) for x in f),a.limit,a.max_new_tokens)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(p,ensure_ascii=False) for p in predictions)+'\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
