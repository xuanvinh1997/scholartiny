import hashlib
import json
import re
import unicodedata
from .protocol import canonical_json

class RejectRow(ValueError): pass

def normalize_key(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFC',str(text))).strip()

def sha(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def split_for(group_key,val_fraction=.01,test_fraction=.01,seed=42):
    if not 0 <= val_fraction+test_fraction < 1: raise ValueError('Invalid split fractions')
    value = int(sha(f'{seed}:{group_key}')[:16],16)/2**64
    return 'test' if value<test_fraction else ('val' if value<test_fraction+val_fraction else 'train')

def decode_json(value): return json.loads(value) if isinstance(value,str) else value

def xlam_tools(raw):
    types = {'int':'integer','float':'number','str':'string','bool':'boolean','list':'array','dict':'object'}
    result=[]
    for tool in decode_json(raw):
        params=tool.get('parameters',{})
        if params.get('type')=='object' and 'properties' in params:
            schema=params
        else:
            properties,required={},[]
            for name,desc in params.items():
                desc=dict(desc)
                if desc.pop('required',False): required.append(name)
                desc['type']=types.get(desc.get('type'),desc.get('type','string'))
                if desc['type'] not in ('integer','number','string','boolean','array','object','null'):
                    raise RejectRow('Unsupported xLAM parameter type')
                properties[name]=desc
            schema={'type':'object','properties':properties,'required':required,'additionalProperties':False}
        result.append({'name':tool['name'],'description':tool.get('description',''),'parameters':schema})
    return result

def convert(row,spec):
    adapter=spec['adapter']
    out={'tools':[]}
    if adapter=='text':
        text=row.get(spec.get('text_field','text'))
        if not isinstance(text,str): raise RejectRow('Missing text field')
        # Preserve indentation, LaTeX, equations and Unicode; only normalize newlines.
        out.update(kind='pretrain',text=unicodedata.normalize('NFC',text).replace('\r\n','\n').strip())
        group=normalize_key(out['text'])
    elif adapter=='openmath':
        if 'test' in str(row.get('problem_source','')).lower(): raise RejectRow('Test-derived math source')
        problem,solution=row.get('problem'),row.get('generated_solution')
        if not isinstance(problem,str) or not isinstance(solution,str): raise RejectRow('Bad math schema')
        out.update(kind='sft',messages=[{'role':'user','content':problem},{'role':'assistant','content':solution}],
                   expected_answer=row.get('expected_answer'),problem_source=row.get('problem_source'))
        group=normalize_key(problem) # All solutions to one problem stay in one split.
    elif adapter=='xlam':
        calls=decode_json(row['answers']); tools=xlam_tools(row['tools'])
        names={t['name'] for t in tools}
        if not isinstance(calls,list): raise RejectRow('Expected a list of calls')
        for call in calls:
            if call.get('name') not in names or not isinstance(call.get('arguments'),dict):
                raise RejectRow('Invalid tool reference')
        # Empty answers without a textual target cannot train useful refusal behavior.
        if not calls: raise RejectRow('No supervised call or answer')
        out.update(kind='sft',tools=tools,messages=[{'role':'user','content':row['query']},
                   {'role':'assistant','content':'','tool_calls':calls}])
        group=normalize_key(row['query'])
    elif adapter=='messages':
        messages=row.get('messages')
        if not isinstance(messages,list): raise RejectRow('Missing messages')
        if not any(m.get('role')=='assistant' for m in messages): raise RejectRow('No assistant')
        if any(m.get('role') not in ('system','user','assistant','tool') for m in messages):
            raise RejectRow('Unknown role')
        if any(not isinstance(m.get('content',''),str) for m in messages): raise RejectRow('Non-text message')
        out.update(kind='sft',messages=messages,tools=row.get('tools',[]))
        users=[m['content'] for m in messages if m['role']=='user']
        if not users: raise RejectRow('No user question')
        group=normalize_key(users[0])
    else: raise ValueError(f'Unknown adapter: {adapter}')
    payload=out.get('text') or canonical_json(out['messages'])
    if not spec.get('min_chars',40) <= len(payload) <= spec.get('max_chars',200000):
        raise RejectRow('Length filter')
    if '\x00' in payload: raise RejectRow('NUL byte')
    for key,values in spec.get('filters',{}).items():
        if row.get(key) not in values: raise RejectRow(f'Filter {key}')
    out['group_id']=sha(group)
    out['id']=sha(canonical_json({'text':normalize_key(out.get('text','')),
                               'messages':out.get('messages'), 'tools':out.get('tools')}))
    out['language']=spec.get('language','unknown')
    return out

class DenyOverlap:
    """Exact normalized and 13-word overlap checks, NOT semantic decontamination."""
    def __init__(self,path=None,n=13):
        self.n,self.exact,self.ngrams=n,set(),set()
        if path:
            with open(path,encoding='utf-8') as f:
                for line in f:
                    obj=json.loads(line); text=obj.get('problem') or obj.get('text') or obj.get('question')
                    if not text: raise ValueError('Deny rows need problem, question or text')
                    norm=normalize_key(text).casefold()
                    self.exact.add(sha(norm))
                    self.ngrams.update(self.grams(norm))
    def grams(self,text):
        words=text.split()
        return (sha(' '.join(words[i:i+self.n])) for i in range(len(words)-self.n+1))
    def matches(self,row):
        texts=[row['text']] if row['kind']=='pretrain' else [m.get('content','') for m in row['messages']]
        for text in texts:
            norm=normalize_key(text).casefold()
            if sha(norm) in self.exact or any(g in self.ngrams for g in self.grams(norm)): return True
        return False
