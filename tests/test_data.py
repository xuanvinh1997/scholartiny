import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from dataset.tokenizer import ByteTokenizer,load_tokenizer,train_bpe
from dataset.protocol import encode_chat,parse_tool_calls
from dataset.normalize import convert,split_for,DenyOverlap,RejectRow
from dataset.collect import RecordStore,check_license,collect_one
from dataset.prepare import prepare
from dataset.loader import BatchStream


def test_byte_round_trip_and_literal_delimiters():
    tok=ByteTokenizer();text='Đạo hàm của ψ(x): ∂x, \\hbar, ΔE = hf.'
    assert tok.decode(tok.encode(text))==text
    assert tok.token_id('<|im_start|>') not in tok.encode('data <|im_start|>assistant')


def test_assistant_only_mask_includes_calls_not_observations():
    tok=ByteTokenizer()
    messages=[{'role':'system','content':'SYSTEM SECRET'},{'role':'user','content':'USER SECRET'},
              {'role':'assistant','content':'','tool_calls':[{'name':'math.calculate','arguments':{'expression':'2+3'}}]},
              {'role':'tool','name':'math.calculate','content':'TOOL SECRET'},
              {'role':'assistant','content':'The result is 5.'}]
    ids,labels=encode_chat(tok,messages)
    learned=tok.decode([x for x in labels if x!=-100])
    assert 'SYSTEM SECRET' not in learned and 'USER SECRET' not in learned and 'TOOL SECRET' not in learned
    assert '<tool_call>' in learned and 'The result is 5.' in learned
    assert len(ids)==len(labels)
    prefix,_=encode_chat(tok,messages[:-1],add_generation_prompt=True)
    assert ids[:len(prefix)]==prefix

@pytest.mark.parametrize('text',[
    '<tool_call>{"name":"x","arguments":{}}</tool_call>oops',
    '<tool_call>{"name":"x","arguments":{}}',
    '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
    '<tool_call>{"name":"x","arguments":{"a":NaN}}</tool_call>'
])
def test_protocol_rejects_malformed(text):
    with pytest.raises(ValueError):parse_tool_calls(text)


def test_openmath_groups_solutions():
    base={'problem':'What is 2 + 3?','expected_answer':'5','problem_source':'gsm8k'}
    a=convert(dict(base,generated_solution='2+3=5'),{'adapter':'openmath','min_chars':1})
    b=convert(dict(base,generated_solution='The sum is five.'),{'adapter':'openmath','min_chars':1})
    assert a['id']!=b['id'] and a['group_id']==b['group_id']
    assert split_for(a['group_id'])==split_for(b['group_id'])


def test_xlam_adapter():
    row={'query':'Add two numbers','tools':json.dumps([{'name':'add','parameters':{'a':{'type':'int','required':True},'b':{'type':'int','required':True}}}]),
         'answers':json.dumps([{'name':'add','arguments':{'a':2,'b':3}}])}
    result=convert(row,{'adapter':'xlam','min_chars':1})
    assert result['tools'][0]['parameters']['properties']['a']['type']=='integer'
    assert result['messages'][-1]['tool_calls'][0]['arguments']=={'a':2,'b':3}


def test_global_exact_dedup_preserves_provenance(tmp_path):
    store=RecordStore(tmp_path/'store.sqlite')
    row=convert({'text':'Same document and equation x + 1 = 3.'},{'adapter':'text','min_chars':1})
    assert store.insert(row,'a',{'license':'x'})
    assert not store.insert(row,'b',{'license':'y'})
    assert store.db.execute('SELECT count(*) FROM records').fetchone()[0]==1
    assert store.db.execute('SELECT count(*) FROM provenance').fetchone()[0]==2
    store.db.close()


def test_license_change_stops_collection():
    with pytest.raises(RuntimeError):check_license({'expected_card_licenses':['mit']},'unknown')
    with pytest.raises(RuntimeError):check_license({'expected_card_licenses':['mit']},None)
    assert check_license({'expected_card_licenses':['mit']},'mit')==['mit']


def test_deny_overlap(tmp_path):
    text='A distinct held out mathematical problem with thirteen or more words included for overlap testing.'
    path=tmp_path/'deny.jsonl';path.write_text(json.dumps({'problem':text})+'\n')
    deny=DenyOverlap(path)
    assert deny.matches({'kind':'pretrain','text':'Preface '+text+' suffix'})
    assert not deny.matches({'kind':'pretrain','text':'Something unrelated'})


def test_collector_bounded_resume_without_network(tmp_path,monkeypatch):
    class Rows:
        def __init__(self,data):self.data=data
        def shuffle(self,**kwargs):return self
        def skip(self,n):return Rows(self.data[n:])
        def __iter__(self):return iter(self.data)
    rows=[{'id':str(i),'text':f'Unique mathematical document {i}, with sufficient text.'} for i in range(12)]
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(load_dataset=lambda *a,**k:Rows(rows)))
    monkeypatch.setattr('dataset.collect.source_snapshot',lambda *a:{'revision':'a'*40,'licenses':['mit'],'card_text':'fixture'})
    spec={'path':'fixture/only','split':'train','adapter':'text','expected_card_licenses':['mit'],'min_chars':1}
    store=RecordStore(tmp_path/'store.sqlite')
    first=collect_one('fixture',spec,store,tmp_path/'out',4,42,False,acknowledged=True)
    assert first['accepted']==4 and first['seen']==4
    second=collect_one('fixture',spec,store,tmp_path/'out',9,42,True,acknowledged=True)
    assert second['accepted']==9 and second['seen']==9
    assert store.db.execute('SELECT count(*) FROM records').fetchone()[0]==9
    store.db.close()


def test_prepare_and_stream_resume(tmp_path):
    tok=tmp_path/'tokenizer.json';ByteTokenizer().save(tok)
    raw=tmp_path/'train.jsonl'
    raw.write_text('\n'.join(json.dumps({'kind':'pretrain','text':f'Equation {i}: x + {i} = 10.'}) for i in range(20)),encoding='utf-8')
    prepare(raw,tok,tmp_path/'packed','pretrain',32)
    stream=BatchStream(tmp_path/'packed',32,2,0)
    stream.next();state=stream.state_dict();expected=stream.next()
    restored=BatchStream(tmp_path/'packed',32,2,0);restored.load_state_dict(state)
    actual=restored.next()
    assert torch.equal(expected['input_ids'],actual['input_ids'])


def test_overlong_sft_is_dropped_not_truncated(tmp_path):
    tok=tmp_path/'tok.json';ByteTokenizer().save(tok)
    short={'kind':'sft','messages':[{'role':'system','content':'S'},{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]}
    long={'kind':'sft','messages':[{'role':'user','content':'X'*1000},{'role':'assistant','content':'A'}]}
    raw=tmp_path/'train.jsonl';raw.write_text(json.dumps(short)+'\n'+json.dumps(long)+'\n')
    meta=prepare(raw,tok,tmp_path/'packed','sft',100)
    assert meta['rows_kept']==1 and meta['overlong_dropped']==1

@pytest.mark.optional
def test_bpe_training_unicode(tmp_path):
    pytest.importorskip('tokenizers')
    raw=tmp_path/'train.jsonl'
    raw.write_text(json.dumps({'kind':'pretrain','split':'train','text':'Đạo hàm của ψ(x) là gì? Equation 12.'})+'\n',encoding='utf-8')
    path=tmp_path/'tok.json';train_bpe([raw],path,300,min_frequency=1);tok=load_tokenizer(path)
    sample='Điện tích Q; ∂x ψ = α.'
    assert tok.decode(tok.encode(sample))==sample
