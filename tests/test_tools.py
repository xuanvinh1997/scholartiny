import json
import pytest
from agent.tools import safe_expression,compute_math,ToolExecutor
from agent.rag import build_index,search,citation_ids

@pytest.mark.parametrize('bad',['__import__("os").system("id")','open("/etc/passwd")',
                                 'x.__class__','[x for x in [1]]','2**1000000','1/0'])
def test_math_parser_rejects_code(bad):
    with pytest.raises(Exception):safe_expression(bad)


def test_exact_tools():
    assert compute_math('math.calculate',{'expression':'1/3+1/6'})['exact']=='1/2'
    assert compute_math('math.differentiate',{'expression':'x**3+2*x'})['derivative']=='3*x**2 + 2'
    result=compute_math('math.eigenvalues',{'matrix':[[2,1],[0,3]]})
    assert [x['value'] for x in result['eigenvalues']]==['2','3']


def test_executor_process_and_allowlist():
    executor=ToolExecutor(timeout=10)
    assert executor.execute({'name':'python','arguments':{'code':'print(1)'}})['ok'] is False
    result=executor.execute({'name':'math.calculate','arguments':{'expression':'2+3'}})
    assert result['ok'] and result['result']['exact']=='5'
    assert executor.execute({'name':'math.calculate','arguments':{'expression':'2','extra':1}})['ok'] is False


def test_retrieval_has_provenance_and_ids(tmp_path):
    raw=tmp_path/'docs.jsonl';raw.write_text(json.dumps({'id':'abc123456789','source':'fixture','text':'The resonator energy is E = h f. Photon frequency is f.',
            'provenance':{'url':'https://example.org/paper'}})+'\n')
    build_index([raw],tmp_path/'rag.sqlite')
    found=search(tmp_path/'rag.sqlite','resonator photon')
    assert len(found)==1 and found[0]['url']=='https://example.org/paper'
    assert citation_ids(f"Evidence [{found[0]['chunk_id']}].")=={found[0]['chunk_id']}
    assert search(tmp_path/'rag.sqlite','" OR 1=1 --',top_k=2)==[]
