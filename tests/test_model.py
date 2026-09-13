from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from model.config import ModelConfig
from model.model_scholar import ScholarLM,optimizer_groups
from model.ssm import Mamba2Reference,new_ssm_cache

def config(**overrides):
    values=dict(vocab_size=37,hidden_size=32,num_hidden_layers=4,intermediate_size=48,
                num_attention_heads=4,num_key_value_heads=2,ssm_backend='reference',
                state_size=4,ssm_expand=2,ssm_head_dim=8,max_seq_len=64,loss_chunk_size=5)
    values.update(overrides)
    return ModelConfig(**values)

@pytest.mark.parametrize('pattern',['MMMA','A'])
def test_causality(pattern):
    torch.manual_seed(7);model=ScholarLM(config(layer_pattern=pattern)).eval()
    ids=torch.randint(0,37,(2,12));other=ids.clone();other[:,7:]=torch.randint(0,37,(2,5))
    with torch.no_grad():
        a=model(ids)['logits'];b=model(other)['logits']
    torch.testing.assert_close(a[:,:7],b[:,:7],rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize('pattern',['MMMA','A'])
def test_cache_matches_full(pattern):
    torch.manual_seed(8);model=ScholarLM(config(layer_pattern=pattern)).eval()
    ids=torch.randint(0,37,(1,13))
    with torch.no_grad():
        full=model(ids)['logits'];cache=model.new_cache(1)
        outputs=[model(ids[:,:4],cache=cache)['logits']]
        for i in range(4,ids.shape[1]):outputs.append(model(ids[:,i:i+1],cache=cache)['logits'])
    torch.testing.assert_close(torch.cat(outputs,1),full,rtol=2e-5,atol=3e-6)
    assert len(cache.attention)==config(layer_pattern=pattern).layers.count('A')
    assert all(c['k'].shape==(1,2,13,8) for c in cache.attention.values())
    for hist,state in cache.ssm.key_value_memory_dict.values():
        assert state.shape==(1,8,8,4)
        assert hist.shape[-1]==4


def test_attention_cached_chunk_mask():
    model=ScholarLM(config(layer_pattern='A')).eval();ids=torch.randint(0,37,(1,14))
    with torch.no_grad():
        full=model(ids)['logits'];cache=model.new_cache()
        model(ids[:,:5],cache=cache)
        actual=model(ids[:,5:],cache=cache)['logits']
    torch.testing.assert_close(actual,full[:,5:],rtol=2e-5,atol=3e-6)


def test_loss_shift_and_mask():
    torch.manual_seed(9);model=ScholarLM(config()).eval()
    ids=torch.randint(0,37,(2,11));labels=ids.clone();labels[:,:4]=-100;labels[1,8:]=-100
    with torch.no_grad():
        logits=model(ids)['logits'];result=model(ids,labels=labels)
        expected=F.cross_entropy(logits[:,:-1].reshape(-1,37),labels[:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(result['loss'],expected)
    assert result['token_count'].item()==11


def test_checkpointing_gradient_parity():
    torch.manual_seed(10);a=ScholarLM(config());b=ScholarLM(config(gradient_checkpointing=True))
    b.load_state_dict(a.state_dict());ids=torch.randint(0,37,(2,9));labels=ids.clone();labels[:,:2]=-100
    a(ids,labels=labels)['loss'].backward();b(ids,labels=labels)['loss'].backward()
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==name2 and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-6)


def test_initialization_and_weight_decay():
    model=ScholarLM(config())
    assert model.lm_head.weight is model.embedding.weight
    excluded={id(p) for g in optimizer_groups(model,.1) if g['weight_decay']==0 for p in g['params']}
    for block in model.layers:
        if block.kind=='M':
            for name in ('A_log','D','dt_bias'):assert id(getattr(block.mixer,name)) in excluded
            delta=F.softplus(block.mixer.dt_bias)
            assert delta.min()>=.00099 and delta.max()<=.101


def test_bad_cache_and_context_fail_closed():
    model=ScholarLM(config()).eval()
    with pytest.raises(ValueError,match='inference-only'):
        model(torch.ones(1,2,dtype=torch.long),cache=model.new_cache())
    with torch.no_grad(),pytest.raises(ValueError,match='Context limit'):
        model(torch.ones(1,65,dtype=torch.long))
    with pytest.raises(ValueError,match='No supervised tokens'):
        model(torch.ones(1,4,dtype=torch.long),labels=torch.full((1,4),-100))


def test_padding_does_not_affect_real_prefix():
    model=ScholarLM(config()).eval();ids=torch.randint(0,37,(1,8))
    with torch.no_grad():
        short=model(ids)['logits']
        padded=model(torch.cat([ids,torch.zeros((1,5),dtype=torch.long)],1))['logits']
    torch.testing.assert_close(short,padded[:,:8],rtol=1e-5,atol=2e-6)


def test_optimizer_can_reduce_tiny_batch_loss():
    torch.manual_seed(30);model=ScholarLM(config());ids=torch.randint(0,37,(1,8))
    opt=torch.optim.AdamW(optimizer_groups(model,.01),lr=.003)
    first=model(ids,labels=ids)['loss'].item()
    for _ in range(6):
        opt.zero_grad();loss=model(ids,labels=ids)['loss'];loss.backward();opt.step()
    assert model(ids,labels=ids)['loss'].item()<first

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_official_mamba2_reference_parity():
    pytest.importorskip('mamba_ssm')
    torch.manual_seed(3)
    cfg=config(hidden_size=128,intermediate_size=256,ssm_head_dim=64,state_size=16,
               num_attention_heads=4,ssm_chunk_size=32)
    from model.ssm import make_ssm
    ref=Mamba2Reference(cfg,0).cuda()
    official=make_ssm(replace(cfg,ssm_backend='mamba2'),0).cuda()
    official.load_state_dict(ref.state_dict(),strict=True)
    x=torch.randn(2,64,128,device='cuda')*.1
    with torch.no_grad():
        expected=ref(x);actual=official(x)
    torch.testing.assert_close(actual,expected,atol=3e-3,rtol=3e-3)

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_official_hybrid_cache_backward():
    pytest.importorskip('mamba_ssm')
    cfg=config(hidden_size=128,intermediate_size=256,ssm_head_dim=64,state_size=16,
               ssm_chunk_size=32,ssm_backend='mamba2')
    model=ScholarLM(cfg).cuda();ids=torch.randint(0,37,(1,32),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(ids,labels=ids)['loss']
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        expected=model(ids)['logits'];cache=model.new_cache()
        model(ids[:,:16],cache=cache)
        for i in range(16,32):actual=model(ids[:,i:i+1],cache=cache)['logits']
    torch.testing.assert_close(actual[:,-1],expected[:,-1],atol=3e-3,rtol=3e-3)
