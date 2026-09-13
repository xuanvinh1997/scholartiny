import json
from pathlib import Path
import torch
from dataset.tokenizer import ByteTokenizer
from dataset.prepare import prepare
from trainer.train import parser,run,trusted_load


def test_exact_training_resume(tmp_path):
    raw=tmp_path/'train.jsonl'
    raw.write_text('\n'.join(json.dumps({'kind':'pretrain','text':f'Academic example {i}: x + 2 = 3. Theory and calculation.'}) for i in range(12)))
    tokenizer=tmp_path/'tok.json';ByteTokenizer().save(tokenizer)
    packed=tmp_path/'packed';prepare(raw,tokenizer,packed,'pretrain',16)
    config=Path(__file__).parents[1]/'configs/debug.yaml'
    common=['--config',str(config),'--tokenizer',str(tokenizer),'--data',str(packed),
            '--seq-len','16','--batch-size','2','--grad-accum','2','--max-steps','4',
            '--warmup-steps','1','--lr','0.001','--device','cpu','--cpu-threads','1','--save-every','10']
    run(parser().parse_args(common+['--out',str(tmp_path/'full')]))
    run(parser().parse_args(common+['--out',str(tmp_path/'resume'),'--stop-after','2']))
    run(parser().parse_args(common+['--out',str(tmp_path/'resume'),'--resume',str(tmp_path/'resume/last.pt')]))
    full=trusted_load(tmp_path/'full/last.pt');resumed=trusted_load(tmp_path/'resume/last.pt')
    assert full['step']==resumed['step']==4
    assert full['supervised_seen']==resumed['supervised_seen']
    for name,value in full['model'].items():torch.testing.assert_close(value,resumed['model'][name],atol=0,rtol=0)
