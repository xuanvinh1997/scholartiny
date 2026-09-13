"""Normalize -> token files. No silent SFT truncation or conversation packing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .tokenizer import load_tokenizer,fingerprint
from .protocol import encode_chat

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def prepare(input_path,tokenizer_path,out,kind,max_seq_len=8192):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Prepared output must be new/empty; avoid stale mixed artifacts')
    out.mkdir(parents=True,exist_ok=True)
    tok=load_tokenizer(tokenizer_path)
    counts={'rows_seen':0,'rows_kept':0,'overlong_dropped':0,'no_labels_dropped':0,'tokens':0,'supervised_tokens':0}
    if kind not in ('pretrain','sft'): raise ValueError('Unknown dataset kind')
    try:
        with open(input_path,encoding='utf-8') as src, (out/'tokens.bin').open('wb') as ids_out, \
             (out/'labels.bin').open('wb') as label_out, (out/'offsets.bin').open('wb') as offset_out:
            if kind=='sft': np.array([0],dtype='<u8').tofile(offset_out)
            for line in src:
                row=json.loads(line);counts['rows_seen']+=1
                if row['kind']!=kind: raise ValueError('Dataset kind mismatch')
                if kind=='pretrain':
                    ids=tok.encode(row['text'])+[tok.token_id('<|eos|>')]
                    labels=None
                else:
                    ids,labels=encode_chat(tok,row['messages'],row.get('tools'))
                    if len(ids)>max_seq_len:
                        counts['overlong_dropped']+=1;continue
                    if not any(v!=-100 for v in labels[1:]):
                        counts['no_labels_dropped']+=1;continue
                    np.asarray(labels,dtype='<i4').tofile(label_out)
                    counts['supervised_tokens']+=sum(v!=-100 for v in labels[1:])
                np.asarray(ids,dtype='<u4').tofile(ids_out)
                counts['tokens']+=len(ids);counts['rows_kept']+=1
                if kind=='sft': np.array([counts['tokens']],dtype='<u8').tofile(offset_out)
        if not counts['rows_kept']: raise ValueError(f'No usable samples: {counts}')
        manifest={'format_version':1,'kind':kind,'vocab_size':tok.vocab_size,
                  'tokenizer_sha256':fingerprint(tokenizer_path),'input_sha256':file_sha(input_path),
                  'input_file':str(input_path),'max_seq_len':max_seq_len,**counts,
                  'boundary_policy':'EOS concatenation; no hard resets' if kind=='pretrain' else 'one conversation per row; right padding only'}
        (out/'meta.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        return manifest
    except BaseException:
        (out/'INCOMPLETE').write_text('Preparation failed. Remove this directory and rerun.',encoding='utf-8')
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--out',required=True);p.add_argument('--kind',choices=['pretrain','sft'],required=True)
    p.add_argument('--max-seq-len',type=int,default=8192)
    a=p.parse_args();print(json.dumps(prepare(a.input,a.tokenizer,a.out,a.kind,a.max_seq_len),indent=2))
if __name__=='__main__':main()
