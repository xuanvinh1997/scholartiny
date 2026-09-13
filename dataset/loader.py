import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from .normalize import sha
from .protocol import canonical_json

class PreparedDataset:
    def __init__(self,path,seq_len):
        self.path=Path(path)
        if (self.path/'INCOMPLETE').exists(): raise ValueError(f'Incomplete prepared data: {path}')
        self.meta=json.loads((self.path/'meta.json').read_text(encoding='utf-8'))
        self.kind=self.meta['kind'];self.seq_len=seq_len
        if seq_len<2: raise ValueError('seq_len must be at least 2')
        self.tokens=np.memmap(self.path/'tokens.bin',dtype='<u4',mode='r')
        if self.kind=='sft':
            self.labels=np.memmap(self.path/'labels.bin',dtype='<i4',mode='r')
            self.offsets=np.memmap(self.path/'offsets.bin',dtype='<u8',mode='r')
            if int(np.diff(self.offsets).max())>seq_len:
                raise ValueError('SFT sample exceeds training context. Re-prepare with the target limit.')
        if len(self)==0: raise ValueError(f'Dataset too short for seq_len={seq_len}: {path}')

    def __len__(self):
        return (len(self.tokens)-1)//(self.seq_len-1) if self.kind=='pretrain' else len(self.offsets)-1

    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        if self.kind=='pretrain':
            start=index*(self.seq_len-1);end=start+self.seq_len
            ids=torch.tensor(np.asarray(self.tokens[start:end],dtype=np.int64))
            return ids,ids.clone()
        start,end=map(int,self.offsets[index:index+2])
        return (torch.tensor(np.asarray(self.tokens[start:end],dtype=np.int64)),
                torch.tensor(np.asarray(self.labels[start:end],dtype=np.int64)))

class BatchStream:
    """Deterministic, replacement sampling with serializable per-rank RNG.

    Pretrain source weights are approximately token weights because every window
    has equal length. SFT weights are example weights, NOT token percentages.
    Sampling is not an epoch without replacement; count effective epochs separately.
    """
    def __init__(self,data,seq_len,batch_size,pad_id,seed=42,rank=0):
        path=Path(data)
        entries=(yaml.safe_load(path.read_text(encoding='utf-8'))['sources']
                 if path.suffix in ('.yaml','.yml') else [{'path':str(path),'weight':1.0}])
        self.datasets=[PreparedDataset(e['path'],seq_len) for e in entries]
        self.weights=[float(e['weight']) for e in entries]
        if any(w<=0 for w in self.weights) or not self.weights: raise ValueError('Positive mixture weights required')
        if len({d.kind for d in self.datasets})!=1: raise ValueError('Do not mix pretrain and SFT data')
        if len({d.meta['tokenizer_sha256'] for d in self.datasets})!=1: raise ValueError('Tokenizer mismatch')
        self.signature=sha(canonical_json({'meta':[d.meta for d in self.datasets],'weights':self.weights,
                                          'seq_len':seq_len,'batch_size':batch_size}))
        self.rng=random.Random(seed+1000003*rank)
        self.batch_size,self.pad_id=batch_size,pad_id
    def state_dict(self): return {'signature':self.signature,'rng_state':self.rng.getstate()}
    def load_state_dict(self,state):
        if state['signature']!=self.signature: raise ValueError('Dataset/mixture/shape changed on exact resume')
        self.rng.setstate(state['rng_state'])
    def next(self):
        rows=[]
        for _ in range(self.batch_size):
            ds=self.rng.choices(self.datasets,weights=self.weights,k=1)[0]
            rows.append(ds[self.rng.randrange(len(ds))])
        length=max(len(x) for x,_ in rows)
        ids=torch.full((len(rows),length),self.pad_id,dtype=torch.long)
        labels=torch.full_like(ids,-100)
        count=0
        for i,(x,y) in enumerate(rows):
            ids[i,:len(x)]=x;labels[i,:len(y)]=y;count+=len(x)
        return {'input_ids':ids,'labels':labels,'input_tokens':count}
