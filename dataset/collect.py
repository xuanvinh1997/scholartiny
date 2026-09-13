"""Bounded HF streaming collector with snapshot provenance and transactional resume."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import yaml
from .normalize import convert,RejectRow,split_for,DenyOverlap,sha
from .protocol import canonical_json

class RecordStore:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,group_id TEXT,source TEXT,split TEXT,payload TEXT);
        CREATE INDEX IF NOT EXISTS idx_source_split ON records(source,split);
        CREATE TABLE IF NOT EXISTS provenance(record_id TEXT,source TEXT,payload TEXT,
                                             PRIMARY KEY(record_id,source));
        CREATE TABLE IF NOT EXISTS runs(source TEXT PRIMARY KEY,payload TEXT);
        ''')
    def state(self,name):
        r=self.db.execute('SELECT payload FROM runs WHERE source=?',(name,)).fetchone()
        return json.loads(r[0]) if r else None
    def save_state(self,name,state):
        self.db.execute('INSERT OR REPLACE INTO runs VALUES(?,?)',(name,canonical_json(state)))
    def insert(self,record,name,metadata,val_fraction=.01,test_fraction=.01,seed=42):
        split=split_for(record['group_id'],val_fraction,test_fraction,seed)
        record=dict(record,source=name,split=split,provenance=metadata)
        self.db.execute('INSERT OR IGNORE INTO provenance VALUES(?,?,?)',
                        (record['id'],name,canonical_json(metadata)))
        cur=self.db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?)',
                            (record['id'],record['group_id'],name,split,canonical_json(record)))
        return cur.rowcount==1
    def export(self,name,out):
        dest=Path(out)/name;dest.mkdir(parents=True,exist_ok=True)
        counts={}
        for split in ('train','val','test'):
            tmp=dest/(split+'.jsonl.tmp');n=0
            with tmp.open('w',encoding='utf-8') as f:
                for (payload,) in self.db.execute('SELECT payload FROM records WHERE source=? AND split=? ORDER BY id',(name,split)):
                    f.write(payload+'\n');n+=1
            tmp.replace(dest/(split+'.jsonl'));counts[split]=n
        state=self.state(name) or {}
        (dest/'manifest.json').write_text(json.dumps(dict(state,exported=counts),ensure_ascii=False,indent=2),encoding='utf-8')
        return counts


def check_license(spec,licenses):
    licenses=[licenses] if isinstance(licenses,str) else list(licenses or [])
    allowed=set(spec.get('expected_card_licenses',[]))
    if not licenses or not set(licenses).issubset(allowed):
        raise RuntimeError(f"Dataset card license changed/missing: {licenses}; expected {sorted(allowed)}. Review manually.")
    return licenses

def source_snapshot(spec,token):
    from huggingface_hub import HfApi,hf_hub_download
    info=HfApi().dataset_info(spec['path'],revision=spec.get('revision','main'),token=token)
    card=info.card_data.to_dict() if info.card_data else {}
    licenses=check_license(spec,card.get('license'))
    readme=hf_hub_download(spec['path'],'README.md',repo_type='dataset',revision=info.sha,token=token)
    return {'revision':info.sha,'licenses':licenses,'card_text':Path(readme).read_text(encoding='utf-8'),
            'dataset_url':f"https://huggingface.co/datasets/{spec['path']}",
            'retrieved_at':datetime.now(timezone.utc).isoformat()}

def collect_one(name,spec,store,out,limit,seed,resume,deny=None,acknowledged=False):
    if not acknowledged:
        raise RuntimeError('Read dataset cards and terms; pass --acknowledge-source-terms after review. '
                           'Public access does not imply unrestricted reuse.')
    token=os.environ.get('HF_TOKEN')
    signature=sha(canonical_json({'spec':spec,'seed':seed,'normalizer_version':1,
                                 'deny':sorted((deny.exact|deny.ngrams) if deny else [])}))
    state=store.state(name)
    if state:
        if not resume: raise RuntimeError(f'{name} already exists; use --resume or a new database')
        if state['signature']!=signature: raise RuntimeError('Source/filter/seed changed; use a fresh data directory')
    else:
        snapshot=source_snapshot(spec,token)
        state={'signature':signature,'spec':spec,'snapshot':snapshot,'seen':0,'accepted':0,'rejected':{},'seed':seed}
        store.save_state(name,state);store.db.commit()
    from datasets import load_dataset
    ds=load_dataset(spec['path'],name=spec.get('config'),split=spec['split'],streaming=True,
                    revision=state['snapshot']['revision'],token=token)
    if spec.get('shuffle_buffer',0): ds=ds.shuffle(seed=seed,buffer_size=spec['shuffle_buffer'])
    ds=ds.skip(state['seen']) # Same pinned snapshot and shuffle before skipping.
    rejected=Counter(state['rejected'])
    try:
        for raw in ds:
            if state['accepted']>=limit: break
            state['seen']+=1
            try:
                row=convert(raw,spec)
                if deny and deny.matches(row): raise RejectRow('Benchmark overlap')
                meta=raw.get('metadata') or {}
                provenance={'dataset':spec['path'],'config':spec.get('config'),'upstream_split':spec['split'],
                            'revision':state['snapshot']['revision'],'card_licenses':state['snapshot']['licenses'],
                            'upstream_id':raw.get('id',raw.get('prompt_id')),
                            'url':raw.get('url',meta.get('url') if isinstance(meta,dict) else None),
                            'title':raw.get('title'),'seen_index':state['seen']-1}
                inserted=store.insert(row,name,provenance,spec.get('val_fraction',.01),spec.get('test_fraction',.01),seed)
                if inserted: state['accepted']+=1
                else: rejected['Exact duplicate']+=1
            except (RejectRow,KeyError,TypeError,json.JSONDecodeError) as exc:
                rejected[str(exc)]+=1
            if state['seen']%128==0:
                state['rejected']=dict(rejected);store.save_state(name,state);store.db.commit()
                print(json.dumps({'source':name,'seen':state['seen'],'accepted':state['accepted']}),flush=True)
            if state['seen']>=spec.get('max_seen',max(limit*20,10000)): break
    finally:
        state['rejected']=dict(rejected);store.save_state(name,state);store.db.commit()
        store.export(name,out)
    if state['accepted']==0: raise RuntimeError(f'{name}: no accepted samples; inspect manifest/schema/access')
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='configs/data_sources.yaml')
    p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--out',default='data/normalized');p.add_argument('--db',default='data/collection.sqlite')
    p.add_argument('--limit',type=int,default=10000,help='Total accepted rows per source including resumed rows')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p.add_argument('--deny-file');p.add_argument('--acknowledge-source-terms',action='store_true')
    args=p.parse_args()
    if args.limit<1: p.error('--limit must be positive')
    sources=yaml.safe_load(Path(args.manifest).read_text(encoding='utf-8'))['sources']
    store=RecordStore(args.db);deny=DenyOverlap(args.deny_file)
    try:
        for name in args.sources:
            if name not in sources: raise ValueError(f'Unknown source {name}')
            collect_one(name,sources[name],store,args.out,args.limit,args.seed,args.resume,deny,args.acknowledge_source_terms)
    finally: store.db.close()

if __name__=='__main__': main()
