"""Small, offline SQLite FTS5 retriever. This is lexical BM25, not dense/GraphRAG."""
import json
from pathlib import Path
import re
import sqlite3
from dataset.normalize import sha

def build_index(input_paths,db_path,chunk_words=180,overlap=40,limit=0):
    if not 0<=overlap<chunk_words:raise ValueError('Expected 0 <= overlap < chunk_words')
    path=Path(db_path)
    if path.exists():raise FileExistsError('Refusing to overwrite an existing retrieval index')
    path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path)
    db.execute('CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED,source UNINDEXED,url UNINDEXED,text,tokenize="unicode61")')
    count,docs=0,0
    try:
        for input_path in input_paths:
            with open(input_path,encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if not row.get('text'):continue
                    words=row['text'].split();doc_id=row.get('id',sha(row['text']))
                    for start in range(0,len(words),chunk_words-overlap):
                        chunk=' '.join(words[start:start+chunk_words])
                        cid=f"D{doc_id[:12]}_{start}"
                        db.execute('INSERT INTO chunks VALUES(?,?,?,?)',(cid,row.get('source',str(input_path)),
                                   (row.get('provenance') or {}).get('url') or '',chunk));count+=1
                        if start+chunk_words>=len(words):break
                    docs+=1
                    if limit and docs>=limit:break
            if limit and docs>=limit:break
        db.commit()
    finally:db.close()
    return {'documents':docs,'chunks':count,'retrieval':'SQLite FTS5 BM25'}

def search(db_path,query,top_k=3):
    if not 1<=top_k<=5:raise ValueError('top_k out of range')
    words=re.findall(r'\w+',query,flags=re.UNICODE)[:24]
    if not words:return []
    expr=' OR '.join('"'+word+'"' for word in words)
    uri=Path(db_path).resolve().as_uri()+'?mode=ro'
    db=sqlite3.connect(uri,uri=True)
    try:
        rows=db.execute('SELECT chunk_id,source,url,text,bm25(chunks) FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?',
                        (expr,top_k)).fetchall()
        return [dict(zip(['chunk_id','source','url','text','bm25'],r)) for r in rows]
    finally:db.close()

def citation_ids(text):return set(re.findall(r'\[([A-Za-z0-9_-]+)\]',text))
