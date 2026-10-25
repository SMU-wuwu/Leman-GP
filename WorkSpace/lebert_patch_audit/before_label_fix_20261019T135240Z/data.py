"""AgCNER flat-NER data; all evaluation gold retained in original coordinates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_bio(path):
    rows=[];chars=[];tags=[]
    def finish():
        if not chars:return
        gold=[];active=None
        for i,t in enumerate(tags+['O']):
            typ=t[2:] if t.startswith(('B-','I-')) else None
            if active and (not t.startswith('I-') or typ!=active[0]):gold.append((active[0],active[1],i));active=None
            if typ and active is None:active=(typ,i)
        rows.append((''.join(chars),gold));chars.clear();tags.clear()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():finish();continue
        token,tag=line.split();tag=tag.replace('_','-')
        chars.extend(token);tags.extend([tag]+[('I-'+tag[2:]) if tag.startswith('B-') else tag]*(len(token)-1))
    finish();return rows


def label_names(types):return ['O']+[f'{tag}-{t}' for t in types for tag in ('B','I','E','S')]


def extract_spans(labels):
    spans=[];active=None
    for i,tag in enumerate(labels+['O']):
        prefix=tag[0];kind=tag[2:] if tag!='O' else None
        if active and (prefix not in ('I','E') or kind!=active[0]):spans.append((active[0],active[1],i));active=None
        if prefix=='S':spans.append((kind,i,i+1))
        elif prefix in ('B','I','E'):
            if active is None:active=(kind,i)
            if prefix=='E':spans.append((active[0],active[1],i+1));active=None
    return spans


class WindowDataset(Dataset):
    def __init__(self,rows,tokenizer,word_ids,types,max_chars=254,overlap=64,max_words=5):
        self.samples=rows;self.items=[];self.labels=label_names(types);labelids={v:i for i,v in enumerate(self.labels)}
        lengths=sorted({len(w) for w in word_ids if 0<len(w)<=max_chars},reverse=True)
        self.uncovered_gold=0
        for sid,(text,gold) in enumerate(rows):
            starts=list(range(0,max(1,len(text)-max_chars+1),max_chars-overlap))
            if starts[-1]+max_chars<len(text):starts.append(max(0,len(text)-max_chars))
            self.uncovered_gold+=sum(not any(s<=a and b<=s+max_chars for s in starts) for _,a,b in gold)
            for start in sorted(set(starts)):
                part=text[start:start+max_chars];n=len(part);matches=[[] for _ in part]
                for a in range(n):
                    for length in lengths:
                        if a+length>n:continue
                        wid=word_ids.get(part[a:a+length])
                        if wid is not None:
                            for pos in range(a,a+length):matches[pos].append((-length,a,wid))
                lexical=[[w for _,_,w in sorted(set(m))[:max_words]] for m in matches]
                labs=['O']*n
                for t,a,b in gold:
                    if start<=a<b<=start+n:
                        a-=start;b-=start
                        if b-a==1:labs[a]='S-'+t
                        else:
                            labs[a]='B-'+t;labs[b-1]='E-'+t
                            for j in range(a+1,b-1):labs[j]='I-'+t
                # The author model is character-based BERT, not wordpiece span pooling.
                ids=tokenizer.convert_tokens_to_ids(list(part))
                self.items.append({'sid':sid,'offset':start,'ids':[tokenizer.cls_token_id]+ids+[tokenizer.sep_token_id],
                    'words':[[]]+lexical+[[]],'labels':[labelids[x] for x in labs]})
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]


def collate(rows):
    b=len(rows);n=max(len(r['labels']) for r in rows);w=max(1,max((len(x) for r in rows for x in r['words']),default=0))
    input_ids=torch.zeros(b,n+2,dtype=torch.long);token_mask=torch.zeros(b,n+2,dtype=torch.bool)
    words=torch.zeros(b,n+2,w,dtype=torch.long);mask=torch.zeros(b,n,dtype=torch.bool);labels=torch.zeros(b,n,dtype=torch.long)
    for i,r in enumerate(rows):
        length=len(r['labels']);input_ids[i,:length+2]=torch.tensor(r['ids']);token_mask[i,:length+2]=True;mask[i,:length]=True
        labels[i,:length]=torch.tensor(r['labels'])
        for j,ids in enumerate(r['words']):
            if ids:words[i,j,:len(ids)]=torch.tensor(ids)
    return {'input_ids':input_ids,'token_mask':token_mask,'word_ids':words,'char_mask':mask,'labels':labels,
            'meta':[(r['sid'],r['offset']) for r in rows]}


def score(rows,pred,types):
    counts={t:[0,0,0] for t in types}
    for sid,(_,entities) in enumerate(rows):
        gold=set(entities);p=pred.get(sid,set())
        for j,s in enumerate((gold&p,p,gold)):
            for t,_,_ in s:counts[t][j]+=1
    def metric(v):
        t,p,g=v;return {'tp':t,'fp':p-t,'fn':g-t,'precision':t/p if p else 0.,'recall':t/g if g else 0.,'f1':2*t/(p+g) if p+g else 0.,'support':g}
    pt={t:metric(v) for t,v in counts.items()}
    return {'micro':metric([sum(v[i] for v in counts.values()) for i in range(3)]),'macro_f1':sum(v['f1'] for v in pt.values())/len(types),'per_type':pt}
