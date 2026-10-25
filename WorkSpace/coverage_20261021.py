"""Candidate-cap coverage from the cached development features, no model inference."""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'short_ablation'))
from config import Config
from data import entity_types,prepare_split,load_split,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=Config.make(a.root,'F',42);lex=json.loads((Path(c.resource_dir)/'lexicon.json').read_text());types=entity_types(Path(c.data_dir)/'label.txt');words={w:i for i,w in enumerate(lex['words'])};rows=[]
 for cap in (3,4,5):
  cfg=replace(c,max_words=cap,output_dir=c.resource_dir);d=load_split(prepare_split(cfg,'dev',lex,types));positions=defaultdict(int);available=set();eligible=set()
  for it in d.items:
   sid,off=it['sid'],it['offset'];text=d.samples[sid][0]
   for pos,ids in enumerate(it['word_ids']):positions[(sid,off+pos)]=max(positions[(sid,off+pos)],sum(int(w)>0 for w in ids))
   for t,s,e in it['spans']:
    ent=(sid,t,off+s,off+e);eligible.add(ent);wid=words.get(text[off+s:off+e],-1)
    if all(any(int(w)==wid and int(b)==(3 if e-s==1 else 0 if k==s else 2 if k==e-1 else 1) for w,b in zip(it['word_ids'][k],it['boundary_ids'][k])) for k in range(s,e)):available.add(ent)
  rows.append({'cap':cap,'unique_characters':len(positions),'candidate_count_histogram':dict(Counter(positions.values())),'gold_entities':len(eligible),'gold_available_all_characters':len(available)})
 write_json(a.out,rows);print(json.dumps(rows))
if __name__=='__main__':main()
