"""Independent BIO recount, text-cluster paired uncertainty and error mechanisms."""
import argparse,csv,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from evidence_core import read_bio,read_predictions,score,write_json,sha256

def csvout(path,rows):
 if not rows:return
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def counts(samples,pred):return np.asarray([[len(g&pred[i]),len(pred[i]),len(g)] for i,(_,g) in enumerate(samples)],dtype=np.int64)
def bootstrap(samples,a,b,n=4000):
 ids={};clusters=[]
 for i,(text,_) in enumerate(samples):
  if text not in ids:ids[text]=len(clusters);clusters.append([])
  clusters[ids[text]].append(i)
 aa=np.array([a[c].sum(0) for c in clusters]);bb=np.array([b[c].sum(0) for c in clusters]);r=np.random.default_rng(20261020);d=[]
 for _ in range(n):
  k=r.integers(0,len(clusters),len(clusters));x=aa[k].sum(0);y=bb[k].sum(0)
  d.append((2*y[0]/max(1,y[1]+y[2])-2*x[0]/max(1,x[1]+x[2]))*100)
 return {'low_pp':float(np.quantile(d,.025)),'high_pp':float(np.quantile(d,.975)),'replicates':n,'text_clusters':len(clusters)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--base',type=Path);p.add_argument('--out',type=Path);p.add_argument('--bootstrap',type=int,default=4000);a=p.parse_args()
 base=a.base or Path(a.root)/'output/leman_short_mechanism_v1';out=a.out or base/'analysis';out.mkdir(parents=True,exist_ok=True)
 raw=Path(a.root)/'data';samples=read_bio(raw/'dev2.txt');train=read_bio(raw/'train2.txt');types=sorted({t for _,g in samples for t,_,_ in g})
 freq=Counter(text[s:e] for text,g in train for _,s,e in g);metrics=[];sub=[];errors=[];trans=[];cases=[];curves=[];pairs=[];loaded={};manifest={}
 sources=[]
 for v in sorted((base/'fixed_interventions').glob('*')):
  if (v/'dev_predictions.jsonl').exists():sources.append(('fixed',v.name,v/'dev_predictions.jsonl'))
 for v in sorted(base.glob('*_seed*')):
  if (v/'PRIMARY_ENDPOINT.json').exists():
   j=json.loads((v/'PRIMARY_ENDPOINT.json').read_text());pred=v/f'epoch_{j["epoch"]:03d}_ema/dev_predictions.jsonl'
   if pred.exists():sources.append(('adapted',v.name.split('_seed')[0],pred))
  if (v/'history.json').exists():
   for row in json.loads((v/'history.json').read_text()):
    for w in ('live','ema'):
     z=row['dev_'+w];curves.append({'variant':v.name,'epoch':row['epoch'],'weight_source':w,'micro_f1':z['micro']['f1']*100,'macro_f1':z['macro_f1']*100,'loss':row['train_loss']})
 for scope,v,path in sources:
  pred=read_predictions(path,samples);loaded[(scope,v)]=pred;manifest[str(path)]=sha256(path);z=score(samples,pred,types)
  metrics.append({'scope':scope,'variant':v,**{k:x for k,x in z['micro'].items()},'macro13':z['macro13']})
  buckets=defaultdict(lambda:[0,0]);err=Counter()
  for i,(text,gold) in enumerate(samples):
   for entity in gold:
    t,s,e=entity;f=freq[text[s:e]]
    groups=['seen' if f else 'unseen','length_'+('1-2' if e-s<=2 else '3-5' if e-s<=5 else '6-10' if e-s<=10 else '11+'),
            'frequency_'+('0' if not f else '1-5' if f<=5 else '6-20' if f<=20 else '21+'),'type_'+t]
    for group in groups:buckets[group][0]+=entity in pred[i];buckets[group][1]+=1
   for t,s,e in gold-pred[i]:
    label=('type_confusion' if any(aa==s and bb==e for _,aa,bb in pred[i]) else 'boundary_shift' if any(tt==t and max(s,aa)<min(e,bb) for tt,aa,bb in pred[i]) else 'missed')
    err['FN_'+label]+=1
   for t,s,e in pred[i]-gold:
    label=('type_confusion' if any(aa==s and bb==e for _,aa,bb in gold) else 'boundary_shift' if any(tt==t and max(s,aa)<min(e,bb) for tt,aa,bb in gold) else 'spurious')
    err['FP_'+label]+=1
  for g,(tp,n) in sorted(buckets.items()):sub.append({'scope':scope,'variant':v,'group':g,'tp':tp,'support':n,'recall':tp/n})
  for k,n in err.items():errors.append({'scope':scope,'variant':v,'category':k,'count':n})
 for (scope,v),pred in loaded.items():
  ref=('fgm_control' if v in ('no_fgm','clean_two_pass') else 'weighted_control' if v=='unit_loss' else 'gp_reset' if v=='crf' else 'F')
  if (scope,ref) not in loaded or ref==v:continue
  control=loaded[(scope,ref)];delta=(score(samples,pred,types)['micro']['f1']-score(samples,control,types)['micro']['f1'])*100
  pairs.append({'scope':scope,'variant':v,'reference':ref,'delta_micro_pp':delta,**bootstrap(samples,counts(samples,control),counts(samples,pred),a.bootstrap)})
  c=Counter()
  for i,(text,gold) in enumerate(samples):
   for t,s,e in gold:
    ent=(t,s,e);old=ent in control[i];new=ent in pred[i];label=('retained' if old and new else 'lost' if old else 'rescued' if new else 'missed_both')
    c[(t,label)]+=1
    if label in ('lost','rescued'):cases.append({'scope':scope,'variant':v,'reference':ref,'sid':i,'type':t,'start':s,'end':e,'entity':text[s:e],'transition':label,'seen':bool(freq[text[s:e]]),'sentence':text})
  for (t,k),n in c.items():trans.append({'scope':scope,'variant':v,'reference':ref,'type':t,'transition':k,'count':n})
 for name,rows in [('metrics',metrics),('subgroup_recall',sub),('error_categories',errors),('gold_transitions',trans),('changed_cases',cases),('training_curves',curves),('paired_bootstrap',pairs)]:csvout(out/(name+'.csv'),rows)
 write_json(out/'AUDIT.json',{'data_hashes':{n:sha256(raw/n) for n in ('train2.txt','dev2.txt')},'prediction_hashes':manifest,
   'sentences':len(samples),'gold':sum(len(g) for _,g in samples),'models':len(sources),'test_read':False,
   'scope':'conditional fixed/adapted dev comparisons; bootstrap is sample uncertainty, not training seed variance',
   'boundary_error_definition':'disjoint priority: same span wrong type, otherwise same-type overlap, otherwise missed/spurious; not a one-to-one alignment'})
 print(json.dumps({'models':len(sources),'output':str(out)}))
if __name__=='__main__':main()
