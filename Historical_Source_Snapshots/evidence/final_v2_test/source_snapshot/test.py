"""Evaluate the frozen 12-source final-v2 system on test2.txt, with no fitting.

Default: one idle CUDA GPU, BF16 encoder / FP32 custom layers, batch 16.
--preflight: CPU-only frozen-dev decoder replay and resource checks.
--wait-for-ablation: acquire the existing suite lock after all 20 jobs finish.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,copy,csv,fcntl,gc,gzip,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import frozen_pipeline as fp
import torch,joblib
from data import entity_types
from evidence_core import read_bio,read_predictions,score
HERE=Path(__file__).resolve().parent
ROOT=fp.ROOT;FROZEN=fp.FROZEN
OUTPUT=ROOT/'output/leman_final_v2_test_20261021'
NAMES=['old','fgm43','fgm44','ema43','fgm46','soup2','soup3','parent','v4ema','v2ema','rdrop','fgm47']
JOBS=['F','no_lexicon','no_attention','dot_attention','gp_reset','crf','max_words3','max_words5','no_char','no_boundary','no_recurrence','no_gru_linear','no_bmes','uniform_lexicon','fixed_gate','no_rope','encoder_gp','no_fgm','clean_two_pass','unit_loss']

def now():return datetime.now(timezone.utc).isoformat()
def save(n,z):fp.save(OUTPUT/n,z)
def status(phase,**extra):save('STATUS.json',dict(phase=phase,updated_utc=now(),**extra));print(phase,extra,flush=True)
def idle_gpu():
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if apps:raise RuntimeError('GPU process exists; use --wait-for-ablation or run after the GPU is free: '+apps)

def inputs():
 freeze=json.loads((FROZEN/'FREEZE.json').read_text());assert freeze['models']==NAMES
 paths=['analysis_v20.json','DEPENDENCIES.json','v6_train_gazetteer.json','train_boundary_priors.joblib','wide_context6_consensus8/hgb_full_dev.joblib','wide_context6_consensus8/decoder_full.json','wide_context6_consensus8/full_dev_predictions.jsonl','source_snapshot/context_priors_v22.py','source_snapshot/predict_hgb_v22.py']
 hashes={}
 for name in paths:
  actual=fp.digest(FROZEN/name);assert actual==freeze['files'][name],name;hashes[name]=actual
 for name,h in freeze['data_sha256'].items():assert fp.digest(ROOT/'data'/name)==h,name
 lineage=json.loads((ROOT/'output/manuscript_revision_20261020/CANDIDATE_LINEAGE.json').read_text())['sources']
 dependencies=json.loads((FROZEN/'DEPENDENCIES.json').read_text())
 floors={}
 for name in NAMES:
  z=lineage[name];p=Path(z['path']);assert p.is_file() and p.stat().st_size==z['bytes'],name
  cache=FROZEN/'caches'/f'{name}.jsonl.gz';assert fp.digest(cache)==freeze['files'][f'caches/{name}.jsonl.gz']
  assert z['cache_sha256']==freeze['files'][f'caches/{name}.jsonl.gz']
  header=dependencies[name]['header'];floors[name]=header.get('floor',header.get('policy',{}).get('floor'));assert floors[name] is not None
 types=entity_types(ROOT/'data/label.txt')
 lexpath=ROOT/'output/leman_finetune_v4_seed42/lexicon.json';assert fp.digest(lexpath)==dependencies['old']['header']['checkpoint']['lexicon_sha256']
 assets={'hgb':joblib.load(FROZEN/'wide_context6_consensus8/hgb_full_dev.joblib'),'decoder':json.loads((FROZEN/'wide_context6_consensus8/decoder_full.json').read_text())}
 assert assets['hgb'].n_features_in_==99
 gaz=json.loads((FROZEN/'v6_train_gazetteer.json').read_text());priors=joblib.load(FROZEN/'train_boundary_priors.joblib')
 return freeze,lineage,floors,types,lexpath,assets,gaz,priors,hashes

def replay(assets,gaz,priors,types):
 d=fp.load_split(ROOT/'output/leman_finetune_v4_seed42/cache/dev_d7ab938f313842cd.pkl');assert d.types==types
 records=[fp.read_cache(FROZEN/'caches'/f'{n}.jsonl.gz')[1] for n in NAMES]
 blind=[(text,[]) for text,_ in d.samples]
 pred,count=fp.postprocess(records,blind,types,gaz,priors,assets,OUTPUT/'preflight_dev_replay.jsonl')
 expected=fp.read_predictions(FROZEN/'wide_context6_consensus8/full_dev_predictions.jsonl',types)
 assert pred==expected,'Frozen 12-source decoder does not replay exactly'
 result={'exact_predictions':True,'sentences':len(d.samples),'candidate_rows':count,'test_read':False,'new_fitting':False}
 save('PREFLIGHT.json',result);return result

def evaluate(bundle):
 freeze,lineage,floors,types,lexpath,assets,gaz,priors,hashes=bundle
 idle_gpu();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.cuda.set_device(0);device=torch.device('cuda:0')
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False);torch.backends.cudnn.benchmark=False
 script_hashes={p.name:fp.digest(p) for p in HERE.glob('*.py')}
 model_sources={p.name:fp.digest(p) for p in (ROOT/'leman_gp_fgm_seed46_dual4090').glob('*.py')}
 testsha=fp.digest(ROOT/'data/test2.txt');actual_weights={}
 for n in NAMES:
  status('verifying_checkpoint',source=n)
  z=lineage[n];h=fp.digest(z['path']);assert h==z['sha256'],n;actual_weights[n]={'path':z['path'],'sha256':h,'floor':floors[n],'weight_source':z['weight_source']}
 identity={'freeze_sha256':fp.digest(FROZEN/'FREEZE.json'),'test_sha256':testsha,'source_hashes':script_hashes,'model_source_hashes':model_sources,'checkpoint_sources':actual_weights,'asset_hashes':hashes,'precision':'CUDA BF16 encoder / FP32 custom layers; TF32 enabled','batch_size':16,'candidate_order':NAMES,'test_fitting':False}
 if (OUTPUT/'RUN_IDENTITY.json').exists():assert json.loads((OUTPUT/'RUN_IDENTITY.json').read_text())==identity,'Input or code changed since this test run'
 save('RUN_IDENTITY.json',identity)
 ck=torch.load(lineage['old']['path'],map_location='cpu',weights_only=False,mmap=True);cfg=fp.Config()
 for k,v in ck['config'].items():setattr(cfg,k,v)
 cfg.root=str(ROOT);cfg.data_dir=str(ROOT/'data');cfg.encoder_dir=str(ROOT/'model');cfg.output_dir=str(OUTPUT);cfg.preprocessing_workers=4
 lex=json.loads(lexpath.read_text());status('preparing_test')
 cache=fp.prepare_split(cfg,'test',lex,types);dataset=fp.load_split(cache);samples=copy.deepcopy(dataset.samples)
 independent=read_bio(ROOT/'data/test2.txt')
 assert len(independent)==len(samples)
 for (ta,ga),(tb,gb) in zip(samples,independent):assert ta==tb and {(types[t],a,b) for t,a,b in ga}==gb
 # Gold is used only for coverage and final scoring. Clear it from inference inputs.
 dataset.samples=[(text,[]) for text,_ in samples]
 for item in dataset.items:item['spans']=[]
 model=fp.LemanGP(cfg,len(types),ck['vocab_size'],char_vocab_size=ck['char_vocab_size'],encoder_config=ck['encoder_config'],pretrained=False).to(device).eval()
 del ck;gc.collect()
 records=[];timings=[]
 for idx,n in enumerate(NAMES,1):
  path=OUTPUT/'candidates'/f'{n}.jsonl.gz';path.parent.mkdir(exist_ok=True)
  receipt=path.with_suffix('.receipt.json')
  if receipt.exists():
   z=json.loads(receipt.read_text());assert z['sha256']==fp.digest(path)
   head,rows=fp.read_cache(path);assert head['identity']==identity and head['source']==n
   assert set(rows)==set(range(len(samples)));records.append(rows);timings.append(z);continue
  status('test_inference',source=n,source_number=idx,total_sources=12,sentences=len(samples),windows=len(dataset))
  t=time.monotonic();ck=torch.load(lineage[n]['path'],map_location='cpu',weights_only=False,mmap=True);state=ck['model']
  if n in ('ema43','v4ema','v2ema'):state={k:ck['ema']['shadow'].get(k,v) for k,v in state.items()}
  model.load_state_dict(state,strict=True);del ck,state;gc.collect()
  rows=fp.candidates(model,dataset,device,floors[n],batch_size=16)
  torch.cuda.synchronize();seconds=time.monotonic()-t
  tmp=path.with_name(path.name+'.tmp')
  with gzip.open(tmp,'wt',encoding='utf-8') as f:
   f.write(json.dumps({'identity':identity,'source':n,'types':types})+'\n')
   for sid in sorted(rows):f.write(json.dumps({'sid':sid,'rows':rows[sid]})+'\n')
  os.replace(tmp,path);z={'source':n,'seconds':seconds,'sha256':fp.digest(path),'candidate_rows':sum(map(len,rows.values()))};fp.save(receipt,z)
  records.append(rows);timings.append(z)
 del model;gc.collect();torch.cuda.empty_cache()
 status('frozen_postprocessing')
 pred,candidate_count=fp.postprocess(records,dataset.samples,types,gaz,priors,assets,OUTPUT/'test_predictions.jsonl')
 result=fp.score_predictions(samples,pred,types)
 audited=score(independent,read_predictions(OUTPUT/'test_predictions.jsonl',independent),types)
 assert audited['micro']['tp']==result['micro']['true_positive']
 assert audited['micro']['predicted']==result['micro']['predicted']
 assert audited['micro']['support']==result['micro']['support']
 assert abs(audited['micro']['f1']-result['micro']['f1'])<1e-12
 assert abs(audited['macro13']-result['macro_f1'])<1e-12
 # Report split-text overlap transparently; the original test remains the primary endpoint.
 traintext={x[0] for x in read_bio(ROOT/'data/train2.txt')};devtext={x[0] for x in read_bio(ROOT/'data/dev2.txt')}
 cleanids=[i for i,(text,_) in enumerate(independent) if text not in traintext|devtext]
 clean=score([independent[i] for i in cleanids],{j:read_predictions_cache for j,read_predictions_cache in enumerate([set((types[t],a,b) for t,a,b in pred[i]) for i in cleanids])},types)
 overlap={'train_text_overlap_sentences':sum(t in traintext for t,_ in independent),'dev_text_overlap_sentences':sum(t in devtext for t,_ in independent),'disjoint_text_sentences':len(cleanids),'disjoint_text_secondary_metrics':clean}
 save('INDEPENDENT_RECOUNT.json',audited);save('SPLIT_OVERLAP.json',overlap)
 with (OUTPUT/'per_type.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['type',*next(iter(audited['per_type'].values()))]);w.writeheader()
  for n,z in audited['per_type'].items():w.writerow({'type':n,**z})
 result.update({'system':'frozen final-v2 12-source + 99D HGB + weighted interval','split':'test2.txt','test_sha256':testsha,'independent_recount':True,'new_training':False,'new_fitting':False,'thresholds_frozen_on_dev':True,'candidate_rows':candidate_count,'coverage':dataset.stats,'sources':timings,'prediction_sha256':fp.digest(OUTPUT/'test_predictions.jsonl'),'completed_utc':now(),'split_overlap':overlap})
 for n,h in hashes.items():assert fp.digest(FROZEN/n)==h
 assert fp.digest(ROOT/'data/test2.txt')==testsha
 save('TEST_RESULT.json',result);save('COMPLETE.json',{'completed_utc':now(),'test_result_sha256':fp.digest(OUTPUT/'TEST_RESULT.json'),'micro_f1':result['micro']['f1'],'macro13':result['macro_f1'],'new_training':False,'new_fitting':False})
 status('complete',micro_f1=result['micro']['f1'],macro13=result['macro_f1']);print(json.dumps(result['micro']),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--wait-for-ablation',action='store_true');a=p.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 with (OUTPUT/'.test.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUTPUT/'COMPLETE.json').exists():print((OUTPUT/'COMPLETE.json').read_text());return
  suite=None
  try:
   bundle=inputs();replay(bundle[5],bundle[6],bundle[7],bundle[3])
   if a.preflight:print('PREFLIGHT PASSED; no test inference or GPU use',flush=True);return
   if a.wait_for_ablation:
    base=ROOT/'output/leman_short_mechanism_v1';suite=(base/'.suite.lock').open('a')
    status('waiting_for_ablation',condition='suite lock released AND all 20 new jobs complete; no GPU allocation while waiting')
    fcntl.flock(suite,fcntl.LOCK_EX)
    missing=[n for n in JOBS if not (base/f'{n}_seed42/COMPLETE.json').exists()]
    if missing:raise RuntimeError('Ablation exited before all jobs completed; test not launched. Missing: '+','.join(missing))
   evaluate(bundle)
  except Exception as e:
   save('FAILED.json',{'time_utc':now(),'error':str(e),'traceback':traceback.format_exc()});raise
  finally:
   if suite:suite.close()
if __name__=='__main__':main()
