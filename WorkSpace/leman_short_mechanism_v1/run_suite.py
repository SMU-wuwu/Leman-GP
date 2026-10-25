"""Serial two-GPU queue; complete evidence and compact only this suite's completed resume files."""
import argparse,fcntl,json,os,signal,subprocess,sys,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
from config import VARIANTS
from data import sha256,write_json
HERE=Path(__file__).resolve().parent
CORE=['F','no_lexicon','no_attention','dot_attention','gp_reset','crf','max_words3','max_words5']
EXTRA=['no_char','no_boundary','no_recurrence','no_gru_linear','no_bmes','uniform_lexicon','fixed_gate','no_rope','encoder_gp','no_fgm','clean_two_pass','unit_loss']

def verify():
 m=json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())
 for n,h in m['files'].items():
  if sha256(HERE/n)!=h:raise ValueError('Package changed: '+n)
 return {p.name:sha256(p) for p in HERE.glob('*.py')}

def collect(base):
 files=[p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.npz','.log','.pdf','.svg','.png','.py','.sh','.md') and '.tmp' not in p.name]
 manifest={'files':{str(p.relative_to(base)):sha256(p) for p in files},'weights_included':False,'test_read':False}
 name=base.parent/('return_short_mechanism_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
 with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(base)))
  z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,indent=2))
 print('RETURN',name,sha256(name),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--group',choices=['all','core','mechanisms'],default='all');p.add_argument('--variants',nargs='+',choices=VARIANTS);p.add_argument('--compact-completed',action='store_true');p.add_argument('--collect-only',action='store_true');a=p.parse_args()
 root=Path(a.root);base=root/'output/leman_short_mechanism_v1';base.mkdir(parents=True,exist_ok=True)
 with (base/'.suite.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.collect_only:collect(base);return
  sources=verify();env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',LEMAN_GP_WORLD_SIZE='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
  env.setdefault('CUDA_VISIBLE_DEVICES','0,1')
  active=[None]
  def stop(sig,frame):
   (base/'STOP').touch()
   if active[0]:(active[0]/'STOP').touch()
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  queue=a.variants or (CORE if a.group=='core' else EXTRA if a.group=='mechanisms' else CORE+EXTRA)
  if (base/'STOP').exists():print('STOP: remove only after confirming intended resume');return 75
  if any(v in queue for v in ('no_fgm','clean_two_pass','unit_loss')) and not all((base/f'{v}_seed42/COMPLETE.json').exists() for v in ('fgm_control','weighted_control')):
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   if apps:raise RuntimeError('GPU process already running: '+apps)
   cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(HERE/'import_controls.py'),'--root',str(root)]
   with (base/'reuse_controls.log').open('ab') as f:subprocess.run(cmd,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  for v in queue:
   folder=base/f'{v}_seed42';folder.mkdir(exist_ok=True);active[0]=folder
   if (base/'STOP').exists() or (folder/'STOP').exists():print('STOP: remove only after confirming intended resume');return 75
   receipt=folder/'COMPLETE.json'
   if receipt.exists():
    info=json.loads(receipt.read_text());assert info['source_hashes']==sources,'Completed source mismatch'
    endpoint=json.loads((folder/'PRIMARY_ENDPOINT.json').read_text());assert sha256(folder/'final_ema.pt')==endpoint['sha256']
    print('Verified completed:',v,flush=True)
   else:
    if shutil.disk_usage(base).free<24*2**30:raise RuntimeError('Less than 24 GiB free; archive completed weights before resuming')
    # The queue's own run is serial. Never coexist with unrelated GPU workloads.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('GPU process already running: '+apps)
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(HERE/'train.py'),'--root',str(root),'--variant',v,'--seed','42']
    print('START',v,'log',folder/'train.log',flush=True)
    with (folder/'train.log').open('ab',buffering=0) as f:rc=subprocess.call(cmd,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    if rc:print('Stopped:',v,'return code',rc,flush=True);return rc
    assert receipt.exists()
   if a.compact_completed and not (folder/'COMPACTED.json').exists():
    info=json.loads(receipt.read_text());endpoint=json.loads((folder/'PRIMARY_ENDPOINT.json').read_text())
    assert sha256(folder/'final_ema.pt')==endpoint['sha256']
    if 'best_sha256' in info:assert sha256(folder/'best.pt')==info['best_sha256']
    removed=[]
    for f in [folder/'last.pt',*folder.glob('epoch_*.pt')]:
     if f.exists():removed.append({'name':f.name,'bytes':f.stat().st_size,'sha256':sha256(f)})
    write_json(folder/'COMPACTED.json',{'removed_completed_resume_states':removed,'kept':['best.pt','final_ema.pt'],'reason':'fixed budget completed; intermediate optimizer state no longer needed; explicit CLI option'})
    for row in removed:(folder/row['name']).unlink()
   print('COMPLETE',v,flush=True)
  active[0]=None
  for script,extra in [('analyze.py',['--root',str(root)]),('plot_results.py',['--base',str(base)])]:
   subprocess.run([sys.executable,str(HERE/script),*extra],cwd=HERE,env=env,check=True)
  collect(base)
if __name__=='__main__':raise SystemExit(main())
