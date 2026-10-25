"""User-authorized priority test: wait for a safe training stop, test, resume."""
import os,sys,fcntl,json,subprocess,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/data/WorkSpace');HERE=Path(__file__).resolve().parent
BASE=ROOT/'output/leman_short_mechanism_v1';OUT=ROOT/'output/leman_final_v2_test_20261021'
def now():return datetime.now(timezone.utc).isoformat()
def write(n,z):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 assert (BASE/'STOP').exists(),'Expected user-authorized safe stop flag'
 with (OUT/'.priority.lock').open('w') as priority:
  fcntl.flock(priority,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (BASE/'.suite.lock').open('a') as suite:
   write('PRIORITY_STATUS.json',{'phase':'waiting_for_safe_checkpoint','utc':now()});print('Waiting for safe checkpoint and suite exit',flush=True)
   fcntl.flock(suite,fcntl.LOCK_EX)
   # The original queue can only release this lock after its DDP subprocess exits.
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not apps,apps
   import torch
   folder=BASE/'no_boundary_seed42';path=folder/'last.pt';assert not (folder/'COMPLETE.json').exists(),'Configuration completed before pause: inspect resume target'
   ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
   source={p.name:digest(p) for p in (ROOT/'leman_short_mechanism_v1').glob('*.py')}
   assert ck['source_hashes']==source
   assert ck['world_size']==2 and ck['config']['variant']=='no_boundary'
   assert all(k in ck for k in ['model','optimizer','scheduler','ema','rng_by_rank','partial_by_rank'])
   pause={'utc':now(),'variant':'no_boundary','checkpoint':str(path),'checkpoint_sha256':digest(path),'global_step':ck['global_step'],'epoch_zero_based':ck['epoch'],'next_batch':ck['next_batch'],'world_size':ck['world_size'],'source_hashes_match':True,'complete_training_state_present':True}
   write('SAFE_PAUSE.json',pause);del ck
   print('SAFE PAUSE',json.dumps(pause),flush=True)
   write('PRIORITY_STATUS.json',{'phase':'test_running','utc':now(),'paused_global_step':pause['global_step']})
   rc=None
   try:
    with (OUT/'priority_test.log').open('ab',buffering=0) as log:
     rc=subprocess.call([sys.executable,'-u',str(HERE/'test.py')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    print('TEST EXIT',rc,flush=True)
   finally:
    # Resume unchanged original sources, configuration and random/optimizer state.
    assert digest(path)==pause['checkpoint_sha256'],'Paused checkpoint changed during inference'
    assert not (folder/'STOP').exists(),'A separate config STOP was added; do not override it'
    (BASE/'STOP').unlink()
    fcntl.flock(suite,fcntl.LOCK_UN)
    log=(ROOT/'output/short_suite_launcher.log').open('ab',buffering=0)
    child=subprocess.Popen(['bash','run.sh','--group','all','--compact-completed'],cwd=ROOT/'leman_short_mechanism_v1',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();write('RESUME_DISPATCH.json',{'utc':now(),'pid':child.pid,'test_exit_code':rc,'paused_global_step':pause['global_step'],'command':'bash run.sh --group all --compact-completed','original_checkpoint_unchanged':True})
    write('PRIORITY_STATUS.json',{'phase':'training_resume_dispatched','utc':now(),'test_exit_code':rc,'pid':child.pid})
    print('RESUME DISPATCH',child.pid,flush=True)
   if rc:raise SystemExit(rc)
if __name__=='__main__':main()
