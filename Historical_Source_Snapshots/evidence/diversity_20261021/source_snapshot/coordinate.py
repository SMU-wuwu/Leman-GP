"""Safely pause the user's existing queue, infer/select/evaluate, always resume."""
import os,fcntl,json,subprocess,sys,time,traceback
from pathlib import Path
from settings import ROOT,OUT,SOURCES,GROUPS
from evidence_core import sha256,write_json
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;BASE=ROOT/'output/leman_short_mechanism_v1'
def now():return datetime.now(timezone.utc).isoformat()
def save(n,z):write_json(OUT/n,z)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.coordinator.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'RESUME_DISPATCH.json').exists(),'This completed interruption must not be repeated'
        for n in SOURCES:assert (OUT/f'PREFLIGHT_{n}.json').exists()
        protocol=dict(created_utc=now(),new_sources=SOURCES,groups=GROUPS,
            model_families=['legacy 180/15/35/L2=2/negative=1.5','balanced 180/15/70/L2=4/negative=1','compact 200/7/100/L2=8/negative=1'],
            threshold_fit=['nested inner-3 OOF','self fitting-fold logits'],total_candidates=18,
            outer_validation='fixed original two text-hash folds; held labels never enter calibrator or threshold fitting',
            selection='highest outer Micro F1; exact ties earlier case; test only if delta vs original >=0.03 percentage points',
            test_policy='freeze one selected version before new test inference; keep every dev outcome; no test-error-guided retuning',
            new_neural_training=False,source_hashes={p.name:sha256(p) for p in HERE.glob('*.py')})
        save('PROTOCOL.json',protocol)
        launcher=(ROOT/'output/short_suite_launcher.log').read_text().splitlines()
        active=[line.split()[1] for line in launcher if line.startswith('START ')][-1]
        folder=BASE/f'{active}_seed42';assert folder.is_dir()
        token='leman_diversity_20261021 '+now()
        with (BASE/'STOP').open('x') as f:f.write(token)
        save('STATUS.json',dict(phase='waiting_for_safe_checkpoint',active=active,utc=now()))
        print('WAIT_SAFE_STOP',active,flush=True)
        with (BASE/'.suite.lock').open('a') as suite:
            fcntl.flock(suite,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            snapshots={};pause=dict(utc=now(),active=active,complete_before_pause=(folder/'COMPLETE.json').exists())
            if not pause['complete_before_pause']:
                import torch
                p=folder/'last.pt';c=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
                source={q.name:sha256(q) for q in (ROOT/'leman_short_mechanism_v1').glob('*.py')}
                assert c['source_hashes']==source and c['world_size']==2 and c['config']['variant']==active
                assert all(k in c for k in ['model','optimizer','scheduler','ema','rng_by_rank','partial_by_rank'])
                snapshots[str(p)]=sha256(p)
                pause.update(global_step=c['global_step'],epoch=c['epoch'],next_batch=c['next_batch'],checkpoint_sha256=snapshots[str(p)],complete_resume_state=True)
                del c
            save('SAFE_PAUSE.json',pause);print('SAFE_PAUSE',json.dumps(pause),flush=True)
            rc=0
            try:
                commands=[['infer.py','--split','dev'],['select_fusion.py'],['infer.py','--split','test'],['test.py']]
                for command in commands:
                    if command[0] in ['test.py'] or command[-1]=='test':
                        if not (OUT/'SELECTED_FREEZE.json').exists():print('NO_DEV_IMPROVEMENT: skip test',flush=True);break
                    save('STATUS.json',dict(phase='running',command=command,utc=now()))
                    print('RUN',command,flush=True)
                    with (OUT/(command[0].replace('.py','')+('_'+command[-1] if len(command)>1 else '')+'.log')).open('ab',buffering=0) as f:
                        rc=subprocess.call([sys.executable,'-u',str(HERE/command[0]),*command[1:]],cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
                    if rc:raise RuntimeError(f'{command} exited {rc}')
            except Exception:
                rc=rc or 1;save('FAILED.json',dict(utc=now(),traceback=traceback.format_exc()));raise
            finally:
                for p,h in snapshots.items():assert sha256(p)==h,'Paused checkpoint changed'
                assert (BASE/'STOP').read_text()==token,'STOP ownership changed'
                assert not (folder/'STOP').exists(),'Separate per-configuration STOP exists'
                (BASE/'STOP').unlink();fcntl.flock(suite,fcntl.LOCK_UN)
                with (ROOT/'output/short_suite_launcher.log').open('ab',buffering=0) as log:
                    child=subprocess.Popen(['bash','run.sh','--group','all','--compact-completed'],cwd=ROOT/'leman_short_mechanism_v1',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save('RESUME_DISPATCH.json',dict(utc=now(),pid=child.pid,experiment_exit_code=rc,paused_state=pause,original_checkpoint_unchanged=True))
                save('STATUS.json',dict(phase='training_resume_dispatched',pid=child.pid,experiment_exit_code=rc,utc=now()))
                print('RESUME_DISPATCH',child.pid,'rc',rc,flush=True)

if __name__=='__main__':main()
