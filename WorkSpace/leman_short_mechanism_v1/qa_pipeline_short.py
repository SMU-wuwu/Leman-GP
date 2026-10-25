"""Run the real short trainer with synthetic parent/data; verify exact interruption recovery."""
from pathlib import Path
import argparse,importlib.util,json,shutil,subprocess,sys,os
import torch
from config import Config
from model import LemanGP
from train import prepare_resources
from data import entity_types,write_json,sha256
from runtime import input_identity,seed_everything

HERE=Path(__file__).resolve().parent

def same(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b,(a,b)

def launch(root,v,stop=None):
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LEMAN_GP_WORLD_SIZE='2',TOKENIZERS_PARALLELISM='false')
 if stop:env['LEMAN_QA_STOP_AFTER']=str(stop)
 else:env.pop('LEMAN_QA_STOP_AFTER',None)
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(HERE/'train.py'),'--root',str(root),'--variant',v,'--qa-config',str(root/'qa_config.json')]
 with (root/f'{v}_{stop or "run"}.log').open('ab') as log:rc=subprocess.call(cmd,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
 if stop:assert rc!=0
 elif rc:raise RuntimeError(f'Trainer failed: {root}/{v}_run.log')
 return root/'output/leman_short_mechanism_v1'/f'{v}_seed42'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('oldqa','/root/data/WorkSpace/leman_controlled_ablation/qa.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 root=a.output/'continuous';old.fixture(root)
 c=Config.make(root,'F',42);over=json.loads((root/'qa_config.json').read_text())
 for k,v in over.items():setattr(c,k,v)
 c.probe_sentences=4;c.workers=0;c.gp_type_weights=[1.,1.]
 types=entity_types(Path(c.data_dir)/'label.txt');identity=input_identity(c);prepare_resources(c,identity,types)
 lexpath=Path(c.resource_dir)/'lexicon.json';lex=json.loads(lexpath.read_text())
 seed_everything(42);m=LemanGP(c,2,len(lex['words']),char_vocab_size=len(lex['characters']))
 parent=a.output/'synthetic_parent.pt'
 torch.save({'config':{'variant':'F','seed':42},'encoder_config':m.encoder.config.to_dict(),'model':m.state_dict(),
              'input_identity':identity,'lexicon_sha256':sha256(lexpath)},parent)
 over.update(warm_start=str(parent),warm_start_sha256=sha256(parent),probe_sentences=4,workers=0)
 (root/'qa_config.json').write_text(json.dumps(over))
 resumed=a.output/'resumed';shutil.copytree(root,resumed)
 full=launch(root,'F');part=launch(resumed,'F',2)
 ck=torch.load(part/'last.pt',weights_only=False,map_location='cpu');assert ck['global_step']==2;del ck
 resumed_path=launch(resumed,'F')
 a0=torch.load(full/'last.pt',weights_only=False,map_location='cpu');b0=torch.load(resumed_path/'last.pt',weights_only=False,map_location='cpu')
 for k in ('model','scheduler','ema','best_f1','best_epoch','global_step','epoch','next_batch'):same(a0[k],b0[k])
 optmax=0.
 for k,values in a0['optimizer']['state'].items():
  for field,value in values.items():
   other=b0['optimizer']['state'][k][field]
   if torch.is_tensor(value):
    optmax=max(optmax,float((value-other).abs().max()))
    torch.testing.assert_close(value,other,atol=1e-9,rtol=1e-5)
   else:assert value==other
 same(a0['optimizer']['param_groups'],b0['optimizer']['param_groups'])
 for x,y in zip(a0['history'],b0['history']):same(x['train_loss'],y['train_loss']);same(x['dev_ema']['micro'],y['dev_ema']['micro'])
 for v in ('crf','no_boundary','no_gru_linear'):
  out=launch(root,v);assert (out/'PRIMARY_ENDPOINT.json').exists() and (out/'mechanisms/epoch_002_ema/PANEL.json').exists()
 write_json(a.output/'QA_RESULT.json',{'synthetic_only':True,'F_exact_resume_model_scheduler_ema':True,'optimizer_max_absolute_difference':optmax,'optimizer_tolerance':{'atol':1e-9,'rtol':1e-5},'optimizer_not_bitwise_identical':optmax>0,'probes_run_inside_real_trainer':True,'CRF_no_boundary_no_gru_linear_trainers':True,'fixed_ema_and_best_exported':True})
if __name__=='__main__':main()
