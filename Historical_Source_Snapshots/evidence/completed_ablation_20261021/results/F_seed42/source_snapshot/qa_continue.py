from qa_pipeline_short import *
from types import SimpleNamespace
a=SimpleNamespace(output=Path('/root/data/WorkSpace/output/short_pipeline_qa_v1'))
root=a.output/'continuous'
full=root/'output/leman_short_mechanism_v1/F_seed42'
resumed_path=a.output/'resumed/output/leman_short_mechanism_v1/F_seed42'
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
