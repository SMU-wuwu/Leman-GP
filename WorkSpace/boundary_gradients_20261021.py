"""No-update gradients at the shared recurrent representation, fixed train panel."""
import argparse,hashlib,json,sys,csv
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'short_ablation'))
from frozen_mechanisms import state_digest
from config import Config
from model import LemanGP
from data import load_split,entity_types,Collator,sha256,write_json
from train import prepare_resources
from parent_model import boundary_loss
from runtime import amp_context,to_device

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);device=torch.device('cuda:0');c=Config.make(a.root,'F',42);c.workers=0
 run=Path(c.output_dir);identity=json.loads((run/'run_config.json').read_text())['input_identity'];types=entity_types(Path(c.data_dir)/'label.txt')
 paths=prepare_resources(c,identity,types);data=load_split(paths[0]);lex=json.loads((Path(c.resource_dir)/'lexicon.json').read_text())
 ck=torch.load(run/'final_ema.pt',map_location='cpu',mmap=True,weights_only=False)
 c.gp_type_weights=ck['config']['gp_type_weights'];c.gp_positive_weight=ck['config']['gp_positive_weight'];c.boundary_weight=ck['config']['boundary_weight']
 m=LemanGP(c,len(types),len(lex['words']),char_vocab_size=len(lex['characters']),encoder_config=ck['encoder_config'],pretrained=False);m.load_state_dict(ck['model'],strict=True);del ck;m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 before=state_digest(m);cache={}
 h1=m.sequence_norm.register_forward_hook(lambda mod,inp,out:out.detach().requires_grad_(True))
 def capture(mod,inp):cache['sequence']=inp[0]
 h2=m.attention.register_forward_pre_hook(capture)
 selected=sorted(range(len(data.samples)),key=lambda i:(hashlib.sha256(data.samples[i][0].encode()).hexdigest(),i))[:64]
 first={}
 for it in data.items:first.setdefault(it['sid'],it)
 write_json(a.out/'PROTOCOL.json',{'split':'train','sentence_ids':selected,'text_sha256':[hashlib.sha256(data.samples[i][0].encode()).hexdigest() for i in selected],'window':'first window per selected sentence','measurement':'gradients w.r.t. shared recurrent representation before attention and boundary branches','main':'GP objective','auxiliary':'boundary_weight times boundary loss','mode':'eval; dropout disabled','loss_type_weights':c.gp_type_weights,'positive_weight':c.gp_positive_weight,'boundary_weight':c.boundary_weight,'optimizer_steps':0,'source_sha256':sha256(__file__)})
 rows=[];collate=Collator(types,data.pad_id)
 for i,sid in enumerate(selected):
  batch=collate([first[sid]]);inputs=to_device(batch,device)
  with amp_context(device):total,scores=m(**inputs)
  seq=cache['sequence'];mainloss=m.loss_fn(scores,inputs['labels'],inputs['char_mask']);aux=c.boundary_weight*boundary_loss(m.boundary(seq).reshape(*seq.shape[:2],m.num_types,2),inputs['labels'],inputs['char_mask'])
  gm=torch.autograd.grad(mainloss,seq,retain_graph=True)[0].double().flatten();ga=torch.autograd.grad(aux,seq)[0].double().flatten();nm=gm.norm();na=ga.norm();cos=float(torch.dot(gm,ga)/(nm*na).clamp_min(1e-30))
  rows.append({'sid':sid,'offset':first[sid]['offset'],'characters':first[sid]['length'],'gold_spans_in_window':len(first[sid]['spans']),'main_loss':float(mainloss.detach()),'weighted_boundary_loss':float(aux.detach()),'main_gradient_norm':float(nm),'aux_gradient_norm':float(na),'aux_to_main_norm_ratio':float(na/nm.clamp_min(1e-30)),'cosine':cos})
  if i%16==0:print('gradient window',i,flush=True)
 h1.remove();h2.remove();after=state_digest(m);assert before==after
 with (a.out/'representation_gradients.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write_json(a.out/'COMPLETE.json',{'windows':len(rows),'weight_state_before':before,'weight_state_after':after,'weights_unchanged':True,'optimizer_steps':0,'positive_alignment_windows':sum(r['cosine']>0 for r in rows),'no_parameter_gradients_accumulated':all(p.grad is None for p in m.parameters())})
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
