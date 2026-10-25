"""Figures from actual exported CSVs only; editable SVG, PDF, PNG plus source tables."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,path):
 for ext in ('pdf','svg','png'):fig.savefig(str(path)+'.'+ext,dpi=250,bbox_inches='tight')
 plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();out=a.base/'figures';out.mkdir(exist_ok=True)
 tables=a.base/'analysis'
 if (tables/'metrics.csv').exists():
  df=pd.read_csv(tables/'metrics.csv');scopes=list(df.scope.unique());fig,axes=plt.subplots(1,len(scopes),figsize=(max(5,5*len(scopes)),5),squeeze=False)
  for ax,scope in zip(axes[0],scopes):
   d=df[df.scope==scope];ax.scatter(d.f1*100,range(len(d)),color='#416782');ax.set_yticks(range(len(d)),d.variant);ax.invert_yaxis();ax.set_xlabel('Exact-span Micro F1 (%)');ax.set_title(scope+' dev; F42 parent');ax.grid(axis='x',alpha=.15)
  fig.tight_layout();save(fig,out/'performance')
 if (tables/'paired_bootstrap.csv').exists():
  df=pd.read_csv(tables/'paired_bootstrap.csv');fig,ax=plt.subplots(figsize=(7,max(3,len(df)*.23)))
  for i,r in df.iterrows():ax.plot([r.low_pp,r.high_pp],[i,i],color='#7c94a4');ax.scatter(r.delta_micro_pp,i,color='#416782',s=18)
  ax.set_yticks(range(len(df)),[f'{r.scope}: {r.variant} vs {r.reference}' for _,r in df.iterrows()]);ax.axvline(0,color='grey',lw=.7);ax.invert_yaxis();ax.set_xlabel('Micro F1 difference (pp); paired text-cluster 95% CI');fig.tight_layout();save(fig,out/'paired_effects')
 if (tables/'training_curves.csv').exists():
  df=pd.read_csv(tables/'training_curves.csv');fig,ax=plt.subplots(figsize=(8,4))
  for v,d in df[df.weight_source=='ema'].groupby('variant'):ax.plot(d.epoch,d.micro_f1,marker='.',label=v)
  ax.set_xlabel('Adaptation epoch');ax.set_ylabel('EMA raw-dev Micro F1 (%)');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=6);fig.tight_layout();save(fig,out/'recovery_curves')
 panels=sorted(a.base.glob('fixed_interventions/*/mechanisms/*/PANEL.json'))+sorted(a.base.glob('*_seed*/mechanisms/*/PANEL.json'))
 for pf in panels:
  panel=pf.parent/'attention_example.npz'
  if not panel.exists():continue
  w=np.load(panel)['weights'];fig,ax=plt.subplots(figsize=(4.4,3.6));im=ax.imshow(w.mean(0),vmin=0,cmap='Blues',aspect='auto');fig.colorbar(im,ax=ax,label='Mean attention weight');ax.set_xlabel('Key character index');ax.set_ylabel('Query character index');ax.set_title(panel.parents[2].name+' / '+panel.parent.name);fig.tight_layout();save(fig,out/('attention_'+panel.parents[2].name+'_'+panel.parent.name))
 # Aggregate mechanisms across fixed panel, with explicit sample sizes, no pseudo-replicate CI.
 rows=[]
 for pf in panels:
  f=pf.parent/'characters.csv'
  if not f.exists():continue
  d=pd.read_csv(f);v=f.parents[2].name+' / '+f.parent.name
  for col in ('gate_mean','lexicon_relative_change','char_relative_residual','attention_relative_change'):
   x=d[col].dropna();rows.append({'variant':v,'metric':col,'mean':x.mean() if len(x) else np.nan,'n_window_positions':len(x)})
 if rows:
  d=pd.DataFrame(rows);d.to_csv(out/'mechanism_aggregates.csv',index=False)
  fig,axes=plt.subplots(1,4,figsize=(14,max(4,d.variant.nunique()*.18)))
  for ax,(metric,g) in zip(axes,d.groupby('metric',sort=False)):
   ax.scatter(g['mean'],range(len(g)),color='#416782');ax.set_yticks(range(len(g)),g.variant);ax.invert_yaxis();ax.set_title(metric.replace('_','\n'),fontsize=8);ax.set_xlabel('Panel mean; no causal attribution')
  fig.tight_layout();save(fig,out/'mechanism_panel')
 # Every mechanism trajectory is tied to an actual panel export and its weight tag.
 trajectory=[]
 for pf in panels:
  for name,cols in [('boundary_metrics',['average_precision','precision','recall']),('gold_span_margins',['boundary_margin','detected']),('attention_heads',['entropy','mean_distance','within_entity_mass','uniform_within_entity_mass']),('characters',['candidate_count','lexicon_entropy','gate_mean'])]:
   f=pf.parent/(name+'.csv')
   if not f.exists():continue
   d=pd.read_csv(f)
   for col in cols:
    x=d[col].dropna();trajectory.append(dict(variant=pf.parents[2].name,tag=pf.parent.name,metric=name+'.'+col,mean=x.mean() if len(x) else np.nan,n_rows=len(x)))
 if trajectory:
  d=pd.DataFrame(trajectory);d.to_csv(out/'mechanism_trajectories.csv',index=False)
  for metric,g in d.groupby('metric'):
   fig,ax=plt.subplots(figsize=(8,max(3,len(g)*.16)))
   ax.scatter(g['mean'],range(len(g)),s=12,color='#416782');ax.set_yticks(range(len(g)),g.variant+' / '+g.tag,fontsize=5);ax.invert_yaxis();ax.set_xlabel(metric+'; descriptive panel row mean');fig.tight_layout();save(fig,out/metric.replace('.','_'))
 for fname,value in [('subgroup_recall','recall'),('error_categories','count')]:
  f=tables/(fname+'.csv')
  if not f.exists():continue
  d=pd.read_csv(f);col='group' if fname=='subgroup_recall' else 'category'
  for scope,g in d.groupby('scope'):
   mat=g.pivot(index='variant',columns=col,values=value).fillna(0)
   fig,ax=plt.subplots(figsize=(max(7,len(mat.columns)*.34),max(3,len(mat)*.27)));im=ax.imshow(mat.values,aspect='auto',cmap='Blues');fig.colorbar(im,ax=ax,label=value);ax.set_yticks(range(len(mat)),mat.index);ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=70,ha='right',fontsize=6);ax.set_title(scope+'; supports in source CSV');fig.tight_layout();save(fig,out/(fname+'_'+scope))
 grad=[];objectives=[]
 for f in sorted(a.base.glob('*_seed*/gradient_norms.jsonl')):
  for line in f.read_text().splitlines():
   r=json.loads(line)
   for module,z in r['groups'].items():grad.append(dict(variant=f.parent.name,step=r['step'],module=module,**z))
   for key,val in r.get('last_forward_objective_parts',{}).items():objectives.append(dict(variant=f.parent.name,step=r['step'],objective=key,value=val))
 for name,rows,group,y in [('gradient_trajectories',grad,'module','gradient_norm'),('objective_trajectories',objectives,'objective','value')]:
  if not rows:continue
  d=pd.DataFrame(rows);d.to_csv(out/(name+'.csv'),index=False)
  for v,g in d.groupby('variant'):
   fig,ax=plt.subplots(figsize=(7,3))
   for label,z in g.groupby(group):ax.plot(z.step,z[y],label=label,lw=.8)
   ax.set_xlabel('Optimizer step');ax.set_ylabel(y+' (rank 0)');ax.set_title(v);ax.legend(bbox_to_anchor=(1.02,1),fontsize=6);fig.tight_layout();save(fig,out/(name+'_'+v))
if __name__=='__main__':main()
