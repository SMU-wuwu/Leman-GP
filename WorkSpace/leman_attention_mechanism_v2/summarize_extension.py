"""Automatically export paired results and mechanism trajectories after training."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evidence_core import read_bio,read_predictions,score,sha256
from analyze import counts,bootstrap,csvout

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');a=p.parse_args();root=Path(a.root);base=root/'output/leman_attention_mechanism_v2';out=base/'analysis';out.mkdir(exist_ok=True)
 samples=read_bio(root/'data/dev2.txt');types=sorted({t for _,g in samples for t,_,_ in g});rows=[];curves=[];mechanisms=[];support=[]
 for seed in (42,43):
  control=read_predictions(base/f'F_seed{seed}/epoch_003_ema/dev_predictions.jsonl',samples)
  for variant in ('F','attention_norm_only','uniform_attention'):
   run=base/f'{variant}_seed{seed}';assert (run/'COMPLETE.json').exists()
   endpoint=json.loads((run/'PRIMARY_ENDPOINT.json').read_text());assert endpoint['epoch']==3
   pred=read_predictions(run/'epoch_003_ema/dev_predictions.jsonl',samples);z=score(samples,pred,types);assert abs(z['micro']['f1']-endpoint['dev']['micro']['f1'])<1e-12
   delta=100*(z['micro']['f1']-score(samples,control,types)['micro']['f1'])
   ci=bootstrap(samples,counts(samples,control),counts(samples,pred)) if variant!='F' else {'low_pp':0.,'high_pp':0.,'replicates':4000,'text_clusters':5959}
   rows.append({'variant':variant,'adaptation_seed':seed,**z['micro'],'macro13':z['macro13'],'delta_micro_pp':delta,**ci,'prediction_sha256':sha256(run/'epoch_003_ema/dev_predictions.jsonl')})
   for h in json.loads((run/'history.json').read_text()):curves.append({'variant':variant,'seed':seed,'epoch':h['epoch'],'micro_f1':100*h['dev_ema']['micro']['f1']})
   for panel in sorted((run/'mechanisms').glob('*')):
    per={}
    with (panel/'gold_span_margins.csv').open() as f:
     for r in csv.DictReader(f):per.setdefault((int(r['sid']),r['type'],r['start'],r['end']),[]).append(float(r['boundary_margin']))
    sent={}
    for key,values in per.items():sent.setdefault(key[0],[]).append(float(np.mean(values)))
    for sid,vals in sent.items():mechanisms.append({'variant':variant,'seed':seed,'epoch_tag':panel.name,'sid':sid,'gold_entities':len(vals),'mean_boundary_margin':np.mean(vals)})
   support.append({'variant':variant,'seed':seed,'endpoint':3,'new_epochs':3,'parent_sha256':endpoint['parent_sha256']})
 csvout(out/'paired_results.csv',rows);csvout(out/'epoch_curves.csv',curves);csvout(out/'boundary_margin_by_sentence.csv',mechanisms);csvout(out/'run_identity.csv',support)
 plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.8),layout='constrained');colors={'F':'#3B7293','attention_norm_only':'#B96843','uniform_attention':'#7B8791'}
 for ax,seed in zip(axs,(42,43)):
  for v,col in colors.items():
   ss=[r for r in curves if r['seed']==seed and r['variant']==v];ax.plot([r['epoch'] for r in ss],[r['micro_f1'] for r in ss],'o-',label=v.replace('_',' '),color=col)
  ax.set_title(f'Adaptation seed {seed}');ax.set_xticks([1,2,3]);ax.set_xlabel('Epoch');ax.set_ylabel('Exact-span Micro F1 (%)');ax.legend(fontsize=7)
 for ext in ('png','svg','pdf'):fig.savefig(out/f'attention_training.{ext}',dpi=300)
 plt.close(fig)
 lines=['# 注意力机制补充实验结果','', '每项3 epoch，以最终EMA为主终点；两个种子共享同一训练父模型。差值均与相同适应训练种子的完整模型配对。置信区间来自句子文本簇重采样。','', '| 配置 | 种子 | F1 (%) | 差值 (pp) | 95% CI |','|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['variant']} | {r['adaptation_seed']} | {100*r['f1']:.4f} | {r['delta_micro_pp']:+.4f} | [{r['low_pp']:.4f}, {r['high_pp']:.4f}] |")
 lines+=['','![训练曲线](attention_training.png)','', '每个运行目录的mechanisms/保存词典权重、门控、注意力分配、边界分数、辅助边界检测指标；gradient_norms.jsonl保存各模块训练梯度。归一化对照没有注意力权重，故不导出虚构的权重热图。']
 (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
 print(out/'RESULTS.md')
if __name__=='__main__':main()
