"""Dev-only source-diversity study, with a fixed finite selection protocol."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np
from settings import ROOT,OUT,FROZEN,OLD,OLD_NAMES,GROUPS
import study_core as core
from evidence_core import read_bio,read_predictions,score,sha256,write_json
sys.path.insert(0,str(ROOT/'leman_final_v2_test_20261021'))
import frozen_pipeline as fp

def save(n,z):write_json(OUT/n,z)
def main():
    if (OUT/'SUMMARY.json').exists():raise RuntimeError('Completed selection is immutable')
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    for n,h in protocol['source_hashes'].items():assert sha256(Path(__file__).parent/n)==h,n
    gold=read_bio(ROOT/'data/dev2.txt');types=list(json.loads((OLD/'decoder_full.json').read_text())['thresholds'])
    samples=[(text,{(types.index(t),a,b) for t,a,b in spans}) for text,spans in gold]
    blind=[(text,[]) for text,_ in samples];ids=np.arange(len(samples))
    outer=np.load(OLD/'features.npz')['outer']
    inner=np.array([int.from_bytes(hashlib.sha256(('leman-inner-20261021\0'+text).encode()).digest()[:8],'big')%3 for text,_ in samples])
    gaz=json.loads((FROZEN/'v6_train_gazetteer.json').read_text());priors=joblib.load(FROZEN/'train_boundary_priors.joblib')
    records={n:fp.read_cache(FROZEN/'caches'/f'{n}.jsonl.gz')[1] for n in OLD_NAMES}
    for n in GROUPS[-1]:
        p=OUT/'candidates/dev'/f'{n}.jsonl.gz';receipt=json.loads(p.with_suffix('.receipt.json').read_text());assert sha256(p)==receipt['sha256']
        records[n]=fp.read_cache(p)[1]
    # Rebuild old features and verify exact frozen predictions before new fitting.
    oldkeys,oldx=fp.feature_matrix([records[n] for n in OLD_NAMES],blind,types,gaz,priors)
    saved=np.load(OLD/'features.npz');assert np.array_equal(oldkeys,saved['keys']) and np.allclose(oldx,saved['X'],rtol=0,atol=1e-12)
    baseline=read_predictions(OLD/'crossfit_predictions.jsonl',gold);base=score(gold,baseline,types)
    outcomes=[];coverage=[];best_bundle=None;best_score=-1.
    for group_index,new in enumerate(GROUPS,1):
        names=OLD_NAMES+new;keys,x=fp.feature_matrix([records[n] for n in names],blind,types,gaz,priors);keys=np.asarray(keys,dtype=np.int32)
        y=np.asarray([int((t,a,b) in samples[sid][1]) for sid,t,a,b in keys],dtype=np.uint8)
        cover=dict(new_sources=new,candidate_rows=len(keys),covered_gold=int(y.sum()),gold=sum(len(g) for _,g in samples),feature_dimensions=x.shape[1])
        cover['candidate_recall']=cover['covered_gold']/cover['gold'];coverage.append(cover);print('POOL',json.dumps(cover),flush=True)
        for spec in core.CASES[:3]:
            tag=f'plus{group_index}_{spec["name"]}';print('BEGIN',tag,flush=True)
            values=np.full(len(keys),np.nan);predictions={mode:{} for mode in ('nested','self')};reports={mode:[] for mode in predictions}
            for held in (0,1):
                fitids=ids[outer!=held];heldids=ids[outer==held]
                tr=outer[keys[:,0]]!=held;va=~tr;innervalues=np.full(len(keys),np.nan)
                for h in range(3):
                    it=tr & (inner[keys[:,0]]!=h);iv=tr & (inner[keys[:,0]]==h)
                    m=core.fit_model(x[it],y[it],spec);innervalues[iv]=core.logits(m,x[iv])
                m=core.fit_model(x[tr],y[tr],spec);values[va]=core.logits(m,x[va]);selfvalues=np.full(len(keys),np.nan);selfvalues[tr]=core.logits(m,x[tr])
                for mode,v in [('nested',innervalues),('self',selfvalues)]:
                    dec=core.calibrate(keys,v,samples,fitids,len(types))
                    pred=core.predict(keys[va],values[va],dec['thresholds_array'],len(samples))
                    predictions[mode].update({int(i):pred[i] for i in heldids});reports[mode].append(dict(held=held,decoder=dec))
            for mode in ('nested','self'):
                name=tag+'_'+mode;report=score(gold,core.convert(predictions[mode],types),types)
                result=dict(name=name,new_sources=new,model=spec,threshold_fit=mode,crossfit=report,folds=reports[mode],delta_pp=100*(report['micro']['f1']-base['micro']['f1']))
                outcomes.append(result);save(name+'.json',result)
                core.write_predictions(OUT/(name+'_crossfit.jsonl'),predictions[mode],blind,types)
                print('RESULT',name,round(report['micro']['f1']*100,6),'delta',round(result['delta_pp'],6),flush=True)
                if report['micro']['f1']>best_score+1e-14:
                    best_score=report['micro']['f1'];best_bundle=(result,keys.copy(),x.copy(),y.copy(),values.copy(),predictions[mode])
    best,keys,x,y,values,pred=best_bundle
    adoption=best['delta_pp']>=.03
    summary=dict(completed_utc=core.now(),baseline=base,coverage=coverage,cases=outcomes,best=best['name'],adopt=adoption,
        paired_bootstrap=core.bootstrap(gold,baseline,core.convert(pred,types)),test_read=False,seconds=time.monotonic()-start)
    if adoption:
        model=core.fit_model(x,y,best['model'])
        finalvalues=values if best['threshold_fit']=='nested' else core.logits(model,x)
        decoder=core.calibrate(keys,finalvalues,samples,ids,len(types));decoder['thresholds']={t:v for t,v in zip(types,decoder['thresholds_array'])}
        decoder['threshold_source']=best['threshold_fit']+' dev logits; no test fitting'
        joblib.dump(model,OUT/'selected_hgb.joblib');save('selected_decoder.json',decoder)
        np.savez_compressed(OUT/'selected_dev_features.npz',keys=keys,X=x,y=y,outer=outer,oof_scores=values)
        freeze=dict(frozen_utc=core.now(),selected_case=best,new_sources=best['new_sources'],models=OLD_NAMES+best['new_sources'],types=types,
            dev_sha256=sha256(ROOT/'data/dev2.txt'),original_freeze_sha256=sha256(FROZEN/'FREEZE.json'),
            files={n:sha256(OUT/n) for n in ['PROTOCOL.json','selected_hgb.joblib','selected_decoder.json']},
            source_hashes=protocol['source_hashes'],test_read_during_selection=False,
            evaluation_policy='only selected frozen version; no test-error-driven threshold edits')
        save('SELECTED_FREEZE.json',freeze)
    save('SUMMARY.json',summary);print('SELECTED',best['name'],'ADOPT',adoption,'F1',best_score,flush=True)

if __name__=='__main__':main()
