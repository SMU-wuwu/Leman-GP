"""Evaluate only the dev-selected version using authenticated candidate caches."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,sys,time
from pathlib import Path
import numpy as np,joblib
import study_core as core
from settings import ROOT,OUT,FROZEN,OLD_NAMES
from evidence_core import read_bio,read_predictions,score,sha256,write_json
sys.path.insert(0,str(ROOT/'leman_final_v2_test_20261021'))
import frozen_pipeline as fp

def main():
    dest=OUT/'test';dest.mkdir(exist_ok=True)
    if (dest/'RESULT.json').exists():print((dest/'RESULT.json').read_text());return
    freeze=json.loads((OUT/'SELECTED_FREEZE.json').read_text())
    for n,h in freeze['files'].items():assert sha256(OUT/n)==h,n
    for n,h in freeze['source_hashes'].items():assert sha256(Path(__file__).parent/n)==h,n
    types=freeze['types'];gold=read_bio(ROOT/'data/test2.txt');blind=[(text,[]) for text,_ in gold]
    previous=ROOT/'output/leman_final_v2_test_20261021'
    orig=json.loads((previous/'TEST_RESULT.json').read_text());assert orig['test_sha256']==sha256(ROOT/'data/test2.txt')
    records=[];hashes={}
    for n in freeze['models']:
        p=(previous/'candidates' if n in OLD_NAMES else OUT/'candidates/test')/f'{n}.jsonl.gz'
        receipt=json.loads(p.with_suffix('.receipt.json').read_text());assert sha256(p)==receipt['sha256']
        head,rows=fp.read_cache(p);assert head['types']==types and head['source']==n
        h=(head['identity']['test_sha256'] if n in OLD_NAMES else head['split_sha256'])
        assert h==orig['test_sha256'] and set(rows)==set(range(len(gold)))
        records.append(rows);hashes[n]=receipt['sha256']
    keys,x=fp.feature_matrix(records,blind,types,json.loads((FROZEN/'v6_train_gazetteer.json').read_text()),joblib.load(FROZEN/'train_boundary_priors.joblib'))
    decoder=json.loads((OUT/'selected_decoder.json').read_text());model=joblib.load(OUT/'selected_hgb.joblib')
    assert model.n_features_in_==x.shape[1]
    pred=core.predict(keys,core.logits(model,x),decoder['thresholds_array'],len(gold))
    core.write_predictions(dest/'test_predictions.jsonl',pred,blind,types)
    loaded=read_predictions(dest/'test_predictions.jsonl',gold);assert loaded==core.convert(pred,types)
    result=score(gold,loaded,types);baseline=read_predictions(previous/'test_predictions.jsonl',gold);base=score(gold,baseline,types)
    tp=sum(len(g & loaded[i]) for i,(_,g) in enumerate(gold));n=sum(map(len,loaded.values()));ng=sum(len(g) for _,g in gold)
    assert (tp,n,ng)==tuple(result['micro'][k] for k in ['tp','predicted','support'])
    texts={t for split in ['train','dev'] for t,_ in read_bio(ROOT/'data'/f'{split}2.txt')}
    clean=[i for i,(t,_) in enumerate(gold) if t not in texts]
    result.update(dict(selected_case=freeze['selected_case']['name'],new_sources=freeze['new_sources'],models=freeze['models'],
        baseline=base,delta_pp=100*(result['micro']['f1']-base['micro']['f1']),achieved_95=result['micro']['f1']>=.95,
        paired_bootstrap=core.bootstrap(gold,baseline,loaded),
        disjoint_text_secondary=dict(sentences=len(clean),metrics=score([gold[i] for i in clean],{j:loaded[i] for j,i in enumerate(clean)},types)),
        independent_counts=dict(tp=tp,predicted=n,gold=ng,f1=2*tp/(n+ng)),
        test_sha256=orig['test_sha256'],selected_freeze_sha256=sha256(OUT/'SELECTED_FREEZE.json'),candidate_hashes=hashes,
        prediction_sha256=sha256(dest/'test_predictions.jsonl'),completed_utc=core.now(),test_fitting=False,
        test_status='reused benchmark; previous aggregate results known; no test-error-directed fitting',
        new_neural_training=False,one_selected_version_evaluated=True))
    write_json(dest/'RESULT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
