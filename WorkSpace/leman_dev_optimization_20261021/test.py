"""One CPU evaluation of the dev-selected frozen system on saved BF16 candidates.

Run only after SELECTED_FREEZE.json exists. No fitting or variant selection.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json, sys, time
from pathlib import Path
import joblib
import numpy as np
import optimize as study
from evidence_core import read_bio, read_predictions, score, sha256, write_json

def main():
    root=study.ROOT; out=study.OUT; start=time.monotonic()
    freeze_path=out/'SELECTED_FREEZE.json'
    freeze=json.loads(freeze_path.read_text())
    assert json.loads((out/'SUMMARY.json').read_text())['adopt']
    assert freeze['test_read_during_selection'] is False
    dest=out/'test'; dest.mkdir(exist_ok=True)
    if (dest/'RESULT.json').exists():
        print((dest/'RESULT.json').read_text()); return
    assert sha256(study.FROZEN/'FREEZE.json')==freeze['original_freeze_sha256']
    for n,h in freeze['files'].items(): assert sha256(out/n)==h,n
    for n,h in freeze['source_sha256'].items(): assert sha256(Path(__file__).parent/n)==h,n
    # Imports support existing CPU feature construction; no GPU inference.
    sys.path.insert(0,str(root/'leman_final_v2_test_20261021'))
    import frozen_pipeline as fp
    import torch
    torch.set_num_threads(2)
    types=freeze['types']
    prior_test=root/'output/leman_final_v2_test_20261021'
    prior_result=json.loads((prior_test/'TEST_RESULT.json').read_text())
    assert sha256(root/'data/test2.txt')==prior_result['test_sha256']
    gold=read_bio(root/'data/test2.txt')
    blind=[(text,[]) for text,_ in gold]
    records=[]; cache_hashes={}
    for name in freeze['models']:
        p=prior_test/'candidates'/f'{name}.jsonl.gz'
        receipt=json.loads(p.with_suffix('.receipt.json').read_text())
        assert sha256(p)==receipt['sha256']
        header,rows=fp.read_cache(p)
        assert header['source']==name and header['types']==types
        assert header['identity']['test_sha256']==prior_result['test_sha256']
        assert set(rows)==set(range(len(blind)))
        records.append(rows);cache_hashes[name]=receipt['sha256']
    gaz=json.loads((study.FROZEN/'v6_train_gazetteer.json').read_text())
    priors=joblib.load(study.FROZEN/'train_boundary_priors.joblib')
    keys,x=fp.feature_matrix(records,blind,types,gaz,priors)
    keys=np.asarray(keys,dtype=np.int32)
    original_model=joblib.load(study.OLD/'hgb_full_dev.joblib')
    original_decoder=json.loads((study.OLD/'decoder_full.json').read_text())
    original_pred=study.predict(keys,study.logits(original_model,x),[original_decoder['thresholds'][t] for t in types],len(blind))
    baseline=read_predictions(prior_test/'test_predictions.jsonl',gold)
    assert study.convert(original_pred,types)==baseline,'Original frozen test replay differs'
    model=joblib.load(out/'selected_hgb.joblib')
    decoder=json.loads((out/'selected_decoder.json').read_text())
    if freeze['selected_case']['conflicts']: x=study.augment(keys,x)
    pred=study.predict(keys,study.logits(model,x),decoder['thresholds_array'],len(blind))
    study.write_predictions(dest/'test_predictions.jsonl',pred,blind,types)
    reloaded=read_predictions(dest/'test_predictions.jsonl',gold)
    assert reloaded==study.convert(pred,types)
    result=score(gold,reloaded,types); base=score(gold,baseline,types)
    # Independently tally exact entity matches using serialized predictions.
    tp=sum(len(g & reloaded[i]) for i,(_,g) in enumerate(gold))
    n=sum(map(len,reloaded.values())); ng=sum(len(g) for _,g in gold)
    assert (tp,n,ng)==tuple(result['micro'][k] for k in ('tp','predicted','support'))
    traintexts={text for text,_ in read_bio(root/'data/train2.txt')}
    devtexts={text for text,_ in read_bio(root/'data/dev2.txt')}
    clean=[i for i,(text,_) in enumerate(gold) if text not in traintexts|devtexts]
    clean_result=score([gold[i] for i in clean],{j:reloaded[i] for j,i in enumerate(clean)},types)
    result.update(dict(system='dev-selected '+freeze['selected_case']['name'],split='test2.txt',
        baseline=base,delta_vs_baseline_pp=100*(result['micro']['f1']-base['micro']['f1']),
        achieved_95=result['micro']['f1']>=.95,
        paired_bootstrap=study.bootstrap(gold,baseline,reloaded),
        disjoint_text_secondary=dict(sentences=len(clean),metrics=clean_result),
        candidate_rows=len(keys),candidate_hashes=cache_hashes,
        selected_freeze_sha256=sha256(freeze_path),test_sha256=sha256(root/'data/test2.txt'),
        prediction_sha256=sha256(dest/'test_predictions.jsonl'),evaluator_sha256=sha256(__file__),
        original_test_replay_exact=True,independent_counts=dict(tp=tp,predicted=n,gold=ng,f1=2*tp/(n+ng)),
        new_neural_training=False,new_neural_inference=False,fit_on_test=False,one_selected_version_evaluated=True,
        test_status='Reused benchmark: original test aggregate was known before this dev-only study; no test-error-guided tuning',
        completed_utc=study.now(),seconds=time.monotonic()-start))
    write_json(dest/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
