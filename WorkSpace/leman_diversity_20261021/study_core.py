"""Bounded dev-only nested calibration study; never reads test data.

Neural weights, candidate pool, train lexicon and contexts are unchanged.
Text-grouped outer folds match final-v2. Thresholds use three inner OOF folds.
The winning model is frozen before separate evaluation can read test assets.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from fast_intervals import GridIntervals
from predict_hgb_v22 import decode
from evidence_core import read_bio, read_predictions, score, sha256, write_json

ROOT = Path('/root/data/WorkSpace')
FROZEN = ROOT/'output/leman_final_v2_20261019'
OLD = FROZEN/'wide_context6_consensus8'
OUT = ROOT/'output/leman_dev_optimization_20261021'
CASES = [
    dict(name='legacy_oof', iterations=180, leaves=15, minimum=35, l2=2., negative=1.5, conflicts=False),
    dict(name='balanced_oof', iterations=180, leaves=15, minimum=70, l2=4., negative=1., conflicts=False),
    dict(name='compact_oof', iterations=200, leaves=7, minimum=100, l2=8., negative=1., conflicts=False),
    dict(name='conflicts_oof', iterations=180, leaves=15, minimum=70, l2=4., negative=1., conflicts=True),
]

def now(): return datetime.now(timezone.utc).isoformat()
def save(name, value): write_json(OUT/name, value)

def augment(keys, x):
    """Eight label-free features measure competing types and boundaries."""
    own = x[:, :72:6].max(1)
    extra = np.zeros((len(keys), 8))
    extra[:, 0] = (x[:, 1:72:6] != -16.).mean(1)
    for sid in np.unique(keys[:, 0]):
        ids = np.flatnonzero(keys[:, 0] == sid)
        for i in ids:
            _, t, a, b = keys[i]
            overlap = ids[(keys[ids, 2] < b) & (keys[ids, 3] > a) & (ids != i)]
            rivals = [overlap[(keys[overlap, 2] == a) & (keys[overlap, 3] == b) & (keys[overlap, 1] != t)],
                      overlap[keys[overlap, 1] == t], overlap]
            values = [float(own[r].max()) if len(r) else -4. for r in rivals]
            extra[i, 1:4] = values
            extra[i, 4:7] = own[i] - np.asarray(values)
            extra[i, 7] = np.log1p(len(overlap))
    return np.column_stack((x, extra))

def fit_model(x, y, spec):
    model = HistGradientBoostingClassifier(max_iter=spec['iterations'], learning_rate=.05,
        max_leaf_nodes=spec['leaves'], min_samples_leaf=spec['minimum'],
        l2_regularization=spec['l2'], max_bins=127, early_stopping=False, random_state=42)
    model.fit(x, y, sample_weight=np.where(y, 1., spec['negative']))
    return model

def logits(model, x):
    p = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1-1e-7)
    return np.log(p)-np.log1p(-p)

def prepared_subset(keys, values, samples, ids):
    ids = np.asarray(ids)
    remap = np.full(len(samples), -1, dtype=int); remap[ids] = np.arange(len(ids))
    mask = remap[keys[:, 0]] >= 0
    kk = keys[mask].copy(); kk[:, 0] = remap[kk[:, 0]]
    return GridIntervals(kk, values[mask], [samples[i] for i in ids], minimum=-4.)

def calibrate(keys, values, samples, ids, ntypes):
    prepared = prepared_subset(keys, values, samples, ids)
    grid = np.arange(-3., 2.0001, .25)
    trials = np.repeat(grid[:, None], ntypes, axis=1)
    scores = prepared.scores(trials); j = int(np.argmax(scores))
    threshold = trials[j].copy(); best = float(scores[j]); global_threshold = float(grid[j])
    support = np.zeros(ntypes, dtype=int)
    for i in ids:
        for t, _, _ in samples[i][1]: support[t] += 1
    # One coarse pass, common classes only. Shrink offsets toward global cutoff.
    for t in range(ntypes):
        if support[t] < 200: continue
        shrink = support[t]/(support[t]+200.)
        opts = np.r_[threshold[t], np.clip(global_threshold + shrink*np.arange(-.75, .7501, .25), -3., 2.)]
        trials = np.repeat(threshold[None, :], len(opts), axis=0); trials[:, t] = opts
        scores = prepared.scores(trials); j = int(np.argmax(scores))
        if scores[j] > best + 1e-14: threshold = trials[j].copy(); best = float(scores[j])
    return dict(strategy='weighted_interval', thresholds_array=threshold.tolist(),
        global_threshold=global_threshold, fit_micro_f1=best, fit_support=support.tolist(),
        threshold_source='inner OOF logits; one support-shrunk coarse type pass')

def predict(keys, values, thresholds, nsamples):
    rows = [[] for _ in range(nsamples)]
    for (sid, t, a, b), value in zip(keys, values):
        if value > -4.: rows[sid].append((int(t), int(a), int(b), float(value)))
    return {i: decode(row, thresholds, 'weighted_interval') for i, row in enumerate(rows)}

def convert(pred, types):
    return {sid: {(types[t], a, b) for t, a, b in spans} for sid, spans in pred.items()}

def write_predictions(path, pred, samples, types):
    with path.open('w') as f:
        for sid, (text, _) in enumerate(samples):
            f.write(json.dumps(dict(sid=sid, text=text, predicted=[dict(type=types[t], start=a,
                end=b, text=text[a:b]) for t, a, b in sorted(pred[sid])]), ensure_ascii=False)+'\n')

def bootstrap(samples, baseline, challenger):
    # Resample identical text groups jointly; both systems share each resample.
    groups = {}
    for sid, (text, gold) in enumerate(samples):
        v = groups.setdefault(text, np.zeros(5, dtype=int))
        v += [len(gold & baseline[sid]), len(baseline[sid]), len(gold & challenger[sid]), len(challenger[sid]), len(gold)]
    counts = np.asarray(list(groups.values())); rng = np.random.default_rng(20261021)
    deltas = []
    for _ in range(2000):
        a, b, c, d, g = counts[rng.integers(len(counts), size=len(counts))].sum(0)
        deltas.append(100*(2*c/(d+g)-2*a/(b+g)))
    return dict(paired_text_cluster_delta_pp_ci95=np.quantile(deltas, [.025, .975]).tolist(),
        resamples=2000, seed=20261021, scope='exploratory interval; does not adjust for model selection')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'SUMMARY.json').exists(): raise RuntimeError('Completed study is immutable')
    t0 = time.monotonic()
    spec = dict(created_utc=now(), cases=CASES, neural_training=False, test_read=False,
        selection='largest outer-crossfit Micro F1; original baseline retained unless improvement >= 0.03 percentage points; ties prefer earlier case',
        validation='fixed two text-grouped outer folds; three inner text-grouped OOF folds fit decoder; no outer held labels enter fitting',
        threshold_grid='global -3 to 2 by .25; one per-type pass ±.75 by .25, support >=200, offsets shrunk by n/(n+200)',
        final_fit='one HGB trained on all dev; final decoder calibrated on two-fold OOF logits of selected family',
        scope='postprocessor crossfit only; neural checkpoints were historically dev-selected',
        threads=2, source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    save('PROTOCOL.json', spec)
    original = read_bio(ROOT/'data/dev2.txt')
    types = list(json.loads((OLD/'decoder_full.json').read_text())['thresholds'])
    samples = [(text, {(types.index(t), a, b) for t, a, b in gold}) for text, gold in original]
    data = np.load(OLD/'features.npz'); keys, x, y, outer = [data[k] for k in ('keys', 'X', 'y', 'outer')]
    assert len(samples) == len(outer)
    assert np.array_equal(y, [int((t,a,b) in samples[sid][1]) for sid,t,a,b in keys])
    textfold = {}
    for (text,_), fold in zip(samples,outer):
        assert textfold.setdefault(text,int(fold)) == fold
    oldmodel = joblib.load(OLD/'hgb_full_dev.joblib')
    oldvalues = logits(oldmodel, x)
    oldthresholds = list(json.loads((OLD/'decoder_full.json').read_text())['thresholds'].values())
    replay = predict(keys, oldvalues, oldthresholds, len(samples))
    assert convert(replay,types) == read_predictions(OLD/'full_dev_predictions.jsonl', original)
    # Validate fast DP against the independent scalar decoder at varied thresholds.
    thresholds = [oldthresholds, [-3.]*len(types), [0.]*len(types), [2.]*len(types)]
    fast = GridIntervals(keys, oldvalues, samples, minimum=-4.)
    for th in thresholds:
        expected = score(original, convert(predict(keys,oldvalues,th,len(samples)),types), types)['micro']
        tp, pred = fast.counts(th)
        assert int(tp[0]) == expected['tp'] and int(pred[0]) == expected['predicted']
    baseline = read_predictions(OLD/'crossfit_predictions.jsonl', original)
    base_report = score(original, baseline, types)
    cover = int(y.sum()); gold = sum(len(g) for _,g in samples)
    save('AUDIT.json', dict(dev_sha256=sha256(ROOT/'data/dev2.txt'), feature_sha256=sha256(OLD/'features.npz'),
        baseline=base_report, exact_frozen_replay=True, fast_scalar_agree=True, candidate_rows=len(keys),
        gold_covered=cover, gold=gold, candidate_recall=cover/gold, oracle_f1=2*cover/(gold+cover)))
    xplus = augment(keys,x)
    inner = np.array([int.from_bytes(hashlib.sha256(('leman-inner-20261021\0'+text).encode()).digest()[:8],'big')%3 for text,_ in samples])
    allids = np.arange(len(samples)); outcomes=[]; bundles={}
    for case in CASES:
        print('BEGIN',case['name'],now(),flush=True)
        xx = xplus if case['conflicts'] else x
        outvalues = np.full(len(keys), np.nan); predictions={}; fold_reports=[]
        for held in (0,1):
            fitids = allids[outer != held]; heldids = allids[outer == held]
            trainmask = outer[keys[:,0]] != held; heldmask = ~trainmask
            innervalues = np.full(len(keys), np.nan)
            for h in range(3):
                tr = trainmask & (inner[keys[:,0]] != h)
                va = trainmask & (inner[keys[:,0]] == h)
                m=fit_model(xx[tr],y[tr],case); innervalues[va]=logits(m,xx[va])
            assert np.isfinite(innervalues[trainmask]).all()
            fitted = calibrate(keys,innervalues,samples,fitids,len(types))
            m=fit_model(xx[trainmask],y[trainmask],case)
            outvalues[heldmask]=logits(m,xx[heldmask])
            pred=predict(keys[heldmask],outvalues[heldmask],fitted['thresholds_array'],len(samples))
            predictions.update({int(i):pred[i] for i in heldids})
            fold_reports.append(dict(held_fold=held,decoder=fitted))
        report=score(original,convert(predictions,types),types)
        result=dict(case=case, crossfit=report, folds=fold_reports,
            delta_vs_baseline_pp=100*(report['micro']['f1']-base_report['micro']['f1']))
        outcomes.append(result);bundles[case['name']]=(outvalues,predictions)
        save(case['name']+'.json',result)
        write_predictions(OUT/(case['name']+'_crossfit.jsonl'),predictions,samples,types)
        print('RESULT',case['name'],report['micro'],flush=True)
    best=max(outcomes,key=lambda z:z['crossfit']['micro']['f1'])
    adopt=best['delta_vs_baseline_pp'] >= .03
    interval=bootstrap(original,baseline,convert(bundles[best['case']['name']][1],types))
    result=dict(completed_utc=now(),baseline=base_report,cases=outcomes,best=best['case']['name'],
        adopt=adopt, paired_bootstrap=interval, seconds=time.monotonic()-t0, test_read=False)
    if adopt:
        case=best['case'];values,_=bundles[case['name']]
        fitted=calibrate(keys,values,samples,allids,len(types))
        fitted['thresholds']={t:v for t,v in zip(types,fitted['thresholds_array'])}
        model=fit_model(xplus if case['conflicts'] else x,y,case)
        joblib.dump(model,OUT/'selected_hgb.joblib'); save('selected_decoder.json',fitted)
        np.savez_compressed(OUT/'selected_dev_oof_scores.npz',keys=keys,scores=values)
        freeze=dict(frozen_utc=now(),selected_case=case,types=types,models=json.loads((FROZEN/'FREEZE.json').read_text())['models'],
            original_freeze_sha256=sha256(FROZEN/'FREEZE.json'),dev_sha256=sha256(ROOT/'data/dev2.txt'),
            selection_delta_pp=best['delta_vs_baseline_pp'],test_read_during_selection=False,
            files={n:sha256(OUT/n) for n in ['PROTOCOL.json','AUDIT.json','selected_hgb.joblib','selected_decoder.json']},
            source_sha256=spec['source_sha256'],evaluation_policy='one evaluation of this selected frozen version; no test-based retuning')
        save('SELECTED_FREEZE.json',freeze)
    save('SUMMARY.json',result)
    print('DONE',json.dumps(dict(best=result['best'],adopt=adopt,delta_pp=best['delta_vs_baseline_pp'],bootstrap=interval)),flush=True)

if __name__ == '__main__': main()
