"""Complete the fixed wide-pool 2x2 feature ablation; no neural training."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path('/root/data/WorkSpace')
sys.path.insert(0, str(ROOT))
import remote_wide_v23 as old
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = ROOT / 'output/manuscript_revision_20261019/h2_consensus93'
FROZEN = ROOT / 'output/leman_final_v2_20261019'
WIDE = ROOT / 'output/leman_wide_v23'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    f = np.load(FROZEN / 'wide_context6_consensus8/features.npz')
    base = np.load(WIDE / 'wide_base85/features.npz')
    context = np.load(WIDE / 'wide_context6/features.npz')
    for other in (base, context):
        for key in ('keys', 'y', 'outer'):
            assert np.array_equal(other[key], f[key]), key
    assert np.array_equal(base['X'], f['X'][:, :85])
    assert np.array_equal(context['X'], f['X'][:, :91])
    X = np.column_stack((f['X'][:, :85], f['X'][:, 91:99]))
    keys = f['keys'].tolist()
    y, outer, sids = f['y'], f['outer'], f['keys'][:, 0]
    plan = json.loads((WIDE / 'PLAN.json').read_text())
    types = json.loads((FROZEN / 'FREEZE.json').read_text())['full_dev']['per_type']
    types = list(types)
    samples = old.read_samples(ROOT / 'data/dev2.txt', types)
    assert np.array_equal(outer, old.folds_for(samples))
    gold = [set(map(tuple, spans)) for _, spans in samples]
    assert np.array_equal(y, [tuple(k[1:]) in gold[k[0]] for k in keys])
    cfg = old.Config.from_environment()
    spec = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Complete pre-specified feature ablation; retain frozen final v2',
        'case': 'wide_consensus8', 'feature_columns': 93,
        'columns': '0:85 base; 91:99 consensus from frozen 99-column matrix',
        'parameters': plan['parameters'], 'negative_weight': 1.5,
        'candidate_rows': len(keys), 'gold_covered': int(y.sum()),
        'frozen_freeze_sha256': digest(FROZEN / 'FREEZE.json'),
        'feature_sha256': digest(FROZEN / 'wide_context6_consensus8/features.npz'),
        'dev_sha256': digest(ROOT / 'data/dev2.txt'),
        'fold_rule': 'sha256(leman-f1-v2-folds-20261016\\0 + text), first 8 bytes big endian modulo 2',
        'script_sha256': digest(__file__),
        'reused_source_sha256': {str(p): digest(p) for p in
            (ROOT/'remote_wide_v23.py', ROOT/'remote_hgb47_v20.py')},
        'environment': {'python': sys.version, 'numpy': np.__version__,
                        'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'neural_training': False, 'test_predictions_generated': False,
        'scope': 'Post-selection dev analysis; HGB and thresholds are fitted on the opposite fold for crossfit. Neural checkpoints were dev-selected.',
    }
    old.write_json(OUT / 'PLAN.json', spec)
    np.savez_compressed(OUT / 'features.npz', keys=f['keys'], X=X, y=y, outer=outer)
    predictions = {'full_dev': {}, 'crossfit': {}}
    scores = {k: np.full(len(keys), np.nan) for k in ('full', 'crossfit')}
    fits = {}
    ids = list(range(len(samples)))
    for part in ('full', 'held0', 'held1'):
        fit_ids = ids if part == 'full' else np.flatnonzero(outer != int(part[-1])).tolist()
        eval_ids = ids if part == 'full' else np.flatnonzero(outer == int(part[-1])).tolist()
        mask = np.ones(len(keys), dtype=bool) if part == 'full' else outer[sids] != int(part[-1])
        model = HistGradientBoostingClassifier(**plan['parameters']).fit(
            X[mask], y[mask], sample_weight=np.where(y[mask] == 1, 1.0, 1.5))
        z = old.logits(model, X)
        cand = old.candidates_from_scores(keys, z, len(samples))
        decoder = old.refine(samples, cand, types, cfg, fit_ids)
        pred = old.predict(cand, decoder, types, eval_ids)
        mode = 'full_dev' if part == 'full' else 'crossfit'
        predictions[mode].update(pred)
        fits[part] = {'decoder': decoder, 'report': old.subset_score(samples, pred, types, eval_ids)}
        model_name = 'hgb_full_dev.joblib' if part == 'full' else f'hgb_excluding_fold{part[-1]}.joblib'
        joblib.dump(model, OUT / model_name)
        old.write_json(OUT / f'decoder_{part}.json', decoder)
        if part == 'full':
            scores['full'][:] = z
        else:
            held = outer[sids] == int(part[-1])
            scores['crossfit'][held] = z[held]
        print(part, json.dumps(fits[part]['report']['micro']), flush=True)
    assert all(np.isfinite(v).all() for v in scores.values())
    for mode, pred in predictions.items():
        old.write_predictions(OUT / f'{mode}_predictions.jsonl', samples, pred, types)
    np.savez_compressed(OUT / 'scores.npz', **scores)
    result = {**spec, 'fits': fits, 'reports': {m: old.score_predictions(samples, p, types)
              for m, p in predictions.items()}, 'seconds': time.perf_counter() - started}
    assert digest(FROZEN / 'FREEZE.json') == spec['frozen_freeze_sha256']
    old.write_json(OUT / 'RESULT.json', result)
    print('COMPLETE', {k: v['micro']['f1'] for k, v in result['reports'].items()}, flush=True)


if __name__ == '__main__':
    main()
