"""Dev-only HGB calibration of the existing 11-model pool plus seed47."""
from pathlib import Path
import hashlib
import json
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path('/root/data/WorkSpace')
sys.path[:0] = [str(ROOT / 'leman_gp_fgm_seed46_dual4090'),
                str(ROOT / 'leman_gp_fgm_v10_analysis'),
                str(ROOT / 'leman_gp_typecal_v9'),
                str(ROOT / 'leman_gp_typecal_v9/frozen_v4')]
from analyze import build_features
from common import named_predictions, read_cache, sha256, subset_score
from calibration_core import candidates_from_scores
from config import Config
from data import read_samples, write_json
from decoding import decode_sentence
from evaluate import score_predictions, write_predictions
from search import folds_for, predict, select_fit


PATHS = {
    'old': ROOT / 'leman_gp_typecal_v9/inputs/v6_upgraded_features.jsonl.gz',
    'fgm43': ROOT / 'output/leman_fgm_v10_analysis_seed43/fgm_features.jsonl.gz',
    'fgm44': ROOT / 'output/leman_fgm_v14_analysis_seed44/fgm_features.jsonl.gz',
    'ema43': ROOT / 'output/leman_ema_v12_seed43/ema_features.jsonl.gz',
    'fgm46': ROOT / 'output/leman_fgm_seed46_dual4090/dev_calibration/upgraded/dev_scored_candidates.jsonl.gz',
    'soup2': ROOT / 'output/leman_soup_v16/soup2_features.jsonl.gz',
    'soup3': ROOT / 'output/leman_soup_v16/soup3_features.jsonl.gz',
    'parent': ROOT / 'output/leman_inference_v7_seed42/parent_features.jsonl.gz',
    'v4ema': ROOT / 'output/leman_snapshot_v8_seed42/final_v4_features.jsonl.gz',
    'v2ema': ROOT / 'output/leman_snapshot_v8_seed42/final_v2_features.jsonl.gz',
    'rdrop': ROOT / 'output/leman_rdrop_v17_seed45/dev_calibration/upgraded/dev_scored_candidates.jsonl.gz',
    'fgm47': ROOT / 'output/leman_fgm_seed47_dual4090/dev_calibration/upgraded/dev_scored_candidates.jsonl.gz',
}
CASES = {
    'legacy12': {'max_iter': 180, 'learning_rate': 0.05, 'max_leaf_nodes': 15,
                 'min_samples_leaf': 35, 'l2_regularization': 2.0,
                 'negative_weight': 1.5, 'nested_decoder': False},
    'compact12': {'max_iter': 140, 'learning_rate': 0.05, 'max_leaf_nodes': 7,
                  'min_samples_leaf': 150, 'l2_regularization': 8.0,
                  'negative_weight': 1.0, 'nested_decoder': True},
    'balanced12': {'max_iter': 140, 'learning_rate': 0.05, 'max_leaf_nodes': 15,
                   'min_samples_leaf': 100, 'l2_regularization': 4.0,
                   'negative_weight': 1.0, 'nested_decoder': True},
}


def logits(model, x):
    p = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def refine(samples, candidates, types, cfg, fit_ids):
    """Reproduce the old HGB threshold search, using only the fitting IDs."""
    result = select_fit(samples, {0.0: candidates}, types, cfg, fit_ids, 'expanded')
    thresholds = np.asarray([result['thresholds'][t] for t in types], dtype=float)
    gold = {sid: set(map(tuple, samples[sid][1])) for sid in fit_ids}

    def counts(sid, th):
        pred = decode_sentence(candidates[sid], th, result['strategy'])
        return np.asarray([len(pred & gold[sid]), len(pred), len(gold[sid])])

    rows = {sid: counts(sid, thresholds) for sid in fit_ids}
    total = sum(rows.values(), np.zeros(3, dtype=np.int64))
    current = 2.0 * total[0] / (total[1] + total[2])
    for step, radius in ((0.25, 2.0), (0.05, 0.4), (0.01, 0.08)):
        for _ in range(2):
            changed = False
            for typ in range(len(types)):
                affected = [sid for sid in fit_ids if any(r[0] == typ for r in candidates[sid])]
                fixed = total - sum((rows[sid] for sid in affected), np.zeros(3, dtype=np.int64))
                center = thresholds[typ]
                best = None
                for value in np.arange(center-radius, center+radius+step*0.49, step):
                    proposed = thresholds.copy()
                    proposed[typ] = value
                    updates = {sid: counts(sid, proposed) for sid in affected}
                    newtotal = fixed + sum(updates.values(), np.zeros(3, dtype=np.int64))
                    f1 = 2.0 * newtotal[0] / (newtotal[1] + newtotal[2])
                    if f1 > current + 1e-12:
                        best = (value, updates, newtotal)
                        current = f1
                if best is not None:
                    thresholds[typ], updates, total = best
                    rows.update(updates)
                    changed = True
            if not changed:
                break
    result.update(thresholds=dict(zip(types, thresholds.tolist())), fit_micro_f1=current)
    return result


def main():
    started = time.perf_counter()
    output = ROOT / 'output/leman_hgb_v20_seed47'
    output.mkdir(exist_ok=True)
    caches = {name: read_cache(path) for name, path in PATHS.items()}
    types = caches['old'][0]['types']
    assert all(cache[0]['types'] == types for cache in caches.values())
    samples = read_samples(ROOT / 'data/dev2.txt', types)
    gaz = json.loads((ROOT / 'leman_gp_typecal_v9/inputs/v6_train_gazetteer.json').read_text())
    keys, values = build_features([cache[1] for cache in caches.values()], samples, types, gaz)
    key_array = np.asarray(keys, dtype=np.int32)
    x = np.column_stack([values, np.eye(len(types))[key_array[:, 1]]])
    gold = [set(map(tuple, spans)) for _, spans in samples]
    y = np.asarray([tuple(k[1:]) in gold[k[0]] for k in keys], dtype=np.uint8)
    sids = key_array[:, 0]
    outer = np.asarray(folds_for(samples), dtype=np.int8)
    inner = np.asarray([int.from_bytes(hashlib.sha256(
        ('leman-hgb-v20-inner-20261019\0' + text).encode()).digest()[:8], 'big') % 2
        for text, _ in samples])
    cfg = Config.from_environment()
    ids = list(range(len(samples)))
    baseline = named_predictions(ROOT / 'output/leman_rdrop_v17_analysis_seed45/crossfit_predictions.jsonl', types)
    report = {'models': list(PATHS), 'cache_sha256': {name: sha256(p) for name, p in PATHS.items()},
              'script_sha256': sha256(Path(__file__)),
              'dev_sha256': sha256(ROOT / 'data/dev2.txt'),
              'candidate_rows': len(keys), 'gold_covered': int(y.sum()),
              'gold': sum(map(len, gold)), 'baseline_crossfit': score_predictions(samples, baseline, types),
              'cases': {}, 'test_accessed': False,
              'scope': 'Exploratory dev calibration; neural checkpoints already dev-selected. '
                       'full_dev is evaluated on the HGB fitting data. Each crossfit prediction '
                       'uses an HGB trained on the opposite fixed text-hash fold.'}
    write_json(output / 'plan.json', {**report, 'configurations': CASES})
    np.savez_compressed(output / 'features.npz', keys=key_array, X=x, y=y, outer=outer, inner=inner)
    print(f'Candidates={len(keys)}, covered={int(y.sum())}, features={x.shape[1]}', flush=True)

    for name, spec in CASES.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        params = {k: v for k, v in spec.items() if k not in ('negative_weight', 'nested_decoder')}
        params.update(max_bins=127, early_stopping=False, random_state=42)

        def fit(rows):
            weights = np.where(y[rows] == 1, 1.0, spec['negative_weight'])
            return HistGradientBoostingClassifier(**params).fit(x[rows], y[rows], sample_weight=weights)

        cross = {}
        outer_scores = np.full(len(keys), np.nan)
        fit_oof = np.full((2, len(keys)), np.nan)
        fold_records = []
        for held in (0, 1):
            fit_ids = np.flatnonzero(outer != held).tolist()
            held_ids = np.flatnonzero(outer == held).tolist()
            train_rows = np.flatnonzero(outer[sids] != held)
            held_rows = np.flatnonzero(outer[sids] == held)
            if spec['nested_decoder']:
                for inner_held in (0, 1):
                    tr = train_rows[inner[sids[train_rows]] != inner_held]
                    val = train_rows[inner[sids[train_rows]] == inner_held]
                    m = fit(tr)
                    fit_oof[held, val] = logits(m, x[val])
                assert np.isfinite(fit_oof[held, train_rows]).all()
                fit_candidates = candidates_from_scores(
                    [keys[i] for i in train_rows], fit_oof[held, train_rows], len(samples))
            model = fit(train_rows)
            all_scores = logits(model, x)
            outer_scores[held_rows] = all_scores[held_rows]
            candidates = candidates_from_scores(keys, all_scores, len(samples))
            if spec['nested_decoder']:
                decoder = select_fit(samples, {0.0: fit_candidates}, types, cfg, fit_ids, 'expanded')
            else:
                decoder = refine(samples, candidates, types, cfg, fit_ids)
            pred = predict(candidates, decoder, types, held_ids)
            cross.update(pred)
            held_report = subset_score(samples, pred, types, held_ids)
            base = subset_score(samples, baseline, types, held_ids)
            delta = 100.0 * (held_report['micro']['f1'] - base['micro']['f1'])
            fold_records.append({'held': held, 'decoder': decoder, 'report': held_report,
                                 'delta_v17_pp': delta, 'threshold_fit': 'inner_oof' if spec['nested_decoder'] else 'in_sample'})
            joblib.dump(model, folder / f'hgb_excluding_fold{held}.joblib')
            write_json(folder / f'decoder_excluding_fold{held}.json', decoder)
            print(f'{name} held{held}: {100*held_report["micro"]["f1"]:.6f}%, delta={delta:+.6f}pp', flush=True)

        assert np.isfinite(outer_scores).all()
        model = fit(np.arange(len(keys)))
        full_scores = logits(model, x)
        full_candidates = candidates_from_scores(keys, full_scores, len(samples))
        # Keep the old full-dev fitting/score convention visible for direct comparison.
        full_decoder = refine(samples, full_candidates, types, cfg, ids)
        full = predict(full_candidates, full_decoder, types, ids)
        # Also save the full model with thresholds fit to OOF scores, for deployment comparison.
        oof_candidates = candidates_from_scores(keys, outer_scores, len(samples))
        oof_decoder = select_fit(samples, {0.0: oof_candidates}, types, cfg, ids, 'expanded')
        oof_threshold_full = predict(full_candidates, oof_decoder, types, ids)
        write_predictions(folder / 'crossfit_predictions.jsonl', samples, cross, types)
        write_predictions(folder / 'full_dev_predictions.jsonl', samples, full, types)
        write_predictions(folder / 'full_model_oof_thresholds_predictions.jsonl', samples, oof_threshold_full, types)
        joblib.dump(model, folder / 'hgb_full_dev.joblib')
        write_json(folder / 'decoder_full_dev.json', full_decoder)
        write_json(folder / 'decoder_oof_thresholds.json', oof_decoder)
        np.savez_compressed(folder / 'scores.npz', full=full_scores, crossfit=outer_scores, fit_oof=fit_oof)
        report['cases'][name] = {
            'parameters': spec, 'full_dev': score_predictions(samples, full, types),
            'crossfit': score_predictions(samples, cross, types), 'folds': fold_records,
            'full_model_oof_thresholds': score_predictions(samples, oof_threshold_full, types),
            'full_decoder': full_decoder, 'oof_decoder': oof_decoder}
        report['seconds'] = time.perf_counter() - started
        write_json(output / 'analysis_v20.json', report)
        row = report['cases'][name]
        print(f'{name}: full={100*row["full_dev"]["micro"]["f1"]:.6f}% '
              f'crossfit={100*row["crossfit"]["micro"]["f1"]:.6f}%', flush=True)


if __name__ == '__main__':
    main()
