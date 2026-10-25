"""Fit expanded GP thresholds and a boundary penalty using fitting rows only."""
import hashlib
import numpy as np

from decoding import decode, decode_sentence, fit as original_fit, objective, sentence_counts
from evaluate import score_predictions


POLICY = {
    'floor': -4.0, 'upper': 16.0, 'step': 0.25,
    'boundary_weights': [0.0, 0.25, 0.5, 1.0],
    'min_support': 200, 'type_radius': 1.0, 'type_passes': 1,
    'primary': 'exact character-span Micro F1', 'secondary': 'Macro13',
    'boundary_penalty': '(logsigmoid(start_logit) + logsigmoid(end_logit)) / 2',
    'refined_score': 'sqrt(gp_head_size) * (GP_native_logit + lambda * boundary_penalty)',
    'window_merge': 'max AFTER adding the boundary penalty, separately for each lambda',
    'fold_rule': 'sha256(leman-f1-v2-folds-20261016\\0 + text), first 8 bytes big endian modulo 2',
}


def folds_for(samples):
    return [int.from_bytes(hashlib.sha256(('leman-f1-v2-folds-20261016\0' + text).encode()).digest()[:8],
                           'big') % 2 for text, _ in samples]


def scored_candidates(records, weight, scale):
    """Do not merge GP and boundary maxima from different windows."""
    candidates = {}
    for sid, rows in records.items():
        merged = {}
        for typ, a, b, score, penalty in rows:
            key = (typ, a, b)
            value = score + scale * weight * penalty
            if value > POLICY['floor']:
                merged[key] = max(merged.get(key, -float('inf')), value)
        candidates[sid] = [(*key, score) for key, score in sorted(merged.items())]
    return candidates


def fit_expanded(samples, candidates, types, ids, policy=POLICY):
    """Same objective, tie breaks and one type pass as the old fitter; larger upper bound."""
    ids = list(ids)
    size = len(types)
    gold = {sid: set(map(tuple, samples[sid][1])) for sid in ids}
    count = lambda sid, pred: sentence_counts(gold[sid], pred, size)
    grid = np.arange(policy['floor'], policy['upper'] + 1e-8, policy['step']).tolist()
    best_key, best = None, None
    for strategy in ('all', 'weighted_interval'):
        for threshold in grid:
            thresholds = [threshold] * size
            rows = {sid: count(sid, decode_sentence(candidates[sid], thresholds, strategy)) for sid in ids}
            counts = sum(rows.values(), np.zeros((size, 3), dtype=np.int64))
            key = (*objective(counts), threshold, int(strategy == 'all'))
            if best_key is None or key > best_key:
                best_key, best = key, (strategy, threshold, thresholds, rows, counts)
    strategy, global_threshold, thresholds, rows, counts = best
    eligible = [typ for typ in range(size) if counts[typ, 2] >= policy['min_support']]
    for typ in eligible:
        affected = [sid for sid in ids if any(t == typ for t, _, _, _ in candidates[sid])]
        fixed = counts - sum((rows[sid] for sid in affected), np.zeros_like(counts))
        best_objective = objective(counts)
        trials = [x for x in grid if abs(x - global_threshold) <= policy['type_radius'] + 1e-8]
        for trial in trials:
            proposal = thresholds.copy()
            proposal[typ] = trial
            updates = {sid: count(sid, decode_sentence(candidates[sid], proposal, strategy)) for sid in affected}
            proposed_counts = fixed + sum(updates.values(), np.zeros_like(counts))
            if objective(proposed_counts) > best_objective:
                selected = (trial, updates, proposed_counts)
                best_objective = objective(proposed_counts)
        if best_objective > objective(counts):
            thresholds[typ], updates, counts = selected
            rows.update(updates)
    micro, macro = objective(counts)
    return {'strategy': strategy, 'global_threshold': global_threshold,
            'thresholds': dict(zip(types, thresholds)), 'fit_sentences': len(ids),
            'fit_micro_f1': micro, 'fit_macro_f1': macro,
            'eligible': [types[t] for t in eligible],
            'fit_support': {name: int(counts[t, 2]) for t, name in enumerate(types)},
            'min_support': policy['min_support'],
            'global_upper_bound_hit': global_threshold == policy['upper']}


def predict(candidates, fitted, types, ids):
    return decode(candidates, [fitted['thresholds'][typ] for typ in types], fitted['strategy'], ids)


def select_fit(samples, all_candidates, types, cfg, ids, family, policy=POLICY):
    """Lambda, strategy and thresholds are all chosen on these fitting IDs."""
    ids = list(ids)
    weights = [0.0] if family in ('baseline', 'expanded') else policy['boundary_weights']
    choices = []
    for weight in weights:
        candidates = all_candidates[weight]
        # Keep the exact previous fitter as a candidate as well as the expanded one.
        fitted_options = [original_fit(samples, candidates, types, cfg, ids)]
        if family != 'baseline':
            fitted_options.append(fit_expanded(samples, candidates, types, ids, policy))
        for fitted in fitted_options:
            fitted = {**fitted, 'boundary_weight': weight}
            prediction = predict(candidates, fitted, types, ids)
            counts = sum((sentence_counts(set(map(tuple, samples[sid][1])), prediction[sid], len(types))
                          for sid in ids), np.zeros((len(types), 3), dtype=np.int64))
            micro, macro = objective(counts)
            fitted.update(fit_micro_f1=micro, fit_macro_f1=macro)
            # If the metrics tie, prefer no boundary penalty and the original grid.
            key = (micro, macro, -weight, -int(fitted.get('global_upper_bound_hit') is not None))
            choices.append((key, fitted))
        print(f'  fit {family}, lambda={weight:g}, sentences={len(ids)}', flush=True)
    return max(choices, key=lambda pair: pair[0])[1]


def compare(samples, records, types, cfg, scale, policy=POLICY):
    all_candidates = {w: scored_candidates(records, w, scale) for w in policy['boundary_weights']}
    ids = list(range(len(samples)))
    folds = folds_for(samples)
    reports, predictions = {}, {}
    for family in ('baseline', 'expanded', 'boundary'):
        fitted = select_fit(samples, all_candidates, types, cfg, ids, family, policy)
        full = predict(all_candidates[fitted['boundary_weight']], fitted, types, ids)
        cross, fold_records = {}, []
        for held in (0, 1):
            fit_ids = [sid for sid in ids if folds[sid] != held]
            held_ids = [sid for sid in ids if folds[sid] == held]
            fit_half = select_fit(samples, all_candidates, types, cfg, fit_ids, family, policy)
            cross.update(predict(all_candidates[fit_half['boundary_weight']], fit_half, types, held_ids))
            fold_records.append({'held_out_fold': held, 'evaluated_sentences': len(held_ids), 'fit': fit_half})
        full_metrics = score_predictions(samples, full, types)
        cross_metrics = score_predictions(samples, cross, types)
        for name, metrics in [('full_dev', full_metrics), ('crossfit', cross_metrics)]:
            metrics['overlap_policy'] = ('selected on full dev' if name == 'full_dev'
                                         else 'selected on opposite dev half')
        reports[family] = {'decoder': fitted, 'full_dev': full_metrics, 'crossfit': cross_metrics,
                           'fold_records': fold_records}
        predictions[family] = {'full_dev': full, 'crossfit': cross}
        print(f'{family}: full Micro={full_metrics["micro"]["f1"]:.6f}, '
              f'crossfit Micro={cross_metrics["micro"]["f1"]:.6f}, '
              f'crossfit Macro={cross_metrics["macro_f1"]:.6f}', flush=True)
    baseline = reports['baseline']
    for report in reports.values():
        report['delta_vs_baseline_pp'] = {
            name: {'micro': 100 * (report[name]['micro']['f1'] - baseline[name]['micro']['f1']),
                   'macro': 100 * (report[name]['macro_f1'] - baseline[name]['macro_f1'])}
            for name in ('full_dev', 'crossfit')}
    # Retain the previous decoder if the exploratory settings do not help cross-fit Micro.
    eligible = [family for family, report in reports.items()
                if report['full_dev']['micro']['f1'] >= baseline['full_dev']['micro']['f1']
                and (family == 'baseline' or report['crossfit']['micro']['f1'] > baseline['crossfit']['micro']['f1'])]
    chosen = max(eligible, key=lambda family: (reports[family]['crossfit']['micro']['f1'],
                                              reports[family]['full_dev']['micro']['f1'],
                                              family == 'baseline'))
    return {'chosen': chosen, 'reports': reports, 'folds': folds,
            'policy': policy, 'test_accessed': False,
            'scope': 'Exploratory dev-only comparison. Each half fits lambda, strategy and thresholds '
                     'without its held-out labels. Checkpoint and final family were selected using dev; '
                     'this is not an independent test estimate.'}, predictions
