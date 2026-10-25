"""Dev-only threshold fitting and globally optimal flat interval decoding."""
from bisect import bisect_right
import hashlib
import numpy as np

from evaluate import score_predictions


def decode_sentence(candidates, thresholds, strategy):
    spans = [(t, a, b, score - thresholds[t]) for t, a, b, score in candidates
             if score > thresholds[t]]
    if strategy == 'all':
        return {(t, a, b) for t, a, b, _ in spans}
    if strategy != 'weighted_interval':
        raise ValueError(strategy)
    spans.sort(key=lambda x: (x[2], x[1], x[0]))
    ends = [x[2] for x in spans]
    optimum = [0.0] * (len(spans) + 1)
    previous, take = [], []
    for i, (_, start, _, weight) in enumerate(spans):
        prev = bisect_right(ends, start, hi=i)
        chosen = optimum[prev] + weight > optimum[i]
        optimum[i + 1] = optimum[prev] + weight if chosen else optimum[i]
        previous.append(prev)
        take.append(chosen)
    selected, i = set(), len(spans)
    while i:
        if take[i - 1]:
            selected.add(tuple(spans[i - 1][:3]))
            i = previous[i - 1]
        else:
            i -= 1
    return selected


def decode(candidates, thresholds, strategy, ids=None):
    return {sid: decode_sentence(candidates[sid], thresholds, strategy)
            for sid in (sorted(candidates) if ids is None else ids)}


def sentence_counts(gold, pred, num_types):
    counts = np.zeros((num_types, 3), dtype=np.int64)
    for t, _, _ in gold & pred:
        counts[t, 0] += 1
    for t, _, _ in pred:
        counts[t, 1] += 1
    for t, _, _ in gold:
        counts[t, 2] += 1
    return counts


def objective(counts):
    tp, pred, gold = counts.sum(0)
    micro = 2 * tp / (pred + gold) if pred + gold else 0.0
    denominator = counts[:, 1] + counts[:, 2]
    per_type = np.divide(2 * counts[:, 0], denominator, out=np.zeros(len(counts)), where=denominator > 0)
    return float(micro), float(per_type.mean())


def fit(samples, candidates, types, cfg, ids=None):
    """Select global threshold/strategy, then one coordinate pass, all by dev Micro.

    Per-type fitting requires >=200 gold in the fitting partition. Rare types
    keep the global threshold. A type only changes if total Micro improves.
    """
    ids = sorted(candidates) if ids is None else list(ids)
    gold = {sid: set(map(tuple, samples[sid][1])) for sid in ids}
    num_types = len(types)
    count = lambda sid, pred: sentence_counts(gold[sid], pred, num_types)
    grid = np.arange(cfg.calibration_floor, -cfg.calibration_floor + 1e-8,
                     cfg.calibration_step).tolist()
    best_key, best = None, None
    for strategy in ('all', 'weighted_interval'):
        for threshold in grid:
            thresholds = [threshold] * num_types
            rows = {sid: count(sid, decode_sentence(candidates[sid], thresholds, strategy)) for sid in ids}
            counts = sum(rows.values(), np.zeros((num_types, 3), dtype=np.int64))
            key = (*objective(counts), threshold, int(strategy == 'all'))
            if best_key is None or key > best_key:
                best_key, best = key, (strategy, threshold, thresholds, rows, counts)
    strategy, global_threshold, thresholds, rows, counts = best
    supports = {name: int(counts[t, 2]) for t, name in enumerate(types)}
    eligible = [t for t in range(num_types) if counts[t, 2] >= cfg.calibration_min_support]
    for typ in eligible:
        # Other sentences cannot change under this type's threshold adjustment.
        affected = [sid for sid in ids if any(t == typ for t, _, _, _ in candidates[sid])]
        fixed = counts - sum((rows[sid] for sid in affected), np.zeros_like(counts))
        selected_threshold, selected_rows, selected_counts = thresholds[typ], None, counts
        selected_objective = objective(counts)
        trials = sorted(set([global_threshold] + [x for x in grid if abs(x - global_threshold) <= 1.0 + 1e-8]))
        for trial in trials:
            trial_thresholds = thresholds.copy()
            trial_thresholds[typ] = trial
            updates = {sid: count(sid, decode_sentence(candidates[sid], trial_thresholds, strategy)) for sid in affected}
            trial_counts = fixed + sum(updates.values(), np.zeros_like(counts))
            trial_objective = objective(trial_counts)
            if trial_objective > selected_objective:
                selected_threshold, selected_rows = trial, updates
                selected_counts, selected_objective = trial_counts, trial_objective
        thresholds[typ] = selected_threshold
        if selected_rows is not None:
            rows.update(selected_rows)
            counts = selected_counts
    return {'strategy': strategy, 'global_threshold': global_threshold,
            'thresholds': dict(zip(types, thresholds)), 'fit_support': supports,
            'eligible': [types[t] for t in eligible], 'min_support': cfg.calibration_min_support,
            'selection_metric': 'exact original character-span micro F1',
            'fit_sentences': len(ids), 'fit_micro_f1': objective(counts)[0]}


def crossfit(samples, candidates, types, cfg):
    # Same text always stays in one half. Split does not consult any gold label.
    fold = [int.from_bytes(hashlib.sha256(('leman-f1-v2-folds-20261016\0' + text).encode()).digest()[:8], 'big') % 2
            for text, _ in samples]
    predictions, records = {}, []
    for held in (0, 1):
        train_ids = [i for i, x in enumerate(fold) if x != held]
        eval_ids = [i for i, x in enumerate(fold) if x == held]
        fitted = fit(samples, candidates, types, cfg, train_ids)
        thresholds = [fitted['thresholds'][t] for t in types]
        predictions.update(decode(candidates, thresholds, fitted['strategy'], eval_ids))
        records.append({'held_out_fold': held, 'fit': fitted, 'evaluated_sentences': len(eval_ids)})
    metrics = score_predictions(samples, predictions, types)
    return metrics, predictions, {'folds': fold, 'records': records,
                                 'scope': 'threshold/decode cross-fit; checkpoint already selected with full dev'}
