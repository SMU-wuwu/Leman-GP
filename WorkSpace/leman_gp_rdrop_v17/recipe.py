"""Freeze mild type emphasis using original training mentions, before windowing."""
from collections import Counter
from statistics import median


def training_recipe(samples, types, cfg, train_sha256):
    counts = Counter(typ for _, spans in samples for typ, _, _ in spans)
    reference = float(median(counts.get(typ, 0) for typ in range(len(types))))
    weights = [min(cfg.rare_type_cap, max(1.0, ((reference + 1) / (counts.get(typ, 0) + 1))
                                        ** cfg.rare_type_exponent)) for typ in range(len(types))]
    return {
        'source': 'train2.txt original gold mentions; each mention counted once before windowing',
        'train_sha256': train_sha256,
        'types': types,
        'counts': {name: counts.get(i, 0) for i, name in enumerate(types)},
        'median_type_count': reference,
        'type_weights': dict(zip(types, weights)),
        'ordered_weights': weights,
        'formula': f'clip(((median_train_type_count + 1)/(n_t + 1))**{cfg.rare_type_exponent:g}, 1, {cfg.rare_type_cap:g})',
        'loss': f'sum_b,t w_t * ({cfg.gp_positive_weight:g} * positive_logsumexp + negative_logsumexp) / (B * sum_t w_t)',
        'positive_weight': cfg.gp_positive_weight,
        'rare_type_exponent': cfg.rare_type_exponent,
        'rare_type_cap': cfg.rare_type_cap,
        'boundary_loss': 'unchanged parent MLCCE; boundary_weight=0.1',
        'test_accessed': False,
    }
