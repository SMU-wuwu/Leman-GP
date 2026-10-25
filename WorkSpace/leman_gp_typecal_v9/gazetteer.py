"""Add two soft train-mention features; never force an entity prediction."""
from collections import Counter, defaultdict
import numpy as np

from calibration_core import feature_rows


def build_gazetteer(train_samples, types, train_sha256):
    counts = defaultdict(Counter)
    for text, spans in train_samples:
        for typ, start, end in spans:
            counts[text[start:end]][types[typ]] += 1
    return {'train_sha256': train_sha256, 'types': types,
            'sentences': len(train_samples),
            'mention_count': sum(sum(row.values()) for row in counts.values()),
            'surfaces': {surface: dict(sorted(row.items())) for surface, row in sorted(counts.items())},
            'source': 'train2 original gold mentions, counted once before windowing',
            'test_accessed': False}


def make_features(records, texts, types, family, gazetteer):
    keys, values = feature_rows({}, records, 'single_v4')
    if family in ('v5', 'single_v4'):
        return keys, values
    if family != 'train_gazetteer':
        raise ValueError(f'Unknown feature family: {family}')
    surfaces = gazetteer['surfaces']
    extra = []
    for sid, typ, start, end in keys:
        counts = surfaces.get(texts[sid][start:end], {})
        same = counts.get(types[typ], 0)
        other = sum(counts.values()) - same
        extra.append((np.log1p(same), np.log1p(other)))
    return keys, np.column_stack((values, np.asarray(extra, dtype=np.float64)))
