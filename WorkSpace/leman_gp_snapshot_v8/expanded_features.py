"""Logistic score fusion using completed EMA snapshots only."""
import numpy as np


ROUTES = {'v4_final_ensemble': ['upgraded', 'final_v4'],
          'v2_final_ensemble': ['upgraded', 'final_v2'],
          'three_snapshot_ensemble': ['upgraded', 'final_v4', 'final_v2']}


def features(parent, upgraded, samples, types, family, gazetteer, floor=-32.0):
    records = {'upgraded': upgraded, **parent}
    keys, values = [], []
    for sid, (text, _) in enumerate(samples):
        models = []
        for name in ROUTES[family]:
            spans = {}
            for typ, start, end, score, penalty in records[name][sid]:
                key = (typ, start, end)
                spans[key] = max(spans.get(key, (-float('inf'), -float('inf'))), (score, penalty))
            models.append(spans)
        for span in sorted(set().union(*models)):
            typ, start, end = span
            row = []
            for model in models:
                score, penalty = model.get(span, (floor, -16.0))
                row.extend((score/8.0, penalty, np.exp(penalty)))
            counts = gazetteer['surfaces'].get(text[start:end], {})
            same = counts.get(types[typ], 0)
            row.extend((np.log1p(end-start), *[float(span in model) for model in models],
                        np.log1p(same), np.log1p(sum(counts.values())-same)))
            keys.append((sid, *span))
            values.append(row)
    return keys, np.asarray(values, dtype=np.float64)
