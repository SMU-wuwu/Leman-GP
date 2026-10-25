"""Keep each model's GP and boundary scores from the same inference window."""
import numpy as np

from gazetteer import make_features


def best_window(rows):
    result = {}
    for typ, start, end, score, penalty in rows:
        key = (typ, start, end)
        if key not in result or (score, penalty) > result[key]:
            result[key] = (score, penalty)
    return result


def features(parent, upgraded, samples, types, family, gazetteer, floor=-32.0):
    texts = [text for text, _ in samples]
    if family == 'expanded_v4':
        return make_features(upgraded, texts, types, 'train_gazetteer', gazetteer)
    assert family == 'expanded_fusion'
    keys, values = [], []
    surfaces = gazetteer['surfaces']
    for sid in sorted(upgraded):
        p, u = best_window(parent[sid]), best_window(upgraded[sid])
        for typ, start, end in sorted(p.keys() | u.keys()):
            span = (typ, start, end)
            row = []
            for model in (p, u):
                score, penalty = model.get(span, (floor, -16.0))
                row.extend((score / 8.0, penalty, np.exp(penalty)))
            counts = surfaces.get(texts[sid][start:end], {})
            same = counts.get(types[typ], 0)
            row.extend((np.log1p(end-start), float(span in p), float(span in u),
                        np.log1p(same), np.log1p(sum(counts.values())-same)))
            keys.append((sid, typ, start, end))
            values.append(row)
    return keys, np.asarray(values, dtype=np.float64)
