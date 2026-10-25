"""Apply a context-enhanced span calibrator to unlabeled model candidates.

Inputs: sentences JSONL with sid/text; JSON mapping the 12 model names to
candidate gzip files in the existing header + {sid, rows} format.
"""
from bisect import bisect_right
from pathlib import Path
import argparse
import gzip
import json

import joblib
import numpy as np


def decode(rows, thresholds, strategy):
    spans = [(t, a, b, z-thresholds[t]) for t, a, b, z in rows if z > thresholds[t]]
    if strategy == 'all':
        return {(t, a, b) for t, a, b, _ in spans}
    if strategy != 'weighted_interval':
        raise ValueError(strategy)
    spans.sort(key=lambda x: (x[2], x[1], x[0]))
    ends = [x[2] for x in spans]
    optimum = [0.0]
    previous, taken = [], []
    for i, (_, start, _, weight) in enumerate(spans):
        prev = bisect_right(ends, start, hi=i)
        take = optimum[prev] + weight > optimum[i]
        optimum.append(optimum[prev]+weight if take else optimum[i])
        previous.append(prev)
        taken.append(take)
    result, i = set(), len(spans)
    while i:
        if taken[i-1]:
            result.add(tuple(spans[i-1][:3]))
            i = previous[i-1]
        else:
            i -= 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', type=Path, required=True)
    parser.add_argument('--caches', type=Path, required=True)
    parser.add_argument('--calibrator', type=Path, required=True)
    parser.add_argument('--gazetteer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--boundary-priors', type=Path, required=True)
    parser.add_argument('--consensus-summary', action='store_true')
    args = parser.parse_args()
    report = json.loads(args.parent_manifest.read_text())
    supplied = json.loads(args.caches.read_text())
    samples = {int(r['sid']): r['text'] for r in map(json.loads, args.sentences.read_text().splitlines())}
    assert sorted(samples) == list(range(len(samples)))
    caches, types = [], None
    for name in report['models']:
        with gzip.open(supplied[name], 'rt', encoding='utf-8') as stream:
            header = json.loads(next(stream))
            if types is None:
                types = header['types']
            assert header['types'] == types
            records = {int(r['sid']): r['rows'] for r in map(json.loads, stream)}
        assert records.keys() == samples.keys()
        caches.append(records)
    gaz = json.loads(args.gazetteer.read_text())
    keys, features = [], []
    for sid, text in samples.items():
        models = []
        for cache in caches:
            merged = {}
            for typ, a, b, score, penalty in cache[sid]:
                key = (typ, a, b)
                merged[key] = max(merged.get(key, (-float('inf'), -float('inf'))), (score, penalty))
            models.append(merged)
        for typ, a, b in sorted(set().union(*models)):
            counts = gaz['surfaces'].get(text[a:b], {})
            same = counts.get(types[typ], 0)
            extra = (np.log1p(b-a), np.log1p(same), np.log1p(sum(counts.values())-same))
            row = []
            for model in models:
                score, penalty = model.get((typ, a, b), (-4.0, -16.0))
                row.extend((score/8.0, penalty, np.exp(penalty), *extra))
            row.extend(float(t == typ) for t in range(len(types)))
            keys.append((sid, typ, a, b))
            features.append(row)
    model = joblib.load(args.calibrator/'hgb_full_dev.joblib')
    rows = {sid: [] for sid in samples}
    if keys:
        from context_priors_v22 import candidate_features
        base = np.asarray(features, dtype=np.float64)
        contexts = candidate_features(np.asarray(keys), [(samples[sid], []) for sid in range(len(samples))],
                                      joblib.load(args.boundary_priors))
        values = [base, contexts]
        if args.consensus_summary:
            gp, boundary = base[:,:72:6], base[:,2:72:6]
            values.append(np.column_stack((gp.max(1),gp.mean(1),gp.std(1),
                np.quantile(gp,0.25,axis=1),np.quantile(gp,0.75,axis=1),
                (gp>0).mean(1),boundary.mean(1),boundary.std(1))))
        matrix = np.column_stack(values)
        prob = np.clip(model.predict_proba(matrix)[:, 1], 1e-7, 1-1e-7)
        scores = np.log(prob)-np.log1p(-prob)
        for (sid, typ, a, b), z in zip(keys, scores):
            if z > -4.0:
                rows[sid].append((typ, a, b, float(z)))
    fitted = json.loads((args.calibrator/'decoder_full.json').read_text())
    thresholds = [fitted['thresholds'][t] for t in types]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        for sid, text in samples.items():
            spans = decode(rows[sid], thresholds, fitted['strategy'])
            stream.write(json.dumps({'sid': sid, 'text': text,
                'predicted': [{'type': types[t], 'start': a, 'end': b, 'text': text[a:b]}
                              for t, a, b in sorted(spans)]}, ensure_ascii=False)+'\n')
    print(f'Predictions saved: {args.output}')


if __name__ == '__main__':
    main()
