"""Independent character-span reader/counts for the revision evidence tables."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_bio(path):
    rows, letters, labels = [], [], []

    def finish():
        if not letters:
            return
        spans, active = set(), None
        for i, tag in enumerate(labels + ['O']):
            typ = tag[2:] if tag.startswith(('B-', 'I-')) else None
            if active and (not tag.startswith('I-') or typ != active[0]):
                spans.add((active[0], active[1], i))
                active = None
            if typ and active is None:
                active = (typ, i)
        rows.append((''.join(letters), spans))
        letters.clear()
        labels.clear()

    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            finish()
            continue
        token, tag = line.split()
        tag = tag.replace('_', '-')
        assert tag == 'O' or tag.startswith(('B-', 'I-'))
        letters.extend(token)
        labels.extend([tag] + [('I-' + tag[2:]) if tag.startswith('B-') else tag] * (len(token) - 1))
    finish()
    return rows


def read_predictions(path, samples):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        sid = row['sid']
        assert sid not in result and row['text'] == samples[sid][0]
        pred = {(e['type'], e['start'], e['end']) for e in row['predicted']}
        assert len(pred) == len(row['predicted'])
        for e in row['predicted']:
            assert 0 <= e['start'] < e['end'] <= len(row['text'])
            assert e['text'] == row['text'][e['start']:e['end']]
        if 'gold' in row:
            assert {(e['type'], e['start'], e['end']) for e in row['gold']} == samples[sid][1]
        result[sid] = pred
    assert set(result) == set(range(len(samples)))
    return result


def metrics(tp, predicted, gold):
    return {'tp': tp, 'fp': predicted-tp, 'fn': gold-tp, 'predicted': predicted,
            'support': gold, 'precision': tp/predicted if predicted else 0.,
            'recall': tp/gold if gold else 0., 'f1': 2*tp/(predicted+gold) if predicted+gold else 0.}


def score(samples, pred, types):
    counts = {t: [0, 0, 0] for t in types}
    for sid, (_, gold) in enumerate(samples):
        for j, spans in enumerate((gold & pred[sid], pred[sid], gold)):
            for t, _, _ in spans:
                counts[t][j] += 1
    per_type = {t: metrics(*v) for t, v in counts.items()}
    return {'micro': metrics(*(sum(v[i] for v in counts.values()) for i in range(3))),
            'macro13': sum(v['f1'] for v in per_type.values()) / len(types), 'per_type': per_type}
