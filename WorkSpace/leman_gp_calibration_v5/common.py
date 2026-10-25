"""Shared paths and immutable input identities for the v5 decoder experiment."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'frozen_v4'))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paths():
    root = Path(os.environ.get('LEMAN_GP_ROOT', '/root/data/WorkSpace')).resolve()
    output = Path(os.environ.get('LEMAN_CALIBRATION_OUTPUT',
                                root / 'output/leman_calibration_v5_seed42')).resolve()
    return root, output


def read_cache(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        records = {row['sid']: row['rows'] for row in map(json.loads, stream)}
    return header, records


def verify_package():
    manifest = read_json(PACKAGE / 'PACKAGE_MANIFEST.json')
    for name, expected in manifest['files'].items():
        if sha256(PACKAGE / name) != expected:
            raise ValueError(f'Package file changed: {name}')
    return manifest


def subset_score(samples, predictions, types, ids):
    from evaluate import score_predictions
    ids = list(ids)
    return score_predictions([samples[sid] for sid in ids],
                             {i: predictions[sid] for i, sid in enumerate(ids)}, types)


def named_predictions(path, types):
    mapping = {name: i for i, name in enumerate(types)}
    predictions = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        predictions[row['sid']] = {(mapping[s['type']], s['start'], s['end']) for s in row['predicted']}
    return predictions
