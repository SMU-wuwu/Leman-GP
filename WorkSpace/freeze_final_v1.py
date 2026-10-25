"""Freeze the user-selected v20 legacy12 result without copying large encoders."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import platform
import shutil

import numpy
import sklearn

ROOT = Path('/root/data/WorkSpace')
SOURCE = ROOT / 'output/leman_hgb_v20_seed47'
TARGET = ROOT / 'output/leman_final_v1_20261019'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if TARGET.exists():
        manifest = json.loads((TARGET/'FREEZE.json').read_text())
        for name, expected in manifest['files'].items():
            assert digest(TARGET/name) == expected, name
        print('Final version 1 already frozen; all hashes match:', TARGET)
        return
    TARGET.mkdir()
    for name in ['analysis_v20.json', 'features.npz', 'plan.json', 'RESULTS.md',
                 'independent_recount.json', 'dev_text_only.jsonl']:
        shutil.copy2(SOURCE/name, TARGET/name)
    shutil.copytree(SOURCE/'legacy12', TARGET/'legacy12')
    shutil.copytree(SOURCE/'source_snapshot', TARGET/'source_snapshot')
    caches = json.loads((SOURCE/'dev_caches.json').read_text())
    (TARGET/'caches').mkdir()
    dependencies, frozen_caches = {}, {}
    for name, original in caches.items():
        source = Path(original)
        dest = TARGET/'caches'/f'{name}.jsonl.gz'
        shutil.copy2(source, dest)
        with gzip.open(source, 'rt', encoding='utf-8') as stream:
            header = json.loads(next(stream))
        dependencies[name] = {'original_cache': original, 'cache_sha256': digest(source),
                              'header': header}
        frozen_caches[name] = str(dest)
    (TARGET/'dev_caches.json').write_text(json.dumps(frozen_caches, indent=2))
    (TARGET/'DEPENDENCIES.json').write_text(json.dumps(dependencies, ensure_ascii=False, indent=2))
    selected = json.loads((SOURCE/'analysis_v20.json').read_text())['cases']['legacy12']
    manifest = {
        'name': '最终定稿版本1', 'version': 'final_v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(SOURCE), 'selected_case': 'legacy12',
        'full_dev': selected['full_dev'], 'crossfit': selected['crossfit'],
        'parameters': selected['parameters'],
        'selection_authorization': 'User selected this round as final version 1 before further exploration.',
        'primary_report': 'full-dev calibrated exact character-span Micro F1',
        'test_accessed': False,
        'data_sha256': {name: digest(ROOT/'data'/name) for name in ['train2.txt','dev2.txt','label.txt']},
        'runtime': {'python': platform.python_version(), 'numpy': numpy.__version__,
                    'sklearn': sklearn.__version__},
        'large_checkpoint_policy': 'Original neural checkpoints are referenced by cache provenance; '
                                   'the dev caches and fitted HGB models are copied into this snapshot.',
        'files': {str(f.relative_to(TARGET)): digest(f) for f in sorted(TARGET.rglob('*')) if f.is_file()},
    }
    (TARGET/'FREEZE.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for file in TARGET.rglob('*'):
        if file.is_file():
            file.chmod(0o444)
    for folder in sorted((p for p in TARGET.rglob('*') if p.is_dir()), reverse=True):
        folder.chmod(0o555)
    TARGET.chmod(0o555)
    print('Frozen:', TARGET)
    print('Files:',len(manifest['files']))
    print('Full-dev:',100*selected['full_dev']['micro']['f1'])
    print('Crossfit:',100*selected['crossfit']['micro']['f1'])
    print('Freeze SHA256:',digest(TARGET/'FREEZE.json'))


if __name__ == '__main__':
    main()
