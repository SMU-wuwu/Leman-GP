"""Bind the new seed/output to the unchanged completed v10 training code."""
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(os.environ.get('LEMAN_GP_ROOT', '/root/data/WorkSpace'))
    own = json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())
    for name, expected in own['files'].items():
        assert sha(HERE/name) == expected
    plan = json.loads((HERE/'EXPERIMENT_PLAN.json').read_text())
    base = root/'leman_gp_fgm_v10'
    assert sha(base/'PACKAGE_MANIFEST.json') == plan['base_package_manifest_sha256']
    for name, expected in json.loads((base/'PACKAGE_MANIFEST.json').read_text())['files'].items():
        assert sha(base/name) == expected
    assert sha(root/plan['training']['parent_path']) == plan['training']['parent_checkpoint_sha256']
    assert sha(root/'output/leman_fgm_v10_analysis_seed43/selection.json') == plan['baseline_selection_sha256']
    output = root/plan['training']['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    record = {'plan_sha256': sha(HERE/'EXPERIMENT_PLAN.json'), 'plan': plan}
    destination = output/'execution_plan.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == record
    else:
        destination.write_text(json.dumps(record, indent=2)+'\n')
    print('v14: seed44, 4 GPUs, 32/GPU, 3 epochs, existing v10 training code.', flush=True)


if __name__ == '__main__':
    main()
