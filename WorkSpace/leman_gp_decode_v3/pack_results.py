"""Return decoder reports, scored dev candidates and sources; no weights."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile


def main():
    root = Path(os.environ.get('LEMAN_GP_ROOT', '/root/data/WorkSpace'))
    seed = os.environ.get('LEMAN_GP_SEED', '42')
    output = Path(os.environ.get('LEMAN_REFINE_OUTPUT', str(root / 'output' / f'leman_decode_v3_seed{seed}')))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = output / f'return_leman_decode_v3_{stamp}.zip'
    files = sorted(p for p in output.rglob('*') if p.is_file()
                   and not any(part in ('cache', '__pycache__') for part in p.relative_to(output).parts)
                   and p.suffix in ('.py', '.sh', '.json', '.jsonl', '.md', '.gz', '.log'))
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for path in files:
            name = path.relative_to(output).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr('RETURN_MANIFEST.json', json.dumps({'files': hashes, 'checkpoint_files_included': False,
                                                           'test_executed_by_collector': False}, indent=2))
    print(f'Return ZIP: {target}', flush=True)


if __name__ == '__main__':
    main()
