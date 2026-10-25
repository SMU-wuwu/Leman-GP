"""Collect v5 results without model weights or test execution."""
from datetime import datetime, timezone
import json
import zipfile

from common import paths, sha256


def main():
    _, output = paths()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = output / f'return_leman_calibration_v5_{stamp}.zip'
    files = [p for p in output.rglob('*') if p.is_file()
             and not any(part in ('cache', '__pycache__') for part in p.relative_to(output).parts)
             and p.suffix in ('.json', '.jsonl', '.csv', '.log', '.py', '.sh', '.md', '.txt')]
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(output).as_posix())
        archive.writestr('RETURN_MANIFEST.json', json.dumps({
            'files': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(files)},
            'checkpoint_files_included': False, 'test_executed_by_collector': False,
        }, ensure_ascii=False, indent=2))
    print(target, flush=True)


if __name__ == '__main__':
    main()
