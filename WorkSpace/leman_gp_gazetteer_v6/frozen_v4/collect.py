"""Create a small return ZIP without checkpoints, encoder weights or feature caches."""
from datetime import datetime, timezone
from pathlib import Path
import json
import zipfile

from config import Config
from data import sha256


def main():
    cfg = Config.from_environment()
    run = Path(cfg.output_dir)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = run / f'return_leman_finetune_v4_{stamp}.zip'
    files = [p for p in run.rglob('*') if p.is_file()
             and not any(part in ('cache', '__pycache__') for part in p.relative_to(run).parts)
             and p.suffix in ('.json', '.jsonl', '.csv', '.log', '.py', '.sh', '.md', '.gz')]
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(run).as_posix())
        manifest = {'files': {p.relative_to(run).as_posix(): sha256(p) for p in sorted(files)},
                    'checkpoint_files_included': False, 'test_executed_by_collector': False}
        archive.writestr('RETURN_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print(target)


if __name__ == '__main__':
    main()
