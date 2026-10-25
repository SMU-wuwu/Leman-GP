#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd "$(dirname "$0")"
source ./env.sh
bash train.sh "$@"
# A graceful STOP returns success after saving. Calibrate only a finished run.
if ! python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ['LEMAN_GP_OUTPUT'])
results = json.loads((p / 'training_results.json').read_text()) if (p / 'training_results.json').exists() else {}
config = json.loads((p / 'run_config.json').read_text())['config']
raise SystemExit(0 if results.get('completed_epochs', 0) >= config['epochs'] and not (p / 'STOP').exists() else 1)
PY
then
    echo 'Training is paused; last.pt is saved. Rerun this command to resume.'
    exit 0
fi
bash calibrate.sh
python collect.py
