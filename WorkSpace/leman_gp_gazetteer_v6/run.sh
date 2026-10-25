#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
python run_calibration.py 2>&1 | tee -a "$LEMAN_CALIBRATION_OUTPUT/calibrate.log"
python collect_results.py
