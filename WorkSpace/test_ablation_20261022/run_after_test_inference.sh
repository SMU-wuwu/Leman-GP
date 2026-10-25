#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/WorkSpace
REMOTE_OUT="$ROOT/output/test_ablation_20261022"
REMOTE_ANALYSIS="$ROOT/output/test_ablation_analysis_20261022"
VARIANTS=(F no_lexicon no_bmes no_recurrence no_gru_linear no_rope no_attention max_words3 max_words5 gp_reset crf uniform_lexicon fixed_gate no_char no_boundary fgm_control no_fgm clean_two_pass weighted_control unit_loss encoder_gp)

while true; do
  complete=0
  for variant in "${VARIANTS[@]}"; do
    if [[ -f "$REMOTE_OUT/$variant/COMPLETE.json" ]]; then
      complete=$((complete+1))
    fi
  done
  [[ "$complete" -eq "${#VARIANTS[@]}" ]] && break
  sleep 45
done

for variant in "${VARIANTS[@]}"; do
  python3 - "$REMOTE_OUT/$variant" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
meta=json.loads((p/'COMPLETE.json').read_text())
protocol=json.loads((p/'PROTOCOL.json').read_text())
assert meta['split']==protocol['split']=='test'
assert meta['all_windows_evaluated'] is True
assert meta['optimizer_steps']==0 and protocol['optimizer_steps']==0
assert protocol['training'] is False and protocol['device']=='cpu'
rows=(p/'test_predictions.jsonl').read_text().splitlines()
assert len(rows)==5974
PY
done

mkdir -p "$REMOTE_ANALYSIS"
cp "$ROOT/test_ablation_20261022/staging/FINAL_TEST_METRICS.json" "$REMOTE_ANALYSIS/FINAL_TEST_METRICS.json"
python3 "$ROOT/test_ablation_20261022/analyze_test.py" --input "$REMOTE_OUT" --out "$REMOTE_ANALYSIS" --bootstrap 4000 > "$REMOTE_ANALYSIS/analyze.log" 2>&1

python3 "$ROOT/test_ablation_20261022/probe_test.py" > "$REMOTE_ANALYSIS/probe.log" 2>&1
python3 "$ROOT/test_ablation_20261022/summarize_test_mechanisms.py" \
  --panel "$REMOTE_OUT/test_mechanism_panel" \
  --predictions "$REMOTE_OUT/F/test_predictions.jsonl" \
  --out "$REMOTE_ANALYSIS" > "$REMOTE_ANALYSIS/summarize_mechanisms.log" 2>&1

for shard in {0..7}; do
  python3 "$ROOT/test_ablation_20261022/frozen_interventions_test.py" \
    --shard "$shard" --shards 8 --out "$REMOTE_ANALYSIS/intervention_shards/shard_$shard" \
    > "$REMOTE_ANALYSIS/intervention_shard_$shard.log" 2>&1 &
done
wait
python3 "$ROOT/test_ablation_20261022/merge_interventions.py" \
  --input "$REMOTE_ANALYSIS/intervention_shards" --shards 8 \
  --out "$REMOTE_ANALYSIS/frozen_interventions"
python3 "$ROOT/test_ablation_20261022/analyze_test_interventions.py" \
  --input "$REMOTE_ANALYSIS/frozen_interventions" \
  --full "$REMOTE_OUT/F/test_predictions.jsonl" \
  --out "$REMOTE_ANALYSIS" > "$REMOTE_ANALYSIS/analyze_interventions.log" 2>&1

python3 - "$REMOTE_ANALYSIS" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/'TEST_AUDIT.json').read_text())
m=json.loads((p/'MECHANISM_TEST_AUDIT.json').read_text())
i=json.loads((p/'FROZEN_INTERVENTIONS_TEST_AUDIT.json').read_text())
assert a['split']=='AgCNER test' and a['weights_changed'] is False and a['optimizer_steps']==0
assert m['split']=='AgCNER test' and m['training'] is False and m['optimizer_steps']==0
assert i['split']=='AgCNER test' and i['training'] is False and i['optimizer_steps']==0
(p/'PIPELINE_COMPLETE.json').write_text(json.dumps({'test_inference_variants':a['variants'],'test_sentences':a['sentences'],'test_mechanism_panel_sentences':m['panel_sentences'],'frozen_intervention_conditions':len(i['conditions']),'weights_changed':False,'optimizer_steps':0,'split':'AgCNER test'},indent=2)+'\n')
PY
