"""Archive the user-selected v23 result as final version 2; no fitting."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import joblib
import numpy
import sklearn

ROOT = Path('/root/data/WorkSpace')
SOURCE = ROOT / 'output/leman_wide_v23'
PARENT = ROOT / 'output/leman_final_v1_20261019'
TARGET = ROOT / 'output/leman_final_v2_20261019'
CASE = 'wide_context6_consensus8'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def verify(folder, hashes):
    for name, expected in hashes.items():
        assert digest(folder / name) == expected, name


def main():
    if TARGET.exists():
        verify(TARGET, read(TARGET / 'FREEZE.json')['files'])
        print('Version 2 already frozen and verified:', TARGET)
        return
    parent = read(PARENT / 'FREEZE.json')
    summary = read(SOURCE / 'summary.json')
    plan = read(SOURCE / 'PLAN.json')
    parent_hash = digest(PARENT / 'FREEZE.json')
    assert parent_hash == summary['parent_freeze_sha256'] == plan['parent_freeze_sha256']
    verify(PARENT, parent['files'])
    source_hashes = read(SOURCE / 'ARTIFACT_HASHES.json')
    verify(SOURCE, source_hashes)
    caches = read(SOURCE / 'dev_caches.json')
    for name, path in caches.items():
        assert digest(Path(path)) == plan['cache_sha256'][name], name
    data_hashes = {name: digest(ROOT / 'data' / name) for name in parent['data_sha256']}
    assert data_hashes == parent['data_sha256']
    selected = summary['cases'][CASE]
    recount = read(SOURCE / 'independent_recount.json')
    for label in ['full_dev', 'crossfit']:
        assert selected['reports'][label]['micro']['f1'] == recount[f'{CASE}/{label}']['micro_f1']
    assert abs(selected['reports']['full_dev']['micro']['f1'] - .9525971360266596) < 1e-14

    stage = TARGET.with_name(TARGET.name + '.preparing')
    stage.mkdir()
    (stage / 'provenance').mkdir()
    for name in ['summary.json', 'PLAN.json', 'EXPLORATION_BEST.json',
                 'ARTIFACT_HASHES.json', 'independent_recount.json', 'RESULTS.md',
                 'dev_caches.json', 'REPLAY_COMMAND.txt']:
        shutil.copy2(SOURCE / name, stage / 'provenance' / name)
    shutil.copy2(PARENT / 'FREEZE.json', stage / 'provenance' / 'parent_FREEZE.json')
    shutil.copytree(SOURCE / CASE, stage / CASE)
    shutil.copytree(SOURCE / 'source_snapshot', stage / 'source_snapshot')
    shutil.copy2(Path(__file__), stage / 'source_snapshot' / 'freeze_final_v2.py')
    for name in ['train_boundary_priors.joblib', 'v6_train_gazetteer.json']:
        shutil.copy2(SOURCE / name, stage / name)
    for name in ['analysis_v20.json', 'dev_text_only.jsonl']:
        shutil.copy2(PARENT / name, stage / name)
    (stage / 'caches').mkdir()
    dependencies, frozen_caches = {}, {}
    for name, original in caches.items():
        relative = f'caches/{name}.jsonl.gz'
        shutil.copy2(original, stage / relative)
        with gzip.open(stage / relative, 'rt', encoding='utf-8') as stream:
            header = json.loads(next(stream))
        dependencies[name] = {'original_cache': original, 'cache_sha256': digest(stage / relative),
                              'header': header}
        frozen_caches[name] = relative
    write(stage / 'dev_caches.json', frozen_caches)
    write(stage / 'DEPENDENCIES.json', dependencies)
    write(stage / 'SELECTED_RESULT.json', selected)
    (stage / 'requirements_replay.txt').write_text(
        f'numpy=={numpy.__version__}\nscikit-learn=={sklearn.__version__}\njoblib=={joblib.__version__}\n')
    replay_script = '''#!/usr/bin/env bash
set -euo pipefail
snapshot_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$snapshot_dir"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -B source_snapshot/predict_hgb_v22.py \\
  --sentences dev_text_only.jsonl --caches dev_caches.json \\
  --calibrator wide_context6_consensus8 --gazetteer v6_train_gazetteer.json \\
  --parent-manifest analysis_v20.json --boundary-priors train_boundary_priors.joblib \\
  --consensus-summary --output "${1:-../leman_final_v2_dev_replay.jsonl}"
'''
    (stage / 'replay_dev.sh').write_text(replay_script)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1')
    output = stage / 'replay_verified.jsonl'
    subprocess.run([sys.executable, '-B', 'source_snapshot/predict_hgb_v22.py',
                    '--sentences', 'dev_text_only.jsonl', '--caches', 'dev_caches.json',
                    '--calibrator', CASE, '--gazetteer', 'v6_train_gazetteer.json',
                    '--parent-manifest', 'analysis_v20.json',
                    '--boundary-priors', 'train_boundary_priors.joblib',
                    '--consensus-summary', '--output', str(output)],
                   cwd=stage, env=env, check=True)
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    expected = [json.loads(line) for line in (stage / CASE / 'full_dev_predictions.jsonl').read_text().splitlines()]
    assert len(actual) == len(expected) == 5963
    for a, b in zip(actual, expected):
        assert a['sid'] == b['sid'] and a['text'] == b['text']
        spans = lambda row: {(r['type'], r['start'], r['end']) for r in row['predicted']}
        assert spans(a) == spans(b), a['sid']
    assert sum(len(r['predicted']) for r in actual) == 20629
    write(stage / 'REPLAY_VERIFICATION.json', {
        'gold_free_prediction_replay_equal': True, 'sentences': len(actual),
        'predicted_spans': 20629, 'replayed_sha256': digest(output),
        'source_prediction_sha256': digest(stage / CASE / 'full_dev_predictions.jsonl'),
        'fit_or_search_performed': False})
    reports = selected['reports']
    deltas = {name: 100 * (reports[name]['micro']['f1'] - parent[name]['micro']['f1'])
              for name in ['full_dev', 'crossfit']}
    (stage / 'README.md').write_text(f'''# 最终定稿版本2

已按用户指示将 v23 / {CASE} 保存为最终定稿版本2。

| 指标 | 定稿版本1 | 定稿版本2 | 变化（百分点） |
|---|---:|---:|---:|
| Full-dev Micro F1 | 95.0034% | 95.2597% | +{deltas['full_dev']:.4f} |
| 两折 cross-fit Micro F1 | 94.3927% | 94.4762% | +{deltas['crossfit']:.4f} |
| Full-dev Macro13 | 89.1796% | 90.2001% | +1.0205 |

评价使用原始字符精确跨度和实体类型，共 5,963 句、20,782 个 gold 实体。
Full-dev 使用全量 dev 拟合的 HGB 和类型阈值，并在该 dev 上评分；cross-fit 使用两折留出预测。
以上是 dev 指标。该冻结过程未读取 test，也未重新训练或重新搜索参数。

## 固定配置

12 路候选融合，seed46/47 使用 floor=-32 的宽候选；共 23,769 个候选。
85 维原特征 + 6 维 train2 边界上下文先验 + 8 维共识统计，共 99 维。
HGB 参数见 FREEZE.json，类型阈值见 {CASE}/decoder_full.json。

## 保存内容

- {CASE}/：全量和两折 HGB 模型、阈值、特征、分数与预测。
- caches/：完整 12 路 dev 候选；dev_text_only.jsonl：无标签输入。
- train_boundary_priors.joblib、v6_train_gazetteer.json：训练集统计和词典。
- source_snapshot/：实验源码、推理入口和本次归档脚本。
- FREEZE.json：定稿身份、指标、参数、数据哈希及逐文件 SHA256。
- DEPENDENCIES.json：神经 checkpoint 的原始来源和缓存头内已有哈希。
- provenance/：历史探索记录，保留原始状态；本版本身份以 FREEZE.json 为准。

大体积神经 checkpoint 仍引用服务器原文件；本快照保存了重放 dev 结果所需的候选及全部校准模型。
REPLAY_VERIFICATION.json 记录了新目录下无标签重放：5,963 句、20,629 个预测跨度全部一致。

## 重放

使用 requirements_replay.txt 对应依赖，在此目录运行 `bash replay_dev.sh`。
默认输出为上级目录的 leman_final_v2_dev_replay.jsonl，也可将绝对输出路径作为第一个参数。

服务器归档位置：{TARGET}
''')
    manifest = {
        'name': '最终定稿版本2', 'version': 'final_v2', 'status': 'frozen',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(SOURCE), 'selected_case': CASE,
        'selection_authorization': '用户：好的，将新的最优结果保存为定稿版本2。',
        'parent_version': str(PARENT), 'parent_freeze_sha256': parent_hash,
        'source_artifact_manifest_sha256': digest(SOURCE / 'ARTIFACT_HASHES.json'),
        **reports, 'delta_vs_final_v1_percentage_points': deltas,
        'primary_report': 'Full-dev fitted exact character-span Micro F1',
        'models': plan['models'], 'parameters': plan['parameters'],
        'negative_weight': plan['negative_weight'], 'feature_columns': selected['feature_columns'],
        'context_prior': {'source': 'train2', 'strength': 20.0, 'cap': 8.0},
        'candidate_rows': plan['candidate_rows'], 'gold_covered': plan['gold_covered'],
        'data_sha256': data_hashes, 'test_accessed_during_freeze': False,
        'new_neural_training': False, 'new_calibrator_fitting': False,
        'runtime': {'python': platform.python_version(), 'numpy': numpy.__version__,
                    'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'checkpoint_policy': 'Existing neural weights referenced by DEPENDENCIES.json; '
                             'all 12 dev caches and fitted calibrators copied into snapshot.',
        'files': {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob('*')) if p.is_file()},
    }
    write(stage / 'FREEZE.json', manifest)
    verify(stage, manifest['files'])
    stage.rename(TARGET)
    for path in TARGET.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in TARGET.rglob('*') if p.is_dir()), reverse=True):
        path.chmod(0o555)
    TARGET.chmod(0o555)
    index = {'name': manifest['name'], 'snapshot': str(TARGET),
             'freeze_sha256': digest(TARGET / 'FREEZE.json'),
             'file_count_excluding_manifest': len(manifest['files']),
             'full_dev_micro_f1': reports['full_dev']['micro']['f1'],
             'crossfit_micro_f1': reports['crossfit']['micro']['f1']}
    write(TARGET.with_name(TARGET.name + '_INDEX.json'), index)
    assert digest(PARENT / 'FREEZE.json') == parent_hash
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
