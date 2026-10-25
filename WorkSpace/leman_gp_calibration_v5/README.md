# LeMan-GP v5：跨度置信度校准

本版复用已完成的 v4 epoch-1 EMA 权重，在跨度 GP 分数、辅助边界置信度、字符长度和类型之间拟合一个小型 ridge logistic 校准器。模型结构、权重和词典保持原样。当前选择结果只需加载 v4，一个模型前向即可完成推理。

## 当前结果

| 方法 | Full-dev Micro % | Crossfit Micro % | Crossfit Macro13 % |
|---|---:|---:|---:|
| v4 边界解码 | 94.044479 | 94.013701 | 87.439674 |
| v5 单模型校准 | 94.113354 | 94.038639 | 87.562215 |
| 原权重与 v4 的双模型 stacking | 94.107940 | 93.997483 | 87.652163 |

最终 full-dev、两个拟合半区都选择单模型校准。两个 held-out 半区的 Micro 增益分别是 +0.006107 和 +0.043936 个百分点，合并后的增益是 +0.024938 个百分点。双模型 stacking 不采用。另一次固定比例分数融合实验也未超过 v4 的 crossfit Micro，记录保留在工作区 `experiments/f1_fusion_v5/`。

v5 主要改善精确率，同时降低召回率；它带来的是小幅 dev 增益。这里的 crossfit 指校准交叉拟合，仍使用之前按全量 dev 选择的 checkpoint，后续路线也参考过 dev。此阶段不读取 test。

## 运行

上传 ZIP 到 `/root/data/WorkSpace`，解压后执行：

```bash
cd /root/data/WorkSpace
unzip -q leman_gp_calibration_v5.zip
cd leman_gp_calibration_v5
bash run.sh
```

这一步使用包内已经回传并核对过的跨度缓存及服务器的 `data/dev2.txt`，在 CPU 上重新拟合、生成预测、比较并收集产物。无需训练、无需 GPU、无需下载新依赖。中断后重复 `bash run.sh`，已经完成的 family/fold 拟合会直接复用。

输出目录：`/root/data/WorkSpace/output/leman_calibration_v5_seed42/`。回传其中的 `return_leman_calibration_v5_*.zip` 即可，不含 pt。`selection.json` 保存部署用的全量 dev 校准参数；`comparison.json` 包含各方法、逐半区与最终选择的指标。

## 用四张 4090 重新前向

若需要在服务器确认实际推理，完成上述步骤后执行：

```bash
bash predict.sh --split dev
```

此命令使用 4 张 GPU、BF16、每卡 evaluation batch 16；当前选择只做一次 v4 模型前向。执行前需要保留：

```text
/root/data/WorkSpace/data/dev2.txt
/root/data/WorkSpace/model/                    原始 ERNIE 权重与 tokenizer
/root/data/WorkSpace/output/leman_finetune_v4_seed42/best.pt
/root/data/WorkSpace/output/leman_finetune_v4_seed42/lexicon.json
```

best.pt SHA256：`55b803b48f2aeee3fbfda60422ac3918ab6d87603cdd2f5039b7312f1ec300e4`。词典 SHA256：`982e8dfb75e3e2195e06907861cddd3684594f75db94ea0807a263c6da33f7d1`。CPU 校准重放不需要加载这些权重。

预测入口默认 `--split dev`，只在明确传入 `--split test` 时才读取 test。常规运行脚本不自动启动 test。

## 定义与选择规则

每个跨度、每个模型先选择 raw GP 分数最大的窗口，若分数相同选择边界 log-probability 较大的窗口。三个特征来自同一个窗口：GP 原生分数、起止边界 log-probability 的均值、该均值的指数。另加 `log1p(字符长度)`。双模型候选附加两个模型的候选存在标志；单模型候选不读取原模型分数。

辅助边界特征来自训练时同一组 typed start/end logits。既有缓存只保留放大后 GP 分数大于 -4 的跨度；单模型使用自己的候选集，双模型使用并集。双模型中缺失的一侧使用明确的 censored 特征值，并附带存在标志，不将其解释成真实 dense logit。

Logistic 拟合使用 summed binary logistic loss，加系数平方和的 ridge 项 `10/2 * ||w||²`，全局截距不惩罚。连续特征的均值、标准差仅取拟合句子；拟合半区 gold 支持度至少 200 的类型增加一个受 ridge 约束的截距。拟合完成后按原版 decoder 的 Micro/Macro 顺序，选择阈值及 `all`/`weighted_interval`。

同一文本沿用 v4 的固定哈希划分。Logistic 参数、标准化、类型资格、阈值和候选 family 的选择都只看拟合半区，预测另一半。最终合并两半预测后计算 F1。全量 dev 单独拟合部署参数。

只有 full-dev Micro 不降、合并 crossfit Micro 提升、且两个 held-out 半区 Micro 都不降时才采用 v5，否则 `selection.json` 回退到 v4。

`POLICY.json` 在本轮校准结果产生前写入，记录固定特征、正则强度、family 与选择规则。未修改的 16 个 v4 Python 文件全部复制到 `frozen_v4/`；新增逻辑只在外层执行，checkpoint 的源码身份核对仍指向原来的完整 Python 快照。
