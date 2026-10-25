# Leman-GP Decode v3

复用四卡 F1 v2 已训练的 best.pt，在 dev 上探索解码改进。运行入口是 `bash run.sh`；不训练，不重新生成词典，不自动评估 test。

## 本轮依据

数据来自 `return_leman_f1_v2_20261017T092218Z.zip` 的 dev 产物。

- 当前 best 是 epoch 3 的 live 权重；后续训练 loss 下降而 dev loss 上升。因此本轮先复用 best，不延长训练。
- 原 full-dev 全局阈值为 4.0，正好位于原搜索网格上界。阈值空间由 `[-4, 4]` 扩到 `[-4, 16]`，步长仍为 0.25；分类型调整仍为全局阈值附近 ±1，仍为一遍、至少 200 个拟合 gold。
- 原 full-dev 解码有 1,334 个漏检 gold：696 个没有重叠预测，452 个为同类型边界偏移，137 个同时涉及边界与类型，49 个边界相同但类型错误。这些类别描述漏检实体，不是互不重复的句子数。
- 已训练的 typed start/end 头在旧推理流程中没有参与解码；本轮尝试用其起止分数辅助 GP。

## 三组比较

1. `baseline`：完全复用旧解码拟合函数、网格及交叉拟合规则。
2. `expanded`：只扩大阈值上界；旧拟合结果也保留为候选。
3. `boundary`：在扩大阈值搜索的基础上，选择边界权重 λ∈{0, 0.25, 0.5, 1}。

对原始窗口内类型 t、闭区间 [i,j]：

```text
p = [logsigmoid(boundary_start[i,t]) + logsigmoid(boundary_end[j,t])] / 2
s = sqrt(gp_head_size) * (GP_native_logit[i,j,t] + λ * p)
```

这里的 logsigmoid 是辅助头分数的单调变换，不声称它是经过概率校准的边界概率。λ=0 精确退化为原 GP 打分。边界项只惩罚起止分数较弱的跨度，重新拟合阈值后可相应调整精确率与召回率。

所有窗口先各自组合 GP 与边界项，再对同一句的重复跨度取最大分数。不会把不同窗口的 GP 最大值与边界最大值拼在一起。保存 GP 原始分数>-4 的候选已足够：λ≥0 且 p≤0，低于 -4 的 GP 候选不可能在调整后越过本轮任何阈值。

模型源码的 13 个 Python 文件直接复制到 `frozen_source/`，逐字节保持不变。新工具用 attention 的前向 pre-hook 读取训练时辅助头所用的同一份 sequence，并调用已训练的 boundary 线性层；不额外执行 encoder。模型处于 eval/inference_mode，权重不更新。

## 选择与指标

- 主指标继续为 exact character-span Micro F1，同时报告 Macro13 与各类型指标。
- 继续沿用原来按文本 SHA256 固定的两个 dev 分区，重复文本始终在同一分区。
- 每个拟合半区独立选择 λ、策略与阈值，再在另一半上评估。不是先在全 dev 上挑 λ 再计算交叉拟合。
- 新组必须提高 cross-fit Micro，且 full-dev Micro 不低于 baseline，才能替换原解码。没有提高就保留 baseline。
- checkpoint 和最终解码组的选择都使用了 dev；这些是开发集探索结果。cross-fit 只隔离本轮解码拟合标签，不是独立测试集评估。
- 以前已评估过 test；本轮默认不读取 test，也不按已知 test 类型分数设定搜索。

`dev_search.json` 给出三组指标、逐折拟合参数和提升百分点。若原 run 的 `dev_calibration/upgraded/calibration.json` 还在，会额外给出 baseline 重放差值。`selection.json` 保存后续推理所需的冻结设置。

## 服务器文件与执行

使用现有四张 RTX 4090 和原 Python 环境，不增加依赖。必须保留：

```text
/root/data/WorkSpace/data/dev2.txt
/root/data/WorkSpace/data/label.txt
/root/data/WorkSpace/model/
/root/data/WorkSpace/output/leman_f1_4x4090_seed42/best.pt
/root/data/WorkSpace/output/leman_f1_4x4090_seed42/lexicon.json
```

当前 best.pt 的 SHA256 是 `22a4efe6a31021bda0d62fc6c1b73294dadfca79f00e79ab83bb89a9c9177bc1`。本轮不需要旧 A100 checkpoint。

上传本包到 `/root/data/WorkSpace`，执行：

```bash
cd /root/data/WorkSpace
unzip -q leman_gp_decode_v3.zip
cd leman_gp_decode_v3
bash run.sh
```

默认每卡推理 batch 16，与此前校准相同。GPU 分布式前向只执行一次，之后 rank0 在 CPU 上搜索解码设置。进度条同一行刷新，搜索阶段每完成一组拟合打印一行。

结果独立存入：

```text
/root/data/WorkSpace/output/leman_decode_v3_seed42/
```

完成后回传终端打印的：

```text
return_leman_decode_v3_*.zip
```

回传包包含三组 dev 预测、逐折参数、指标、源码和压缩的 dev 候选分数；不含 pt。这样后续本地可以直接分析候选分数，不必再次申请 GPU。中断后重复 `bash run.sh`：若完整分数缓存已写入，复用它并重新完成 CPU 搜索；若前向尚未完成，只需重做一次 dev 前向。

需要改变位置时，支持环境变量：`LEMAN_GP_ROOT`、`LEMAN_GP_DATA`、`LEMAN_GP_ENCODER`、`LEMAN_GP_OUTPUT`（现有训练目录）、`LEMAN_REFINE_OUTPUT`（本轮新结果目录）。

`test.sh` 是单独入口，只用已冻结的 `selection.json` 解码，不做阈值搜索。先完成本轮 dev 分析，再决定是否执行。

## 本地功能检查

使用 tiny BERT 做过真实前向，检查边界读取位置、GP 分数、权重与 RNG 不变；用独立标量公式复算边界项；检查重复窗口合并顺序、旧解码重放和 held-out 标签隔离；走通报告、缓存、冻结解码及回传 ZIP。CUDA 四卡计算由上述运行入口直接执行。
