# Leman-GP F1 fine-tune v4

从已完成的 F1 v2 best.pt 开始短程微调，训练后用同一套 decode-v3 流程比较原权重与微调权重。

## 直接运行

服务器为 4×RTX 4090 24GB，根目录 `/root/data/WorkSpace`。保持上一轮 Python 环境即可。

需要保留：

- `data/train2.txt`、`data/dev2.txt`、`data/label.txt`
- `model/` 中原 ERNIE 权重、配置和 tokenizer 文件
- `output/leman_f1_4x4090_seed42/best.pt`
- `output/leman_f1_4x4090_seed42/lexicon.json`

上传 ZIP 后执行：

```bash
cd /root/data/WorkSpace
unzip -q leman_gp_finetune_v4.zip
cd leman_gp_finetune_v4
bash run.sh
```

脚本依次执行四卡训练、两份权重的 dev 解码比较、回传打包。默认流程只读取 train/dev。
如果是新镜像，在运行前执行 `bash install.sh`。

## 训练方案

| 设置 | 值 |
|---|---|
| 初始权重 | F1 v2 seed42 best.pt，原最佳 epoch 3 |
| 每卡 batch / 全局 batch | 32 / 128 |
| 新增训练 | 5 epoch，预期 382 step/epoch |
| Encoder LR | 1e-6 |
| 其余已有参数 LR | 1e-5 |
| 精度 | ERNIE BF16；自定义层、GP loss FP32；梯度范数 FP64 |
| Warmup / schedule | 5% / linear decay |
| EMA | 0.995，沿用原更新公式 |
| Checkpoint | 每 250 step 及每轮保存 last.pt；第 5 轮保存 epoch_005.pt；单独保留 best.pt |

全部模型权重严格继承，词典和字符 ID 原样复制。模型结构、dropout、边界辅助损失、窗口规则和原 F1 v2 相同。

GP 的 MLCCE 分成正例项与负例项：

```text
L_pos(b,t) = log(1 + sum_positive exp(-score))
L_neg(b,t) = log(1 + sum_negative exp(score))
w_t = clip(((median_type_train_count + 1)/(train_count_t + 1))**0.25, 1, 1.5)
L_GP = sum_b,t w_t * (1.10 * L_pos(b,t) + L_neg(b,t)) / (B * sum_t w_t)
L_total = L_GP + 0.10 * L_boundary_original
```

类型计数只使用 train2 的原始实体 mention，每个实体在分窗前统计一次。实际计数与权重写入 `training_recipe.json`。

## 模型及解码选择

1. 训练前评估原权重，作为 epoch 0 写入 best.pt。
2. 每轮分别评估 live 与 EMA 权重，按零阈值、精确字符跨度的 dev Micro F1 更新 best.pt。
3. 对原 F1 v2 best.pt 和新 best.pt 分别运行原 decode-v3 的 baseline、expanded、boundary 三组搜索，沿用原 folds、网格和选择规则。
4. 只有新权重所选解码的 crossfit Micro 严格提高，且 full-dev Micro 不低于原权重，`selection.json` 才选择新权重；否则选择原权重。Macro13、precision、recall、分类型指标同时保留。

这些比较用于开发集上的实验选择；每半的阈值在另一半拟合，checkpoint 与最终候选选择仍使用 dev。

## 中断及续训

中断后回到本目录，再执行 `bash run.sh`，会从 last.pt 恢复训练位置、优化器、学习率、EMA 和各 rank 的随机数状态。
意外断电从最近一次完整写入的 checkpoint 恢复；最多回退至上次保存位置。

主动暂停时，在另一终端执行：

```bash
touch /root/data/WorkSpace/output/leman_finetune_v4_seed42/STOP
```

训练会在完成一个 optimizer step 后保存退出。恢复：

```bash
rm -f /root/data/WorkSpace/output/leman_finetune_v4_seed42/STOP
cd /root/data/WorkSpace/leman_gp_finetune_v4
bash run.sh
```

## 回传

新产物目录：`/root/data/WorkSpace/output/leman_finetune_v4_seed42/`。
完成后回传其中的 `return_leman_finetune_v4_*.zip`，包括训练记录、原/新权重的预测、解码搜索、crossfit 与打分候选。包内不包含 pt。

重新收集：

```bash
cd /root/data/WorkSpace/leman_gp_finetune_v4
python collect.py
```

`test.sh` 是单独的显式评估入口，按 selection.json 使用冻结解码；本轮先回传 dev 结果。

## 权重身份

- 原 best.pt SHA256：`22a4efe6a31021bda0d62fc6c1b73294dadfca79f00e79ab83bb89a9c9177bc1`
- 原 lexicon.json SHA256：`982e8dfb75e3e2195e06907861cddd3684594f75db94ea0807a263c6da33f7d1`
- 原模型源码文件哈希：`PARENT_SOURCE_HASHES.json`
- 当前交付文件及继承关系：`PACKAGE_MANIFEST.json`

本地使用 CPU 小模型检查损失公式、完整权重继承、前向一致性、断点恢复和解码流程；未在本地运行 CUDA/NCCL 训练。
