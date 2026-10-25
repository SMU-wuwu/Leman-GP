# F1 v17 — R-Drop Span Consistency

在现有 v4 最佳 EMA 权重上微调 3 个 epoch，增加同一样本两次 dropout 输出的一致性损失。模型结构、字符表示、训练词典、正例加权 MLCCE 和边界辅助损失均直接复用；推理结构不变。

## 执行

服务器代码已上传后：

```bash
cd /root/data/WorkSpace/leman_gp_rdrop_v17
bash train.sh
```

训练完成后告知 Codex，由 Codex 通过 SSH 完成 dev 校准、融合和比较。

## 本轮设置

| 项目 | 设置 |
|---|---|
| 硬件 | 2×RTX 4090 24GB |
| 父模型 | `output/leman_finetune_v4_seed42/best.pt`，epoch1 EMA |
| seed / epochs | 45 / 3 |
| 每卡原始样本数 | 16 |
| 实际单次前向样本数 | 32；每条原始样本复制两份 |
| 梯度累积 | 4 次；有效原始样本 batch 为 128 |
| 学习率 | encoder `3e-6`，其余参数 `1e-5` |
| 一致性权重 | 1.0 |
| EMA / warmup / clip | 0.995 / 总更新步数的 5% / 1.0 |
| 精度 | encoder BF16，自定义层 FP32，梯度范数 FP64 |
| FGM | 本轮关闭 |
| 输出 | `output/leman_rdrop_v17_seed45` |

每个样本、每个类型，把有效跨度 logits 与一个固定为 0 的 null 类一起做 softmax。有效跨度同时要求 query/key 有效、上三角、长度不超过模型 max_span。对两份随机输出计算 `0.5 × (KL(p||q) + KL(q||p))`，在样本和类型上平均，再与原监督损失相加。两个分布都参与反向传播。

这是对 R-Drop 的跨度排名适配。原多标签监督负责所有 gold 实体；一致性分布不改变推理解码。普通层和逐样本 dropout 对两个副本独立采样；原模型按词表生成的 word-dropout mask 在同一次加倍 batch 前向内共享。

非最后一个累积 microbatch 使用 DDP `no_sync()`；完成累积窗口后同步、裁剪、更新 AdamW、更新 EMA 和 scheduler。最后不足一个窗口时按实际样本数加权。

## 保存和续训

重复执行 `bash train.sh` 自动恢复 `last.pt`，包括优化器、scheduler、EMA、各 rank 随机状态和数据位置。保持两卡、batch16 和累积4不变。

`last.pt` 每250次优化器更新、每个 epoch 和正常中断时保存。`best.pt` 依照阈值0的 raw-dev 精确字符跨度 Micro 选择；父模型作为 epoch0候选，每轮比较 live 和 EMA。每满5 epoch额外保存快照；本轮3 epoch仍会保存每轮 last 和全程 best。

需要主动停止时，在另一终端创建：

```bash
touch /root/data/WorkSpace/output/leman_rdrop_v17_seed45/STOP
```

训练会在一个完整优化器更新后保存并退出。恢复前删除 STOP，再执行原命令。强制断电恢复到最近一次保存点。

## 比较方案

当前 v14 full-dev Micro 为94.630467%，完整流程 crossfit为94.451181%。训练结束后按固定文本哈希 folds，比较 v14、R-Drop单模型、v4+R-Drop、v14三份权重+R-Drop四条路线。归一化、类型支持度、拟合、阈值和路线选择均只使用相应拟合区。

采用条件为 full-dev Micro不下降、完整流程 crossfit严格提高、两个 held-out半区均不下降。具体规则与数据身份见 `EXPERIMENT_PLAN.json`。这是多轮dev探索；95%是目标，收益由实际结果决定。

双卡无参数更新检查已完成：最坏长度247 encoder tokens、192字符，每卡16条原始样本，峰值约18.82 GiB；检查预留了Adam两个矩和EMA显存，执行了一次非同步和一次同步反向，未启动正式训练。

参考：Wu et al., *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS 2021，https://arxiv.org/abs/2106.14448 。
