# F1 v10 — Short FGM Fine-tuning

从 v4 最佳 EMA 权重继续做 3 个 epoch 的短程微调。模型前向、字符映射、词典、跨度损失及边界辅助损失均复用 v4；本轮增加 FGM 对抗训练。

## 固定方案

| 项目 | 设置 |
|---|---|
| 根目录 | `/root/data/WorkSpace` |
| GPU | 4×RTX 4090 24GB |
| 初始化 | `output/leman_finetune_v4_seed42/best.pt`，epoch 1 EMA |
| 初始化 SHA256 | `55b803b48f2aeee3fbfda60422ac3918ab6d87603cdd2f5039b7312f1ec300e4` |
| 训练种子 | 43；这是从 seed42 权重出发的新微调轨迹 |
| epoch | 3 |
| batch | 每卡 32，全局 128 |
| 学习率 | encoder `3e-6`，其余复用参数 `1e-5` |
| FGM | encoder 输入 embedding；每个 rank 当前 batch 的有效非 padding token 行 |
| 扰动 | 沿 clean loss 梯度方向，所有有效行合计 L2 范数为 `0.5`；范数用 FP64 计算 |
| 目标 | `0.5 × clean_loss + 0.5 × adversarial_loss` |
| dropout | 两次前向独立采样，沿用模型训练模式 |
| DDP | clean forward/backward 在 `no_sync()` 内；adversarial backward 同步两次累积梯度 |
| 精度 | encoder BF16，自定义层 FP32，梯度范数 FP64 |
| 其他训练项 | 沿用 v4：正例项权重 1.10、训练集类型权重、EMA 0.995、clip 1.0 |

每个 batch 两次前向、两次反向、一次优化器更新。3 个 epoch 的计算量约相当于 6 个普通 epoch，实际时间取决于数据和 GPU。FGM 不增加推理开销。

## 执行与续训

包已上传服务器时，直接执行：

```bash
cd /root/data/WorkSpace/leman_gp_fgm_v10
bash train.sh
```

中断后重复同一命令，自动从 `last.pt` 恢复优化器、scheduler、EMA、各 rank RNG、epoch 与 batch 位置。正常退出请求会在完成一个优化器 step 后保存；强制断电从最近保存点恢复。不要通过改变 batch、epoch 或 GPU 数量来续训。

输出：`/root/data/WorkSpace/output/leman_fgm_v10_seed43`。

`last.pt` 每 250 step、每个 epoch 和正常中断时保存。`best.pt` 保留最优 raw-dev Micro 模型，初始 v4 权重作为 epoch0 候选；每满 5 epoch 另存快照。本轮共 3 epoch，仍有每轮 `last.pt` 和最优 `best.pt`。

启动只训练并评估 dev。训练完成后由 Codex 通过 SSH 读取结果、进行校准、比较当前 v9。`run.sh` 仍为旧版一体化工具，本轮使用上述 `train.sh`。

## 与当前结果的关系

当前 v9 完整 dev Micro 为 94.2520%，按拟合半区选择校准方法的 crossfit Micro 为 94.1557%。本轮目标是改善召回；是否采用，待训练完成后按同一份 dev、同一组 folds 和 v9 校准流程比较。

这些是已用于 checkpoint 选择和多轮探索的 dev 结果。95% 是优化目标，不预设实验结果。
