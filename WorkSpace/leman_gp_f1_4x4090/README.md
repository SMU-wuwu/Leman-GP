# Leman GP F1 v2 — 4×RTX 4090 warm-start run

此版本从已经完成的 `leman_paper_a100_seed42/best.pt` 迁移训练权重，而不是重新训练 ERNIE。旧训练目录及其 checkpoint 保持原样。它保留训练好的 ERNIE、词典 embedding（按词字符串映射）、Lexicon Adapter 的 bilinear 层、双层 reset-before BiGRU、Manhattan Q/K/V 和 RoPE GlobalPointer 权重；用较低学习率继续微调 12 个 epoch。

## 相对上一轮的改动

- ERNIE 子词输出投影到每个原始字符，并添加 train-only 字符身份 embedding，减少多个汉字共用一个 ERNIE 子词向量带来的边界损失。
- 词典最长词从 8 字扩到 40 字，新增词向量随机初始化，旧词向量按字符串精确搬迁；训练中同时执行整词 dropout 和整句词典分支 dropout，减轻词典记忆。
- 词典融合改成门控残差 + LayerNorm；无候选词的字符维持原表示。
- 恢复带残差和 LayerNorm 的上下文 Manhattan attention，Q/K 按 head LayerNorm，加入可学习温度并按 head 维缩放；所有距离仍是精确 L1。RoPE GP logits 除以 `sqrt(64)`，稳定 logits 尺度。
- 加入类型化 start/end 多标签分类 CE 辅助项，权重 0.1；主 GP 仍是标准 multilabel categorical CE。
- 新增 EMA，在每轮 dev 上对比 live 与 EMA 权重；再用 dev 交叉拟合阈值/解码器。最终候选选择首先依据 crossfit Micro F1，平手再看 full-dev Micro、Macro13。
- 窗口 192 字、重叠 64。新窗口对 train/dev 的 gold 全覆盖；最长词 40 字，GP 候选跨度上限 64，当前 train/dev 的最长 gold 实体为 35 字。旧模型保持其原始跨度规则；真实数据窗口覆盖已核对。

BEGD 与 TCBS 未纳入此版本。上一轮测试集结果已观察到，但此版本没有用 test 选择结构、超参、checkpoint 或阈值。test 只会在你单独运行 `test.sh` 后评估一次。

## Server setup and run

服务端根目录固定为 `/root/data/WorkSpace`，预期已有 `data/`、`model/` 和上一轮 `output/leman_paper_a100_seed42/`。压缩包解开后目录为 `leman_gp_f1_4x4090/`。训练、校准、test 均为四卡 DDP/分布式推理。

```bash
cd /root/data/WorkSpace
unzip -o leman_gp_f1_4x4090.zip
cd leman_gp_f1_4x4090
bash run.sh
```

`run.sh` 自动进行 warm start、训练、dev full calibration 和 crossfit calibration，并在结束后运行 `collect.py`。默认自动测每卡 batch，依次尝试 4/8/12/16/24/32/48/64；四张卡全部通过才接受，最终使用显存最紧张的一张也能容纳的 batch。探测使用 train 中最长 token 和字符窗口，执行真实前向/反向，同时为 Adam 两份动量、DDP 梯度桶和 EMA shadow 预分配显存，并保留 14% 余量。实际 batch 以启动日志为准，例如每卡 16 时全局 batch=64，每卡 24 时全局 batch=96。

每张卡默认 eval batch=16，DataLoader workers=2，CPU 计算线程=2。四卡合计 8 个训练 workers。学习率、12 epoch、模型结构和旧权重迁移方式与 A100 新版一致；实际全局 batch 取决于显存探测，因此优化步数会重新计算，学习率不按卡数自动放大。无需重训旧模型，只需保留旧 best.pt 与 lexicon.json。

如重新租用环境后缺少依赖，先运行 `bash install.sh`，再运行 `bash run.sh`。保持原有 `data/`、`model/` 和旧 checkpoint 文件即可。

如果中断，原命令即可从 `last.pt` 自动续训：

```bash
cd /root/data/WorkSpace/leman_gp_f1_4x4090
bash run.sh
```

每轮结束和每 1000 个 optimizer steps 自动写 `last.pt`；收到 SIGTERM/SIGINT 或存在 `output/leman_f1_4x4090_seed42/STOP` 时，在当前 optimizer step 完成后保存并退出。正式 epoch snapshot 每 5 个 epoch 保存。已有同目录的 `.training.lock` 防止重复启动。要重新开始而非续训，请把 `LEMAN_GP_OUTPUT` 指向新的 output 子目录。续训时保持四卡和原训练配置不变。

完整训练并完成 calibration 后，再运行 test：

```bash
cd /root/data/WorkSpace/leman_gp_f1_4x4090
bash test.sh
python collect.py
```

可以通过 `LEMAN_GP_EPOCHS`、`LEMAN_GP_WORKERS`、`LEMAN_GP_EVAL_BATCH_SIZE` 设置运行参数。默认保留自动 batch 探测；手动设置 `LEMAN_GP_BATCH_SIZE` 会跳过探测。沿用已完成 CPU 检查的新版模型；本次另行检查四卡启动配置、显存探测和断点恢复链路。实际 CUDA/BF16/NCCL 训练需在服务器执行。

## Output

- `weight_transfer.json`: 每个迁移/新增参数和词条映射计数。
- `history.json`, `best.pt`, `last.pt`, 每 5 epoch 的 `epoch_*.pt`。
- `dev_calibration/{upgraded,parent}/calibration.json`: 同样数据和 decoder 搜索口径下的旧模型与 warm-start 模型对照；包含 raw/full-dev/crossfit 和分类型结果。GP 输出阈值比较统一换算为旧模型原始 logit 单位。
- `selection.json`: 只按 dev 冻结的最终 checkpoint 和阈值策略。
- `test_evaluation/<candidate>/test_results.json`: 选择冻结之后运行 test 才生成。
- `return_leman_f1_v2_*.zip`: 运行 `python collect.py` 生成，不含 weights/cache。

这里的 crossfit 是阈值/解码器交叉拟合；模型 checkpoint 本身仍由 full-dev best Micro F1 选择，报告会标明这一点。

## Four-card adaptation

四卡包使用独立代码和输出目录；原双卡包保留。`model.py`、`paper_core.py`、`data.py`、`transfer.py`、`inference.py` 和解码/评价算法直接复制，模型定义与新版 A100 包相同。四卡包是从旧正式模型进行新一轮 12 epoch warm start，不是恢复旧 40 epoch 的优化器状态。后续四卡运行生成的 last.pt 可自动续训。
