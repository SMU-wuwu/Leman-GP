# FGM seed46 双卡互补微调

本轮从已冻结的 v4 seed42 最佳 EMA 权重启动一条新的 FGM 微调轨迹，使用当前两张 RTX 4090。模型、损失、FGM 扰动、学习率与 EMA 规则沿用已完成的 FGM seed43/44；新增随机 seed 为 46，输出目录独立为 `output/leman_fgm_seed46_dual4090`。

## 训练设置

| 项目 | 设置 |
|---|---|
| GPU | 2×RTX 4090 24GB |
| 父权重 | `output/leman_finetune_v4_seed42/best.pt`，v4 epoch1 EMA |
| 父权重 SHA256 | `55b803b48f2aeee3fbfda60422ac3918ab6d87603cdd2f5039b7312f1ec300e4` |
| 随机种子 | 46 |
| 轮数 | 3 |
| 每卡 batch | 32 |
| 全局 batch | 64（原 4 卡 FGM 参考运行是 128） |
| Encoder / reused head LR | `3e-6` / `1e-5` |
| FGM | 每 rank 当前 batch 的有效非 padding token embedding，FP64 范数归一化，epsilon `0.5` |
| 目标 | `0.5 × clean loss + 0.5 × adversarial loss` |
| loss / EMA / clip | FGM seed43/44 同定义；EMA `0.995`，梯度裁剪 `1.0` |
| 精度 | encoder BF16，自定义层 FP32，梯度范数 FP64 |

双卡限制使全局 batch 从历史 128 降为 64；每个 epoch 的优化器更新数因此增加。其余训练定义保持 FGM seed43/44 的配置。这个模型是探索性互补候选，不保证 F1 提升；结果会与既有 FGM/v32/v42 折内结果按同口径比较。训练脚本仅读取 train/dev，不运行 test。

## 服务器执行

上传并解压此目录到 `/root/data/WorkSpace/leman_gp_fgm_seed46_dual4090`，保证服务器已有 `data/`、`model/`、`output/leman_finetune_v4_seed42/best.pt` 及对应 `lexicon.json`。依赖沿用当前双卡环境。

```bash
cd /root/data/WorkSpace/leman_gp_fgm_seed46_dual4090
bash run.sh
```

`run.sh` 先训练，再执行 dev 校准/交叉拟合和回传归档，不运行 test。中断后重新执行相同命令会从 `last.pt` 继续，保持 2 卡、每卡 batch32 和当前代码不变。训练每250个 optimizer steps、每个 epoch及正常停止时保存 `last.pt`；best checkpoint 以 threshold=0 的 raw-dev exact character-span Micro F1 选择。3 epoch 内每轮都会保存 `last.pt`，正常停止时保留独立 `best.pt`。

输出在 `/root/data/WorkSpace/output/leman_fgm_seed46_dual4090/`。完成后回传 `return_leman_fgm_seed46_dual4090_*.zip` 供后续折内互补性分析。

如需检查暂停恢复：

```bash
ls -lh /root/data/WorkSpace/output/leman_fgm_seed46_dual4090/last.pt
```

由 STOP 文件暂停的操作沿用训练脚本：在另一个 shell 创建该输出目录下的 `STOP`，恢复前删除 STOP 并重跑 `bash run.sh`。
