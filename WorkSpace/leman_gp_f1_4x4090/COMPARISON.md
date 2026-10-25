# 上一轮结果与 F1 v2 实施说明

上一轮完整训练 40 epoch，best checkpoint 位于第 36 epoch。所有值按原始字符跨度和实体类型精确匹配。

| 指标 | Dev | Test（上一轮已经报告） |
|---|---:|---:|
| Micro F1 | 92.7221 | 92.6346 |
| Macro13 F1 | 85.4545 | 86.3531 |
| Precision | 94.4051 | 94.4279 |
| Recall | 91.0981 | 90.9082 |

Dev 的 CUL Precision=91.71%，Recall=58.46%，F1=71.41%。下一轮优先处理边界召回和词典覆盖，而不是继续增加原配置的训练轮数。

## 保留的优点

- 字符级标签、原始跨度评价、窗口重叠去重，保留完整 gold 分母。
- 标准 GlobalPointer multilabel categorical cross-entropy，原版 reset-before 双层 BiGRU。
- 训练集独立建词典，旧词 embedding 按字符串映射，所有原有模型参数均迁移到对应层。
- 精确 L1 及 PyTorch 2.7 CUDA cdist 分块修复，自定义层 FP32，FP64 范数裁剪。
- 四卡 DDP、BF16 编码器、完整 optimizer/scheduler/RNG 断点恢复。

## 实施的改进

1. **字符身份**：为 train 中的字符建立 96 维 embedding，经投影加入 ERNIE 对齐表示，初始残差系数 0.1。多个字被同一个子词覆盖时仍能具有不同的输入特征。
2. **词典覆盖与正则**：最长词 8→40；词表 45,283→48,942，增加 3,659 个超过 8 字的词条。旧词向量精确迁移；新增词以新 head 学习率学习。加入 gate、LayerNorm 和整词/整句词典 dropout。
3. **上下文结构**：恢复 Q/K 归一化、可学习 L1 温度、维度缩放和上下文残差。保留原 Q/K/V/out 权重。GP 训练分数使用 /sqrt(64)，保留原投影及 RoPE；跨度上限 64，窗口 192、重叠 64。
4. **边界与泛化**：在 attention 前加入类型化 start/end 辅助头，用多标签 categorical CE，权重 0.1。加入 EMA；每轮在 live/EMA 中选择 raw-dev Micro 更好的权重。
5. **解码**：固定 dev 搜索网格 -4 到 4、步长 0.25（统一换算到原始 dot-score 单位），比较保留全部跨度与最优非重叠区间解码；支持度不少于 200 的类别做一次局部阈值调整。固定文本哈希两折交叉拟合。旧模型和新版用同一拟合方法，最终依据 crossfit Micro 选择候选，再将所选候选的 full-dev 阈值用于 test。

这些改动构成一个整体升级版本；本轮可比较整体效果，单项因果收益需要后续消融。

## 训练设置与时间

- 从上一轮第 36 epoch 的 best.pt 迁移，微调 12 epoch。
- ERNIE LR=5e-6；复用 head LR=3e-5；新增 head/词 embedding LR=1e-4。
- weight decay=0.01（bias/1D 参数为0），warmup=8%，clip=1，EMA上限0.995。
- 四张 RTX 4090 24GB，默认自动测每卡 batch，候选4/8/12/16/24/32/48/64，显存预算86%；eval batch16/GPU。四张卡都要通过；探测额外预留 Adam 两份动量、DDP 梯度桶和 EMA shadow。实际全局batch为每卡batch×4，学习率不按卡数放大。
- 新 train 窗口数 48,884、dev 6,099。Gold覆盖分别 165,686/20,782，均无窗口丢失。
- 编码器允许 PyTorch fused SDPA；DataLoader workers 跨 epoch 常驻；减少大 checkpoint 的重复写盘。更长窗口增加单步成本，实际总耗时由服务器结果给出。
- 每5 epoch保留完整快照、每轮保留last.pt、始终保留best.pt。新实验输出到独立目录。

## 本地验证范围

真实 ERNIE 3.0 + 真实 train 样本（包括192字窗口）的 CPU 前向/反向有限；覆盖完整训练集和dev的特征缓存。小模型检查包含全部旧参数/词条迁移、独立边界损失公式、padding、EMA/optimizer/scheduler/RNG精确重放、区间解码穷举核对、训练中断续跑和冻结推理全链路。服务器 CUDA/BF16/DDP 的实际训练效果由正式运行产生。
