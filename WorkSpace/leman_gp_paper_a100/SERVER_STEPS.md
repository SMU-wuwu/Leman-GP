# Leman-GP 论文模型双 A100 执行包

服务器根目录：`/root/data/WorkSpace`。使用现有 `data/` 和 `model/`。
硬件：2×A100 80GB。Python 3.10–3.12。模型文件从本地读取。

## 覆盖 fix2：修复大批量 L1 距离 CUDA 反向索引

上传 `leman_gp_paper_a100_fix2.zip` 到 `/root/data/WorkSpace` 后直接执行：

```bash
cd /root/data/WorkSpace
unzip -o leman_gp_paper_a100_fix2.zip
cd leman_gp_paper_a100
bash train.sh
```

此次停止发生在自动选批量阶段，没有更新模型参数；已经建立的词典和特征缓存会复用。
fix2 将精确 L1 距离按样本×head分组，每次反向的临时缓冲区限制为512 MiB以内，避免 PyTorch 2.7.0 的32位偏移乘法溢出。
这不拆分优化器 batch，不使用梯度累积，也不改变注意力覆盖的key或损失归一化。
不需要重装依赖或另跑验证命令。临时显存降低后会重新测量批量，以新日志为准，不手动固定128或192。
后续本修正版内部正常断点续训仍使用同一条 `bash train.sh` 命令。
若已有其他版本成功训练的 checkpoint，源码一致性检查仍会拒绝混用；本修复不绕过该检查。

## 直接训练

上传 `leman_gp_paper_a100.zip` 到服务器根目录，执行：

```bash
cd /root/data/WorkSpace
unzip -o leman_gp_paper_a100.zip
cd leman_gp_paper_a100
bash install.sh
bash train.sh
```

可在 tmux 中执行最后一条命令，以免 SSH 断开影响任务。
无需另跑 smoke、QA、freeze 或手动构建词典。首次启动自动构建 train-only 词典、缓存 train/dev 特征，测量每卡批量，再开始正式训练。
目录需包含 `data/train2.txt`、`dev2.txt`、`label.txt`，以及完整的 ERNIE 3.0 `model/`。测试时再使用 `test2.txt`。

## A100 配置

- 训练批量自动从每卡16、32、64、96、128、192、256、384、512中选择，最多每卡512、全局1024；使用全训练集观测到的最大 token/字符长度测量前向和反向。
- 显存预算为每张卡容量的88%，扣除其他进程占用，并预留 Adam 状态与 DDP 梯度空间；结果写入 `run_config.json`。测量不更新权重，结束后恢复随机数状态。
- ERNIE 投影使用 BF16，attention 使用 math SDPA 后端并以 FP32 中间量计算；FP32 主参数。字符池化、整个 Lexicon Adapter、GRU、Manhattan attention 和 Global Pointer 均在关闭 autocast 后计算；允许 TF32。关闭 FP16/BF16 GEMM 和 math SDPA 的低精度归约。此路径的显存和速度可能不同于原先自动选择的融合后端。
- 梯度范数以 FP64 归约，检查通过后才裁剪；自动选批量同时检查 loss 和梯度有限性。真实 NaN/Inf 会在更新前停止并保存每个 rank 的诊断，不替换成0，不跳过该批次。
- 精确 L1 距离分组执行，每组反向临时张量最多 `2^27` 个FP32元素（512 MiB）。所有query仍覆盖全部key，数学定义与未分组时一致。
- ERNIE 学习率 `3e-5`，其他参数 `1e-4`；AdamW、weight decay `1e-4`、10% warm-up、linear decay、梯度裁剪1.0。
- 40 epochs；每个 epoch 双卡 dev 评估；以精确字符跨度 Micro F1 选择最优模型。训练不读取 test。
- CPU 特征预处理8线程；每卡4个 DataLoader worker。特征只计算一次并落盘。
- 字符窗口128、重叠32；编码器 token 上限512。长句按固定窗口处理并合并预测，评价分母保留原句全部 gold。超出任何窗口可完整覆盖范围的实体数量另行记录。

需要手动指定批量时，在第一次训练前使用，例如：

```bash
LEMAN_GP_BATCH_SIZE=256 bash train.sh
```

固定批量跳过自动显存选择。默认自动选择能适应不同 A100 实例的可用显存；显存占用不等于 GPU 利用率100%。

## 保存与断点续训

默认输出：`/root/data/WorkSpace/output/leman_paper_a100_seed42/`。

| 文件 | 内容 |
|---|---|
| `best.pt` | 最优 dev checkpoint 的模型权重、结构配置与词典绑定信息 |
| `last.pt` | 每100步及每个 epoch 末更新的完整恢复状态 |
| `epoch_005.pt`、`epoch_010.pt`… | 每5个 epoch 保留的完整恢复快照 |
| `lexicon.json` | 仅由训练集生成的词典及固定ID |
| `history.json` | 每轮训练损失、dev结果、耗时和各卡显存峰值 |
| `source_snapshot/` | 实际运行源码 |
| `startup_record.json`、`startup_source_snapshot/` | 自动选批量前就保存的当前启动配置与源码，避免早期失败只回传旧记录 |

完整恢复状态包含模型、AdamW状态、scheduler、当前epoch、下一批位置、global step、两卡独立的Python/NumPy/CPU/CUDA随机数状态、已完成的历史与当前epoch部分损失统计，以及该时点的最优模型副本。恢复历史快照时也能正确恢复当时的best.pt。BF16无需GradScaler。
所有 checkpoint 先写临时文件再原子替换。每5轮的文件尽可能用硬链接保存；后续替换 `last.pt` 不会改动历史快照。

中断后直接再次执行：

```bash
cd /root/data/WorkSpace/leman_gp_paper_a100
bash train.sh
```

自动从 `last.pt` 恢复，包括 epoch 中间的 batch 位置。突然断电恢复到最后已落盘状态，最多重算尚未保存的99步；正常停止可保存当前完成步。

推荐在另一终端请求正常停止：

```bash
touch /root/data/WorkSpace/output/leman_paper_a100_seed42/STOP
```

程序在完成当前优化步后保存并退出。随后恢复：

```bash
rm /root/data/WorkSpace/output/leman_paper_a100_seed42/STOP
cd /root/data/WorkSpace/leman_gp_paper_a100
bash train.sh
```

也处理SIGINT/SIGTERM；强制关机、SIGKILL或torchrun提前终止进程时，使用已保存的 `last.pt`。不能从只有模型权重的 `best.pt` 恢复优化器。

明确从每5轮快照另开目录恢复：

```bash
mkdir -p /root/data/WorkSpace/output/leman_paper_recovered
cp /root/data/WorkSpace/output/leman_paper_a100_seed42/lexicon.json /root/data/WorkSpace/output/leman_paper_recovered/
LEMAN_GP_OUTPUT=/root/data/WorkSpace/output/leman_paper_recovered \
bash train.sh --resume /root/data/WorkSpace/output/leman_paper_a100_seed42/epoch_010.pt
```

迁移服务器时保留整个运行目录（至少完整checkpoint、`best.pt`、`lexicon.json`和源码）、相同数据与 `model/`。新目录路径可用 `LEMAN_GP_ROOT`、`LEMAN_GP_DATA`、`LEMAN_GP_ENCODER`、`LEMAN_GP_OUTPUT` 指定。续训保持批量、训练轮数及学习率不变。

## 双卡测试

训练结束后：

```bash
cd /root/data/WorkSpace/leman_gp_paper_a100
bash test.sh
python collect.py
```

测试只有一次遍历，不训练 epoch。两张GPU分担窗口，每批只前向一次。所有跨度通过批量 `nonzero` 提取；不对CUDA张量逐位置执行Python循环。
推理保留阈值大于0的所有类型跨度，包括重叠跨度；跨窗口相同结果去重；按原句字符位置计算精确Micro和Macro F1，不再投影为BIO覆盖其他实体。

结果位于 `test_evaluation/`：`test_results.json`、`entity_metrics.csv`、`test_predictions.jsonl`。
回传 `collect.py` 打印的 `return_leman_paper_a100_*.zip`，不需要上传pt和编码器权重。

如果再次出现数值错误，程序会打印 `diagnostics/numerical_failure_*.json` 路径。
直接执行 `python collect.py` 回传小包即可；收集不需要 GPU，也不会执行 test。
诊断包含失败步骤、参数名、NaN/Inf 数量、有限梯度范数、batch 句子ID/窗口位置和版本信息。

## 模型与公式对应

1. ERNIE：使用现有目录中的实际 ERNIE 3.0 权重，隐藏维度从配置读取；对每个字符覆盖的子词表示取均值。
2. 词典：只读train。所有训练实体词与频次≥2的jieba词合并，保留1–8字符词项。每字符最多4个候选，按长度降序、起点升序、词ID排序；每候选分别保留BMES，padding另用ID4。
3. Eq.(4)–(10)：词与边界嵌入拼接，`Linear → tanh → Linear`，通过双线性注意力汇聚候选，再加到字符表示。无候选时增量为0。
4. BiGRU：两层，每方向384，层间dropout0.2；严格实现Eq.(11)–(14)的reset-before递推与更新门方向，输出直接进入Manhattan attention。输入投影批量计算，递推由TorchScript执行。
5. Eq.(17)–(20)：8头，`softmax(-L1(Q,K))V`，拼接后线性输出。无额外融合门、可学习温度、距离平滑或DOT分支。
6. Eq.(22)–(23)：按实体类型独立投影、RoPE和内积。按论文公式使用未除以 `sqrt(d)` 的GP分数。上三角含对角线，排除padding。
7. 原论文未写出的损失在本实现中明确为标准Global Pointer multilabel categorical cross-entropy：对每个样本及类型计算 `log(1+sum_neg exp(s)) + log(1+sum_pos exp(-s))`，然后对batch×type取平均。空正集或负集仍有定义。

窗口、字符/子词映射、候选排序、损失和新训练参数属于需写入修订方法与实验设置的实现细节。当前编码器实际配置由运行记录保存，不能用原表中“vocab size 8000”等不符数值替代。
