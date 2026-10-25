# Leman：当前结构的统一消融执行包

第一项运行完整模型 **F / seed42**。一次命令依次完成 A、B、C 三个阶段；中断后重复同一命令，自动跳过已经完成的阶段并恢复当前阶段。根目录为 `/root/data/WorkSpace`，使用两张 RTX4090，每卡 batch32，全局 batch64。

## 开始训练

执行包已部署至服务器；现有 Python 环境可直接使用。

```bash
cd /root/data/WorkSpace/leman_controlled_ablation
nohup bash train.sh --variant F --seed 42 > /root/data/WorkSpace/output/F_seed42_launcher.log 2>&1 < /dev/null &
tail -f /root/data/WorkSpace/output/F_seed42_launcher.log
```

前台启动时进度条保持原位刷新；后台启动时写入日志，各阶段另存按行日志。按Ctrl+C退出`tail -f`只关闭查看，不会停止后台训练。训练后自动生成紧凑回传ZIP；无需再运行收集命令。可以直接告知训练完成，由已授权SSH连接读取结果。

## 阶段与模型选择

| 阶段 | Epoch | Encoder LR | Head LR | 学习目标与起点 |
|---|---:|---:|---:|---|
| A | 20 | 3e-5 | 1e-4 | 公开ERNIE 3.0编码器，新初始化任务模块；标准GP MLCCE |
| B | 5 | 1e-6 | 1e-5 | 同配置同seed的A-best；positive项1.1与train类型权重 |
| C | 3 | 3e-6 | 1e-5 | 同配置同seed的B-best；FGM epsilon0.5，0.5 clean + 0.5 adversarial |

完整模型使用当前版本的字符对齐与字符残差、gated lexicon adapter、2层BiGRU、归一化Manhattan attention、RoPE Global Pointer和boundary辅助头。该包复用当前F1版本的前向代码；不读取历史任务模型作为A阶段的起点。

共同设置：字符窗口192、重叠64、跨度上限64、词候选4、attention heads8、GP head size64、GRU每向384；AdamW wd0.01，bias及1维参数wd0；每阶段线性schedule与5% warmup；FP64梯度范数计算、上限1；编码器BF16、自定义层FP32。EMA沿用现有实现：实际decay为 `min(0.995,(1+updates)/(10+updates))`，每个阶段重置。

新阶段训练随机流从`seed+rank`开始，断点续训恢复保存的随机状态。当前训练集形成48,884个窗口，对应双卡global batch64下每epoch764个step。数据准备已核对所有train/dev gold均被完整窗口和跨度上限覆盖。

每epoch评估live与EMA，以原始阈值0的精确字符跨度Micro F1选模。相同F1保留更早记录，同epoch并列选live。B/C把阶段入口纳入epoch0候选，因此阶段内微调不会丢失入口模型。最终固定模型为C的best，可能是阶段入口。Micro、Macro13及逐类计数均保留。训练默认只读取train/dev，不运行test。

## 断点恢复与停止

自动保存：每250个优化器step、每epoch结束保存完整`last.pt`；每5epoch保存`epoch_005.pt`等快照；`best.pt`独立保存。C只有3epoch，使用逐epoch的last及best。快照在同一文件系统上使用硬链接保留对应保存点，之后更新last不会改动旧快照。

完整断点包含模型、优化器、scheduler、EMA、每rank随机数状态、数据epoch/批次位置及历史best。BF16没有GradScaler。恢复须使用同一执行包、配置、数据和两张GPU；重复原命令即可。突然断电从最近一次已完成的保存点恢复，不保证保留尚未保存的step。

在另一个终端请求有序停止：

```bash
touch /root/data/WorkSpace/output/leman_controlled_ablation_v1/F_seed42/STOP
```

训练器完成当前优化器step后保存并退出，启动器不会继续下一阶段。恢复：

```bash
rm -f /root/data/WorkSpace/output/leman_controlled_ablation_v1/F_seed42/STOP
cd /root/data/WorkSpace/leman_controlled_ablation
bash train.sh --variant F --seed 42
```

启动器收到Ctrl+C时也会写入STOP并等待保存。已完成阶段的`COMPLETE.json`记录最佳权重哈希；重复执行时会验证并复用。

## 八个配置

| 命令参数 | 结构变化 |
|---|---|
| `F` | 完整模型，max_words4 |
| `no_lexicon` | 去掉整个词典适配分支 |
| `no_attention` | 去掉附加attention残差块 |
| `dot_attention` | 归一化q/k的负L1替换为DOT，其余投影、正温度、1/√d、残差与norm不变 |
| `crf` | GP替换为BIOES线性层+CRF；保留boundary辅助头 |
| `max_words3` | 最大词候选3 |
| `max_words5` | 最大词候选5 |
| `encoder_gp` | 同一编码器字符对齐后直接GP，去掉其他任务分支及boundary辅助头 |

先构建完整初始化，再移除消融模块；同seed共享参数保持相同初值。CRF和encoder+GP的新输出头在隔离的初始化随机流中创建。所有配置的词表、字符表及词ID相同，词候选数量仅影响匹配输入。

CRF以其原生序列NLL训练，B阶段的GP权重不作用于CRF；C同样使用FGM。训练窗口中只给完整包含的实体标BIOES，窗口截断实体不构造新的片段实体；boundary标签与GP数据一致。评价将同句各窗口emissions平均后整句Viterbi，使用完整原始gold计数。此配置是输出头及对应训练目标的对照。

第一项完成后，再逐项使用上述参数及seed42/43/44。总矩阵是24个配置—seed任务，每项最多28epoch。默认命令只启动一个任务；LEBERT不在本包的训练列表中。

## 输出与资源

任务目录：`/root/data/WorkSpace/output/leman_controlled_ablation_v1/F_seed42/`。

- `A/`、`B/`、`C/`：各阶段best/last/快照、训练日志、history、配置及源码快照。
- `*/initialization.json`：阶段入口来源与逐参数哈希。
- `*/best_dev/`：所选模型逐句预测、Micro/Macro及逐类结果。
- `SELECTED_MODEL.json`：最终模型路径、哈希和原始dev成绩。
- 上一级 `return_controlled_F_seed42_*.zip`：完整小文件及权重身份；权重留服务器。

手动重新收集（不训练）：

```bash
cd /root/data/WorkSpace/leman_controlled_ablation
bash train.sh --variant F --seed 42 --collect-only
```

共享资源位于`output/leman_controlled_ablation_v1/resources/`。词典仅由train gold和train分词统计构建；字符表仅由train生成。3/4/5词候选使用不同缓存键，跨seed复用缓存。数据准备脚本`prepare.py`不进行训练。

完整断点包含optimizer、EMA和best备份，比单独推理权重大。按当前模型尺寸，一个完整F任务保存全部阶段及周期快照预计约50 GiB，临时原子保存另需约8 GiB空间；实际大小以产物为准。当前服务器空间足以运行第一项，后续矩阵按任务分批运行并归档，不同时展开全部24项。

## 实验记录与后续分析

`COPIED_FILES.json`记录逐字节复用文件及其来源；`PACKAGE_MANIFEST.json`记录交付文件哈希；`PROTOCOL.json`记录配置、seed及选择规则。双卡合成QA检查前向复用、共享初始化、FGM梯度、CRF解码、阶段切换和断点恢复；合成结果不作为论文成绩。

本包完成结构矩阵的训练与原始dev评价。统一阈值/解码分析及最终test评价在所需权重齐备后执行。CMeEE采用另已固定的inner-dev/official-dev协议，需要独立适配窗口与嵌套输出，不由本AgCNER启动器代跑。
