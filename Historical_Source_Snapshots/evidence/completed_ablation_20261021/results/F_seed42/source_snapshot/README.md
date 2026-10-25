# 短周期模块干预与机制验证执行包

本包复用已经完成的F42权重，在同一父模型下分析模块的即时作用及3-5轮适应结果。正式训练由用户启动。原统一消融包、F42权重、历史定稿2均不改动。

## 实验回答什么

主问题是：在当前已训练模型中，某项计算如何影响词汇融合、上下文整合、边界判断及最终预测；同预算继续训练后，其他模块能补偿多少？本实验的结论是条件于F42父权重的功能干预与适应效应，不能称为独立从头训练的结构必要性证明。保留原24项从公共预训练起点的矩阵定义；本包不偷偷把其未完成项标为完成。若论文要保留“从头训练也更优”的主张，应继续执行相应原矩阵。

主终点统一为固定最后一轮EMA，次终点为包含epoch0入口的dev-best。所有正负方向如实输出。不同适应seed仍共享F42父权重，不等于独立预训练起点。所有正式输出仅使用train/dev，不读取test。

## 矩阵与机制

| 配置 | 从哪里开始 | 轮数 | 干预与结论范围 |
| --- | --- | --- | --- |
| F | C-best | 5 | 完整模型继续训练，共同预算对照 |
| no_lexicon | C-best | 5 | 整体移除词典分支；统计候选覆盖、Seen/Unseen、词频和错误转移 |
| no_attention | C-best | 5 | 去附加attention整个残差块，序列表示直接进入GP |
| dot_attention | C-best | 5 | L1换DOT，保留q/k LayerNorm、温度、缩放、投影及残差 |
| max_words3 / max_words5 | C-best | 各5 | 相同固定词ID，重用已有各自缓存；检验候选宽度和熵 |
| no_char | C-best | 5 | 删除字符身份残差及对应参数 |
| no_boundary | C-best | 5 | 删除辅助头/损失；推理初始预测应与F一致；检验后续优化作用 |
| no_recurrence | C-best | 5 | 每位置前驱隐状态归零；保留原输入投影/更新门，隔离递归上下文 |
| no_gru_linear | C-best | 5 | 整个BiGRU改为新线性1024→768；同时改变映射，应结合no_recurrence解释 |
| no_bmes | C-best | 5 | BMES嵌入置零并冻结，词身份及其投影保留 |
| uniform_lexicon | C-best | 5 | 有效候选等权；冻结双线性相关性参数 |
| fixed_gate | C-best | 5 | 元素门固定0.5；检验自适应门而不是移除所有词汇信息 |
| no_rope | C-best | 5 | 去GP旋转位置变换，保留类型投影和跨度缩放 |
| gp_reset | C-best | 5 | 重置GP头；作为CRF重置头的匹配比较 |
| crf | C-best | 5 | 新BIOES CRF头与对应NLL；合并窗口emission后整句Viterbi |
| encoder_gp | C-best的encoder | 5 | 编码器直接接新GP；表征维度变更，作为复合对照 |
| no_fgm | B-best | 3 | 与原C阶段预算、LR、目标一致，去对抗步骤 |
| clean_two_pass | B-best | 3 | 两次独立dropout的clean前向，各权重0.5；控制FGM双前向与计算差异 |
| unit_loss | A-best | 5 | 与原B阶段同预算、LR，改为单位正例/类型权重 |
| fgm_control | 已完成C末EMA | 0新训练 | 复用3轮C的最终EMA并重新推理；不是C选模成绩替代末轮 |
| weighted_control | 已完成B末EMA | 0新训练 | 复用5轮B的最终EMA并重新推理；不是B-entry best |

共22配置，其中20项需要新训练：17项结构/功能配置各5轮，no_fgm与clean_two_pass各3轮，unit_loss5轮，总96轮。两个原阶段完整控制直接复用。默认每配置seed42。原训练每轮约10-13分钟（C含FGM约21分钟），本次大部分为单clean前向；整套约16-22小时是历史耗时估计，真实速度以日志为准。不能保证任一新头在5轮内达到其充分训练上限；通过完整逐轮曲线、同预算gp_reset和epoch0→末轮区分解释。

## 一次运行保存什么

1. 原始父权重SHA、数据/词ID/字符ID、每个复用/新增/移除tensor、参数数量、源码快照。
2. epoch0干预预测（冷启动新头只测固定panel，不在全dev导出无意义的大量随机正跨度）、每epoch EMA全dev预测与计数、live/EMA曲线、固定末轮EMA与dev-best各一份权重。
3. 每50step记录模块梯度范数、参数范数、主目标/加权边界目标及FGM或双clean诊断。它们是优化过程描述，不直接等同于模块效应大小。
4. 固定128句dev机制panel，由文本SHA256选择，不按模型是否预测正确挑选。各配置、各epoch使用同一规则。保存词候选数、熵、门值、词典/字符/attention相对表示变化；逐head温度、熵、注意距离、实体内质量和uniform参考；typed start/end AP、precision、recall；gold跨度相对邻近错误边界的margin。不同窗口可能重复位置，panel不充当全语料分母。
5. 完整dev预测独立复算；逐类、频次、Seen/Unseen、长度召回率；FP/FN类型混淆/边界偏移/漏检；逐gold rescued/lost和完整案例。子群仅报有明确定义的Recall，不把FP无法归属的子群写成F1。
6. 每个主对照以完全相同文本簇配对bootstrap 4,000次，输出差值和95%区间。区间是固定预测的样本变化，不是seed SD；大量探索性比较不写成已校正多重检验的显著性结论。
7. 队列结束自动生成CSV、PDF/SVG/PNG图、回传ZIP和manifest。注意力图仅展示固定样本计算模式，需结合干预和错误转移支持作用解释。

## 运行命令

服务器目录：`/root/data/WorkSpace/leman_short_mechanism_v1`。

完整队列（正式训练，由用户执行）：

```bash
cd /root/data/WorkSpace/leman_short_mechanism_v1
nohup bash run.sh --group all --compact-completed > /root/data/WorkSpace/output/short_suite_launcher.log 2>&1 < /dev/null &
tail -f /root/data/WorkSpace/output/short_suite_launcher.log
```

跟踪当前配置：查看`/root/data/WorkSpace/output/leman_short_mechanism_v1/<variant>_seed42/train.log`。launcher在每个配置开始/完成打印状态。仅先运行原表核心对照时用`--group core`；补功能模块用`--group mechanisms`。机制组应保留已完成F/gp_reset参照，分析时才会计算对应配对差值。默认all先完整模型再各配置，串行使用两张GPU，遇到外部GPU进程、STOP、磁盘不足或训练错误即停止。

`--compact-completed`仅清理本新实验中已经完成并核对末轮EMA及best哈希的last/epoch优化器断点，保留final_ema.pt、best.pt和全部证据，并登记被清理文件的SHA。原F42和任何历史checkpoint不清理。运行中完整断点每250step及每epoch保存。约124 GiB剩余空间不能保留20项的全部优化器历史；该选项支持分项完成后释放空间。若希望保留这些新任务的所有优化器状态，去掉选项并另安排存储。

有序停止：

```bash
touch /root/data/WorkSpace/output/leman_short_mechanism_v1/STOP
```

当前配置会在optimizer step结束后保存退出。恢复前确认停止原因，移除suite和对应配置的STOP文件，再执行同一命令。没有STOP的普通中断直接执行同命令即可；完成配置核验后跳过，未完成配置从last严格恢复。保留launcher历史时将重定向改成`>>`。

只收集已有结果（不训练）：

```bash
bash run.sh --collect-only
```

重新分析与绘图（不训练）：

```bash
/root/miniconda3/bin/python analyze.py --root /root/data/WorkSpace
/root/miniconda3/bin/python plot_results.py --base /root/data/WorkSpace/output/leman_short_mechanism_v1
```

## 代码入口

`model.py`明确定义每个干预，`transfer_parent`逐项拒绝非预定missing/unexpected tensors；`train.py`承接已验证的双卡、EMA和resume流程；`probes.py`执行固定panel、保存及恢复RNG；`clean_control.py`提供FGM双clean控制；`inspect_fixed.py`仅固定权重推理；`import_controls.py`仅提取原阶段末EMA并推理；`analyze.py`从原始BIO重新计数；`plot_results.py`仅绘制实际CSV。

未改文件逐字节复制，身份记录在COPIED_FILES。缓存schema沿用原始协议以重用数据，并不表示本短周期方案与原24项从头训练矩阵是同一个实验。新的experiment/protocol字段、父权重和输出根目录分别保存。

## 已完成的免训练检查

13组固定权重干预已回传并独立复算。两个原阶段末轮EMA控制已完成重新推理：C末为94.5811%，B末为94.2738%；B末轮不是B-entry best的94.4942%。每项控制新增训练更新为0，重新推理的Micro计数与原history逐项一致。最终词候选记录已在两项控制上检查，每项19,688行、449个gold跨度，候选权重和最大偏差低于1.2e-7。

`lexical_slots.csv`包含词ID/文字/BMES/权重和是否被实际消费；`gold_span_margins.csv`另外区分词表覆盖与所有字符处候选可用。新增字段不追溯填写到早期固定干预数据。训练后的图表覆盖每epoch机制、梯度与目标组成；全部源数据保留，不按结果正负筛选。
