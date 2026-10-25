# 达到95%：14路候选融合的冻结测试结果

**完整测试集Micro F1为95.023222%，目标已达到。** 比原版94.814887%提高0.208334个百分点。本机与服务器独立计数一致。该成绩属于**14路候选融合＋111维HGB校准＋加权区间解码**，不能标作单个F/seed42神经网络的分数。

测试完成：北京时间2026年10月21日08:46:57。原消融已恢复，08:49:05确认训练超过暂停位置。

## 1. 测试结果与口径

| 指标 | 原定稿2：12路 | 第一轮校准优化：12路 | 本轮选定：14路 |
|---|---:|---:|---:|
| Micro F1 | 94.814887% | 94.813622% | **95.023222%** |
| Precision | 95.236691% | 94.943326% | 95.344845% |
| Recall | 94.396804% | 94.684272% | 94.703761% |
| Macro F1（13类） | 89.544283% | 90.041227% | 90.003595% |
| TP | 19374 | 19433 | 19437 |
| FP | 969 | 1035 | 949 |
| FN | 1150 | 1091 | 1087 |

完整测试集含5,974句、20,524个真实实体。评价必须同时匹配字符起止位置与实体类型。新方案比原版多识别63个真实实体、减少20个误报。F1按`2×19437/(20386+20524)`计算，未改变测试标签、测试分母或评分规则。

按文本簇配对Bootstrap 2,000次，相对原版Micro F1差值的95%区间为[+0.058456,+0.361403]个百分点。该区间是探索性的固定预测比较，没有对历史模型选择及重复基准评估作多重校正。

### 去重子集单独报告

沿用原审计，完整测试集中有125句与训练集或开发集文本完全相同。排除它们后的5,849句、20,119个实体上：

| 指标 | 原定稿2 | 本轮14路 |
|---|---:|---:|
| Micro F1 | 94.774685% | **94.979674%** |
| TP | 18,981 | 19,042 |
| FP | 955 | 936 |
| FN | 1,138 | 1,077 |

因此，“达到95%”限定于原协议的完整测试集；去重子集为94.979674%，尚未达到95%。论文中两种口径需要分别列示，不能混用。

## 2. 优化来自哪里

新加入的两份权重均已训练完成：

1. **F42_C**：统一F/seed42的C阶段best，权重来源为EMA，开发集原记录94.5811%。
2. **F42_B**：同一训练链的B阶段best，实际上是从A阶段继承的入口最佳权重，开发集原记录94.4942%。它不是独立随机种子。

旧12路同样包含共享父权重的微调、EMA和soup。因此本系统应称多来源候选融合，不应表述为14个独立训练重复。

没有增加新的神经训练；新增操作是固定权重前向、开发集CPU HGB拟合和解码校准。三个新来源的候选门槛在比较前固定为GP输出乘8后大于−32。无词典分支也生成了开发集候选并参与比较，最终没有进入测试方案。

| 逐步增加来源 | 开发集候选数 | 覆盖真实实体数 | 未覆盖真实实体数 | 候选召回 |
|---|---:|---:|---:|---:|
| 原12路 | 23,769 | 20,170 | 612 | 97.0551% |
| + F42_C | 25,176 | 20,389 | 393 | 98.1089% |
| + F42_B | 25,563 | 20,412 | 370 | 98.2196% |
| + no_lexicon | 26,230 | 20,438 | 344 | 98.3447% |

选定组合比原12路多覆盖242个开发集真实实体。融合还利用新增分数重新判断共有候选，并不只是把新增候选直接加入最终预测。

111维特征包括14×6维来源特征、13维实体类型、6维训练集边界上下文和8维来源共识统计；原词典与边界先验仍只来自训练集。最终HGB参数：180轮、学习率0.05、最多15叶、每叶最少70样本、L2=4、正负例等权、random_state=42。预测概率转为logit，按冻结类别阈值进行加权区间解码。

## 3. 如何选定方案

比较范围在运行前保存于`PROTOCOL.json`：3个来源组合×3个HGB容量×2种阈值拟合方式，共18项。外层沿用原两个文本哈希分组折。每个外层验证折的标签不参与校准器及阈值拟合。

nested模式在外层训练折内用三折折外分数拟合阈值；self模式在外层训练折内部使用训练内分数拟合阈值。两种模式都通过外层验证评价。阈值使用粗网格及一次支持数收缩的类别调整，不进行测试集错误特例修补。

唯一选择指标是开发集外层Micro F1。最终选中`plus2_balanced_oof_self`，开发集后处理交叉验证Micro F1为**94.923674%**，比原版94.476154%提高0.447521个百分点。开发集配对区间为[+0.291008,+0.612436]个百分点。

最终HGB使用全部开发集拟合；因选中self模式，最终阈值使用该全开发集模型在开发集上的分数拟合。这一阶段不是独立评估。`SELECTED_FREEZE.json`在北京时间08:44:50保存，随后才为选中的两份新权重生成测试候选、执行冻结测试。

基础神经权重历史上曾用开发集选模，所以这里是后处理交叉验证。测试基准此前的汇总结果已知，本轮没有用测试错误选择参数，但也不能把它称为从未观察过的新测试集。

| 开发集方案 | 外层Micro F1 | 相对原版/百分点 |
|---|---:|---:|
| plus1_legacy_oof_nested | 94.8109% | +0.3347 |
| plus1_legacy_oof_self | 94.8225% | +0.3463 |
| plus1_balanced_oof_nested | 94.7795% | +0.3033 |
| plus1_balanced_oof_self | 94.8161% | +0.3400 |
| plus1_compact_oof_nested | 94.8601% | +0.3840 |
| plus1_compact_oof_self | 94.8369% | +0.3608 |
| plus2_legacy_oof_nested | 94.8288% | +0.3526 |
| plus2_legacy_oof_self | 94.8584% | +0.3823 |
| plus2_balanced_oof_nested | 94.8707% | +0.3945 |
| plus2_balanced_oof_self | 94.9237% | +0.4475 |
| plus2_compact_oof_nested | 94.8004% | +0.3243 |
| plus2_compact_oof_self | 94.8556% | +0.3795 |
| plus3_legacy_oof_nested | 94.7703% | +0.2941 |
| plus3_legacy_oof_self | 94.8933% | +0.4171 |
| plus3_balanced_oof_nested | 94.8849% | +0.4087 |
| plus3_balanced_oof_self | 94.8343% | +0.3582 |
| plus3_compact_oof_nested | 94.9001% | +0.4239 |
| plus3_compact_oof_self | 94.9231% | +0.4469 |

## 4. 消融恢复与复核

- 安全暂停位置：no_boundary，第5轮，下一batch=189，global_step=3245；保存模型、优化器、调度器、EMA和每个rank的随机数状态。
- 推理期间原断点SHA256不变，原训练源码与配置没有修改。
- 08:47:03派发原队列恢复；08:49:05确认已到batch=233，梯度记录step=3250，两张GPU继续训练。
- 新旧权重身份、词表身份及候选缓存SHA256均记录。两份新增权重都执行了本轮真实测试集前向。
- 服务器读回逐句预测计分后，本机又用原始BIO文件独立重算，Micro、Macro及各类指标完全一致。
- 现已达到完整测试集95%的既定目标，本轮不继续根据测试表现搜索参数；原版、第一轮结果和全部18项开发集比较均保留。

## 5. 查看与复算

服务器代码目录：`/root/data/WorkSpace/leman_diversity_20261021`。

```bash
cd /root/data/WorkSpace/leman_diversity_20261021
python test.py
python recount.py
```

`test.py`显示这次已完成的冻结测试结果；`recount.py`重新读取原始测试标签和预测文件、核对SHA256并独立计算F1。前者的原始执行使用14路候选缓存、冻结HGB和解码器。若要重新从文本运行全部14路神经前向，需要逐一使用清单中的14份权重及各自词表/预处理；不存在一个单独的`.pt`文件即可代表本系统的95.0232%。

本地代码：`diversity_20261021/`。完整证据：`evidence/diversity_20261021/`。单独保存的定稿3清单及校准资产：`final_versions/leman_final_v3_20261021/`。

关键文件包括`selected_hgb.joblib`、`selected_decoder.json`、`SELECTED_FREEZE.json`、`SUMMARY.json`、`dev_comparison.csv`、`test_comparison.csv`、`test/RESULT.json`、`test/LOCAL_RECOUNT.json`、`test/test_predictions.jsonl`及`RESUME_VERIFIED.json`。

测试SHA256：`287377cead6f2a2990bb21e08793532c850e2c343cda614e0cdca6bc8ee01810`。

新预测SHA256：`95a6f9e3e2e699f54025577a16358df3b9be97ba55ed8fd5f5bacd976a11d6de`。

冻结清单SHA256：`e60586d798e02df07d917f7c707749e6bb0a931e88f77a79ea677237f15def74`。

## 6. 可用于论文的英文结果表述

> The enhanced system combines candidates from the original twelve-source pipeline with two checkpoints from the unified F/seed42 training run. The fusion configuration was selected by development-set cross-validation and frozen before evaluation. On the complete test split (5,974 sentences and 20,524 entities), it achieved an exact character-span-and-type micro-F1 of 95.0232%, with precision of 95.3448% and recall of 94.7038%. Relative to the twelve-source reference, it recovered 63 additional true entities and removed 20 false positives, yielding a 0.2083-percentage-point micro-F1 gain. On the secondary subset excluding sentences whose text also appeared in training or development data, micro-F1 was 94.9797% over 5,849 sentences. These results characterize the enhanced multi-source system; its component checkpoints include shared training lineages.

该段是可核验结果稿，尚未直接写入原论文或覆盖已有审稿回复PDF。后续方法章节和实验表必须同步说明14路融合与111维校准，不能继续使用原12路/99维定义来解释本结果。
