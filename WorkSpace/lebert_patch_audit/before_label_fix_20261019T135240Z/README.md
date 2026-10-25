# LEBERT–CRF：AgCNER 双卡基线

本包用于重新建立论文 Table 3 的词典增强基线。名称为 **LEBERT**，依据 ACL 2021 正式论文 *Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter*。旧稿“LeBERT 74.56”不作为该模型的复现成绩。

## 1. 依据与实现范围

- 正式论文：https://aclanthology.org/2021.acl-long.454/
- DOI：10.18653/v1/2021.acl-long.454
- 作者实现：https://github.com/liuwei1206/LEBERT
- 本次核对的作者版本：`60668c0ac7dea72d684db531e996cdaefff1d634`。
- 论文与作者代码均已在项目 `execution/literature/` 存档。

这是针对 PyTorch 2.7 / Transformers 4.56 的独立兼容实现。核心结构对应作者的 WCBertCRF：BERT 第1层输出 → 两层词向量映射及 tanh → 字词双线性注意力 → 残差、dropout、LayerNorm → 后续 BERT 层 → CRF。它不是此前撤回的 Lex-BERT，也不使用 ERNIE 替代 BERT。

## 2. 固定配置

| 项目 | 本次配置 | 来源/说明 |
|---|---|---|
| 编码器 | google-bert/bert-base-chinese | 论文及作者实现使用 Chinese BERT；下载时记录仓库 commit |
| 外部词向量 | Tencent AI Lab，200维，v0.2.0 | 作者 README 更新后的公开下载链接；不是原实验旧版向量的逐字节复制 |
| 外部词表 | 文件前1,000,000行 | 作者 demo 的扫描预算；按文件固定顺序，无任务数据选择 |
| 词向量 | 可训练，pad行固定为0 | 不使用 dev/test 标注建立词表 |
| 词典适配层位置 | 第1、2个 BERT Transformer 层之间 | 论文及实现 |
| 每字符匹配词上限 | 5 | 作者 demo；本包明确用长词优先，其次起点、词ID排序 |
| 字符表示 | 每原字符对应一个 BERT vocab token；未知字符用 UNK | 与作者逐字符输入方式一致；不做子词跨度合并 |
| 输入长度 | 254字符 + CLS/SEP | 论文最大序列长度256；长句采用64字符重叠窗口 |
| 输出 | BIOES linear-chain CRF | 保留原句全部 gold，按字符边界和类型精确评分 |
| 优化器 | Adam，weight_decay=0 | 论文使用 Adam；作者 demo 代码使用 AdamW，wd=0 |
| 学习率 | BERT 1e-5，其余1e-4 | 论文 |
| epoch | 20，不早停 | 论文训练预算 |
| batch | 16/GPU × 2，有效32 | 本次 AgCNER 适配配置；原论文其他数据集通常为4，MSRA为20 |
| warmup | 总更新步数10%，随后线性衰减 | 本次 AgCNER 适配配置；作者 demo 为190步 |
| 精度 | BERT/adapter BF16，CRF FP32 | 双4090执行配置 |
| seed | 42、43、44；每次从同一预训练BERT独立开始 | 本次重复方案 |
| checkpoint | best dev Micro F1；last每250步/每epoch；每5epoch快照 | 本次运行管理 |

预训练BERT与外部向量的内容身份均写入 `run_config.json`。原论文没有 AgCNER 实验，因此应报告为“基于论文和作者实现的 AgCNER 重新实现”，不声称复现了原文在其数据集上的分数。

与作者旧脚本的明确适配：长句滑窗替代直接截断；窗口 emission 平均后进行整句 CRF Viterbi；CLS/SEP 不进入CRF标签；无匹配词时上下文严格为0；词候选采用确定性排序；训练过程不读取test。以上规则在本次新基线结果出现前固定。没有用原论文结果为这些适配背书。

## 3. 服务器目录

```text
/root/data/WorkSpace/
  data/train2.txt, dev2.txt, label.txt, test2.txt
  lebert_baseline/                  本包
  resources/bert-base-chinese/      新增BERT资源
  resources/tencent_200d_first1m/   新增词向量资源
  output/lebert_agcner_seed42/      正式结果
  output/lebert_agcner_seed43/
  output/lebert_agcner_seed44/
```

现有 `model/` 中的 ERNIE 权重可继续保留。本包单独下载 BERT，不使用该目录。

## 4. 安装与资源准备

代码已放至当前服务器 `/root/data/WorkSpace/lebert_baseline`，不需要再上传。

```bash
cd /root/data/WorkSpace/lebert_baseline
/root/miniconda3/bin/python -m pip install -r requirements.txt
/root/miniconda3/bin/python prepare_resources.py
```

资源准备按固定顺序截取腾讯向量的前100万行，不读取 train/dev/test；若源为官方 `.tar.gz`，使用流式解压，完成前缀后停止下载。资源提取到临时目录，完成后才进入正式目录。重复执行会复用已完成资源。BERT 首次解析下载commit并保存在 `resources/BERT_RESOURCE.json`，后续沿用该commit。

如果服务器无法访问资源站点，可以从对应官方链接下载后传入服务器：BERT文件放至上表路径；腾讯词向量执行：

```bash
/root/miniconda3/bin/python prepare_resources.py --vectors-only --vectors /你存放的/tencent-ailab-embedding-zh-d200-v0.2.0.tar.gz
```

本次未下载完整预训练资源；因此尚无完整 BERT 在本机的速度和峰值显存实测。包内 synthetic QA 使用 tiny BERT，只用于验证实现与运行流程。

## 5. 正式训练与续训

推荐在 tmux 中执行：

```bash
tmux new -s lebert
cd /root/data/WorkSpace/lebert_baseline
bash run.sh 42 43 44
```

三个seed依次运行。`run.sh` 使用两张可见GPU，固定有效batch32，并把日志保存到各seed目录。

中断后重新执行同一命令即可。已完成seed读取last确认20epoch结束后退出；未完成seed恢复模型、Adam状态、调度器、每rank RNG、epoch、下一batch及loss累计值。硬中断最多重做最近250步；正常信号停止会在本步完成后存盘。`best.pt` 只保存最佳模型和评价信息；**续训读取的是 `last.pt`**。

## 6. 输出与后续评价

每个seed产生：

- `run_config.json`、`history.json`、`train.log`；
- `best_dev_metrics.json`、`best_dev_predictions.jsonl`；
- `best.pt`、`last.pt`、每5epoch快照。

训练完成后这些文件留在服务器即可，我会读取并统一汇总。三个seed都保留，主报告均值±样本SD及逐seed值，不从三个seed中挑一个充当基线总体结果。

test 在统一评价矩阵固定后独立执行；`run.sh` 不调用test。未来单seed测试命令为：

```bash
/root/miniconda3/bin/python train.py --root /root/data/WorkSpace --seed 42 --test
```

当前脚本针对 AgCNER BIO 文件。CMeEE的嵌套JSON、Weibo token编码及其他数据标签需要各自适配，不直接套用这条命令。

## 7. 已执行的检查

`qa.py` 使用隔离临时目录中的合成文本、随机tiny BERT和小型词表。检查包括：CRF与穷举路径的NLL/Viterbi一致；padding与单字符梯度；词匹配mask；滑窗gold覆盖；tiny BERT反向；词向量解析；双4090/BF16两epoch；第1步暂停后恢复；独立test路径。结果见本包 `QA_RESULT.json`。

双卡检查修正了单字符batch没有转移边导致的DDP问题：转移项为0时仍参与相同的梯度归约；数学上的CRF得分及损失不变。该场景已纳入QA。
