# Leman-GP 论文源码与复现材料

归档日期：2026-10-25。源码从服务器实际目录获取并校验；本地补入最终论文结果与对应历史源码快照。本仓库不含神经网络权重、原始 AgCNER 数据和大型候选/特征缓存。

## 目录

- WorkSpace/：保留服务器相对目录。包含训练、词典构建、基线、消融、机制验证、候选融合、评价及绘图代码。
- WorkSpace/data/：原始 AgCNER 划分不随本仓库分发，下载说明见 WorkSpace/data/README.md。
- WorkSpace/model/：分词器资源；预训练 encoder 配置与权重不随本仓库分发。
- WorkSpace/output/：配套词表、运行配置、校准器、阈值和身份记录等小型资产。
- Paper_Final_V3/：论文最终系统的清单、校准资产、测试预测与结果。
- Historical_Source_Snapshots/：与原实验一起保存的代码快照，重现历史实验时优先对照。
- Archive_Audit/：环境与检查点清理计划。

## 主要代码入口

单模型完整训练：WorkSpace/leman_controlled_ablation/。
短周期结构消融和模块机制：WorkSpace/leman_short_mechanism_v1/。
补充注意力机制实验：WorkSpace/leman_attention_mechanism_v2/。
最终融合选择与结果：WorkSpace/leman_diversity_20261021/。
固定候选特征/推理：WorkSpace/leman_final_v2_test_20261021/。
测试集消融与绘图：WorkSpace/test_ablation_20261022/ 以及 WorkSpace/ 下 evaluate_test_ablation 等脚本。
LEBERT 基线：WorkSpace/lebert_baseline/。

各目录的 README、config 和实际参数解析器定义相应命令。许多历史脚本使用 /root/data/WorkSpace 的服务器绝对路径；迁移时应配置新的根目录并检查依赖路径。归档不会自动启动训练。

## 结果与模型身份

最终测试 P=95.3448%、R=94.7038%、Micro F1=95.0232%。
Paper_Final_V3/WEIGHT_MANIFEST.json 列出完整系统的 14 份神经检查点、各自参数来源与哈希；这些文件保留在服务器，并不在本源码包内。重新从文本执行全部系统推理需要相应神经权重、词表与预处理资源。已保存的最终预测可独立复算结果；这与重新执行全部神经前向分别进行。

实验使用论文所述的 ERNIE 3.0 编码器配置（hidden_size=1024、18 层、16 个注意力头）。本公开仓库不包含预训练权重和 encoder 配置文件；复现时请按论文描述准备编码器权重，并按目标实验目录的配置加载。

## 环境与使用顺序

requirements_server_snapshot.txt 是归档时服务器的相关包版本；各次历史实验的 run_config.json 与 requirements.txt 保留原版本信息。GPU/CUDA 安装需匹配当前运行设备。

1. 阅读目标实验目录的说明和配置。
2. 核对数据与词表哈希、实体类型顺序、左闭右开字符跨度。
3. 取得 WEIGHT_MANIFEST 指定的权重后，按相应代码核对加载方式与词 ID。
4. 原始数据训练、开发选择和测试评价按各实验协议执行。
5. 论文图表由对应测试结果及机制记录生成，保留 CSV/JSONL 与 SVG。

本包归档第三章相关项目，服务器已有的 chapter4_lexicon_contribution 目录保持独立。
