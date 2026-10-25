# 定稿版本2：冻结系统测试集评估

本入口评估产生95.2422%开发集重新前向结果的完整12路系统。原冻结缓存重放为95.2597%。测试指标必须实际推理产生，不能根据开发集推算。

服务器代码：`/root/data/WorkSpace/leman_final_v2_test_20261021/`。
服务器输出：`/root/data/WorkSpace/output/leman_final_v2_test_20261021/`。

## 固定协议

- 按原顺序加载12路权重，逐文件SHA256匹配历史身份。EMA来源从原last断点的ema.shadow提取，soup使用既有平均权重文件。
- 使用原训练词表、字符表、tokenizer、窗口规则、各路候选floor、99维特征、全dev拟合后冻结的HGB及类型阈值和区间解码。
- 单张空闲RTX4090，batch16，BF16 encoder与FP32自定义模块、TF32开启，与先前完整前向测速协议一致。
- test2.txt只用于生成输入和最后评价；候选生成前清除gold字段，后处理只接收文本。无训练、无HGB拟合、无test阈值搜索。
- 原始完整test为主终点。另报告与train/dev文本完全相同的样本数及排除这些文本的补充评价，保留所有原始预测。
- 原评价代码与独立BIO读取/实体计数必须完全吻合，再写COMPLETE。

## 命令

仅CPU预检，重放完整dev归档预测，不运行test：

```bash
cd /root/data/WorkSpace/leman_final_v2_test_20261021
python test.py --preflight
```

GPU空闲时直接测试：

```bash
python test.py
```

在现有消融队列全部完成后自动测试：

```bash
nohup /root/miniconda3/bin/python -u test.py --wait-for-ablation > /root/data/WorkSpace/output/final_v2_test_launcher.log 2>&1 < /dev/null &
```

等待模式不分配GPU，持有单独测试锁。先等待现有消融suite锁释放，再核对全部20项新任务的COMPLETE，并确认GPU无其他进程；任何条件不满足均退出并写FAILED。测试期间继续持有suite锁，防止同一消融队列重启后抢占GPU。不会修改现有训练代码、启动第二套训练或中止现有进程。

## 输出

- `PREFLIGHT.json`：冻结dev预测逐句一致证据。
- `STATUS.json`：等待、哈希核验、测试推理到哪一路、完成状态。
- `RUN_IDENTITY.json`：数据、12路权重、冻结资产、代码、数值精度身份。
- `candidates/`：真实test各路候选及SHA；中断后仅复用身份一致的已完成候选。
- `test_predictions.jsonl`：逐句最终实体预测。
- `TEST_RESULT.json`：Micro P/R/F1、Macro13、各类指标、覆盖和执行记录。
- `INDEPENDENT_RECOUNT.json`：按原始BIO独立复算TP/FP/FN。
- `SPLIT_OVERLAP.json`、`per_type.csv`、`COMPLETE.json`。

`frozen_pipeline.py`是先前已核验的benchmark_pipeline.py逐字节副本，只复用其函数，不调用原dev测速main。`probe_cpu.py`只在8个dev窗口做过CPU可运行性检查，不产生正式测试结论。正式测试保持GPU BF16协议。
