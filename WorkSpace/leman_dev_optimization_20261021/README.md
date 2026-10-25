# 开发集校准优化与冻结测试

本包复用定稿2的12路神经网络候选。它不训练神经网络，不占用GPU，不改变正在运行的消融源码。

## 预先固定的比较

| 方案 | 作用 |
|---|---|
| legacy_oof | 保留原HGB容量，在折外预测上拟合阈值，减少训练内概率对阈值的影响 |
| balanced_oof | 增大叶节点最小样本数和L2正则，取消负例1.5倍权重 |
| compact_oof | 减少叶节点数并进一步增强正则，检验模型容量的影响 |
| conflicts_oof | 加入8维无标签竞争特征，描述同跨度异类型、同类型不同边界和其他重叠候选 |

开发集外层为原来的两个按文本哈希划分的折；每个外层训练折再分为三个内层折。阈值只利用内层折外预测拟合，外层验证标签只参与计分。全局阈值粗搜索后，仅对支持数不低于200的类别作一次收缩后的粗粒度调整。四个方案比较完后选外层Micro F1最高者；提升不足0.03个百分点则不进入测试。

最终HGB用全部开发集拟合，最终阈值由选定方案的两折折外分数拟合。基础神经权重历史上经过开发集选模，因此这里称“后处理交叉验证”，不能当成整个系统的独立验证。

## 服务器使用

目录：`/root/data/WorkSpace/leman_dev_optimization_20261021`

```bash
cd /root/data/WorkSpace/leman_dev_optimization_20261021
python optimize.py
python test.py
```

本轮优化已由助手执行；完成后不要再次运行`optimize.py`，它会拒绝覆盖既有结果。`test.py`需要`SELECTED_FREEZE.json`，使用已完成的BF16测试候选缓存，在CPU上执行一次冻结模型评估。若测试结果已存在，它直接显示原结果。

输出：`/root/data/WorkSpace/output/leman_dev_optimization_20261021`。

- `PROTOCOL.json`：比较前保存的方案和选择规则。
- `AUDIT.json`：原版精确重放、快速与标量区间解码一致性、候选召回。
- 四个方案的JSON及逐句交叉验证预测：保留全部比较结果。
- `SELECTED_FREEZE.json`：测试前固定的模型身份、选择结果及SHA256。
- `selected_hgb.joblib`、`selected_decoder.json`：新模型与阈值；原版文件不覆盖。
- `test/RESULT.json`、`test/test_predictions.jsonl`：新版本测试结果与逐句预测。

原版测试F1已知，因此新结果属于同一测试基准的再次评估。不得根据本次测试错误继续调参数并把测试集称为从未观察的留出集。本文本不承诺达到95%；以实际结果为准。
