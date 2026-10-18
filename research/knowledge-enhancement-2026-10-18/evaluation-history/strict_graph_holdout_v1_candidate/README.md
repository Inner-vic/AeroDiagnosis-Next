# 严格门控独立验证：冻结数据

20个新来源家族，120候选题经答案隐藏独立AI非专家复核后保留113题，7题在任何预测前排除。严格跨块链仅2题（5/7条语义作用边）；未达到计划的2/3边严格链样本目标，相关统计只作探索。

本批已完成一次四组独立验证，452条排名。双文本及严格门控完整证据@10均99/113，没有完整题改善或退化；严格门控1题丢失部分参考证据，不推广到主排序。本批现已见，后续调参仅作开发。

- `strategy-freeze.json`：来源/问题/预测前固定的候选规则及120份检索源码哈希。
- `source-selection-after-review.json`、`source-freeze.json`：34候选来源复核、固定发现顺序20来源、原文件/规范文本/连续窗口。
- `translated-corpus-final.json`、`translation-adoption.json`：645块中文翻译、118处独立来源修订。
- `source-graph-final.json`、`graph-source-semantic-review.json`、`graph-freeze.json`：260关系在出题前审核与冻结；非无人审核自动图谱准确率验证。
- `blind-question-independent-review.json`、`question-final-adoption-audit.json`、`questions-final.jsonl`：作者答案/图谱/策略隐藏的来源重建，包含所有拒绝理由。
- `question-freeze.json`、`evaluation-freeze.json`：113题全部非空引用与充分集合先校验，再固定全部源代码、源文件、索引及数据指纹。
- `corpus-final.json`、`index-vector-audit.json`：活动语料2580块/135版本，全部645新中文向量复算误差0，93个旧非活动存储行由活动版本范围排除。

索引：`.runtime/strict-graph-holdout-v1/index.sqlite`。

预测与审计：`.runtime/experiments/strict-graph-holdout-v1-20261016/`，包括 `predictions.jsonl`、`summary.json`、`paired-comparisons.json`、`safety-audit.json`、`independent-audit.json`、`failure-analysis.json`、`completion.json`。

执行报告：`docs/upgrade/execution/STRICT-GRAPH-HOLDOUT-2026-10-16.md`。

从集成项目根目录执行只读复核：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/audit_strict_graph_results.py
```

准备流程依次为 `prepare_strict_graph_holdout.py prepare/translate/extract/anchor`、独立来源语义审查、`adopt_strict_graph_semantics.py`、`author_strict_graph_holdout.py author/pack`、独立答案隐藏题目审查、`adopt_strict_graph_questions.py --final`、`evaluate_strict_graph_holdout.py build/freeze/evaluate`。build可在题目盲审时独立运行；graph冻结必须早于author，最终freeze必须早于任何evaluate。

冻结目标已存在时脚本拒绝覆盖。复现须使用隔离副本、保存原始结果与相同相对输入路径，不删除或覆盖本次归档。付费准备沿用共享42元累计账本，密钥仅从用户指定本地文件读入内存，秘密不在本数据目录中。评分与只读审计不调用付费API。
