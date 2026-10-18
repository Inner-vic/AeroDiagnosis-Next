# Aero Evidence Benchmark v1 数据卡

日期：2026-10-13。协议版本1.2。定位：真实公开航空资料中的来源条件检索、证据完整性、拒答与限定关系支持判断。状态为`ai_cross_source_verified`，**不是航空专家人工金标**。

## 已构建资产

| 项目 | 实际数量与统计单位 |
|---|---|
| 来源文件 | 21个artifact，17份逻辑文档，13个文档家族，6家发布机构 |
| 内容与格式 | 19个内容文件、2个官方元数据文件；15 PDF、3 HTML、2 XML、1 CSV |
| 来源语言 | 17英文、3中文、1法文文件；译本与原版共享家族 |
| 检索/拒答问题 | 126题，117个不同`fact_id`；118本源题、6场景改写、2译题 |
| 主划分 | 19 dev本源题，99 test本源题；另8个压力变体全在test单独评分 |
| 回答条件 | 115可回答、11冻结语料证据不足；全部采用`all`证据语义 |
| 问题语言 | 109英文、16中文、1法文；语言变体不增独立事实数 |
| 领域分层 | 120燃气涡轮题、6跨领域基础题；不声称覆盖所有发动机 |
| 图谱声明 | 66个原始声明，另6个成对否定压力题；72记录、66原始声明组 |
| 证据定位 | 241条标注证据引用，168个唯一原文区间；绑定原件、正文与片段SHA-256 |
| 复核记录 | 199项，包含结果产生前剔除的1题；完整保留裁定和修订说明 |
| 测试依赖单位 | 7个源家族连通分量；不是99个统计独立观测 |

FAA教材/认证AC/适航指令/SDR、CAAC法规及指导文件、AAIB事故调查与勘误、Airbus安全文章、BEA原版与译本、EASA历史AD及取代元数据分别保留身份。来源逐文件URL、hash、型号/修订说明与权利状态见[corpus_manifest.json](corpus_manifest.json)。所有来源格式与文件数均来自实际落地文件，不以网页副本凑独立案例。

任务包含：条件适用33、定义7、证据不足11、机制17、多证据18、数值/表格17、程序顺序1、报送案例8、修订冲突5、症状检查9。样本按可获得性和技术价值选取；这不是均衡任务分布、随机行业抽样或舰队故障频率调查。

## 构建与复核路线

1. 冻结真实原件与来源身份，按权限保留本地全文，清洗为NFC/LF正文和可定位结构块。
2. 依据真实信息需求选题，明确型号、飞行阶段、资料修订/历史日期及调查结论强度；答案绑定原文区间，不绑定某种chunk ID。
3. 另一代理隐藏作者答案、标签与正证据，从来源重建回答和最小充分证据；检索池只是辅助，不是无答案判断依据。
4. 对照原文逐项裁定，保留修订记录。明确错误前提、可回答的否定结论与冻结语料证据不足的区别。
5. 按来源家族、内容依赖及事实归组开发/测试。译本、勘误、变体不得跨侧；需要跨dev/test证据的eng-q044在结果产生前排除。
6. 冻结协议、标签、来源和工具hash，再比较策略；模型成绩不得反向决定保留哪些题。

这轮复核实际修正了事故阶段被误当关系条件、部分历史手册参数被误当现行值、错误前提被笼统归为无答案，以及原始报告勘误的数值/单位解释。FR XML表题/列头曾丢失，也已修复清洗器并建立新快照，而非手改证据文本。

初次事故问题复核泄漏了task_type，因此全53题由未看作者标签的另一代理重新重建。新增否定压力题的初审者已知设计方向，仅作内容核验，6条又交给未见设计答案的代理重新重建。修正后核验与新盲审分别记录，不把它们混作独立标注者。详见[标注协议](ANNOTATION_PROTOCOL.md)、[标注来源说明](annotation_provenance.json)和[复核台账](review_ledger.jsonl)。模型精确ID未由会话环境暴露，不声称能逐字复现模型标注。

## 文件与字段

- [queries.jsonl](queries.jsonl)：题目、语言、任务、scope、参考答复、answerability、evidence_requirement、证据、fact_id、变体配对、split及裁定。
- [graph_claims.jsonl](graph_claims.jsonl)：subject/predicate/object、polarity、certainty、型号/阶段/条件/修订、supported/unsupported、unsupported_type及证据；压力题含paired_claim_of/claim_group_id。
- [protocol.json](protocol.json)：事前划分、排除项、指标、消融与统计规则；不是看到结果后选指标。
- [quality_audit.json](quality_audit.json)：可重算计数和检查状态。
- [method_audit.json](method_audit.json)：独立审计的内容、来源复用、分组和局限。
- [validation_report.json](validation_report.json)：软件验证和评分器边界检查；**不是模型效果报告**。
- [release.json](release.json)：正式文件集合与编译器SHA-256。数据卡自身也纳入指纹。
- [答辩说明](DEFENSE_NOTES.md)：能支持的结论、常见质疑和需要主动披露的边界。

证据中的`canonical_start`/`canonical_end`是清洗正文的Unicode字符偏移，末位不含；不是UTF-8字节、PDF字节或页内坐标。`source_locator`保留页/节点定位，canonical/raw/span hash分别核查版本和片段。评分时，相邻chunk的覆盖并集可满足一段证据，重叠不重复计数。

## 复现和使用

原件及派生全文被Git忽略，当前路径为`.runtime/datasets/public-aero-corpus-v2-tablefix-final/`。需要复建时，在已安装项目依赖的**全新runtime目录**运行：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/build_diverse_corpus.py --manifest evaluation/public_aero_corpus_v2/source_manifest.json --runtime-root .runtime/datasets/public-aero-corpus-v2-tablefix-final
```

已有冻结快照禁止覆盖。离线可先从现有冻结快照复制raw与receipt，再加`--offline`；官方动态资源若变更，hash检查会失败，应保留旧资产或另建新版本，不要修改标签来强行通过。其它机器的路径可以不同，但本发布源路径是仓库相对路径。源文档原件、canonical与blocks都须与manifest指纹一致。

验证正式标签无需本地作者草稿：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/build_academic_benchmark.py --protocol evaluation/aero_evidence_benchmark_v1/protocol.json --output evaluation/aero_evidence_benchmark_v1 --validate-release
```

本次验证198条正式样本、21个来源及全部证据通过。清洗器@3使用Python3.13.14、pypdf6.18.1和beautifulsoup4 4.15.0；依赖记录见uv.lock。重现旧V2快照应使用提交5e05548的旧构建器，当前代码修正了两份XML的表题/列头；旧/新hash在protocol中保留。

预测文件每行包含`sample_id`、显式bool `abstained`和按排名排列的`chunks`。每个chunk含document_id、canonical_start、canonical_end、canonical_sha256。ID必须严格匹配所选split/variant，缺失、重复、错误源hash或区间会拒绝评分。示意字段如下，需替换成真实偏移和hash：

```json
{"sample_id":"inv-q001","abstained":false,"chunks":[{"document_id":"aaib-aar-1-2010-g-ymmm-final_report_amended","canonical_start":123,"canonical_end":456,"canonical_sha256":"实际canonical SHA-256"}]}
```

```powershell
.venv/Scripts/python.exe scripts/score_evidence_benchmark.py --dataset evaluation/aero_evidence_benchmark_v1 --predictions .runtime/predictions-test-native.jsonl --split test --variant native --output .runtime/scores-test-native.json
```

场景改写/译题分别使用`--variant scenario_paraphrase`/`translation`。图谱预测行是sample_id与label(supported/unsupported)，加`--graph-predictions`；默认只评native声明，压力题用`--graph-variant negation_stress`，描述性合并用`all`。单类别压力层报告准确率，二分类macro F1为空，避免没有supported样本却声称双类稳健。

论文主指标为指定证据完整命中@5/@10、证据覆盖比例、来源命中与拒答precision/recall。检索成绩先按fact平均；拒答报告明确查询级混淆计数。图谱报告限定声明macro F1、准确率和polarity×label混淆。比较策略时还需配对差值与来源依赖聚类区间，不能对逐题IID采样。

固定长度/结构/粗细切分 × BM25/真实dense/hybrid RRF应共用冻结来源、查询、标签和过滤条件，只在dev调参数，reranker另做变量；给相同上下文token预算并记录时延、内存与模型revision。当前评分器检查偏移和标签指纹，**不自动控制实验runner的token预算，也不提供答案语义评分**。

## 质量边界和可支持结论

- 全198条正式记录经跨作者来源重建/核验和裁定，但作者与复核者属于同一模型体系；没有领域专家人工核验。不会给出专家一致率或“无可挑刺”的保证。
- 已核证据不穷尽所有有效替代文段。126题全为all；不能声称测试any替代证据选择，不能把未标文段判为无关或用本集直接宣布完整nDCG/MRR金标。
- 图谱是限定声明支持任务，不是全文穷尽三元组抽取金标，不能由此计算整个图谱的抽取recall。
- 原66声明中的negative全部supported，偏差仍公开；6个成对反例用于检查标签捷径，不能替代自然分布均衡数据。
- 开发和测试检索共用冻结语料；FAA开发概念在CAAC来源中确有复用，目标信息需求不重复，但测试概念不能宣称完全未见。7个测试依赖分量限制统计泛化。
- 部分历史AD/报告不能作现行维护依据；SDR是报送叙述，不证明因果或发生率。8个来源文件标public_domain、13个权利状态unknown，未分发全文，未解除生产文档准入。
- PDF图表、多栏和OCR的完整语义未全面验收；页/字符hash证明定位一致，不能单独证明视觉提取正确。来源并未覆盖所有型号或真实一线记录。

本集可支持“在冻结公开航空资料和定义的信息需求上，对比不同切分/召回策略的指定证据覆盖与限定关系支持”。如需专家金标、完整rank指标或全文抽取recall，应分别增加真实领域人工核验、系统联合候选池判定和完整段落穷尽标注，并发布新版本后对所有策略重跑。

软件验收：406项测试通过，应用src覆盖率87.96%，Ruff及应用89文件/脚本3文件严格mypy通过。评分器对4个有效split/variant的理想证据和空上下文进行8项边界校验，全部符合预期；理想证据读取标签，不能当作检索模型准确率。
