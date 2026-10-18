# 新来源中文独立测试候选集

日期：2026-10-14。状态：候选集已整理，120题答案隐藏 AI 来源复核及作者核对完成，尚未冻结或运行评测。

共120题、28份采用资料、27个文档来源簇。NASA论文保守合并为一个关联组后共25组；120题不能视为120个独立样本。另有5份已保存备用报告，未计入题目来源数量。

| 资料层 | 题目数 | 证据边界 |
| --- | ---: | --- |
| 新事故/事件调查报告 | 80 | 72题网页正文；8题报告页面摘要（单列辅助层） |
| NASA研究论文 | 15 | 3篇官方PDF派生正文，未声称已下载原始PDF字节 |
| 中文期刊论文 | 15 | 3篇发布方中文摘要，单列辅助层，尚不能当作全文题 |
| MIT开放教材/课程讲义 | 10 | 2份课程正文，同一课程家族只算一个来源簇 |

问题和答案均以中文整理。英文原始证据仅用于审计，不作为双语业务展示。资料编号、型号和算法名保留。每题有来源、证据块、字符偏移与SHA256、题型、适用范围、答案和审核状态；不将人工参考答案写入生产知识库。

## 来源清单

| 编号 | 发布方 | 类型 | 官方资料 |
| --- | --- | --- | --- |
| a23p0020 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2023/a23p0020/a23p0020.html) |
| a95o0232 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/1995/a95o0232/a95o0232.html) |
| a03p0247 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2003/a03p0247/a03p0247.html) |
| a03p0054 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2003/a03p0054/a03p0054.html) |
| a00a0176 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2000/a00a0176/a00a0176.html) |
| a13c0150 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2013/a13c0150/a13c0150.html) |
| a08c0108 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2008/a08c0108/a08c0108.html) |
| a01f0020 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2001/a01f0020/a01f0020.html) |
| a02p0168 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2002/a02p0168/a02p0168.html) |
| a14q0068 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2014/a14q0068/a14q0068.html) |
| a03p0136 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2003/a03p0136/a03p0136.html) |
| a05p0038 | TSB | accident_report | [来源](https://www.tsb.gc.ca/eng/rapports-reports/aviation/2005/a05p0038/a05p0038.html) |
| ao-2016-155 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/ao-2016-155) |
| ao-2019-060 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/ao-2019-060) |
| ao-2018-021 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/ao-2018-021) |
| ao-2015-124 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/ao-2015-124) |
| 200200646 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/200200646) |
| 200205780 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/200205780) |
| ao-2022-066 | ATSB | accident_report | [来源](https://www.atsb.gov.au/investigations/ao-2022-066) |
| aaib-n197dn | AAIB | accident_report | [来源](https://www.gov.uk/aaib-reports/aaib-investigation-to-boeing-767-332-er-n197dn) |
| nasa-20160001158 | NASA | paper | [来源](https://ntrs.nasa.gov/api/citations/20160001158/downloads/20160001158.pdf) |
| nasa-20120007104 | NASA | paper | [来源](https://ntrs.nasa.gov/api/citations/20120007104/downloads/20120007104.pdf) |
| nasa-20050175887 | NASA | paper | [来源](https://ntrs.nasa.gov/api/citations/20050175887/downloads/20050175887.pdf) |
| sjtu-kd-deepgnn | SJTU | paper | [来源](https://xuebao.sjtu.edu.cn/CN/10.16183/j.cnki.jsjtu.2024.486) |
| hkxb-diagnosis-review | HKXB | paper | [来源](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2021.26565) |
| tjjs-onboard-review | TJJS | paper | [来源](https://pubs.cstam.org.cn/article/doi/10.13675/j.cnki.tjjs.2405029?viewType=HTML) |
| mit-thermo-ch5 | MIT | teaching | [来源](https://ocw.mit.edu/ans7870/16/16.unified/thermoF03/chapter_5.htm) |
| mit-propulsion8 | MIT | teaching | [来源](https://live.ocw.mit.edu/ans7870/16/16.unified/propulsionS04/UnifiedPropulsion8/UnifiedPropulsion8.htm) |

## 文件与使用

- `blind_queries.jsonl`：无作者答案题单，供独立来源复核使用。
- `generation_queries.jsonl`：仅题目编号、中文问题和语言，回答端只读取问题文字，不提供题型、证据或答案标签。
- `queries.jsonl`：作者参考答案及证据绑定；仅评分端读取。
- `source_manifest.json`：采用/备用来源、获取方式、完整性、指纹与权利状态。
- `validation.json`：152处证据定位检查、题量、分层和指纹。
- `sources/`：本地保存的官网文本、抓取响应、原始HTML和分块定位。未自动上传或再分发。
- `report_annotations.txt` / `literature_annotations.txt`：作者整理底稿。
- `scripts/build_gas_path_holdout.py`：从底稿与既有来源快照构建作者初稿。当前已审核版本禁止原地重建覆盖，后续变更需另存版本。
- `scripts/validate_gas_path_holdout.py`：校验全部证据、指纹及生成题单字段；`--require-frozen`会阻止尚未冻结的题库用于正式运行。
- `reconciliation.json`：120题独立复核及作者核对摘要，原始重建保存在`.runtime/holdout-review-20261014/`。
- `.runtime/holdout-authoring-20261014/`：实际运行的来源采集脚本和过程记录。

## 独立性和已知限制

采用来源与既有国际来源清单的落地URL无重叠，事件编号也不是此前FAA开发题；这只是来源隔离检查，不是模型预训练未见的保证，也不排除相同通用机理。多个问题可能共用一条因果链，按事件/资料家族成组统计，并做NASA研究关联组敏感性分析。

23道cross_passage题要求结合多个来源块；这不自动等于只能靠图谱解决的多跳题，不以参考答案人为补图谱路径。14道错误前提题和7道明确不确定性题保留反例及“可能/很可能/未确定”的限定。只有2道摘要边界的证据不足题，尚不足以评估一般拒答能力，未来可另建固定缺证据子集，不能用偶然未召回替代真缺证据标签。

教材理想循环问答、论文方法问答和事件故障机理问答分层报告，不能把三层混合分数直接当作气路维修诊断准确率。抽取召回率和nDCG仍需要独立的、尽可能完整的相关证据与关系标注，当前题库不提供这些金标准。

独立AI复核120题均认为在其指定保存资料范围内可接受；根代理也核对了全部120对答案。97题使用报告网页正文、NASA派生PDF正文或MIT教材文字，23题仅摘要辅助层。网页正文和派生PDF文本均不能等同于已核对完整原始PDF。该审核只证明来源范围内一致，仍明确记录为非专家审核；论文中的准确率仅代表其作者在特定数据和任务上的结果，不代表本项目检索增强系统成绩。

## 后续冻结与比较约定

1. 已完成120题答案隐藏来源重建及作者逐条核对。保留原作者版本，细化16题必要事实和3题中文术语；没有删题，也没有看系统预测。后续若入库完整性或适用性暴露问题，须在运行前记录处置，不能按答题结果挑题。
2. 确定各层适用性和完整性，摘要层共23题独立报告（8道报告摘要题、15道中文论文摘要题）。所有采用正文、译文、切分和图谱在隔离评测资料库中预构建；知识构建流程仅读取来源，不能读取作者题目、答案或关系金标准。
3. 检查指定证据及等价证据的入库情况。入库/翻译/抽取失败事先记录，不按答题成绩临时排除；同时报告整体管线覆盖率和覆盖题成绩。
4. 冻结题目、答案、来源及知识快照、模型/提示词/生成参数、检索配置和代码指纹。沿用开发确定的排序实现，不根据本测试反馈调参。若调参，必须记为开发并另设未来测试。
5. 仅文本与文本加图谱使用同一批题、同一知识快照、同一回答模型与提示词、同一top-5和2048证据token预算；关闭案例路由，区别仅图谱开关。生成端只读取generation_queries的query字段，评分端读取参考答案。
6. 比较严格事实正确、方向/限定正确、引用有效、错误前提和拒答表现，列出逐题胜/负/平及来源分层成绩；以来源簇重采样计算区间，并按更保守关联组复核。小差异不直接宣称显著。
7. 同时报告耗时和成本。正式双组生成需要另作具体费用估算；原5元共享试跑账本当前已到约4.47元，本次来源收集和人工/子代理整理没有新增DeepSeek调用，也没有重置账本。

当前没有将本候选集导入8082生产应用、运行双组生成或改动当前图谱排序。先交付资料池和题库，再按已冻结条件比较。
