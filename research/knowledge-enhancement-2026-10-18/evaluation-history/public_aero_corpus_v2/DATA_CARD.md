# Public Aero Corpus V2 — 首批公开资料扩展数据卡

日期：2026-10-13。数据集 `public-aero-diverse-20261013`。本轮按公开资料的覆盖深度选择发动机家族，用户没有指定型号，也没有提供私有手册或维修记录。

## 实际取得与处理结果

- 原件 **31个**，合计 **144,539,049 bytes（137.84 MiB）**：其中30个本轮新增，1个为已有完整FAA 2024 SDR CSV。
- 实际取得 **17份逻辑文档、13个文档家族、6家来源机构**：FAA、CAAC、AAIB、Airbus、BEA、EASA。网页/PDF、译本、勘误和手册章节不会全部视为独立知识来源。
- 原始格式：18 PDF、8 HTML、2 XML、2 JSON、1 CSV；语言涵盖英文、中文、法文。JSON是官方目录元数据，不计作完整技术正文。正文包含PDF/HTML/XML/CSV四类实际格式，没有通过格式转换凑数。
- 31个原件均形成派生文本和blocks，共4,508个来源定位块。18个PDF共1,238页，实际生成1,232个非空页块；页数是格式规模，不是独立样本数。
- **11份逻辑文档（8个家族）**进入自动处理门槛通过的候选快照；**6份已取得文档**只保留本地隔离派生文本，等待使用条件核查。候选不等于已验证知识或gold标签。
- 计划清单还含4份未取得文档：ATSB两项超时，TSB两项403，共8个附件缺失；不计入已取得覆盖，也未绕过访问限制。

## 来源与知识维度

| 来源 | 已取得逻辑文档 | 内容与边界 |
|---|---:|---|
| FAA | 8 | 动力装置教材2/5/6/10章；涡轮发动机振动AC；两份CFM/LEAP正式AD；2024 SDR。通用训练、审定、强制要求和未核实故障报告分别保留身份。 |
| CAAC | 3 | CCAR-33-R2、历史R1、2022发动机审定AC。历史版不作现行依据，中文法规与审定原文已提取。 |
| AAIB | 3 | G-YMMM燃油结冰/推力下降、G-EUOE发动机整流罩/燃油管/火灾、G-POWN维护用药过量与喘振。全文、勘误、公告、网页摘要分别存储。 |
| Airbus | 1 | 起飞推力设置、尾风/侧风和稳定过程。属于公开安全文章，不冒充型号批准手册。 |
| BEA | 1 | GP7270风扇轮毂非包容性失效，法文参考版和英文courtesy translation同一家族。 |
| EASA | 1 | Trent 900历史紧急AD，以及取代/取消关系的页面元数据。明确保留superseded与historical-only限制。 |

具体下载地址、落地页、实际原件SHA256/字节数、失败和重试结果见 [source_manifest.json](source_manifest.json)。[sources-international.json](sources-international.json) 是调研目录；直接构建时使用主manifest。

## 清洗规则与实际发现

原件字节保持不变；正文采用UTF-8、NFC和LF。保持数字、单位、否定和条件语句，不进行NFKC、猜测性断词修复、模型润色或虚构OCR。所有文本区间绑定原件和派生文本SHA256。

HTML按经过实际DOM核对的正文选择器提取，缺失或多匹配会隔离；去掉导航、脚本、表单等噪声，保留标题及表格行/单元格。XML使用JATS/FR标签白名单和原节点索引。PDF保留原页号，记录页文本量；文本页比例门槛80%仅检查可提取性，不能替代内容质量审核。PDF复杂表格、图形、公式和多栏阅读顺序仍待专项核查；两轮pypdf出现相同的一条字体十六进制编码告警，尚未做语义修复。

Federal Register的`.txt`地址实际返回HTML的`pre`包装、HTTP头为`text/plain`，已显式按HTML处理，不误计为纯TXT。实际核对后修正了CAAC/AAIB/EASA的正文选择器，也将热启动/悬挂启动主题从没有对应文字的第5章移到第10章。

SDR采用字段白名单，排除注册号、提交人/运营人标识和机体/发动机序列号。根据 [FAA JASC编码说明](https://www.faa.gov/sites/faa.gov/files/JASC_Code.pdf) 筛选70–79及80章，而非匹配`EngineModel`字段名的关键词。66,080条输入中：2,056条选中，2,046条保留，10条因可能包含身份信息而隔离，64,024条未选中，分区计数互斥且完整。自由叙述仍需审核，筛选不覆盖所有编码在其它章节的发动机相关事件。

保留记录中767条缺少EngineModel；没有补造型号或根据原始飞行阶段代码猜测阶段。型号字符串仅形成未审核家族提示：CFM56 269、Trent 114、LEAP 51、V2500 74、PW1100G 131、CF6 113。它们是事件行的覆盖提示，不能据此宣布资料适用性或知识充分性。优先从CFM56/LEAP、Trent相关公开证据做深入整理，同时保留其它家族作对照。

去重保留来源和定位。同一报告的多格式、译本与勘误共享家族；近重复仅输出候选信号，不直接删技术正文。按家族划分后续开发/测试集，SDR还需事件级重复检查。大量SDR行不能压过少量高价值手册/调查证据，后续评测需按任务/来源分层采样。

## 许可、候选状态与版本

许可按artifact判断：AAIB网页OGL摘要不能将PDF全文自动放行；全文/摘要的完整度与修订字段分层保留；BEA参考版与译本身份分别保留。未知使用条件的资料已实际下载和清洗，供本地审核，排除于自动eligible候选集合；原件与派生正文不提交Git。本数据集没有自动激活应用知识库，也没有生成经人工确认的检索/三元组gold。

V2扩展与V1原实验分开保存，现有论文、NTSB等V1资料仍保留。当前最终本机快照路径为`.runtime/datasets/public-aero-corpus-v2-verified/snapshot.json`，其冻结manifest为`manifest.frozen.json`；首轮快照在`.runtime/datasets/public-aero-corpus-v2/`。每个artifact有raw、receipt、processed文本与blocks。仅元数据、审计和复核队列入Git。

## 复现与验收

两轮使用同一批31原件独立构建；逐文件SHA256、全部4,508个文本区间/文本hash、原件hash和NFC/LF检查通过，31个派生文本的hash全部相同。第二轮修复描述层元数据，保留摘要、历史版、原版/译本身份；没有改写正文。完整测试365通过，覆盖率87.96%；新增构建器29项测试，Ruff和mypy通过。[清洗审计](cleaning_audit.json)记录快照/构建器hash及逐原件结果。[复核队列](curation_review_queue.jsonl)记录17份资料的后续检查项。

在已安装项目依赖的环境，用新的runtime目录复制冻结原件后复现；已有快照禁止覆盖：

```powershell
New-Item -ItemType Directory -Force .runtime/datasets/public-aero-corpus-v2-replay/raw
Copy-Item .runtime/datasets/public-aero-corpus-v2-verified/raw/* .runtime/datasets/public-aero-corpus-v2-replay/raw/
.venv/Scripts/python.exe -X utf8 scripts/build_diverse_corpus.py --manifest evaluation/public_aero_corpus_v2/source_manifest.json --runtime-root .runtime/datasets/public-aero-corpus-v2-replay --offline
```

直接联网可省略`--offline`，但动态网页/元数据可能已改变：hash不同会失败，必须建立新的版本并重审；不能静默更新旧实验。下载上限40MiB/文件，单请求60秒；本轮对个别连接失败使用Windows系统证书验证的curl重试，没有关闭TLS校验。

下一阶段先抽检型号、修订、表格和来源定位，再按“症状解释、检查证据、工况/型号区别、跨段因果、冲突和无答案”整理证据区间级评测题；为不同来源/事件保留分层统计，然后使用固定语料、固定查询及家族隔离的划分比较切分和检索策略。现阶段没有新检索指标，也不预设混合检索一定获胜。


## 后续学术评测版本的表头修复

以上V2审计对应提交`5e05548`及当时的构建器版本；如要重现旧快照字节，应使用该提交。后续数值题复核发现FR XML表格的TTITLE/BOXHD/CHED未进入canonical正文，已在构建器@3保留表题和表头，旧快照保持不动。学术评测使用新的`.runtime/datasets/public-aero-corpus-v2-tablefix-final/`，仅两份XML派生正文改变，原件字节和其它19份研究源canonical不变。当前脚本重建的是修正版，不能声称复现旧快照。详见[学术评测数据卡](../aero_evidence_benchmark_v1/DATA_CARD.md)及其protocol中的旧/新hash。
