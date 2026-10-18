# 开源复用选型与实施调整

日期：2026-10-13。依据用户在执行期间提出的建议，后续通用文档解析、抽取编排、原文对齐与图谱检索优先复用成熟上游。自定义代码集中在气路本体、型号/阶段/修订条件、事件部件角色、否定与不确定性、证据契约和科研评测。

## 已读代码与候选用途

| 项目 | 实际核查的代码/接口 | 本项目用途 | 状态与边界 |
| --- | --- | --- | --- |
| [Docling](https://github.com/docling-project/docling) | DocumentConverter、DoclingDocument；pyproject、LICENSE | PDF布局、表格、标题层级与页面位置的解析基础 | 优先验证；新解析产生新文本快照，旧引用不得直接搬用 |
| [LangExtract](https://github.com/google/langextract) | Resolver.align、Extraction.char_interval、OpenAI provider；pyproject、LICENSE | 来源驱动抽取、原文位置与审核可视化 | 已完成真实库离线定位验证；属性值、角色及关系语义仍须核验 |
| [LightRAG](https://github.com/HKUDS/LightRAG) | ProgramingWithCore、operate.py；pyproject、LICENSE | 图谱与文本联合检索、来源链、抽取后合并前质量入口 | 优先作为可替换检索后端候选；先确认限定词、版本和事件角色能保留 |
| [Docling-Graph](https://github.com/docling-project/docling-graph) | PipelineConfig、run_pipeline、pipeline/orchestrator；pyproject、LICENSE | Pydantic模板驱动的文档到图谱流程 | 作为抽取编排候选，与LangExtract方案择一或分层，避免重复调用模型 |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | LocalSearchMixedContext及文本单元/关系上下文预算；LICENSE | 图谱局部检索研究对照 | 优先复用局部检索设计，社区级全局汇总须有明确实验需求再引入 |

Docling、Docling-Graph、LightRAG和GraphRAG所查代码许可证为MIT；LangExtract为Apache-2.0。具体复用保留版权与许可证记录。代码许可证不替代模型权重、公开语料各自的使用依据。

上述选择是依据现有接口与项目约束的工程判断，尚未证明任何上游框架在本气路数据上效果更好。详细文件与许可证原文保存在本地source-snapshots.json；18个文件均重新计算并匹配Git blob摘要。

## 固定上游提交

| 仓库 | 本次源码核查提交 |
| --- | --- |
| docling-project/docling | e017a6ff95f09869e3af2c1f8c7774c548d3a084 |
| docling-project/docling-graph | bbcd0e4846fadfcadac2800784783e59cd2722a9 |
| google/langextract | f6dc772b738a44cd6f77d0858d6ce3afa16acd15 |
| HKUDS/LightRAG | 4d0afc1bc36e6a34d4a5895b68efe9413748627a |
| microsoft/graphrag | 5faaaf4f5685fa2056fa8c3bf9342cb089f2942f |

来源登记：[upstream-source-registry.json](upstream-source-registry.json)。公共GitHub REST匿名配额用尽后，改用已连接GitHub只读工具取得固定提交文件；未把网页摘要当作源码核查。

## 已实际验证的复用点

在独立虚拟环境安装PyPI langextract==1.7.1，直接使用Resolver.align、io.save_annotated_documents和visualize重放首次付费试跑的G2输出。没有新请求模型，也没有改变候选关系文本。

4个来源单元、10条候选引文，在禁用模糊及部分匹配后，10条均取得字符区间并与原文逐字符一致。该结果只验证原文对齐接口，不是关系精确率、召回率或图谱正确率。此前自写G3校验只有4条候选通过所有必要检查，其余因不确定性证据、实体重复或条件字段失败；二者检查目标不同，不能把10/10与4/10当作质量提升比较。

Windows默认GBK控制台在上游保存提示处出现UnicodeEncodeError；首次产物保留。使用python -X utf8再次运行成功，并生成原文高亮审核HTML。PyPI安装版resolver与本次GitHub主分支文件摘要不同；已保存差异、安装源码和58个实际依赖版本，生产接入锁定实际验证的包/提交。

产物位于.runtime/upstream-review-20261013/langextract-grounding-probe-02/，其中summary.json记录每条引文区间，grounding-review.html为上游生成的交互审核页。复现脚本：[probe_langextract_alignment.py](../../../scripts/probe_langextract_alignment.py)。

## 后续实现的具体落点

1. 解析层：先以Docling SDK处理少量实际PDF难页，比较表头/单位/否定和页面定位，再增加Docling适配器。保留Docling JSON、页面bbox、原文件摘要；规范化文本使用新version_id和反向映射。已冻结v1与开发pilot不覆盖。
2. 抽取层：优先试LangExtract的examples/output_schema与真实原文对齐；Docling-Graph的Pydantic模板作为另一候选。少量航空本体模板和边界检查属于本项目适配；分块、模型并发、重试与对齐复用上游。调用先接统一费用账本，不能让SDK内部重试绕过预算。
3. 图谱层：验证LightRAG合并前kg_extraction_validator钩子，过滤无定位、条件缺失或角色混淆记录。明确实体概念与事件中的物理部件；多个来源描述合并后仍保留断言级证据和修订，不把聚合描述当原文。
4. 检索层：以LightRAG本地/混合候选和GraphRAG局部上下文作为框架对照，同一语料、模型、最终token预算与冻结分组计分。图谱路径只帮助找证据，不能直接代替因果支持。
5. 应用层：沿用现有FastAPI与前端，统一EvidenceBundle、澄清状态和报告陈述对象。上游通过薄适配器接入，不复制整套上游服务来替换已有应用。

原自写检索及抽取代码保留为已登记的简单基线和回归参照。接下来停止扩写通用抽取核心；先验证上游组件能满足来源、限定词与预算契约，再替换生产入口。当前仅完成选型、源码核查和LangExtract定位探针，尚未宣称生产系统已切换。

## 阶段与论文章节口径

已完成的25题开发pilot、12组离线检索、17次小样本API调用和补充5题来源复核继续保留。采用上游组件属于实现复用，论文引用相应项目与方法。候选学术贡献仍是气路条件与角色控制、证据组织及陈述核验的可测收益，不将成熟框架改名声称原创。

正式S2/S5门槛仍需新增来源、依赖划分、等价证据池及完整抽取标注；正式端到端、消融、压力与费用统计待这些门槛通过。当前开发结果不改称正式测试结果。
