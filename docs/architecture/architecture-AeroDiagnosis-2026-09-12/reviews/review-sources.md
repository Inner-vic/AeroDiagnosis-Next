# Technology & Source Review

- **审查日期：** 2026-09-12
- **复核日期：** 2026-09-12（主方案与 spine 修订后）
- **审查对象：** `ARCHITECTURE-SPINE.md`、`docs/upgrade/UPGRADE-PLAN.md`
- **审查范围：** 技术版本真实性与适配性；Self-RAG、CRAG、Adaptive-RAG、GraphRAG、Ragas、OpenTelemetry、GitHub Actions 的来源与表述；工程组合与学术贡献的边界
- **来源约束：** 仅使用项目官方文档、PyPI 和原始论文/正式论文页面
- **机械检查：** `lint_spine.py` 通过，0 项机械问题

## Gate verdict

**通过。复核后剩余 Critical：0，High：0。** 修订稿已补齐 MCP SDK、SQLite checkpointer、OTel exporter/instrumentation 的技术绑定，改用处于 bugfix 维护期的 Python 3.13，明确 LangGraph 每个节点边界显式 Pydantic 校验，并将近期方法改称“知识图谱增强 RAG”。Self-RAG、CRAG、Adaptive-RAG 与 Microsoft GraphRAG 均被明确标记为相关工作或思想借鉴，而非当前实现或原创算法。当前仍有 4 个 Medium 实施约束，均不阻止架构进入实现。

### Re-review closure matrix

| 原发现 | 状态 | 复核证据 |
| --- | --- | --- |
| HIGH-1 Stack 未闭合 | **CLOSED** | Stack 已加入 MCP Python SDK 2.2.0、LangGraph SQLite Checkpoint 3.1.1、OTel OTLP HTTP Exporter 1.44.0 与 FastAPI Instrumentation 0.65b0；版本均真实且是审查日对应发布线最新版本。 |
| HIGH-2 Pydantic state 自动验证误解 | **CLOSED** | AD-2 改为 TypedDict 传输，并强制每个节点入口/出口用版本化 Pydantic 模型显式校验；Upgrade Plan Phase 1 同步落地。 |
| HIGH-3 GraphRAG 方法混同 | **CLOSED** | 项目主题改名为 `Evidence-Grounded Adaptive Diagnostic Agent`；方法明确称“知识图谱增强 RAG”，且声明当前不是 Microsoft GraphRAG 实现。 |
| MEDIUM-1 三种论文方法来源边界 | **CLOSED** | Section 12 对 Self-RAG、CRAG、Adaptive-RAG 均写明“仅借鉴思想、不声称复现”；总则区分借鉴、复现、改进和原创。 |
| MEDIUM-2 Python 3.12 生命周期 | **CLOSED** | Stack 改为 Python 3.13.x，3.12 仅作迁移期兼容测试。 |
| MEDIUM-3 OTel / GitHub Actions 运行约束 | **PARTIAL / OPEN MEDIUM** | OTel 依赖闭包已补齐且 beta 风险已显式隔离；CI action SHA、最小 token 权限、自定义 telemetry 属性规范仍待实现阶段绑定。 |

新增版本核验：[MCP 2.2.0](https://pypi.org/project/mcp/2.2.0/)、[LangGraph SQLite Checkpoint 3.1.1](https://pypi.org/project/langgraph-checkpoint-sqlite/)、[OTLP HTTP Exporter 1.44.0](https://pypi.org/project/opentelemetry-exporter-otlp-proto-http/)、[OpenTelemetry FastAPI Instrumentation 0.65b0](https://pypi.org/project/opentelemetry-instrumentation-fastapi/)。Checkpoint 包的 PyPI 官方说明还要求启用 strict msgpack 或显式 allowed module 列表，故保留为实现阶段 Medium 安全项。

## Version verification

| Technology | 文档绑定 | 截至 2026-09-12 的核验 | 结论 |
| --- | ---: | --- | --- |
| Python | 3.13.x | 3.13 处于 bugfix 维护期，仍有官方二进制发布 | 通过 |
| LangGraph | 1.2.11 | PyPI 最新稳定版 1.2.11，2026-08-11 发布 | 通过 |
| LangGraph SQLite Checkpoint | 3.1.1 | PyPI 最新稳定版 3.1.1，2026-07-30 发布 | 通过；实现时启用 strict msgpack/allowed modules |
| FastAPI | 0.141.1 | PyPI 最新稳定版 0.141.1，2026-07-29 发布 | 通过 |
| Pydantic | 2.13.5 | PyPI 最新稳定版 2.13.5，2026-08-28 发布；2.14.0b2 为预发布版 | 通过 |
| MCP Python SDK | 2.2.0 | PyPI 当前稳定 2.x 线最新版本，支持 2026-07-28 及更早协议版本 | 通过；具体传输与契约测试留作 Medium 实施项 |
| Neo4j Community LTS | 5.26.30 | 5.26 是 LTS；5.26.30 是审查日最新 LTS 补丁。Neo4j 当前功能版另为 2026.08.1 | 标签与版本均准确 |
| ChromaDB | 1.5.9 | PyPI 最新稳定版 1.5.9，2026-05-05 发布 | 通过 |
| Ragas | 0.4.3 | PyPI 最新稳定版 0.4.3，2026-01-13 发布 | 通过 |
| OpenTelemetry SDK | 1.44.0 | PyPI 最新稳定版 1.44.0，2026-07-16 发布 | 通过 |
| OpenTelemetry OTLP HTTP Exporter | 1.44.0 | 与 SDK 同日、同稳定版本线发布 | 通过 |
| OpenTelemetry FastAPI Instrumentation | 0.65b0 | PyPI 最新版本；官方分类仍为 Beta | 通过；spine 已要求保持为可替换适配器 |

核验依据：[Python 版本状态](https://devguide.python.org/versions/)、[Python 3.12.14](https://www.python.org/downloads/release/python-31214/)、[LangGraph PyPI](https://pypi.org/project/langgraph/)、[FastAPI PyPI](https://pypi.org/project/fastapi/)、[Pydantic PyPI](https://pypi.org/project/pydantic/)、[Neo4j 版本与发布日期](https://neo4j.com/current-neo4j-versions/)、[Neo4j 文档归档（5.26 LTS）](https://neo4j.com/docs/reference/docs-archive/)、[ChromaDB PyPI](https://pypi.org/project/chromadb/)、[Ragas PyPI](https://pypi.org/project/ragas/)、[OpenTelemetry SDK PyPI](https://pypi.org/project/opentelemetry-sdk/)。

## Findings

### HIGH-1 — [CLOSED] Stack 不是实现承诺的闭包：MCP、checkpoint 与 telemetry 的关键包未绑定

**复核状态：已关闭。** 修订后的 Stack 已绑定 `mcp==2.2.0`、`langgraph-checkpoint-sqlite==3.1.1`、`opentelemetry-exporter-otlp-proto-http==1.44.0` 和 `opentelemetry-instrumentation-fastapi==0.65b0`。MCP 传输选择和协议协商契约测试降为实现阶段 Medium 项，不再构成依赖闭包 High。

**Initial-review evidence.** 初稿将 `v3-mcp`、可持久化 checkpoint 和 OpenTelemetry 设为正式能力，但当时的 Stack 没有 MCP Python SDK，也没有 SQLite checkpointer、FastAPI instrumentation 或 exporter。官方 MCP Python SDK 当前稳定线已是 2.x，最新为 2.2.0；官方明确说明 2.x 是重大重构，`pip install mcp` 会安装 2.x，而仍停留在 v1 的项目必须显式设 `<2` 上界。[MCP 2.2.0 PyPI](https://pypi.org/project/mcp/2.2.0/)。LangGraph 官方说明 SQLite checkpointer 是单独安装的 `langgraph-checkpoint-sqlite`，并不由基础 checkpointer 自动提供。[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。OpenTelemetry 官方也把 SDK、框架 instrumentation 和 OTLP exporter 分成独立包；仅有 `opentelemetry-sdk` 不会自动形成 FastAPI 到可查看后端的链路。[OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)、[instrumentation libraries](https://opentelemetry.io/docs/languages/python/libraries/)、[exporters](https://opentelemetry.io/docs/languages/python/exporters/)。

**Impact.** 两个团队按当前 spine 独立实现时，可能分别选择 MCP v1/v2、不同 protocol revision、不同 checkpoint 后端和不同 exporter，直接产生不兼容；而“可恢复执行”“MCP 契约测试”“运行时 telemetry”可能在装完表内依赖后仍无法运行。

**Disposition: discuss then autofix.** 在 Phase 0 兼容性试验后，至少绑定：MCP protocol revision、`mcp` SDK major/minor、`langgraph-checkpoint-sqlite`、OTel 的 API/SDK/instrumentation/exporter 组合及其兼容组。MCP 还应明确 stdio 与 Streamable HTTP 哪个用于本地演示、哪个用于远端部署。当前规范将 tools 定义为 server 暴露、client 可发现和调用的带 JSON Schema 能力，因此“把 MCP 作为入站适配器”是符合协议角色的架构选择。[MCP server overview](https://modelcontextprotocol.io/specification/2025-11-25/server/index)、[MCP tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)。

### HIGH-2 — [CLOSED] `Pydantic StateGraph state` 不能提供文档可能暗示的逐节点运行时验证

**复核状态：已关闭。** AD-2 现在明确使用 TypedDict 传输，并在每个节点入口和出口显式校验完整 Pydantic 状态；Upgrade Plan Phase 1 与之相符。

**Evidence.** AD-2 要求状态是 Pydantic 模型，Upgrade Plan 又把 `DiagnosisState` 等 Pydantic 模型作为 Agent 基础。LangGraph 1.2.11 确实支持 Pydantic state schema，但官方列出的限制是：运行时验证只发生在第一个节点的输入；后续节点更新和最终输出不会自动验证，图输出也不是 Pydantic 实例。[LangGraph Graph API — Pydantic state limitations](https://docs.langchain.com/oss/python/langgraph/use-graph-api)。StateGraph 的节点契约本质是 `State -> Partial<State>`。[StateGraph reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)。

**Impact.** 若实现者把 AD-2 理解为“LangGraph 会在每一步保证 state 合法”，非法 `evidence_id`、预算或验证状态仍可从中间节点进入 checkpoint，削弱 AD-5/AD-6 的 fail-closed 目标。

**Disposition: autofix.** 保留 Pydantic 领域对象和边界契约，但在 Rule/Phase 1 验收中明确：节点输出需要显式 `model_validate`、受控构造器或 contract test；或者使用 TypedDict/dataclass 作为图状态容器，在每个领域 payload 边界做 Pydantic 验证。不要把 `StateGraph(PydanticModel)` 本身写成逐节点验证保证。

### HIGH-3 — [CLOSED] “GraphRAG” 名称与当前计划实现的图遍历不是同一个可引用方法

**复核状态：已关闭。** 项目主题已去掉 `GraphRAG`，Section 4.3 使用“知识图谱增强 RAG”并明确当前不是 Microsoft GraphRAG 实现；相关技术表中保留 GraphRAG 作为相关工作不构成实现声明。

**Evidence.** Microsoft 原始 GraphRAG 工作的核心问题是面向整个语料的 global sensemaking / query-focused summarization；它先从源文档构建实体知识图谱，再生成实体社区摘要，并对社区摘要的部分回答做最终汇总。[原始 GraphRAG 论文](https://arxiv.org/abs/2404.16130)。当前方案明确把“全量 Microsoft GraphRAG 社区摘要”放在 Deferred，实际近期工作是 Neo4j 上的受控故障传播路径遍历与来源证据保留。

**Impact.** 项目名 `Evidence-Grounded Adaptive Agentic GraphRAG` 加上“Microsoft GraphRAG：用图结构增强语料理解和检索”的宽泛概括，容易让论文评阅人或面试官误以为已经实现/复现 Microsoft GraphRAG；实际上，任意知识图谱检索并不等同于该论文方法。

**Disposition: discuss.** 若近期不做社区检测、社区摘要和 global-query 实验，应把方法称为“knowledge-graph-enhanced RAG”或“graph-assisted diagnostic RAG”，把 Microsoft GraphRAG 只列为相关工作/未来工作。若保留 `GraphRAG` 品牌名，则必须先定义这里使用的是广义图增强 RAG，并显式声明“不是 Microsoft GraphRAG 实现”。只有真正加入对应索引与 global/local 对照实验后，才使用后者作为方法标签。

### MEDIUM-1 — [CLOSED] Self-RAG、CRAG、Adaptive-RAG 只能标注为研究启发，不能当作已实现基线或模块名

**复核状态：已关闭。** Section 12 已对三者逐项标注“仅借鉴思想”，并增加论文必须区分“借鉴、复现、改进、原创”的总约束。

**Evidence.** 三篇原始工作与本计划的机制并不等价：

- Self-RAG 训练一个语言模型使用特殊 reflection tokens，按需检索并反思检索段落和自身生成；外置 Verifier + Planner 状态机不是 Self-RAG 的实现。[Self-RAG，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)。
- CRAG 用轻量 retrieval evaluator 评估检索质量，按置信度触发不同知识检索动作，并包含 web search 与 decompose-then-recompose；本计划中的领域内证据缺口纠错与它有关联，但并未覆盖完整 CRAG。[CRAG 原始论文](https://arxiv.org/abs/2401.15884)。
- Adaptive-RAG 用训练出的较小 LM 分类器预测问题复杂度，在 no-retrieval、single-step 与 iterative retrieval 间选择；当前“按问题类型、复杂度和证据缺口选择工具”的策略需要自定义策略、训练或评测，不能仅凭条件路由称为复现 Adaptive-RAG。[Adaptive-RAG，NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)。

**Assessment.** Upgrade Plan 已写明工程与研究主题“不应预先宣称原创算法”，这是正确且重要的保护语句；Section 12 的概述也没有直接声称实现三者。但论文材料仍应建立逐项 provenance 表：`paper mechanism → adopted idea → AeroDiagnosis implementation difference → experiment`，并始终使用“inspired by / 借鉴”，除非做了忠实复现。

**Disposition: autofix.** 在相关工作和方法章节加入上述边界；消融组用项目自身机制名（如 `evidence-gap replanning`），不要直接标成 Self-RAG/CRAG/Adaptive-RAG。复现实验若使用论文名，应单独说明偏离及其原因。

### MEDIUM-2 — [CLOSED] Python 3.12 对 2026 年新建基线是可用但偏旧的选择

**复核状态：已关闭。** v3 新基线改为 Python 3.13.x；3.12 只保留迁移期测试。

**Evidence.** Python 3.12 仍接受安全修复直至 2028-10，但已处于 security-only，3.12.10 后不再提供官方二进制安装器；3.12.14 是 source-only 安全版本。Python 3.13/3.14 仍处于 bugfix 维护期。[Python 版本状态](https://devguide.python.org/versions/)、[Python 3.12.14 release](https://www.python.org/downloads/release/python-31214/)。表内主要 Python 包均声明支持 3.12，因此这不是当前兼容性错误。

**Impact.** “新环境一条命令启动”特别是在 Windows 本地开发场景中，若把 `3.12.x` 解析为最新 patch，可能落到无官方安装器的版本；同时评阅人会追问为什么 2026 年新项目锁定 security-only 分支。

**Disposition: discuss.** Phase 0 兼容性矩阵优先测试 3.13 或 3.14；若因模型/向量依赖保留 3.12，则明确 pin 到镜像 digest 或可获得的解释器构建，并记录兼容性理由，而不是笼统写 `3.12.x`。

### MEDIUM-3 — [PARTIAL / OPEN] OpenTelemetry 与 GitHub Actions 的总体用途准确，但复现和供应链规则仍未完全绑定

**复核状态：部分关闭。** OTel SDK、OTLP HTTP exporter 和 FastAPI instrumentation 已形成最小依赖闭包，beta 状态也已显式承认。仍需在实现时固定自定义 attribute schema/基数/脱敏策略，并在 GitHub Actions workflow 中声明最小 `GITHUB_TOKEN` permissions、把第三方 actions 固定到完整 commit SHA。

**Evidence.** OpenTelemetry 确实用于 traces、metrics、logs 等 signals；异常应在未处理并导致 span error 时记录为 span event，而不是把所有业务失败都无差别记录为 exception。[OTel signals](https://opentelemetry.io/docs/concepts/signals/)、[OTel exception convention](https://opentelemetry.io/docs/specs/otel/trace/exceptions/)。因此 AD-10 的 trace ID、耗时、错误与降级事件方向合理，但 token、evidence ID 等属于项目自定义属性，需要稳定命名、基数和脱敏策略。GitHub Actions 官方支持构建/测试 Python 项目，`setup-python` 是推荐方式。[GitHub Actions Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)。官方安全指南同时要求最小化 `GITHUB_TOKEN` permissions，并推荐把 action 固定到完整 commit SHA，因为 tag 可移动。[GitHub Actions secure use](https://docs.github.com/en/actions/reference/security/secure-use)、[workflow permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)。

**Disposition: autofix.** CI 验收规则增加 `permissions: contents: read`（或任务所需最小权限）、第三方 action 的完整 SHA pin 与 Dependabot/Renovate 更新策略；telemetry 规范增加自定义属性命名、低基数约束、内容脱敏和 exporter/collector 的最小可运行拓扑。

## Source-fit checks that pass

1. **LangGraph fit passes.** 官方将 LangGraph 定位为长运行、状态化 workflow/agent 的底层编排运行时，明确提供 durable execution、persistence、streaming 与 human-in-the-loop；使用 `StateGraph`、条件边和 checkpointer 承载诊断工作流符合其用途。[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)、[persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。
2. **MCP boundary passes.** MCP server 负责向 client 暴露可发现、可调用、带 schema 的 tools；把 MCP 作为外部入站适配器，而不是内部模块总线，是合理架构决策。协议不强制内部应用必须通过 MCP 调用自己。[MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)。
3. **Ragas use passes.** Ragas 0.4.3 当前提供 context precision/recall 与 faithfulness 等指标；faithfulness 衡量 response 对 retrieved context 的支持程度，而不是领域诊断的真实正确性。方案明确把 Ragas 设为辅助评估、保留人工复核和非 LLM 指标，表述准确。[Ragas faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)、[context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)、[context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)、[RAGAs 原始 EACL 论文](https://aclanthology.org/2024.eacl-demo.16/)。
4. **工程组合没有被直接宣称为学术原创。** Upgrade Plan 已明确要求在相关工作对比、消融实验和统计检验前不宣称原创算法，并禁止虚构指标；这一点符合学术表达底线。后续必须保持该限定，不应把 LangGraph + hybrid retrieval + Neo4j + verifier + MCP 的集成本身包装成算法创新。

## Required changes before thesis citation or implementation freeze

无 Critical/High 阻塞项。以下为实现冻结前的 Medium 跟踪项：

1. MCP 契约测试固定所支持的 protocol revision，并分别验证本地 stdio 与部署用 Streamable HTTP（若两者都支持）；不要依赖 SSE 作为新默认。
2. checkpoint 初始化启用 `LANGGRAPH_STRICT_MSGPACK=true` 或显式 `allowed_msgpack_modules`，并加入恶意/未知类型反序列化测试。
3. OTel 固定项目自定义 attribute schema、低基数边界、内容脱敏策略和 collector/exporter smoke test。
4. GitHub Actions workflow 显式设置最小 `GITHUB_TOKEN` permissions，第三方 actions 固定到完整 commit SHA，并配置自动依赖更新。
