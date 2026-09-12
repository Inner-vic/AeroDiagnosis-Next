# Reviewer Gate — Rubric Review

**评审对象：** `ARCHITECTURE-SPINE.md`<br>
**对照输入：** `docs/upgrade/UPGRADE-PLAN.md`、仓库 `b318baa` 代码事实<br>
**评审日期：** 2026-09-12
**评审意图：** Validate；本评审不修改架构主干

## 最新复核结论

**PASS WITH MEDIUM FOLLOW-UPS（0 Critical，0 High，6 Medium，2 Low）。**

最新主干已实质关闭初审的四个 High：AD-8 固定了 SQLite 发布权威和跨索引可见性；AD-14 固定了 worker 租约与有限重试；AD-15 固定了 branch-by-abstraction 迁移和旧实现退役；AD-11 固定了公开匿名只读与本地 loopback operator token。AD-3、AD-9、AD-13 也分别补足了契约 envelope、可判定的评测门禁和 checkpoint 恢复版本边界。机械 lint 再次通过，新增 Stack 版本也已核对存在且当前。

最后一个 High 已关闭：AD-13 现在由规划节点生成跨重试稳定的 `call_id`，`operation_id=sha256(run_id + call_id + tool_name + canonical_args)` 明确排除 `attempt`；同时补充了调用前 pending、恢复对账，以及只允许具备原生幂等键或确定性 upsert/read-before-write 的副作用适配器注册。Rule 现在能够实际阻止其 Prevents 所述的恢复重复写入。剩余问题均为 Medium/Low，不阻止该 spine 作为下一阶段构建约束。

### 初审项关闭状态

| 初审项 | 状态 | 最新依据 / 残余 |
| --- | --- | --- |
| H1 跨存储权威与原子发布 | **Closed** | AD-8 固定 SQLite 为 active/visibility 唯一权威、条件发布、active 过滤、崩溃恢复与孤儿清理。构建指纹仍是 Medium 补强项。 |
| H2 ingestion worker 协调 | **Closed** | AD-14 固定 SQLite durable job、条件租约、heartbeat、接管、有限退避和 dead-letter。 |
| H3 brownfield 迁移边界 | **Closed** | AD-15 固定入口特征测试、单入口 flag、逐模块永久切换和最终删除旧目录/flag。 |
| H4 安全模式二选一 | **Closed** | AD-11 固定公开托管匿名只读；本地写入仅 loopback + operator token；公网写入须新增 OIDC/RBAC 决策。 |
| M1 内外契约耦合 | **Closed** | AD-3 明确边界只做映射，定义统一 envelope 与 major schema version。 |
| M2 checkpoint/恢复与预算 | **Closed** | AD-13 已决定运行身份、状态版本、单调预算、至少一次语义、跨重试稳定 call ID、pending 对账和副作用适配器注册条件。 |
| M3 纠错反馈结构 | **Open — Medium** | AD-6 仍未固定 claim-level 反馈 schema 与“有效改变”的判据。 |
| M4 发布门禁不可判定 | **Closed** | AD-9 固定 `eval/policy.yaml`、数据集/基线/统计参数、失败上限与禁止临时 override。 |
| M5 telemetry usage/关联/保留 | **Open — Medium** | AD-10 原 Rule 未变。 |
| M6 运维与外部依赖恢复 | **Open — Medium** | Agent/job 恢复已覆盖；数据库 migration/backup、LLM timeout/rate-limit/fallback 等仍未决定或 Deferred。 |
| M7 UI/SSE/HITL | **Open — Medium** | companion 能力仍未进入 capability map、AD 或 Deferred。 |
| M8 数据治理/构建指纹 | **Open — Medium** | AD-5 改善身份与 lineage；公开 artifact 许可门禁、index build fingerprint 与 provenance namespace 仍不完整。 |
| L1 Binds 词汇 | **Open — Low** | 新 AD 仍混用 `deployment`、`all` 与 frontmatter capability ID。 |
| L2 AD-12 验收主体 | **Open — Low** | Rule 未变。 |

### Critical / High 最终状态

- **Critical：0**
- **High：0**

### 复核 High 关闭记录

#### R-H1 — AD-13 的 `operation_id` 在重试时不稳定【Closed by revised AD-13】

**关联：** AD-3、AD-13
**问题：** AD-13 规定 `operation_id=sha256(run_id + node_id + attempt + call_index + canonical_args)`。`attempt` 会在重试或 checkpoint 恢复后变化，因此同一逻辑副作用会获得不同 operation ID；SQLite 工具账本无法命中原结果，写操作仍会再次执行。该 Rule 不仅未阻止其 Prevents 所述重复写入，反而把重复执行编码进了幂等键。

**关闭依据：** revised AD-13 采用规划节点生成的稳定 `call_id`，`attempt` 仅作观测字段；工具账本在调用前写入 pending 并按 operation ID 恢复对账；不满足幂等/upsert/read-before-write 条件的副作用适配器不得注册到可恢复图。原 High 已完整关闭。

### 仍开放的 Medium

1. **Verifier/Planner 契约：** AD-6 仍需稳定 `claim_id`、supporting/contradicting evidence IDs、gap type、stop reason，以及防止“只改措辞、证据集不变”的有效纠错判据。
2. **Telemetry 语义：** AD-10 不应要求确定性工具记录 token；模型 usage 应区分 provider-reported/estimated/unavailable，并补 `run_id/span_id`、保留期与访问控制。
3. **运维包络：** 默认 demo 的 SQLite/Neo4j/Chroma migration、backup/restore、readiness、容量失败，以及 LLM/embedding timeout、有限重试、rate limit/fallback 尚未决定或 Deferred。
4. **UI/SSE/HITL：** 升级计划中的 SSE 事件、断线回放、终态和人工复核 resume 语义仍未进入 spine；若不属当前构建范围，应明确 Deferred 并同步 companion。
5. **数据治理/构建指纹：** 公开 fixture/eval/report/trace 的许可和敏感数据 gate，以及 embedding/chunker/extractor/graph schema 的 build fingerprint 和重建触发条件仍需绑定。
6. **Checkpoint 反序列化安全：** 已绑定 `langgraph-checkpoint-sqlite` 3.1.1，但未采用其官方安全要求 `LANGGRAPH_STRICT_MSGPACK=true` 或显式 `allowed_msgpack_modules` allowlist。Pydantic 的节点边界校验不能替代反序列化前的类型白名单；应加入 AD-13 或 Security convention。

### 最新机械与版本复核

- `lint_spine.py`：通过，`0` findings。
- 新增 LangGraph SQLite Checkpoint 3.1.1、MCP Python SDK 2.2.0、OpenTelemetry OTLP HTTP Exporter 1.44.0 与 FastAPI Instrumentation 0.65b0 均为评审日存在的当前版本；主干已正确标记 instrumentation 为 beta 且仅可作为可替换适配器。
- LangGraph SQLite Checkpoint 3.1.1 的官方包说明明确要求限制 msgpack 可反序列化类型；这构成上列新增 Medium，而不是版本选择错误。

## 初审记录（已由“最新复核结论”覆盖）

### 初审 Gate 结论

**NEEDS REVISION（无 Critical，4 High，8 Medium，2 Low）。**

主干已经做对了最重要的方向性选择：六边形模块化单体、真正的状态机 Agent、领域工具端口、MCP 仅作入站适配器、证据身份、fail-closed 验证、版本化评测与禁止装饰性基础设施。它也准确回应了当前代码中的固定流水线、验证默认通过、重复检索实现、嵌入式 Chroma/Compose Chroma 不一致、无效 Kafka 和未鉴权写接口等问题。

但它还不是足以让 Phase 0–2 的多个实现单元独立收敛的“构建契约”。最主要的缺口是：没有确定跨 SQLite/Chroma/Neo4j 的数据权威与发布协议；图中新增了 ingestion worker 却没有任务协调模型；没有规定旧目录与新 `src/` 布局的迁移/切换边界；AD-11 把“关闭写操作”和“实现真实认证”两个互斥方案都留在 Rule 中。以上问题会让两个下游单元做出不兼容选择，违反 Reviewer Gate 的核心判据。

## 机械检查与事实核对

- `lint_spine.py`：通过，`0` 个机械问题；无占位符、重复 AD ID、缺失 Binds/Prevents/Rule 或未固定的 Stack 版本。
- 当前仓库确为主干声明的 brownfield 基线 `b318baa`。
- 当前 `MultiAgentSupervisor` 是手写顺序调用与原样诊断重试（`code/python/core/supervisor.py:38-91`），没有实际 LangGraph 状态图；AD-2/AD-6 是明确的目标态纠偏，不是对现状的错误描述。
- 当前 verifier 解析异常后写入 `passed=True`（`code/python/core/verifier.py:70-73`）；AD-6 与升级计划一致。
- 当前向量实现使用 `chromadb.PersistentClient`（`code/python/services/vector_store.py:57-63`），而 Compose 启动 Chroma 服务（`code/docker-compose.yml:22-31`）；目标态选择 HTTP 服务符合升级计划，但主干尚未定义迁移与单一实现的验收方式。
- 当前上传直接以客户端文件名拼接路径，向量/图谱失败后仍返回 `status="success"`（`code/python/api/main.py:267-317`）；AD-8/AD-11 针对的风险真实存在。
- 当前 Kafka/Zookeeper 仅见于依赖、配置和 Compose，没有业务调用方；AD-12 与 Deferred 的删除方向符合 brownfield 事实。
- 版本存在性/当前性已核对：LangGraph 1.2.11、FastAPI 0.141.1、Pydantic 2.13.5、ChromaDB 1.5.9、Ragas 0.4.3、OpenTelemetry SDK 1.44.0 均为评审日可用稳定版本；Neo4j 5.26.30 为官方 5.26 LTS 镜像版本。版本“存在且当前”已满足，但组合兼容性仍需按主干所述在 Phase 0 测试并锁定。

## High findings

### H1 — 跨存储数据权威与“原子发布”协议未被决定【Closed by AD-8】

**关联：** AD-5、AD-8、Structural Seed
**问题：** 主干同时引入 SQLite metadata/checkpoints、Chroma 和 Neo4j，却没有声明谁是文档、版本、chunk、授权信息和 active 状态的唯一事实源。AD-8 的“所有必需索引完成后才切换为 active”没有定义：必需索引集合、ready marker 写在哪里、切换是否带条件、检索如何保证只看到 active 版本、部分发布如何补偿、崩溃后谁负责 reconcile。内容哈希也没有包含 chunker、embedding、实体抽取器和图谱 schema 的构建指纹，索引重建后实验可能不可复现。

**为何导致分歧：** 一个实现可让 Chroma metadata 决定可见性，另一个让 SQLite manifest 决定；一个先发布向量再补图谱，另一个等待全部完成。两者都“符合”当前文字，却会返回不同证据集。

**建议处置：** `discuss → amend AD-8`。至少绑定：SQLite manifest 是 corpus/version/job 的唯一事实源；Chroma/Neo4j 是可重建派生索引；每个 index build 记录构建指纹与 ready marker；只有 manifest 的条件发布可把版本置为 active；检索必须以 active version/publish token 过滤；失败由幂等补偿和 reconciler 收敛。若不采用此方案，也必须选择另一套同等明确的权威与发布协议，不能放入 Deferred。

### H2 — Structural Seed 新增 ingestion worker，但任务协调模型完全空缺【Closed by AD-14】

**关联：** Structural Seed、AD-8、AD-12、Deferred/Kafka、升级计划 Phase 2
**问题：** 部署图规定 API/MCP process 与 ingestion worker 共同访问 SQLite、Chroma、Neo4j；同时 Kafka 被推迟，升级计划要求“失败任务、重试和可恢复状态”。主干没有决定 API 如何提交任务、worker 如何 claim/lease、是否允许多 worker、重复投递如何幂等、超时任务如何回收、谁拥有重试/取消/状态转换。

**为何导致分歧：** 下游可分别选择 FastAPI BackgroundTasks、内存队列、SQLite job table 或同步入库；这些选择在进程重启、并发上传和恢复语义上互不兼容。

**建议处置：** `discuss → add AD`。对当前模块化单体，优先明确“SQLite durable job table + 原子 claim/lease + 有限重试 + 幂等 stage handler + 单 worker 为默认、并发需证明”的最小方案；Kafka 仍可保持 Deferred。若 Phase 0 不引入独立 worker，则应从 seed 图删去它，直到该决定落地。

### H3 — Brownfield 迁移与切换边界缺失，旧实现与新结构可能长期双轨【Closed by AD-15】

**关联：** AD-1、AD-7、AD-12、Structural Seed、升级计划 Phase 0
**问题：** 当前代码位于 `code/python/{api,core,retrieval,services}`，主干 seed 则位于仓库根的 `src/aerodiagnosis/`，且明确要求“项目中只能有一个融合实现”。但没有规定迁移是原地搬迁、并行 strangler、还是一次性切换；旧入口、旧模块和旧数据目录何时失效；现有 REST 路径是否保持兼容；在迁移期间 CI 以哪套包为权威。

**为何导致分歧：** 一个单元可能继续修补 `code/python`，另一个在根目录创建 `src/`；最终出现两套 API 入口、配置、融合器和数据目录，直接复发主干要防止的重复实现。

**建议处置：** `discuss → add migration invariant`。指定唯一目标包与唯一可执行入口、旧代码的适配/迁移窗口、旧入口的退役条件、数据迁移脚本和兼容测试；Phase 0 完成后 CI 应拒绝从旧业务包导入。该决定属于 brownfield 的必要架构边界，不应只留给任务分解。

### H4 — AD-11 的安全 Rule 内含未决的互斥方案【Closed by revised AD-11】

**关联：** AD-4、AD-11、Auth convention、公开 GitHub/MCP 演示
**问题：** “写操作必须关闭**或**通过真实认证授权”允许两个相反实现；“真实认证”也没有绑定身份来源、token/session 形态、凭据存储和本地/公开演示的模式边界。MCP 与 FastAPI 是否共享身份上下文、stdio 本地客户端是否视为可信边界，也未决定。

**为何导致分歧：** API 单元可实现账号系统，MCP 单元可默认开放本地写工具，部署单元则可能关闭所有写入；各自都可声称满足 AD-11，但组合后权限模型不成立。

**建议处置：** `discuss → split mode policy into an AD`。建议当前课题默认绑定两种明确配置：`demo-public` 只读且不注册写工具；`local-admin` 写操作需要单一选定的认证机制，并在工具执行边界再次授权。第三种托管多用户模式留在 Deferred。若本阶段决定完全不做账号系统，应明确删除现有伪认证，而不是保留“或”。

## Medium findings

### M1 — AD-3 把领域契约与外部传输契约强制成同一 Pydantic 模型【Closed by revised AD-3】

AD-3 正确要求单一业务语义和禁止复制业务逻辑，但“FastAPI 和 MCP 只能复用该契约”与 AD-1 的六边形边界存在张力。API/MCP 需要独立的版本化、权限裁剪和错误映射 DTO；直接共享内部模型会把内部字段暴露给外部并阻碍兼容演进。

**建议：** `autofix` 为“领域 command/result 是唯一语义契约；各入站适配器可定义薄传输 DTO，但必须通过显式 mapper 和共享契约测试映射，不得复制用例逻辑”。同时决定外部 schema 的版本兼容策略。

### M2 — AD-2 的 checkpoint/恢复语义和预算结构不足以约束实现【Closed by AD-13；公式缺陷见 R-H1】

“可持久化 checkpoint、可恢复执行”没有绑定 `run_id/thread_id`、状态 schema 迁移、节点副作用幂等、并发恢复锁、取消语义和 checkpoint 保留期。预算也只写抽象名词，未继承升级计划明确的轮次、token、时间和工具次数四维预算。

**建议：** `discuss → amend AD-2`，固定运行身份、四维预算模型、每节点执行记录/幂等键、单运行独占恢复规则与状态版本迁移职责；保留期可在 Config 中设定并记录到运行快照。

### M3 — AD-6 的“改变查询、工具或候选诊断”仍允许无效形式变化

当前 Rule 能阻止完全原样重试，但不能阻止只改写措辞、重复命中同一证据，且没有稳定 `claim_id`、证据充分性判据和停止原因枚举。两个 verifier/planner 可形成不兼容反馈结构。

**建议：** `autofix`：Verifier 输出版本化的 claim-level 结果（claim ID、supporting/contradicting evidence IDs、gap type、recommended action、stop reason）；下一轮动作必须针对 gap type，且重复证据集合不消耗一次“成功纠错”。具体阈值进入版本化实验配置。

### M4 — AD-9 的发布门禁没有可执行的基线、阈值与豁免规则【Closed by revised AD-9】

“质量回归、引用回归或安全契约失败”没有定义比较基线、允许退化阈值、统计不确定性、数据集最小规模、谁批准基线更新，以及外部 LLM 暂时不可用时如何处理。CI 无法仅凭该 Rule 得出唯一 pass/fail。

**建议：** `discuss → amend AD-9`：门禁策略作为提交到仓库的版本化文件，包含指标方向/阈值、基线 result ID、置信区间或容差、安全测试零容忍项、更新审批记录。smoke gate 与论文 full evaluation 可分层，但必须各自确定性判定。

### M5 — 可观测性 Rule 对 token 字段过度绝对，同时缺少关联与数据保留规则

AD-10 要求“每个节点和工具调用记录 token”，但确定性工具没有 token，部分模型/本地推理也可能不给出精确 usage；这会制造伪造的 `0` 或估算值。另一方面，run/trace/span、入库 job 和 evidence lineage 的关联键、采样、保留期、访问控制尚未绑定。

**建议：** `autofix`：token usage 仅对模型调用记录，区分 provider-reported/estimated/unavailable；所有事件强制 `run_id`、`trace_id`、`span_id`、component/action、outcome，敏感字段按 allowlist 采集。保留期和 exporter 选择可配置或 Deferred。

### M6 — 运行环境与运维维度仅画出了容器，未形成完整契约

主干覆盖 test/local 和 Docker demo，也 Deferred 了 Kubernetes，但没有决定数据库 migration、备份/恢复、readiness 与依赖启动、容量/磁盘耗尽、外部 LLM/embedding 的 timeout/retry/rate limit/fallback、优雅停机和恢复演练。升级计划第 8 节明确要求超时、有限重试、熔断/降级显示，只有 AD-10/AD-12 间接触及。

**建议：** `discuss → add operational AD or explicitly defer each item with revisit condition`。至少应在默认 demo 绑定 readiness 与 degraded 状态、有限重试边界、持久卷备份/恢复命令、schema migration 单一入口，以及外部调用不允许无限重试。

### M7 — UI/SSE 与人工复核是 companion 的能力，却在主干中无落点

升级计划 Phase 1 要求 SSE 展示工具、证据与纠错事件，Agent 验收又允许人工确认；主干 scope/Capability Map 没有 UI/streaming 或 review action。若 API 与前端独立构建，会在事件名称、顺序、重连、终态和 checkpoint 恢复上分歧。

**建议：** `defer or decide`。若属于 v3，增加版本化 public event envelope、顺序号、终态、断线重连/回放和人工复核 command 的边界；若首版不做，明确放入 Deferred 并从 Phase 1 退出标准移除。

### M8 — 数据合规与证据幂等范围未吸收升级计划的公开仓库约束

升级计划要求专有手册不得进入公开仓库，只允许脱敏、合成或授权样例。AD-10 只约束 telemetry，AD-11 只约束公开样例读取；没有规定上传数据、测试 fixture、eval 原始结果和 trace artifact 的分类/发布检查。另以纯内容哈希作为幂等键会把“相同内容但不同授权/来源”的文档错误合并。

**建议：** `add data-governance convention/AD`：证据身份应保留独立 provenance，幂等键限定在明确 corpus/source namespace；公开发布 gate 扫描 fixture/eval/report/trace 的数据分类和许可标记。企业级合规可 Deferred，但公开仓库不泄露专有材料不能 Deferred。

## Low findings

### L1 — `Binds` 词汇不一致

Frontmatter 声明的是 `v3-foundation`、`v3-agent-runtime`、`v3-evidence`、`v3-evaluation`、`v3-mcp`，但 AD 中又使用 `API`、`MCP`、`deployment`、`all releases`。这降低了 AD 到 capability/work item 的可追踪性。

**建议：** 统一为一套稳定 scope ID；`all` 只用于真正跨域 invariant，避免同一条同时出现 `deployment, all`。

### L2 — AD-12 的验收主体不清晰

“存在调用方、集成测试、健康检查和故障处理”方向正确，但对纯库依赖并非都适用，也没说明检查由 CI、Compose smoke test 还是人工架构评审承担。

**建议：** 把规则限定为“默认部署中的外部服务/基础设施”，普通库依赖由 lockfile、依赖扫描和单元测试治理；为 Compose service 增加可自动检查的 manifest/CI rule。

## AD 可执行性逐项检查

| AD | 结论 | Binds / Prevents / Rule 评语 |
| --- | --- | --- |
| AD-1 | Pass with note | 依赖方向清楚且与 Prevents 对应；建议用 import-linter/架构测试自动执行，并明确 `tools` 实现属于 application-side 还是 outbound adapter。 |
| AD-2 | Partial | 能排除手写固定流水线，但 checkpoint、恢复并发、幂等与预算结构未绑定，见 M2。 |
| AD-3 | Partial | 单一业务语义可执行；“同一 Pydantic 模型直出所有边界”过度绑定并与适配器职责有张力，见 M1。 |
| AD-4 | Pass | 内部不经 MCP 回环、MCP 仅入站，Rule 能直接防止所述分歧。传输/auth 细节需由 AD-11 补齐。 |
| AD-5 | Partial | 字段集合清楚；证据类型、provenance 独立性、派生证据 lineage、哈希作用域和 index build fingerprint 不足，见 H1/M8。 |
| AD-6 | Partial | fail-closed 与预算耗尽终态明确；纠错反馈 schema 和“有效改变”判据不足，见 M3。 |
| AD-7 | Partial | 能消除两套融合器和隐式降级；应补 embedding/chunker/extractor/index schema 版本，并把 Agent 全局 Prompt/model config 从“retrieval strategy”中分离。 |
| AD-8 | Fail | 目标正确，但缺少事实源、ready/commit/visibility/reconcile 协议，无法实现“原子”，见 H1/H2。 |
| AD-9 | Partial | artifact 要求可执行，pass/fail 政策不可执行，见 M4。 |
| AD-10 | Partial | 禁止思维链和原文默认不入 telemetry 很好；token 要求不适用于所有工具，关联/保留/访问控制缺失，见 M5。 |
| AD-11 | Fail | Prevents 正确，但 Rule 用“关闭或认证”保留互斥架构选择，见 H4。 |
| AD-12 | Pass with note | 与 brownfield 的无效 Kafka 完全吻合；需缩小到外部基础设施并明确自动验收，见 L2。 |

## 架构维度覆盖矩阵

| 维度 | 状态 | 说明 |
| --- | --- | --- |
| 范式、模块边界、依赖方向 | Decided | AD-1 与 paradigm 基本充分。 |
| Agent 编排、状态、纠错 | Partial | 路由/验证方向已定；运行身份、幂等、恢复锁和预算 schema 缺失。 |
| 工具能力与端口 | Partial | 工具语义统一；内部/外部 DTO 分层和 schema 版本策略缺失。 |
| 证据身份与引用 | Partial | 基础字段充分；来源权威、派生 lineage、授权和构建指纹不足。 |
| 数据所有权与跨存储一致性 | Missing | 没有唯一事实源与发布/补偿协议。 |
| 入库后台执行 | Missing | worker 已入图，但任务队列、lease、retry/cancel 未决定。 |
| 检索与实验配置 | Mostly decided | 需把 embedding/chunking/graph extraction 纳入版本快照。 |
| API/MCP 边界 | Partial | MCP 入站定位正确；公共 schema 版本和身份传递缺失。 |
| 安全与权限 | Partial | 最小权限方向正确；默认模式仍是二选一，数据发布治理缺失。 |
| 评测与发布 | Partial | 产物链路已定；门禁阈值与基线治理缺失。 |
| Observability | Partial | 事件内容和隐私底线已定；关联、usage 语义、保留/访问缺失。 |
| 部署与环境 | Partial | local/demo + Compose 已定；worker 协调与外部模型依赖未入图。 |
| 运维、恢复、迁移 | Missing/Partial | 无 schema/data migration、backup/restore、容量与恢复演练契约。 |
| Brownfield cutover | Missing | 旧 `code/python` 与新 `src/` 的权威/退役边界未定。 |
| UI/SSE/HITL | Silent | companion 明确要求，但主干既未决定也未 Deferred。 |
| 论文/公开数据治理 | Partial | 实验可复现有覆盖；专有资料与公开 artifact 管理未进入 spine。 |

## Deferred 完整性检查

现有 Deferred 对 Kafka、微服务、Kubernetes、多租户/RBAC、模型微调和全量 GraphRAG 的处理合理，且都有重审条件。以下事项当前既未决定，也未列入 Deferred；其中前四项会立即造成下游分歧，不应简单推迟：

1. **必须决定：** metadata/system-of-record 与 Chroma/Neo4j 派生索引的发布、回滚、reconcile 规则。
2. **必须决定：** ingestion worker 的持久任务、claim/lease、重试、取消和并发模型。
3. **必须决定：** brownfield 包布局、API/data 迁移与旧实现退役边界。
4. **必须决定：** `demo-public` 与 `local-admin` 的权限模型；不能保留“关闭或认证”。
5. **决定或 Deferred：** API/MCP 公共 schema 版本兼容策略。
6. **决定或 Deferred：** SSE 事件回放和人工复核 command/resume 语义。
7. **决定或 Deferred：** checkpoint/trace/data 的保留期、备份恢复和 schema migration。
8. **决定或 Deferred：** 外部 LLM/embedding 的 timeout、rate-limit、fallback 与离线 CI 策略。
9. **决定或 Deferred：** embedding/chunker/extractor/graph schema 变化触发的全量或增量重建策略。
10. **必须决定：** 专有资料、公开 fixture、eval 输出和 trace artifact 的数据分类/许可门禁。

## 推荐 Gate 处置顺序

1. 先讨论并修订 H1、H2、H3、H4；它们决定 Phase 0–2 是否能并行而不分叉。
2. 直接修正 M1、M3、M5、L1、L2 的清晰表述问题。
3. 对 M2、M4、M6–M8 做明确选择；若当前阶段不实现，写入 Deferred 且给出重审条件。
4. 修订后重新运行机械 lint 与 rubric gate；在 H1/H2/H4 未解决前，不建议把该 spine 作为自动构建或故事拆分的最终约束。
