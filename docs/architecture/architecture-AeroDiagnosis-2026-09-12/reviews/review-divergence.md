# Divergence / Adversarial Review — AeroDiagnosis v3

- **Target:** `ARCHITECTURE-SPINE.md`
- **Lens:** 两个独立实现单元能否仅凭架构主干做出兼容实现
- **Review mode:** Reviewer Gate / independent validate；不修改主方案
- **Verdict:** **PASS WITH IMPLEMENTATION-LEVEL FOLLOW-UPS**
- **Latest re-review:** 2026-09-12，基于 AD-16 构建身份与禁止跨构建恢复后的最终主干

## 0C. 第五轮复核（最终结论）

### 当前 Gate 结论

**Critical：0。High：0。Gate：PASS WITH IMPLEMENTATION-LEVEL FOLLOW-UPS。**

第四轮唯一剩余 High（H4-01）已经关闭。AD-16 现在把 Git commit、容器镜像 digest、依赖 lock digest、工具实现 digest 和全部可变读源共同纳入内容寻址 `RunSnapshot`；未声明实现/读源的工具不得注册，恢复只能在相同构建身份下执行，其他情况 fail closed 为 `build_unavailable`。因此两个独立实现单元不能再分别选择“兼容 schema 即跨构建恢复”与“固定原构建恢复”。

确定性检查再次通过：`lint_spine.py` 返回 0 个发现。先前保留的 canonical encoding、port 所有权、progress fingerprint、SQLite 运行参数等均为 Medium 或实现级 follow-up，不构成当前 initiative spine 的 Critical/High divergence。

### 第五轮 Reviewer Gate 判定

**Gate verdict: PASS WITH IMPLEMENTATION-LEVEL FOLLOW-UPS（0 Critical，0 High）。**

## 0B. 第四轮复核（历史记录）

### 当前 Gate 结论

**Critical：0。High：1。Gate：CHANGES REQUIRED。**

第三轮 H3-02 已关闭；H3-01 的“可变数据源”部分已关闭，但仍缺少同一 schema 下工具可执行行为的构建身份。机械检查继续通过，`lint_spine.py` 返回 0 个发现。

| 第三轮发现 | 当前状态 | 复核结果 |
| --- | --- | --- |
| H3-01：RunSnapshot 未覆盖所有可变工具读源 | **主体关闭，保留一个窄 High** | AD-16 已要求每个只读工具声明全部可变读源，并明确覆盖案例库和参数规则/模型；但 snapshot 仍只记录工具 schema major，没有绑定工具实现/build identity。 |
| H3-02：同 document 的有效 job 发布竞态 | **关闭** | AD-8 已在接收事务分配单调 revision/desired_revision，并要求发布同时通过 desired revision、fencing token、计数与哈希校验；较旧并发 job 进入 superseded。 |

### 剩余 High

#### H4-01 — 恢复运行未绑定工具实现/部署构建身份

AD-16 的 `RunSnapshot` 包含工具 schema major，但兼容 schema 不代表行为不变。实现单元 A 在修复检索/参数算法后保持 schema major 不变并恢复旧 checkpoint；实现单元 B 认为旧 run 必须继续使用原构建。两者均符合现有文字，却会让同一 `snapshot_id` 在两个部署上执行出不同的排序、参数特征或工具结果。AD-9 为 Release 评测保存 Git commit，但普通诊断 run 和 AD-13 的恢复条件没有绑定该 commit/image/tool build；AD-13 甚至允许同 workflow/state major 直接恢复。

**关闭条件：** `RunSnapshot` 增加不可变 execution build identity（至少 Git commit 或 container/image digest）以及各工具 implementation digest/version；恢复时当前构建若不能提供 snapshot 指定的兼容实现，终止为 `snapshot_unavailable`/`resume_incompatible`，不能仅凭 schema major 继续。若选择允许升级后继续，则必须明确这是新 run/派生 run，并保留父 run 与旧 snapshot，而不能沿用原 run 身份。

### 第四轮 Reviewer Gate 判定

**Gate verdict: FAIL（0 Critical，1 High）。** 关闭 H4-01 后，本 divergence lens 可判为 **PASS WITH IMPLEMENTATION-LEVEL FOLLOW-UPS**。

## 0A. 第三轮复核（历史记录）

### 当时 Gate 结论

**Critical：0。High：2。Gate：CHANGES REQUIRED。**

上一轮 H-03、H-04、H-05 已关闭；H-01 与 H-02 的主体风险已关闭，但各自仍留有一个跨单元 High。机械检查再次通过，`lint_spine.py` 返回 0 个发现。

| 上轮 High | 当前状态 | 复核结果 |
| --- | --- | --- |
| H-01：不可变 RunSnapshot | **部分关闭** | AD-16 已固定内容寻址快照、运行内显式传播、恢复禁止重读 active，以及缺失版本的 fail-closed 终态；但快照列举项未覆盖案例库、参数分析规则/模型等所有可变工具读源。 |
| H-02：worker fencing | **部分关闭** | AD-14 已固定单调 fencing token，旧 worker 的状态推进、跨存储写与 active 切换都会被拒绝；但 token 是 job-scoped 时，两个不同且都合法的 job 仍可竞争同一 `document_id`。 |
| H-03：operation identity / 副作用 | **关闭** | AD-13 已令 `call_id` 跨重试稳定、排除物理 attempt、先写 pending 并在恢复时对账，且从可恢复图中拒绝无法原生幂等或确定性对账的副作用工具。 |
| H-04：API/MCP 执行语义 | **关闭** | AD-17 已固定服务端派生 actor、唯一错误映射、单调 deadline 预算、协作取消终态及长任务 job handle；剩余具体码表和接口字段属于公共契约实现，不再是架构选择。 |
| H-05：telemetry 敏感数据与 trace | **关闭** | AD-10 已将原文、Prompt、密钥和非白名单参数改为强制禁止，并固定服务端 trace 身份、入站不信任与传播白名单。 |

### 剩余 High

#### H3-01 — `RunSnapshot` 没有声明覆盖所有可变工具读源

AD-16 明确绑定 active document versions、图谱构建版本、检索配置、Prompt/model/embedding 和工具 schema，但系统还包含独立案例库与气路参数分析工具。两个合法实现仍可分叉：案例检索单元从 snapshot 指定的语料读取，参数工具单元却读取“当前”规则、校准参数或模型 artifact；恢复后同一 run 得到不同数值诊断。仅记录 tool schema major 不能识别行为相同 schema 下的实现/数据变化。

**关闭条件：** 将枚举列表提升为封闭原则：`RunSnapshot` 必须绑定**每个获准读取的可变数据集和行为 artifact**，至少包括案例库 snapshot、参数分析规则/校准/模型 digest、tool implementation/build digest；工具注册时声明其 snapshot dependencies，运行前若任一依赖无法解析则 fail closed。也可明确案例库属于 active document set、参数工具完全纯函数且 artifact 已由 build digest 固定，但必须选择并写明一种。

#### H3-02 — fencing 解决旧 worker，尚未解决两个有效 job 的同文档发布竞态

AD-14 的 fencing token 能拒绝失去租约的旧 worker，但若 token 随 job 单调递增而不是随 `document_id`，两个 job A/B 都可各自持有当前 token。较早提交的版本可能因处理较慢而最后激活，把较新的文档版本回滚为旧内容；实现单元也可分别选择“最后完成者胜出”或“最后接收者胜出”。AD-8 的 SQLite active 指针事务没有指定这项顺序。

**关闭条件：** 固定同一 `document_id` 的发布仲裁：建议在接收时分配单调 `document_revision`，active 切换用 `(document_id, expected_active_revision)` CAS，只有最高未取消 revision 可发布；或明确串行化每个 document 的 job 并令后续版本使前序未发布 job 进入 `superseded/cancelled`。这是 document-scoped 约束，不能只依赖 job-scoped lease token。

### 第三轮 Reviewer Gate 判定

**Gate verdict: FAIL（0 Critical，2 High）。** 关闭 H3-01 与 H3-02 后，当前 divergence lens 可判为 **PASS WITH IMPLEMENTATION-LEVEL FOLLOW-UPS**。

## 0. 修订后复核

### 当前 Gate 结论

**Critical：0。High：5。Gate 仍为 CHANGES REQUIRED。**

首轮 P0 风险已经被实质降级：SQLite 成为发布权威、证据 ID 有了明确算法、恢复明确采用至少一次语义、工具具备统一信封、评测门禁由版本化 policy 决定。主干已从“原则集合”进入“接近可分派建设”的状态，但仍有五个跨单元语义需要固定；否则两个实现团队仍会在长时运行一致性、租约接管、恢复去重、API/MCP 错误和敏感观测上做出不兼容选择。

机械检查已通过：`lint_spine.py` 返回 0 个发现。

### 首轮发现关闭表

| 首轮发现 | 当前状态 | 复核依据 | 剩余风险 |
| --- | --- | --- | --- |
| DVG-01：跨存储原子发布 | **部分关闭；P0 → High** | AD-8 已固定 SQLite 为唯一可见性权威、状态机、active 指针、后端 `version_id/job_id`、发布核对和恢复清理；AD-14 已固定持久任务租约。 | 缺少 run-scoped corpus snapshot、租约 fencing token，以及并发 active 切换的 CAS 前置版本。 |
| DVG-02：证据身份 | **部分关闭；P0 → High** | AD-5 已分别固定四类 ID 的语义和算法，并要求类型化 locator、不可变证据和受引用保护的旧版本。 | 哈希拼接没有 canonical encoding；`immutable_source_ref` 结构未封闭；运行中仍可能跨 active 版本；pipeline artifact 版本未完全进入身份/快照。 |
| DVG-03：恢复与副作用重放 | **部分关闭；P0 → High** | AD-13 已固定至少一次、单调预算、工具账本、版本兼容和 `resume_incompatible`。 | `operation_id` 包含可变化的 `attempt`；外部副作用与 SQLite 账本之间仍有崩溃窗口；checkpoint/ledger/state 提交顺序仍可分叉。 |
| DVG-04：API/MCP 工具契约 | **部分关闭；High 保留** | AD-3 已加入统一请求/结果信封、封闭 status 和 major 版本规则。 | error taxonomy、retryability、deadline/cancel、分页/流式终止映射及 actor 的可信来源仍未固定；AD-4 仍未固定外部 capability catalog。 |
| DVG-05：确定性发布门禁 | **关闭** | AD-9 已把数据集哈希、baseline commit、样本量、重复、阈值、统计检验、效应量、infra failure 和禁止临时 override 固定进 `eval/policy.yaml`。 | 论文级完整复现 manifest 仍应补齐 corpus/model/prompt artifact 身份，但不再影响“谁决定是否发布”的一致性，归入 H-01。 |

### 剩余 High

#### H-01 — 缺少贯穿诊断与实验的不可变 `RunSnapshot`

AD-8 要求“所有检索先读取 active 指针”，但这可能被实现为**每次工具调用重新读取**。因此运行单元 A 在 run 开始时冻结版本，检索单元 B 在每次检索前读取最新 active；一次多轮诊断或 checkpoint 恢复就可能混用两个文档版本。AD-5 的引用保护只能防止删除，不能防止同一 run 跨版本；AD-7/AD-9 也未用一个内容寻址 manifest 同时绑定 corpus、resolved retrieval config、prompt/model/reranker/tool/workflow 版本。

**关闭条件：** 固定不可变、内容寻址的 `RunSnapshot`，至少包含 active `document_id -> version_id` 集、resolved config hash、prompt/model/embedding/reranker/tool/workflow/state schema 版本；run 创建时保存，所有工具、恢复和评测只接受该 snapshot ID。托管模型若无 revision，记录 provider 返回标识、完整参数和响应指纹，并把可复现性表述为可审计重放。

#### H-02 — AD-14 的租约缺少 fencing，过期 worker 仍可能发布

条件更新取得租约只能防止同时“领取”，不能阻止旧 worker 在停顿后恢复。单元 A 租约过期后仍继续写索引/切换版本；单元 B 已接管并重试同一任务。两者可能都通过预期计数与哈希，最后由旧 worker 覆盖新 active 指针。`lease_owner` 本身不构成单调 fencing token，AD-8 的 active 切换也未要求 expected-current-version CAS。

**关闭条件：** 每次接管递增 `lease_generation/fencing_token`；所有 manifest 状态推进、完成记录和 active 切换都必须在同一 SQLite 事务中验证 `(job_id, lease_owner, generation, lease_not_expired)`。同一 `document_id` 的发布还应以预期旧 active version 做 CAS，失去租约的 worker 只能丢弃结果，不能发布或清理新 owner 的数据。

#### H-03 — AD-13 的 operation identity 与外部副作用去重仍不稳定

`operation_id=sha256(run_id + node_id + attempt + call_index + canonical_args)` 中的 `attempt` 没有说明是恢复后保持不变的逻辑调用尝试，还是每次执行递增的物理尝试。两种实现会为同一恢复调用生成不同 ID。即便 ID 稳定，若 Chroma/Neo4j 已写成功而 SQLite 工具账本尚未提交，恢复时仍会重复外部效果；仅在账本中查重不能消除这个窗口。

**关闭条件：** operation ID 必须由持久化的逻辑 invocation ordinal 生成，物理 retry attempt 只作为观测字段，不参与幂等身份；外部写适配器必须把 operation ID 写入目标记录并建立唯一约束/幂等 upsert。固定顺序为“持久化 invocation → 执行/幂等写 → 持久化规范结果 → checkpoint 引用该结果”，并规定预算在何处只增一次。

#### H-04 — API/MCP 的错误、身份与长操作语义仍可分叉

统一信封解决了字段外形，但 `error` 仍可由 FastAPI 单元实现为 HTTP 404/409/503，由 MCP 单元实现为 status=`ok` 的文本错误；`deadline` 的时钟/单位、取消后是否保留部分结果、`degraded` 是否可重试也未封闭。更重要的是，请求信封含 `actor`，但 AD-11 没有声明它只能由可信入站适配器从 operator token/匿名 profile 生成，客户端可能直接伪造。AD-4 也没有明确 MCP 只暴露原子只读工具还是长时诊断 use case。

**关闭条件：** 固定封闭 error code、retryable、details/redaction schema 及 HTTP/MCP/SSE 映射表；规定 deadline 为 UTC 绝对时间或单调剩余毫秒中的一种，并固定 cancel/partial/degraded 语义。`actor/ExecutionContext` 必须由入站适配器创建、从 payload schema 排除且不可覆盖。发布 MCP capability catalog；若含长时 use case，固定 job/stream/cancel/resume 协议。

#### H-05 — AD-10 对敏感 telemetry 仍是可选默认值而非不变量

“原文内容默认不进入 telemetry”允许单元 A 在 debug 配置写全文，单元 B 永远禁止；两者都符合文字，却对专有航空手册和 Prompt injection 内容形成不同泄露面。trace ID 也没有规定跨 FastAPI、MCP、LangGraph 和 ingestion worker 的传播与信任边界。

**关闭条件：** 将原文、完整 Prompt、密钥/token 和私有思维链列为不可持久化字段；若研究调试确需样本内容，必须进入独立、显式授权、短保留期且不外发 OTLP 的 profile。固定 event envelope、数据分级、redaction 责任与 trace propagation；外部传入 trace ID 只能作为 link，内部需生成可信 trace identity。

### 降为 Medium / 可在详细设计中关闭

- AD-1 对 inbound/outbound port 的措辞仍会让团队选择不同 port 所有权；通过允许依赖矩阵、composition root 位置和 import-linter 测试即可关闭。
- AD-5 的字符串拼接必须使用带字段名、长度或 canonical JSON/CBOR 的无歧义编码；所有 hash 前需固定 Unicode normalization、字节编码与算法版本。
- AD-6 尚未把“改变动作”变成 progress fingerprint，也未封闭 claim/terminal status；应在 workflow contract 中固定。
- AD-7 应保存展开默认值后的 canonical config hash，且论文/Release 禁止隐式 fallback。
- AD-12 的基础设施准入仍缺 owner/SLO/移除条件，但可由 ADR 模板和 Compose contract test 管理。
- SQLite 的 WAL、busy timeout、migration leader、backup/restore 和 readiness 属于部署详细设计，但必须由单一 deployment owner 在首个集成里固定。
- AD-15 已解决长期双轨风险；数据迁移的读写兼容窗口、rollback 与 feature-flag 删除证据可在各模块迁移计划中细化。

### 修订后 Reviewer Gate 判定

**Gate verdict: FAIL（0 Critical，5 High）。** 五个 High 都是跨模块/跨适配器接口，不能安全地下放给两个独立实现单元各自决定。关闭 H-01～H-05 后，可把 divergence 结论提升为 **PASS WITH IMPLEMENTATION-LEVEL FOLLOW-UPS**。

## 1. 首轮结论（修订前历史记录）

主干选定的方向是连贯的：六边形模块化单体、持久化状态机、统一领域工具、证据溯源、版本化入库、离线评测门禁与 MCP 入站适配器彼此并不冲突。但它目前更像一组正确的设计原则，还不是足以约束独立实现的“一致性契约”。

最危险的分歧集中在五个跨模块接缝：恢复执行、证据身份、跨存储发布、工具的线协议、发布判定。两个团队可以逐字遵守现有 AD，却仍分别实现出不可恢复的 checkpoint、指向不同内容的相同引用、检索可见性不一致的“active”版本、API/MCP 不同的错误语义，以及相反的发布结论。因此本轮不能判定为可直接分派建设。

严重级别：**P0** 表示若不先固定，数据正确性、可追溯性或恢复语义无法保证；**P1** 表示独立单元将形成接口或运维不兼容；**P2** 表示可以在实现期补足，但必须有唯一所有者和验证方式。

## 2. 最高优先级发现

### DVG-01 [P0] AD-8 的“原子发布”没有可实现的提交协议

**两个实现单元的合法选择：**

- 入库单元 A 先把 SQLite manifest 标为 `active`，再逐个把 Chroma/Neo4j 的 staging 数据改成可检索；失败时依赖补偿清理。
- 检索单元 B 认为只有 Chroma/Neo4j 都完成别名切换才算 `active`，并直接查询各后端当前集合/标签。

两者都符合“先暂存、完成后切换 active”，但在崩溃窗口内 B 会读到新旧混合结果。SQLite、Chroma、Neo4j 之间没有共同事务，当前 Rule 也没有指定唯一提交权威、读路径过滤规则、并发上传仲裁和 crash recovery。

**必须固定的最小契约：**

1. 定义版本状态机及允许的 CAS 转移，例如 `staging -> validating -> active | failed -> deleting -> deleted`；禁止跳转。
2. SQLite manifest 是发布可见性的唯一权威；后端记录必须携带 `version_id`，所有线上查询先取得一次 run-scoped active-version snapshot，再按该 snapshot 过滤，不能读取后端“最新”。
3. 规定 commit barrier：哪些索引为 required、各后端完成证明的字段、事务中写入的发布事件，以及失败/超时后的补偿责任。
4. 规定同一 `document_id` 并发发布的乐观锁或租约、旧 active 版本的切换与保留语义，以及恢复器如何幂等重放。
5. 明确 `content_hash` 幂等键的作用域；至少区分逻辑文档身份、原始字节哈希、解析/切分/embedding/图谱抽取管线版本，否则相同内容或管线升级会被错误折叠。

**验收方式：** 在每个状态转移处注入进程终止和后端故障；恢复后任何诊断运行只能看到完整的旧版本或完整的新版本，且同一操作重放不产生重复节点/向量。

### DVG-02 [P0] AD-5 尚未定义证据 ID 的规范生成与不可变引用

**两个实现单元的合法选择：**

- 检索单元 A 用随机 UUID 作为 `chunk_id/evidence_id`，内容相同但每次重建都会改变引用。
- 入库单元 B 用 `SHA256(chunk_text)` 作为 `chunk_id`，同一段文本在两个文档或两个页码中被合并；`evidence_id` 又可能按查询结果临时生成。

两者都能携带 AD-5 列出的字段，却不能互相解释“稳定”意味着跨重试、跨重建、跨版本还是跨仓库稳定。更严重的是，`get_evidence_detail(evidence_id)` 若默认查当前 active 版本，旧报告可能在版本切换后解析到新内容或查不到内容。

**必须固定的最小契约：**

1. 为 `document_id`、`version_id`、`chunk_id`、`evidence_id` 分别定义规范语义、作用域、生成算法与唯一约束；不要只规定类型为不透明字符串。
2. 证据引用必须绑定不可变版本，建议以规范化的 `(document_id, version_id, chunk_id, locator, content_hash)` 为事实身份；`evidence_id` 若是派生 ID，需规定规范序列化和哈希算法。
3. 区分事实身份与一次检索命中身份。检索分数、query、retriever/ranker 版本和 rank 属于 `retrieval_hit`，不能改变底层 evidence 的身份。
4. 诊断 run 在启动时冻结 corpus/version snapshot；报告与 verifier 只能使用该 snapshot 中的 evidence，且版本 GC 必须受报告/run 引用保护。
5. `locator` 需按来源类型定义可验证结构（页码、段落、表格/单元格、图谱路径及支撑 chunk），不能是适配器自定字符串。

**验收方式：** 对重传、改名、重新切分、管线升级、重复段落、版本切换和版本回收做契约测试；旧报告的每个引用仍解析到同一字节/结构片段，或返回明确的已归档状态，绝不能静默指向新内容。

### DVG-03 [P0] AD-2/AD-6 未固定可恢复执行与副作用重放语义

**两个实现单元的合法选择：**

- 工作流单元 A 在节点执行前 checkpoint；恢复后重新执行整个节点和工具调用。
- 工具单元 B 认为 checkpoint 表示节点已经提交，遇到同一个 `tool_call_id` 就直接返回 success，但没有持久化原结果。

这样会产生重复 LLM/工具调用、不同证据集、预算重复扣减或遗漏、甚至错误地跨过 verifier。现有“状态版本化、可序列化、可恢复”没有定义 checkpoint 的提交边界、run/thread 身份、工作流版本兼容、工具结果幂等性和预算账本。

**必须固定的最小契约：**

1. 状态至少携带 `run_id`、`workflow_version`、`state_schema_version`、单调 `step_seq`、`attempt`、`tool_call_id`、corpus snapshot、配置快照 ID 和累计预算账本。
2. 明确 checkpoint 是“节点开始前”还是“节点提交后”的耐久记录；工具结果和下一状态的提交顺序必须唯一，恢复器据此判定重放或复用结果。
3. 只读工具仍需按 `tool_call_id + canonical_input_hash + tool_version` 保存结果或显式声明可重新求值；时间敏感工具须冻结 snapshot。将来任何写工具都必须另有 idempotency key 和效果日志。
4. 定义工作流/状态迁移矩阵：哪些旧版本可继续、哪些必须迁移、哪些只能终止为 `resume_incompatible`。
5. AD-6 的“改变下一轮”应变成可机检不变量：禁止重复同一 `(tool, canonical_args, evidence_set_hash)`；Verifier 输出要引用稳定 `claim_id` 和 `evidence_id`，终止状态使用封闭枚举并规定优先级。

**验收方式：** 在节点前、工具返回后、checkpoint 写入中和 verifier 后注入崩溃；恢复结果不得重复计费、越过预算、重复副作用或改变已冻结证据。

### DVG-04 [P1] AD-3/AD-4 的 Pydantic 模型不足以构成 API/MCP 共同契约

**两个实现单元的合法选择：**

- FastAPI 单元把类型化领域错误映射为 HTTP 404/409/422/503，并以 SSE 推送部分结果。
- MCP 单元把同样的错误包装成成功工具结果中的 `{error: ...}`，超时后返回已有部分证据；字段别名、空值和分页也使用另一套序列化规则。

两边确实“复用了同一 Pydantic 输入/输出”，但调用者观察到的成功、失败、重试、权限、取消、降级和流式结束语义并不相同。当前图中 `MCP -> Application`，文字又说“暴露同一组工具”；独立团队仍会争论 MCP 暴露的是原子领域工具还是完整诊断用例。

**必须固定的最小契约：**

1. 定义 transport-neutral invocation envelope：`contract_version`、`tool_name/version`、`request_id/tool_call_id`、principal/capabilities、deadline、correlation/trace、input、result/error、warnings/degradation、pagination。
2. 固定封闭错误码和可重试性；HTTP status、MCP error/tool-result、SSE terminal event 只是该错误的映射表，不能各自发明语义。
3. 固定缺省值、null/omitted、枚举、时间、数字精度、最大 payload、分页/cursor 和 JSON Schema 兼容策略；发布生成的 schema snapshot 并在 API/MCP 契约测试中比较。
4. 明确 MCP 工具目录：哪些是原子只读领域工具，哪些是长时诊断 use case；如果暴露后者，须定义异步 job/stream/cancel/resume 语义。
5. 明确部署边界：同进程时共享 composition root；分进程时 MCP 不得直接拼装另一套领域依赖或经自身协议回环，必须调用稳定 application port，并统一 principal context。

**验收方式：** 同一 golden invocation 分别通过直接 application port、FastAPI 和 MCP 执行，比较规范化结果、错误码、授权判定和 trace 关联；只允许传输层表现差异。

### DVG-05 [P1] AD-9 的“发布门禁”没有确定性判定规则

**两个实现单元的合法选择：**

- 评测单元 A 用单次运行均值，只要任一质量指标低于上一 commit 即阻断。
- CI/发布单元 B 用三次运行均值，只在降幅超过 5% 时阻断，并允许 LLM judge 失败时跳过该指标。

两者都保存了逐样本结果、配置、commit 和汇总，也都能声称阻断“质量回归”，但对同一候选会给出相反结论。模型输出和 judge 具有随机性；仅记录版本名称还不足以复现托管模型行为。

**必须固定的最小契约：**

1. 版本化 gate policy：数据集内容哈希与切分、baseline artifact、必测 suite、指标方向、绝对下限/最大允许回归、统计检验、重复次数、随机性与多重比较策略。
2. 区分 deterministic contract gate、离线质量 gate、成本/延迟 guardrail 和人工审阅；规定每类是 hard fail、soft fail 还是 informational。
3. 明确 provider/密钥不可用、超时、配额、LLM judge 解析失败和样本缺失均为 `infra_error`，不能按通过或质量回归处理；不得静默减少分母。
4. 同一机器可读 manifest 同时驱动 CI、论文表格和 README 徽标；override 必须带审批者、理由、过期时间和审计记录。
5. 托管模型记录实际返回的 model revision（若提供）、请求参数和响应指纹；无法保证完全复现时应明确为可审计重放，而非位级复现。

**验收方式：** 给定同一 candidate/baseline artifact，任何环境中的 gate evaluator 必须输出同一类别结论；对 judge/网络故障做注入测试，不能出现 fail-open。

## 3. 逐条 AD 分歧扫描

| AD | 独立实现仍可能分叉的位置 | 风险 | 建议处置 |
| --- | --- | --- | --- |
| AD-1 | 单元 A 把 repository/tool port 放在 `domain`，单元 B 放在 `application`；入站适配器究竟“实现端口”还是调用 use case 也不清楚；composition root 无所有者。 | **P1**：循环依赖可能没有源码 import 却通过 service locator/runtime import 出现。 | 固定允许依赖矩阵、port 所有权、composition root 唯一位置，并用 import-linter/架构测试执行；澄清 inbound adapter 依赖 application use-case port，outbound adapter 实现 outbound port。 |
| AD-2 | 状态/图版本、checkpoint 边界、run/thread ID、并发 resume、预算单位和副作用重放均可自定。 | **P0**：恢复产生不同结论或重复效果。 | 采纳 DVG-03；增加 durable execution 契约与故障注入测试。 |
| AD-3 | “单一 Pydantic 契约”未覆盖调用 envelope、版本兼容、错误、deadline、取消、分页、部分成功和降级。 | **P1**：适配器语义漂移，schema 相同仍不互操作。 | 采纳 DVG-04；建立公共契约包、生成 schema snapshot 与 consumer contract tests。 |
| AD-4 | MCP 暴露原子工具还是诊断 use case 未决；同进程/独立进程的依赖装配、流式/长任务、principal 传播无约束。 | **P1**：出现第二套业务编排、双重授权或网络回环。 | 固定 MCP capability catalog、application port 和部署模式；明确 transport mapping 与 auth context。 |
| AD-5 | 四种 ID 的生成、作用域和稳定期限未定义；score 混合了检索阶段；locator 自由格式；旧版本 GC 未绑定引用。 | **P0**：不可复现引用或错误合并证据。 | 采纳 DVG-02；发布 evidence identity/schema v1。 |
| AD-6 | “改变查询、工具或候选诊断”允许无进展的同义改写；claim 无稳定身份；预算先后顺序、异常重试和终止枚举未定义。 | **P1**：死循环、预算不一致或同一状态不同结果。 | 以 progress fingerprint 禁止重复动作，固定 verifier/action/terminal schema 及预算优先级。 |
| AD-7 | config ID 与内容哈希、默认值展开、后端/模型 artifact revision、fallback 的 fail-open/fail-closed 选择未定；各检索器 score 的意义也可不同。 | **P1**：同名配置不能复现，降级结果混入正式实验。 | 规范化并 hash 完整 resolved config；记录语料 snapshot 和 artifact revision；评测默认禁止隐式 fallback，线上结果显式带 degradation。 |
| AD-8 | active 的权威、跨三库提交顺序、读隔离、并发发布、恢复/补偿、旧版本保留与 pipeline-version 幂等未定义。 | **P0**：读到部分索引或错误折叠版本。 | 采纳 DVG-01；定义 manifest 驱动的 saga/state machine。 |
| AD-9 | baseline、阈值、重复策略、统计规则、基础设施错误和 override 未定。 | **P1**：相同候选在不同 CI/研究者手中得出相反发布结论。 | 采纳 DVG-05；发布 versioned gate policy 和 canonical verdict artifact。 |
| AD-10 | event schema、redaction 分类、摘要算法、保留期、采样、trace 跨 API/MCP/worker 传播未定义；“默认不进入”允许配置泄露。 | **P1**：观测链断裂，或某适配器持久化敏感原文。 | 固定 event envelope 和数据分级；原文/Prompt 默认应为强制禁止，只有显式受控 debug profile 可短期启用；定义 trace propagation。 |
| AD-11 | “真实认证”未规定 principal/capability 模型；API 与 MCP scope 映射、demo/managed profile、二次工具校验的上下文签名均可不同。 | **P1**：同一用户跨入口权限不一致，甚至信任客户端伪造身份。 | 定义 capability matrix 与不可由调用者覆盖的 `ExecutionContext`；启动时校验 deployment profile，demo profile 根本不注册写工具。 |
| AD-12 | “存在调用方/测试/健康检查/故障处理”只有准入原则，没有默认部署 manifest 的唯一所有者和删除/引入流程。 | **P2**：不同分支仍可各自加入“有测试”的装饰组件。 | 让 lockfile + Compose manifest 成为默认部署清单；新增基础设施要求 ADR、owner、SLO/失败策略和移除条件。 |

## 4. AD 之间的跨接缝冲突与缺口

### 4.1 Evidence、run state 与 ingestion version 没有共同 snapshot

AD-5 要求报告引用运行状态证据，AD-7 记录检索配置，AD-8 切换 active version，AD-2 又允许运行中断后恢复；但没有一个不可变 `RunSnapshot` 将以下内容绑定在一起：corpus active-version set、resolved retrieval config、tool contract/version、workflow/state schema、model/prompt revision。结果是长时运行或恢复运行可能跨过入库发布点，并把两版证据混到同一报告。

建议新增一个被 AD-2/5/7/8/9 共同引用的 snapshot 契约，创建后不可修改，只能由新 run 采用新 snapshot。

### 4.2 “领域工具”与“应用用例”的边界没有画清

AD-3 把检索、图谱、案例、参数分析、证据读取称为领域工具；AD-4 让 MCP 暴露同一组工具；结构图却让 FastAPI/MCP 都指向 application。这里至少存在三类接口：

1. 入站 use case（开始/查询/恢复诊断、查询入库任务）；
2. Agent 可选择的受控工具（检索、图遍历、案例、参数分析）；
3. 出站 repository/model ports。

如果不显式命名，团队 A 会让 MCP 直接调用 `tools/`，团队 B 会通过 application service，权限、trace、预算和 snapshot 自然分裂。架构应固定三层接口的方向和哪些接口允许对外发布。

### 4.3 部署图存在没有协议的 ingestion worker

Docker seed 引入独立 `DWORKER`，但没有 API 如何提交任务、worker 如何获取任务、租约/重试/取消/死信如何处理、谁执行 manifest 状态转移。团队 A 可能使用 SQLite 轮询，团队 B 可能同步在 API 中入库再让 worker 仅重建图谱；二者会直接破坏 AD-8 的统一原子性。

这是**P1** 架构缺口。即使暂不选消息中间件，也必须固定“SQLite durable job table + lease/CAS”或“同进程后台任务”等一种工作模型，以及单 writer、幂等 task key、retry ownership、shutdown drain 和健康/就绪语义。否则 `DWORKER` 不应出现在结构 seed。

### 4.4 SQLite 的跨进程所有权未决

图中 API 和 ingestion worker 共用 SQLite volume，同时 SQLite 还承担 metadata/checkpoint。没有声明进程并发模型、WAL/locking、migration owner、busy timeout、备份恢复和 schema compatibility。团队可分别假定“单写者”和“多写者”，导致本地可用、Docker 间歇锁死。

这是**P1** 运维一致性风险。至少固定数据库 owner/migration leader、允许写入者、事务边界、启动顺序与 readiness；如果 checkpoint 与 ingestion manifest 分库，也应明确。

### 4.5 配置版本与契约版本没有兼容治理

主干多处要求“版本化”，但没有规定 semver/hash 的用途、兼容窗口、迁移责任和废弃流程。`state_schema_version=2`、`tool_contract_version=2`、`prompt_version=2` 可以同时出现，却没有 compatibility matrix。应由一个 canonical run manifest 收拢所有内容地址和版本，并规定升级时谁迁移、谁拒绝、谁保留旧适配器。

## 5. 部署与运行包络专项检查

当前已有 local/test 与 Docker demo 两个环境，但以下维度尚未达到“已决定、已延期或开放问题”的 Reviewer Gate 要求：

| 维度 | 当前缺口 | 优先级 |
| --- | --- | --- |
| 进程模型 | API、MCP、worker 同/异进程边界与依赖装配不确定。 | P1 |
| 后台任务 | job store、lease、retry、cancel、recovery、dead-letter 语义缺失。 | P1 |
| 数据迁移 | SQLite/Neo4j/Chroma schema/index migration owner、顺序和 rollback 缺失。 | P1 |
| 就绪与启动 | health check 被 AD-12 提到，但 readiness 依赖、startup ordering 与 partial-ready 未定义。 | P2 |
| 备份与恢复 | manifest/checkpoint、证据版本、Neo4j/Chroma 的一致备份点与恢复校验缺失。 | P1 |
| 资源/安全边界 | CPU/memory/concurrency、上传解压预算、LLM/tool timeout 与 rate limit 没有统一预算层级。 | P1 |
| 配置 profile | test/demo/research 的行为差异与禁止项没有机器可检 manifest。 | P1 |
| 数据保留 | 旧 evidence version、run/checkpoint、trace/eval artifact 的 retention/GC 未定义。 | P1 |

这些不要求现在引入 Kubernetes、broker 或企业级 RBAC；相反，模块化单体更需要把单机/Compose 下的所有权和失败语义固定清楚。

## 6. 评测与论文可复现性专项检查

AD-9 的研究意图正确，但“冻结测试集”应同时冻结或记录：样本内容哈希、标注 schema、排除列表、corpus snapshot、resolved retrieval config、prompt/template content、工具/工作流版本、模型/embedding/reranker revision、seed、judge rubric、依赖/镜像 digest 和硬件/并发设置。只记录 Git commit 与配置快照不能覆盖外部模型和外部数据变化。

此外，论文实验与发布门禁不应完全等同：

- 发布门禁需要快速、确定、可重复的 hard contract suite；
- 论文实验可以昂贵且包含统计重复，但产物必须不可变并接受人工审核；
- LLM-as-judge/Ragas 只能生成指标产物，不能在解析失败时默认通过，也不能单独决定安全契约；
- 调参集、验证集和最终测试集的访问/解冻策略必须受 audit，避免“冻结”只是一句约定；
- 比较基线需指向不可变 artifact，而非会移动的 `main/latest`。

## 7. 建议的修复顺序（供架构作者处理）

1. 先固定 `EvidenceIdentity v1`、`RunSnapshot v1` 与 `IngestionManifest v1`，因为后续状态、检索和评测都依赖它们。
2. 固定 durable execution/checkpoint commit 语义及 workflow/state migration policy。
3. 发布 tool invocation/error/event envelope 和 API/MCP 映射表，补 golden contract tests。
4. 固定 demo 部署的 worker/job/SQLite 所有权与 migration/readiness/backup 规则。
5. 发布 versioned evaluation gate policy，区分 contract、quality、performance 和 infra-error verdict。
6. 最后用以下端到端故障场景复审全部 AD：入库中断、版本切换中诊断、checkpoint 后崩溃、MCP 超时取消、外部模型降级、旧报告解析、CI judge 不可用。

## 8. 首轮 Reviewer Gate 判定（修订前历史记录）

**Gate verdict: FAIL / revise before implementation split.**

本评审没有否定总体范式或技术路线，也没有要求将 seed 膨胀成完整设计文档。所缺的是少量但承重的协议：它们恰好是代码无法替架构作者自动决定、且两个并行实现最容易选择不兼容的部分。完成 DVG-01 至 DVG-05 后，再运行一次 divergence review；如果上述故障注入与跨入口 golden contract 能通过，可将结论提升为“可分派建设”。
