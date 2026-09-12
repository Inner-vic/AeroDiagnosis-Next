# 实现状态与验证记录

更新日期：2026-09-12

## 已完成里程碑

### 自顶向下应用框架与数据主干

- 单一 `bootstrap` 组合根，HTTP、CLI 与 MCP 复用框架无关应用用例；
- SQLite schema v1–v6 前向迁移，默认承载清单、向量、图、案例、checkpoint、会话、模型注册、根因分析会话和迁移账本；
- TXT、Markdown、CSV 确定性解析及明确格式白名单；
- 文档逻辑 ID、不可变 version/chunk/evidence 身份和 active-only 原子发布；
- SQLite hashing 向量基线和两跳双向图遍历；
- 重复摄取幂等、旧版本不可见、失败索引不发布；
- HTTP 写入口默认关闭，可信 operator token 必须由本机环境显式注入。

### LLM 驱动诊断 Agent

- 生产诊断必须使用本机默认或每位使用者自己的 OpenAI-compatible provider，不提供伪 Agent fallback；
- Planner 从手册、图谱、案例、参数四个白名单工具中选择查询和调用路径；
- Generator 只能引用本次冻结证据，Verifier 逐主张决定支持集合；
- verifier 失败后必须改变方案，最多两轮，工具调用有显式预算；
- 运行快照绑定活动语料、工具后端、模型 endpoint 指纹、模型名和 Prompt 版本；
- SQLite checkpoint 可恢复，模型或工作流身份不同会拒绝恢复；
- 工具异常进入错误轨迹，证据不足、引用越界或预算耗尽时 fail closed；
- 支持 `.env.local` 或独立 Key 文件配置本机默认模型，API 状态不会暴露 Key；
- 已使用用户提供的 DeepSeek API 完成真实 Planner、Generator 和 Verifier 请求；
- API Key 只用于 provider 请求；自动化测试确认它不会进入 SQLite。

### 会话、开放入口与低优先级界面

- SQLite 持久化短期会话记忆，消息按序且幂等，可创建、回读和级联删除；
- 官方 MCP Python SDK stdio Server，提供四个只读证据工具和运行状态；
- v2 上传目录迁移器支持默认预检、online backup、哈希账本、幂等更新和失败报告；
- 确定性评测器记录冻结数据集哈希、逐样本状态、claim 术语召回、来源召回和证据绑定；
- 新工作台提供三套主题，以及诊断、根因分析、模型中心、知识图谱、案例库和知识管理六个一级模块；
- 知识管理支持上传、不可变版本浏览和已有逻辑文档更新；
- 首页定位已收敛为知识问答，移除气路参数编辑和独立运行检查器；证据引用改为回答内按需展开；
- 知识问答支持 TXT、Markdown、CSV、JSON 文本附件摘录和浏览器语音输入；
- 根因分析改为可点击的四阶段递进工作台：数据与模型、初步定位、交互排查、分析报告；
- 首次启动幂等写入 1 份通用知识、10 个图谱节点、9 条关系和 4 个已有案例；
- 知识图谱已切换为本地 ECharts 力导向图，支持缩放、平移、节点拖拽、邻接高亮与详情查看；
- wheel 已包含 HTML、CSS 和 JavaScript 前端资源。

### 知识增强根因分析垂直切片

- 系统定位固定为：故障诊断模型给出初步结果，LLM 结合受控知识、观察证据和历史状态提供根因分析与维修支持；
- 模型插件显式绑定数据集 ID、特征画像、预处理参数、标签空间、运行时和 OOD 阈值；
- 数据上传确定性触发 data analysis、model diagnosis、knowledge retrieval、root cause、inspection planner、state update 和 report 角色链；
- 工程师可反馈发现、未发现、不确定或无法检查，候选根因按结构化规则更新后进入下一轮；
- 模型输出、数据画像、候选根因、检查动作、观察、Agent 轨迹和报告均持久化，可回放；
- LLM 只能选择白名单候选与检查动作，报告只能引用允许的维修支持，强制保留人工复核与教学数据限制；
- 内置 `teaching-linear-gaspath` 仅为教学合成的声明式线性模型，用于真实跑通接口和展示，不宣传为深度学习或真实诊断能力；
- 已用本机默认 DeepSeek 实际跑通“上传数据 → 初步诊断 → 根因候选 → 工程师反馈 → 再规划 → 报告”闭环。

## 仍未完成且不得宣传为已实现

- 领域专家标注的论文级金标准、真实模型基线/消融实验及诊断准确率提升；
- Chroma HTTP 与 Neo4j v3 适配器，以及对应外部索引迁移；
- PDF/Office 摄取、持久 ingestion worker、SSE 流式进度和人工审批动作；
- ONNX/TorchScript 深度模型运行时、可信模型包签名以及真实数据集/模型适配；
- 根因知识仍是受控教学包，尚未接入型号手册审核库或形成论文级根因准确率结论；
- 适航验证、机型覆盖或可直接用于维修/放行决策的安全认证；
- 是否采用 LangGraph 仍需由后续对比实验决定，当前是自研显式可恢复状态机。

## 验证口径

完整门禁命令：

```powershell
.\scripts\Test-AeroDiagnosis.ps1 -Full
```

所有自动化测试默认不访问真实 LLM 或外部数据库；工作流测试使用受控模型替身验证路由和
fail-closed 语义。真实 provider 的诊断质量必须在用户提供 API 与合法领域数据后单独评测。

### 2026-09-12 本机最终验证

- Python：uv 管理的 CPython 3.13.14；
- SQLite：schema 6，vector=`sqlite_hashing`，graph=`sqlite_graph`；
- Ruff：通过；
- mypy strict：通过（55 个 source files）；
- pytest：95 passed；
- branch coverage：88.99%（门槛 85%）；
- legacy compileall：通过；
- sdist + wheel：构建通过，wheel 内含三项前端静态资源；
- JavaScript：`node --check` 通过；
- MCP：进程内客户端发现 5 个工具，并实际调用手册检索与参数分析；
- API smoke：`/`、`/api/system`、`/api/documents`、`/api/graph`、`/api/cases` 均通过；
- smoke 状态：schema 6，agent=`llm_orchestrated_v2`，本机默认模型=`deepseek-v4-flash`；
- 合成参数证据的真实模型调用通过严格 `CandidateDiagnosis` 与
  `VerificationDecision` 校验，未向模型发送本地数据库内容；
- 教学 CSV 的真实知识增强调用得到风扇初步定位、4 个候选根因、11 条角色轨迹和带强制限制的持久化报告；
- 当前开发服务运行于 `http://127.0.0.1:8080/`，运行数据保留在 `.runtime`。
