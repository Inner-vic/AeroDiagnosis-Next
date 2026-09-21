# Research Distillation: Fault Diagnosis Agent

研究日期：2026-09-20

本文件提炼自三个参考仓库，用于指导 AeroDiagnosis 的下一步实现，不复制仓库代码。

## 参考来源

1. [zengury/manastone-diag](https://github.com/zengury/manastone-diag)
2. [ScorpioPxg/IndustryGraphRAG](https://github.com/ScorpioPxg/IndustryGraphRAG)
3. [CHAOZHAO-1/LLM-based-PHM](https://github.com/CHAOZHAO-1/LLM-based-PHM)

## 关键结论

垂直故障诊断应用不能只是“上传数据 + RAG 生成一段话”。可信的 Agent 需要至少具备：

1. 确定性故障规则与数值阈值
2. 时间线因果匹配
3. 结构化故障图谱与多跳因果路径
4. LLM 只负责排序、解释和生成受控报告
5. 现场验证后的经验回流
6. 可复现的诊断评测

## 从 manastone-diag 借鉴

### 值得引入

- **三层证据匹配**
  - 症状关键词
  - 日志/文本模式
  - 数值阈值
  - 时间窗口因果链
- **三轮门禁工作流**
  - 数据采集 -> 诊断分析 -> 审核报告
  - 每轮有 gate、checklist、deliverables、validate 和 checkpoint
- **经验库验证闭环**
  - AI 诊断、现场验证结果、实际根因、准确度、教训分离保存
  - 已验证案例提升后续相似案例权重
- **能力边界**
  - 明确记录“哪些信号观测不到、哪些是已知假阳性、哪些结论不可做”
- **领域知识包**
  - 故障 ID、症状、根因、可能原因、维修建议、阈值、因果规则

### 不建议照搬

- 将 LLM 作为外部 coding-agent 通过命令行驱动工具。AeroDiagnosis 已有更严格的状态机，应保留结构化编排。
- 对小型知识库使用 JSON 分片和自定义搜索。当前 SQLite/Neo4j/Chroma 已足够。

## 从 IndustryGraphRAG 借鉴

### 值得引入

- **故障因果图谱 Schema**
  - `Equipment -> HAS_COMPONENT -> Component`
  - `Component -> HAS_SYMPTOM -> Symptom`
  - `Symptom -> CAUSED_BY -> Cause`
  - `Cause -> SOLVED_BY -> Solution`
  - `Cause -> RELATED_TO -> Cause`
  - `Symptom -> HAS_THRESHOLD -> Threshold`
  - `... -> MENTIONED_IN -> Document`
- **多跳因果路径**
  - 从现象出发，经过原因、措施、根因，返回完整路径
  - 多个现象共享同一原因时提升该根因
- **候选路径由 LLM 排序和验证**
  - 图谱没有候选时禁止 LLM 编造
  - LLM 输出失败时回退确定性排序
- **实体对齐与融合**
  - 精确去重、同义词归一化、关系修正、自环过滤
- **诊断评测**
  - `cause_accuracy`、`solution_accuracy`
  - `causal_completeness`
  - `MRR`
  - `answer_coverage`
  - `avg_response_time`
  - 按 easy/medium/hard 分层

## 从 LLM-based-PHM 借鉴

### 学术定位

当前项目更适合引用以下方向：

- PHM/故障诊断大模型综述
- 知识图谱增强 LLM 的故障诊断
- 动态知识图谱与级联关系抽取
- 航空装备故障诊断 KG + LLM
- PHM-Bench 领域评测
- SAFELLM 安全监测
- AeroGPT 航空轴承故障诊断
- CausalKGPT 航空产品质量因果分析

### 设计约束

- 不以“接入了 DeepSeek”作为贡献
- 明确区分检索相关度、故障概率、维修置信度
- 对安全相关结论做 fail-closed 和人工复核

## 融合后的目标架构

```text
监测数据 / 工单 / 手册
        |
        v
确定性摄取与实体抽取
        |
        +--> 版本化文档与证据
        +--> 故障规则库与阈值
        +--> 结构化因果图谱
        |       Equipment/Component/Symptom/Cause/Solution/Threshold
        v
诊断工作流
  - 症状与数值阈值匹配
  - 时间线因果匹配
  - 多跳因果路径检索
  - LLM 排序、解释、验证
  - 工程师反馈与人工审批
        |
        v
现场验证与经验回流
  - 验证结果、实际根因、准确度
  - 更新故障规则权重与相似案例检索
        |
        v
诊断评测
  - cause/solution accuracy
  - causal completeness
  - MRR
  - response time
```

## 对 AeroDiagnosis 的优先级

### P1：结构化故障因果图谱

- 引入 typed node kind 与关系
- 增加多跳因果路径检索
- 将现有演示图谱升级为航空发动机故障因果图谱

### P2：确定性故障规则与能力边界

- 增加 FaultRule、Threshold、TemporalCausalRule 数据模型
- 将参数分析和图谱检索扩展为确定性多证据匹配
- 增加 capability boundary 报告

### P3：经验验证闭环

- 扩展案例模型，保存现场验证结果、实际根因、准确度
- 已验证案例参与检索加权和规则修正

### P4：诊断评测

- 增加冻结诊断用例
- 计算 cause/solution accuracy、causal completeness、MRR、响应时间
- 与纯 RAG 基线对比
