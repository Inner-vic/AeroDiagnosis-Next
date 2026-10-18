# 融合开发交付记录

2026-10-13，实施目录：D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis。

## 已接入的实际应用链路

Docling 2.138.0 / Docling Core 2.101.1 负责 PDF、HTML、DOCX、XLSX、CSV 的结构化解析和官方 HybridChunker 切分。保留标题、表格、Docling JSON、页码和边界框；文本块定位到带 SHA256 的规范快照，原始文件另行不可覆盖归档。当前 PDF 配置保留表格结构，未启用扫描件 OCR。

LangExtract 1.7.1 调用真实 SDK，通过受预算约束的 DeepSeek 提供者抽取带角色、否定、不确定性和条件的陈述。关闭模糊对齐和部分匹配，将精确引文定位、领域必要检查通过的候选映射到已有 SQLite 图谱；拒绝项保留在审计目录。来源对齐不能证明语义蕴含，候选明确标记 machine_candidate / 非专家审核。

这些组件已经接入现有 FastAPI、文档管理上传、BM25 文档检索、图谱检索、引用问答和前端原始文件下载。原文和结构快照下载沿用 Operator Token 权限，不接受跨文档版本访问。原文归档校验文件哈希，所有抽取、重排、生成和核验共用原预算账本。

Docling-Graph、LightRAG 和 GraphRAG 的代码审查用于选型参考，本轮没有声称完成其运行时集成。来源和许可见 [上游审查](UPSTREAM-REUSE-REVIEW.md)。

## 真实联调结果

FAA AMT Powerplant 第10章原始物理第45页派生单页 PDF，经应用 `/api/documents/upload` 实际导入，生成5块、14节点和7条关系；原始文件下载字节完全一致，结构 JSON 下载成功。检索与图谱均返回证据，真实问答返回 evidence_ready、5条经现有 LLM 核验的陈述和5项引用证据。

这是单页工程验收，不是正式冻结测试集的效果成绩。派生文件页码为1；原文件物理第45页的派生关系记录在试跑文件和收据中，不把两种页码混同。没有导入题目或标准答案作为知识资料。

共享账本累计30次供应商调用，费用上界0.226411元，pilot总上限5元；0项待定或未知费用。这里使用峰值价格估计，非供应商账单。脱敏记录见 [链路验收收据](fusion-application-smoke.json)，完整输出位于 `.runtime/fusion-smoke/application-receipt.json`。

## 启动与配置

已生成隔离配置 `.runtime/fusion/profile/.env.local`，应用数据库位于 `.runtime/fusion/application/knowledge.sqlite`。密钥仅引用用户本地 env.txt 路径，Operator Token 保存在本地配置，未写进代码或本文。旧默认配置继续支持现有应用；融合配置已实际运行。

```powershell
uv sync --extra upstream
.venv/Scripts/python.exe -X utf8 scripts/run_upstream_application.py --profile-root .runtime/fusion/profile
```

本地地址 http://127.0.0.1:8082 。知识管理的 Operator Token 使用隔离配置中的值；凭据不要粘贴到问答正文。当前隔离运行选用 BM25，不将 hashing 向量声明为语义检索。

## 验证

全量 pytest 481 passed；Ruff 全 src/tests/scripts 通过；Mypy 全97个源文件通过。新增检查包括真实 LangExtract SDK、真实 Docling HTML 表格解析、精确规范文本定位、原文下载权限/版本归属、归档篡改与路径越界、型号/飞行阶段冲突，以及新增归档字段的旧位置参数兼容性。

本轮未提交、合并或发布。后续全来源建库与正式 S5 对照仍按方案执行；本记录不提前宣称正式评估完成。
