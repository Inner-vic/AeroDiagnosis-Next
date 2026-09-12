# AeroDiagnosis 部署、迁移与运维

本文分别说明 v3 默认本机运行、迁移期 v2 Docker 部署、数据备份与升级。两套路径不能混称为同一实现。

## 1. 新 Windows 电脑：无 Docker 默认方案

### 前置条件

- Windows 10/11；
- Python 3.13；
- uv；
- Git（仅 clone/update 时需要）。

启动和管理数据不需要管理员权限、Docker、Neo4j Desktop、Ollama、Chroma 服务或 API Key；执行 Agent 诊断时需要使用者在前端提供自己的模型 API。建议将源码放在 `D:\Projects\AeroDiagnosis`，默认 `.runtime` 也会在 D 盘。

```powershell
Set-Location D:\Projects
git clone https://github.com/Inner-vic/AeroDiagnosis.git
Set-Location .\AeroDiagnosis
.\scripts\Initialize-AeroDiagnosis.ps1
.\scripts\Test-AeroDiagnosis.ps1 -Full
.\scripts\Start-AeroDiagnosis.ps1
```

初始化脚本是幂等的：它创建目录、按 `uv.lock` 同步 Python 3.13 环境并向前迁移 SQLite，不覆盖数据库。启动脚本只监听 `127.0.0.1`，PID 和日志保存在运行目录。停止脚本会校验 PID 的启动时间，避免误停复用同一 PID 的其他进程。

使用自定义运行目录时，四个脚本都传入同一个参数：

```powershell
$Runtime = "D:\AeroDiagnosisRuntime"
.\scripts\Initialize-AeroDiagnosis.ps1 -RuntimeDir $Runtime
.\scripts\Test-AeroDiagnosis.ps1 -RuntimeDir $Runtime -Full
.\scripts\Start-AeroDiagnosis.ps1 -RuntimeDir $Runtime
.\scripts\Stop-AeroDiagnosis.ps1 -RuntimeDir $Runtime
```

### 当前本机服务范围

当前应用提供系统、文档、证据检索、诊断和短期会话接口。文档写入、版本发布、模型规划、
四类工具、逐主张验证、checkpoint 和会话记忆通过同一个应用核心贯通。状态接口返回
`llm_orchestrated_v2`；这表示模型驱动工作流已实现，不表示已经完成论文评测或适航验证。

本机 CLI 摄取不需要 API token：

```powershell
.\.runtime\.venv\Scripts\aerodiagnosis-ingest.exe .\samples\manual.txt
```

HTTP 写入口默认关闭。确需使用时，由用户在启动服务前自行注入
`AERODIAGNOSIS_OPERATOR_TOKEN`，请求通过 `X-AeroDiagnosis-Operator-Token` 传递；仓库和
日志中不得保存 token。只读证据检索不需要 token。当前 JSON 写入口用于小型文本和测试，
不是大文件上传通道；格式支持限于 TXT、Markdown 和 CSV。

交付模式下，每位使用者在前端输入 provider 地址、模型名和自己的 API Key，配置仅存在当前
标签页。本地展示模式可复制 `.env.example` 为被 Git 忽略的 `.env.local`，一次性填写默认
provider；`AERODIAGNOSIS_LLM_API_KEY_FILE` 可引用单独的密钥文件，避免复制 Key。系统状态只
暴露默认模型是否可用及模型名。服务端不会把 provider Key 写入数据库、checkpoint、会话或日志。

知识管理写入仍要求 `AERODIAGNOSIS_OPERATOR_TOKEN`，并在页面当前标签页输入，以阻止其他网页
静默修改本机知识库。会话问题和压缩后的诊断回复会写入 SQLite，可用会话 DELETE 接口清除。

## 2. 可选 Docker / 学校服务器方案

`code/docker-compose.yml` 当前部署的是迁移期 v2 应用及 Chroma HTTP、Neo4j。它适合在合规学校服务器或允许 Docker 的个人电脑上复现旧原型，不是 v3 本机开发前提：

```powershell
Set-Location .\code
Copy-Item .env.example .env
# 编辑 .env：配置合规 OpenAI-compatible API 和强 Neo4j 密码
docker compose up --build -d
docker compose ps
```

默认 Compose 只把 API 和 Neo4j Browser 绑定到回环地址。开发覆盖文件额外开放回环地址上的 Chroma 和 Bolt，并挂载旧源码：

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

重要边界：v3 的 `chroma_http` 与 `neo4j` 工厂目前会显式拒绝启动，因为相应的新端口适配器尚未实现。不要通过捕获异常后返回空结果来伪造兼容性。完成适配器、契约测试和数据迁移器之后，才会把这两个后端接到 v3。

不要在公司电脑安装未授权 Docker、Neo4j Desktop 或其他常驻服务。外部 LLM 只能使用用户自行配置的合规 OpenAI-compatible API；密钥只放环境或未提交的 `.env`。

## 3. 数据备份与恢复

### v3 SQLite

先停止服务，再复制整个运行目录，至少保留数据库：

```powershell
.\scripts\Stop-AeroDiagnosis.ps1 -RuntimeDir D:\AeroDiagnosisRuntime
Copy-Item -LiteralPath D:\AeroDiagnosisRuntime\data\aerodiagnosis.db `
  -Destination D:\Backups\aerodiagnosis-2026-09-12.db
```

恢复前同样先停止服务，把当前数据库另存为回滚副本，再复制备份。不要在服务写入时用普通文件复制声称获得一致备份。旧上传目录迁移器已经使用 SQLite online backup API 自动生成一致的迁移前副本。

`.venv` 和 `uv-cache` 可由锁文件重建，不是首要备份对象。应优先保护 SQLite、上传原件、实验配置、逐样本结果和未推送源码。SQLite 现在包含会话消息；备份、共享或提交前应按数据治理要求脱敏。API Key 不在数据库内。

### v2 Docker 命名卷

v2 的 Neo4j、Chroma 和上传文件在 Docker 命名卷中。`docker compose down` 保留卷；不要随手使用 `down -v` 或 `docker system prune --volumes`。备份和恢复应使用 Docker 官方命名卷流程，并在备份后校验可读性。

## 4. 旧数据迁移

v3 SQLite schema 已支持 `PRAGMA user_version` 1 → 5 的前向迁移。旧上传目录迁移器默认只预检，
`--apply` 时先在线备份数据库，再用持久账本幂等导入 TXT/Markdown/CSV；原文件不会被移动、
修改或删除：

```powershell
.\.runtime\.venv\Scripts\aerodiagnosis-migrate-v2.exe .\code\uploads
.\.runtime\.venv\Scripts\aerodiagnosis-migrate-v2.exe .\code\uploads --apply
```

出现失败时保留旧文件、迁移前备份和逐文件错误；恢复只需停止服务并换回报告中的
`backup_path`。以下外部存储迁移仍未实现：

- Chroma collection 到 v3 active-version 过滤模型；
- Neo4j 实体/关系到带 `source_ref/version_id` 的 SQLite 图；
- v2 内存案例到 SQLite 版本化案例。

因此现在仍不要删除旧上传目录或 Docker 卷。上传原件只有在迁移报告无失败、计数/哈希与抽样回读均确认后才可另行归档；外部索引继续保持只读备份。

## 5. MCP 本机入口

`aerodiagnosis-mcp` 使用 stdio，适合由本机 MCP Host 启动。它只注册手册检索、故障图谱、
相似案例、参数分析和运行状态五个只读工具，不接收或存储 LLM API Key：

```powershell
.\.runtime\.venv\Scripts\aerodiagnosis-mcp.exe
```

## 6. 安全升级流程

```powershell
.\scripts\Stop-AeroDiagnosis.ps1
Copy-Item -LiteralPath .\.runtime\data\aerodiagnosis.db `
  -Destination .\.runtime\data\aerodiagnosis.before-upgrade.db
git pull --ff-only
.\scripts\Initialize-AeroDiagnosis.ps1
.\scripts\Test-AeroDiagnosis.ps1 -Full
.\scripts\Start-AeroDiagnosis.ps1
```

数据库只自动向前迁移。若数据库 schema 高于当前代码支持版本，进程会 fail closed；请恢复匹配版本的代码或备份，不要手改 `user_version`。

## 7. 故障排查

### `uv` 缓存访问被拒绝或 C 盘增长

只通过项目脚本运行；脚本把 `UV_CACHE_DIR` 与 `UV_PROJECT_ENVIRONMENT` 指向运行目录。若直接执行 `uv`，请先设置同样的环境变量。

### Python 版本不对

初始化固定请求 Python 3.13。运行：

```powershell
uv python find 3.13
```

找不到时按组织允许的方式安装 Python 3.13，不要擅自安装常驻服务。

### 服务未就绪

查看：

```powershell
Get-Content .\.runtime\logs\server.stderr.log -Tail 100
Get-Content .\.runtime\logs\server.stdout.log -Tail 100
.\scripts\Test-AeroDiagnosis.ps1
```

若 PID 文件过期，停止脚本只会清理过期文件；若 PID 已被其他进程复用，脚本会拒绝停止该进程。

### 数据库版本不兼容

错误包含 `newer than supported` 时，当前代码比数据库旧。恢复升级前的代码或匹配备份，不要降低 SQLite `user_version`。

### 选择外部后端后启动失败

这是预期的显式保护。把 `AERODIAGNOSIS_VECTOR_BACKEND` 和 `AERODIAGNOSIS_GRAPH_BACKEND` 恢复为 `sqlite`。v3 外部适配器完成前，Docker 仅用于独立的 v2 迁移期部署。

## 8. 卸载

停止服务后，源码和运行目录可以分别处理。删除 `.venv`/`uv-cache` 可重建；删除 `data/aerodiagnosis.db` 会删除当前 v3 数据，必须先备份。项目脚本不会自动删除这些内容。
