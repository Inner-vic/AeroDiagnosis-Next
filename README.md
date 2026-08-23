# AeroDiagnosis — 航空发动机气路故障多智能体协同诊断平台

## 项目简介

AeroDiagnosis 是一个面向航空发动机气路故障诊断的垂直领域智能平台。系统基于**多智能体协同编排（Multi-Agent Orchestration）**架构，融合**向量语义检索、BM25 关键词检索与知识图谱结构化推理**三路混合 RAG 策略，实现从故障现象输入到诊断报告输出的全流程自动化。平台内置**自验证闭环**与**案例推理引擎**，确保诊断结论有据可查、可追溯。

## 核心功能

1. **多智能体协同诊断** - Supervisor + 6 个专用 Agent（意图识别、实体抽取、混合检索、故障推理、验证、报告生成）

2. **多模态文档解析** - 支持 PDF/DOCX/XLSX/PPTX/图片等 11 种格式，上传即入库并自动抽取知识

3. **混合 RAG 检索** - 三路并行检索：ChromaDB 向量 + BM25 关键词 + Neo4j 图谱遍历，通过 RRF 融合提升召回率

4. **知识图谱可视化** - ECharts 力导向图，支持节点详情查看、关系维护、右键软删除、同名实体自动合并

5. **诊断案例库** - 对话自动记录为新案例，预置 8 个主流机型典型故障案例（CFM56-7B/V2500/GE90/Trent 700）

6. **安全与运维** - Cypher 注入防护、API 鉴权、结构化 APM 日志、一键启动脚本

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Agent 编排 | LangGraph |
| LLM | DeepSeek-Chat / GPT-4o (OpenAI 兼容 API) |
| Embedding | text2vec-base-chinese (本地) |
| 向量数据库 | ChromaDB |
| 知识图谱 | Neo4j 5.28 |
| BM25 检索 | rank-bm25 + jieba |
| API 框架 | FastAPI |
| 前端 | ECharts + marked |

## 快速开始

### 环境要求

- Python 3.10+
- Docker (用于 Neo4j)

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/AeroDiagnosis.git
cd AeroDiagnosis/code/python

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填写你的 API Key

# 5. 启动 Neo4j (Windows)
cd ..\..
start_neo4j.bat

# 6. 启动后端服务
cd python
uvicorn api.main:app --host 0.0.0.0 --port 8081 --reload

# 7. 访问浏览器
# http://localhost:8081
# 默认登录：admin / admin123
```

## 项目结构

```
code/
├── python/                    # Python 后端
│   ├── api/
│   │   └── main.py           # FastAPI 入口 (38 个 REST 端点)
│   ├── core/                 # 核心 Agent 模块
│   │   ├── supervisor.py     # 多 Agent 编排器
│   │   ├── query_agent.py    # 意图识别 + 实体抽取
│   │   ├── retrieval_agent.py # 三路检索统一入口
│   │   ├── diagnostic_agent.py # 故障诊断推理
│   │   ├── verifier.py       # 自验证 Agent
│   │   └── report_agent.py   # 报告生成 Agent
│   ├── retrieval/            # 检索增强模块
│   │   ├── hybrid_engine.py  # 混合检索引擎
│   │   └── hybrid_fusion.py  # RRF 融合算法
│   ├── knowledge/            # 知识抽取模块
│   ├── services/             # 基础设施服务
│   │   ├── vector_store.py   # ChromaDB 封装
│   │   ├── knowledge_graph.py # Neo4j 封装
│   │   └── case_base.py      # 案例推理引擎
│   ├── utils/                # 工具函数
│   ├── config/               # 配置管理
│   └── static/               # 前端 SPA
├── docs/                     # 文档目录
└── .gitignore
```

## 技术亮点

### 多智能体自验证闭环

传统 RAG 系统直接输出结果，无法判断生成内容是否被检索上下文支撑。AeroDiagnosis 引入 **Diagnose→Verify→Retry** 闭环：诊断推理后由独立验证 Agent 逐假设审核，不通过则重试 (max 3)，确保输出可追溯、有据可查。

### 三路混合检索 + RRF 融合

单一向量检索在航空领域专业术语上召回不足。系统将**向量语义 (ChromaDB) + BM25 关键词 (rank_bm25+jieba) + 图谱遍历 (Neo4j)** 三路并行，通过 RRF 融合，兼顾语义、关键词与结构化推理。

### 领域 Few-Shot 实体抽取

通用 NER 模型无法覆盖航空发动机专业实体 (如 EGT、N1、VSV、FOD)。系统使用 **Few-Shot 示例 Prompt**，无需训练专用模型即可准确抽取 7 类实体和 7 种关系。

## API 接口

启动服务后访问 [http://localhost:8081/docs](http://localhost:8081/docs) 查看 Swagger 文档。

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/ingest/upload` | 上传文档 |
| POST | `/api/diagnose/query` | 诊断查询 |
| GET | `/api/graph/nodes` | 获取图谱节点 |
| GET | `/api/cases/list` | 获取案例列表 |
| GET | `/api/health` | 健康检查 |

## License

MIT License
