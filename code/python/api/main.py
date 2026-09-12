"""
FastAPI 入口 — 航空发动机气路故障多智能体协同诊断平台

提供接口:
  1. POST /api/auth/login       — 登录
  2. GET  /api/auth/check       — Token 验证
  3. POST /api/auth/logout      — 登出
  4. POST /api/auth/account     — 修改账户
  5. POST /api/ingest/upload    — 文档上传入库
  6. POST /api/qa/ask           — 智能诊断问答
  7. GET  /api/admin/stats      — 系统统计
  8. GET  /api/admin/graph      — 图谱数据（ECharts）
  9. GET  /api/admin/graph/node/{name} — 节点详情
  10. DELETE /api/admin/graph/node/{name} — 软删除节点
  11. GET  /api/documents        — 文档列表
  12. GET  /api/system           — 系统信息
  13. GET  /api/health           — 健康检查
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from utils.log import logger

# 新模块 imports
from core.supervisor import MultiAgentSupervisor, get_supervisor
from core.query_agent import QueryUnderstandingAgent
from core.diagnostic_agent import DiagnosticReasoningAgent
from core.verifier import VerificationAgent
from core.report_agent import ReportGenerationAgent

# 使用升级版混合检索引擎 (借鉴 MultiAgenticRAG)
from retrieval.hybrid_engine import HybridRetrievalEngine

from knowledge.entity_extractor import EntityExtractor

from services.vector_store import VectorStoreService
from services.knowledge_graph import KnowledgeGraphService
from services.graph_admin import GraphAdminService
from services.case_base import CaseBase
from services.sample_data import seed_data, reset_data

from chat_store import chat_store

# 兼容旧 DocParser 的上传管道
from agents.doc_parser_agent import DocParserAgent

# ── Globals ─────────────────────────────────────────────────────

_start_time = time.time()
kg_available = False
vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
graph_admin = GraphAdminService(knowledge_graph)
doc_parser = DocParserAgent()

supervisor: MultiAgentSupervisor | None = None
hybrid_engine: HybridRetrievalEngine | None = None
case_base = CaseBase()

# ── Simple user store (demo mode, replace with DB in production) ──

_users: dict[str, dict] = {"admin": {"password": "admin123", "role": "admin"}}
_tokens: dict[str, str] = {}  # token → username


def _make_token(username: str) -> str:
    raw = f"{username}:{time.time()}:{os.urandom(8).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supervisor, hybrid_engine

    os.makedirs(settings.upload_dir, exist_ok=True)

    # Initialize vector store
    try:
        await vector_store.init()
        logger.info("VectorStore initialized")
    except Exception:
        logger.exception("VectorStore init failed, service degraded")

    # Initialize knowledge graph
    global kg_available
    kg_available = False
    try:
        await knowledge_graph.init()
        kg_available = True
        logger.info("KnowledgeGraph initialized — %s", knowledge_graph.backend_name)
    except Exception:
        logger.exception("KnowledgeGraph init failed for configured backend")

    # Build multi-agent supervisor with hybrid engine
    sv = get_supervisor()
    hybrid_engine = HybridRetrievalEngine(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph if kg_available else None,
    )
    sv.build_graph(
        query_agent=QueryUnderstandingAgent(),
        hybrid_engine=hybrid_engine,
        diagnostic_agent=DiagnosticReasoningAgent(),
        verifier=VerificationAgent(),
        report_agent=ReportGenerationAgent(),
    )
    supervisor = sv
    logger.info("MultiAgentSupervisor built")

    # Initial BM25 index from existing vector store
    try:
        await hybrid_engine.refresh_bm25()
    except Exception:
        logger.exception("BM25 initial index failed")

    yield

    await knowledge_graph.close()
    logger.info("Server shutdown complete")


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="AeroDiagnosis — 航空发动机气路故障多智能体协同诊断平台",
    description="多Agent编排 + 向量/BM25/图谱三路混合检索 + 专家诊断推理 + 实体级知识图谱交互",
    version="2.0.0",
    lifespan=lifespan,
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── Request / Response Models ────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenCheckRequest(BaseModel):
    token: str


class LogoutRequest(BaseModel):
    token: str


class AccountUpdateRequest(BaseModel):
    username: str
    password: str | None = None


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    intent: str
    sources: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]] | None = None


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


# ── Auth Endpoints ───────────────────────────────────────────────

@app.post("/api/auth/login", tags=["认证"])
async def auth_login(req: LoginRequest):
    """用户登录，返回 token"""
    user = _users.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _make_token(req.username)
    _tokens[token] = req.username
    logger.info(f"User '{req.username}' logged in")
    return {"token": token, "username": req.username, "role": user.get("role", "user")}


@app.get("/api/auth/check", tags=["认证"])
async def auth_check(token: str):
    """验证 token 是否有效"""
    username = _tokens.get(token)
    if username:
        return {"valid": True, "username": username}
    return {"valid": False}


@app.post("/api/auth/logout", tags=["认证"])
async def auth_logout(req: LogoutRequest):
    """登出，清除 token"""
    username = _tokens.pop(req.token, None)
    if username:
        logger.info(f"User '{username}' logged out")
    return {"status": "ok"}


@app.post("/api/auth/account", tags=["认证"])
async def auth_update_account(req: AccountUpdateRequest, request: Request):
    """修改账户信息"""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    current_username = _tokens.get(token)
    if not current_username:
        raise HTTPException(status_code=401, detail="未登录")

    user = _users.get(current_username)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if req.username != current_username:
        # Change username
        _users[req.username] = user
        del _users[current_username]
        _tokens[token] = req.username

    if req.password:
        _users[req.username]["password"] = req.password

    logger.info(f"Account updated for '{req.username}'")
    return {"status": "ok", "username": req.username}


# ── Ingest Endpoints ─────────────────────────────────────────────

@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)):
    """上传并解析文档，自动入库到向量库、BM25索引和知识图谱"""
    save_path = os.path.join(settings.upload_dir, file.filename or "unknown")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 1. 文档解析
    chunks = await doc_parser.parse(save_path)
    logger.info(f"Ingest: parsed {len(chunks)} chunks from {file.filename}")

    # 2. 向量入库
    chunks_count = 0
    try:
        chunks_count = await vector_store.add_chunks(chunks)
    except Exception as e:
        logger.error(f"Vector store failed: {e}")

    # 3. 实体抽取 & 图谱入库
    entities_count = 0
    relations_count = 0
    try:
        extractor = EntityExtractor()
        result = await extractor.extract_batch(chunks)
        for e in result.get("entities", []):
            from agents.knowledge_extract_agent import Entity
            entity = Entity(name=e["name"], type=e["type"], description=e.get("description", ""))
            await knowledge_graph.upsert_entity(entity, source=save_path)
            entities_count += 1
        for r in result.get("relations", []):
            from agents.knowledge_extract_agent import Relation
            relation = Relation(
                head=r["head"], relation=r["relation"], tail=r["tail"],
                confidence=r.get("confidence", 0.9),
            )
            await knowledge_graph.add_relation(relation, source=save_path)
            relations_count += 1
        logger.info(f"Ingest: {entities_count} entities, {relations_count} relations extracted")
    except Exception as e:
        logger.error(f"Entity extraction / graph ingest failed: {e}")

    # 4. 刷新 BM25 索引
    try:
        if hybrid_engine:
            await hybrid_engine.refresh_bm25()
    except Exception as e:
        logger.error(f"BM25 refresh failed: {e}")

    return IngestResponse(
        file_name=file.filename or "unknown",
        chunks_count=chunks_count,
        entities_count=entities_count,
        relations_count=relations_count,
        status="success",
    )


# ── QA Endpoint ──────────────────────────────────────────────────

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能诊断"])
async def ask_question(req: QuestionRequest):
    """智能故障诊断问答 — 多Agent协同 + 三路混合检索"""
    if not supervisor:
        raise HTTPException(status_code=503, detail="诊断引擎未初始化")

    try:
        result = await supervisor.run(req.question)
    except Exception as e:
        logger.error(f"QA failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"诊断流程异常: {str(e)}")

    # 自动记录诊断案例
    intent = result.get("intent", "chat")
    if intent in ("fault_diagnosis", "knowledge_qa"):
        try:
            await case_base.add_case(
                question=req.question,
                intent=intent,
                hypotheses=result.get("hypotheses", []),
                final_answer=result.get("final_answer", ""),
                sources=result.get("sources", []),
            )
        except Exception:
            pass

    return QuestionResponse(
        question=req.question,
        answer=result.get("final_answer", "诊断未能生成答案"),
        intent=result.get("intent", "chat"),
        sources=result.get("sources", []),
        hypotheses=result.get("hypotheses"),
    )


# ── Admin Endpoints ──────────────────────────────────────────────

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取系统统计信息"""
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats() if kg_available else {"total_entities": 0, "total_relations": 0, "status": "neo4j_unavailable"}
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


@app.get("/api/admin/graph", tags=["图谱管理"])
async def get_graph(limit: int = 100):
    """获取全图谱数据（ECharts 可视化），过滤已删除节点"""
    try:
        data = await graph_admin.get_graph_for_viz(limit=limit)
        return data
    except Exception as e:
        logger.error(f"Graph viz failed: {e}")
        import traceback; traceback.print_exc()
        return {"nodes": [], "edges": []}


@app.get("/api/admin/graph/node/{name:path}", tags=["图谱管理"])
async def get_graph_node(name: str):
    """获取实体三层详情"""
    try:
        detail = await graph_admin.get_entity_detail(name)
        if not detail.get("entity"):
            raise HTTPException(status_code=404, detail=f"实体 '{name}' 不存在")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph node detail failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/graph/node/{name:path}", tags=["图谱管理"])
async def delete_graph_node(name: str):
    """软删除图谱实体"""
    try:
        result = await graph_admin.delete_entity(name)
        return result
    except Exception as e:
        logger.error(f"Graph node delete failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Documents Endpoint ──────────────────────────────────────────

@app.get("/api/documents", tags=["文档管理"])
async def list_documents():
    """列出已上传的文档"""
    docs = []
    upload_dir = settings.upload_dir
    if os.path.isdir(upload_dir):
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if os.path.isfile(fpath):
                size_kb = round(os.path.getsize(fpath) / 1024, 1)
                docs.append({
                    "name": fname,
                    "size_kb": size_kb,
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(fpath))),
                })
    return {"documents": docs}


# ── System Endpoint ─────────────────────────────────────────────

@app.get("/api/system", tags=["系统管理"])
async def get_system_info():
    """获取系统信息"""
    uptime_seconds = int(time.time() - _start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {
        "backend": vector_store.backend_name,
        "python_version": sys.version.split()[0],
        "api_version": "2.0.0",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": uptime_seconds,
    }


# ── Health ──────────────────────────────────────────────────────

@app.get("/api/health", tags=["系统管理"])
async def health():
    return {"status": "ok", "service": "AeroDiagnosis", "version": "2.0.0"}


# ── Seed / Reset ───────────────────────────────────────────────

@app.post("/api/admin/seed", tags=["系统管理"])
async def seed_demo_data():
    """一键注入航空发动机故障诊断演示数据"""
    try:
        result = await seed_data(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            hybrid_engine=hybrid_engine,
        )
        return result or {"status": "ok"}
    except Exception as e:
        logger.exception("Seed failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/reset", tags=["系统管理"])
async def reset_demo_data():
    """清除演示数据"""
    try:
        result = await reset_data(
            vector_store=vector_store, knowledge_graph=knowledge_graph
        )
        return result or {"status": "ok"}
    except Exception as e:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Case Base ──────────────────────────────────────────────────

@app.get("/api/cases", tags=["案例库"])
async def get_cases_stats():
    """获取案例库统计信息"""
    return await case_base.get_stats()


@app.get("/api/cases/list", tags=["案例库"])
async def list_cases(page: int = 1, page_size: int = 20, search: str = "", tag: str = ""):
    """分页查询案例列表, 支持关键词搜索和标签过滤"""
    return await case_base.list_cases(page=page, page_size=page_size, search=search, tag=tag)


@app.get("/api/cases/tags", tags=["案例库"])
async def get_case_tags():
    """获取所有案例标签"""
    return {"tags": await case_base.get_all_tags()}


@app.get("/api/cases/{case_id}", tags=["案例库"])
async def get_single_case(case_id: str):
    """获取单个案例详情"""
    case = await case_base.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"案例 '{case_id}' 不存在")
    return case


@app.post("/api/cases/search", tags=["案例库"])
async def search_similar_cases(req: QuestionRequest):
    """搜索与给定问题最相似的历史案例"""
    similar = await case_base.search_similar(req.question, top_k=5)
    return {"query": req.question, "similar": similar}


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
