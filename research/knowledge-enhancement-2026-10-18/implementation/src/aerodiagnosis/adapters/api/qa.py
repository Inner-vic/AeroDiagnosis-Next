# ruff: noqa: RUF001 -- Chinese user-facing punctuation.
"""Knowledge QA HTTP surface, active-source viewing and budget-respecting model binding."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from aerodiagnosis.adapters.persistence import (
    create_embedding_provider,
    create_graph_store,
    create_vector_store,
)
from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_memory import SQLiteConversationMemory
from aerodiagnosis.application.knowledge_applicability import assess_applicability, permits
from aerodiagnosis.application.knowledge_qa_live import GatedKnowledgeQA, display_text
from aerodiagnosis.bootstrap import Application
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import LanguageModel, MessageRole, VectorChunk

from .dependencies import get_application
from .rate_limit import require_diagnosis_rate_limit
from .schemas import ProviderConfiguration

router = APIRouter(prefix="/api/qa", tags=["knowledge-answer"])


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    knowledge_context: KnowledgeQueryContext = Field(default_factory=KnowledgeQueryContext)
    provider: ProviderConfiguration | None = None
    evidence_only: bool = False


class QACatalog:
    def __init__(self, application: Application) -> None:
        self.application = application

    def __call__(self, context: KnowledgeQueryContext) -> dict[str, VectorChunk]:
        records = {}
        for doc in self.application.browse_knowledge.execute():
            if context.document_id and doc.document_id != context.document_id:
                continue
            for version in doc.versions:
                if version.version_id != doc.active_version_id:
                    continue
                metadata = version.metadata.model_dump(mode="json")
                status = assess_applicability(metadata, context)
                if permits(status, context):
                    records[version.version_id] = (
                        doc.document_id,
                        doc.display_name,
                        metadata,
                        status,
                    )
        if not records:
            return {}
        with SQLiteDatabase(self.application.settings.database_path).connect() as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_lexical_chunks WHERE version_id IN ("
                + ",".join("?" for _ in records)
                + ")",
                tuple(records),
            ).fetchall()
        output = {}
        for row in rows:
            document, title, metadata, status = records[row["version_id"]]
            if row["document_id"] != document:
                continue
            c = VectorChunk(
                row["chunk_id"],
                document,
                row["version_id"],
                row["content"],
                {
                    **json.loads(row["metadata_json"]),
                    "qa_title": title,
                    "qa_document_metadata": metadata,
                    "qa_source_url": metadata.get("source_url"),
                    "qa_applicability": status,
                },
            )
            try:
                display_text(c)
            except ValueError:
                continue
            output[c.chunk_id] = c
        return output


def create_qa(application: Application) -> GatedKnowledgeQA:
    settings = application.settings
    provider = create_embedding_provider(settings)
    counter = getattr(provider, "counter", None)
    return GatedKnowledgeQA(
        sparse=SQLiteLexicalStore(settings.database_path),
        dense=create_vector_store(settings),
        graph=create_graph_store(settings),
        catalog=QACatalog(application),
        token_count=(lambda text: len(counter.encode(text).ids))
        if counter
        else (lambda text: len(text.encode())),
    )


def make_model(body: QARequest, application: Application) -> LanguageModel | None:
    if body.evidence_only:
        return None
    settings = application.settings
    if settings.knowledge_extractor_backend != "langextract":
        if not body.provider and not settings.default_provider_available:
            return None
        from .routers import _provider_model

        return _provider_model(body.provider, application)
    if body.provider:
        config = body.provider
        if str(config.base_url).rstrip("/") not in {
            "https://api.deepseek.com",
            "https://api.deepseek.com/v1",
        } or config.model not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise HTTPException(status_code=422, detail="当前研究配置仅允许登记预算的DeepSeek模型")
        key, model = config.api_key.get_secret_value(), config.model
    else:
        if not settings.default_provider_available:
            return None
        assert settings.default_llm_api_key and settings.default_llm_model
        key, model = settings.default_llm_api_key.get_secret_value(), settings.default_llm_model
    from aerodiagnosis.adapters.llm.budgeted import BudgetedLanguageModel
    from aerodiagnosis.adapters.llm.research_client import ResearchClient

    return BudgetedLanguageModel(
        ResearchClient(
            api_key=key,
            runtime=settings.knowledge_budget_runtime,
            model=model,
            cap_cny=min(42.0, settings.knowledge_api_cap_cny),
        ),
        stage="generation_evaluation",
    )


@router.post("/answer")
def answer(
    body: QARequest,
    request: Request,
    application: Annotated[Application, Depends(get_application)],
    _limit: Annotated[None, Depends(require_diagnosis_rate_limit)],
) -> dict[str, Any]:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="问题不能为空")
    if body.session_id:
        try:
            application.conversation_sessions.messages(body.session_id, limit=1)
        except KeyError:
            raise HTTPException(status_code=404, detail="会话不存在") from None
    model = make_model(body, application)
    qa: GatedKnowledgeQA = request.app.state.knowledge_qa
    try:
        result = qa.answer(body.question.strip(), body.knowledge_context, model)
    except ValueError:
        raise HTTPException(
            status_code=503, detail="资料或索引校验未完成，请检查当前知识版本。"
        ) from None
    # Recheck visibility after model latency, before exposing any source-bound answer.
    current = QACatalog(application)(body.knowledge_context)
    if any(
        x["id"] not in current
        or current[x["id"]].version_id != x["version_id"]
        or hashlib.sha256(display_text(current[x["id"]]).encode()).hexdigest() != x["text_sha256"]
        for x in result["evidence"]
    ):
        raise HTTPException(status_code=409, detail="回答期间资料版本已更新，请重新提问。")
    if body.session_id:
        memory = SQLiteConversationMemory(application.settings.database_path)
        for role, content in [
            (MessageRole.USER, body.question),
            (MessageRole.ASSISTANT, json.dumps(result, ensure_ascii=False)),
        ]:
            memory.append(
                session_id=body.session_id,
                message_id=f"{result['run_id']}:{role.value}",
                role=role,
                content=content,
            )
    return result


@router.get("/evidence/{chunk_id}")
def evidence(
    chunk_id: str, application: Annotated[Application, Depends(get_application)]
) -> dict[str, Any]:
    c = QACatalog(application)(KnowledgeQueryContext()).get(chunk_id)
    if c is None:
        raise HTTPException(status_code=404, detail="该片段不属于当前活动资料版本")
    text = display_text(c)
    return {
        "chunk_id": c.chunk_id,
        "document_id": c.document_id,
        "version_id": c.version_id,
        "title": c.metadata.get("qa_title"),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_kind": "chinese_translation"
        if c.metadata.get("display_content_zh")
        else "canonical_source",
        "locator": c.metadata.get("locator", {}),
        "source_url": c.metadata.get("qa_source_url"),
    }
