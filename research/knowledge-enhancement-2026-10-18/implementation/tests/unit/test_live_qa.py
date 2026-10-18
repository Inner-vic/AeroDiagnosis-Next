# ruff: noqa: RUF001 -- Chinese UI punctuation.
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import aerodiagnosis.adapters.api.qa as qa_api
from aerodiagnosis.adapters.api.app import create_app
from aerodiagnosis.application.knowledge_qa_live import GatedKnowledgeQA, display_text
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import GraphEdge, GraphNode, LanguageModelError, VectorChunk, VectorMatch


class Store:
    backend_name = "fixture"

    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls = 0

    def search(self, query, *, top_k, active_version_ids):
        self.calls += 1
        return [m for m in self.hits if m.chunk.version_id in active_version_ids][:top_k]


class Model:
    identity = "deterministic-contract-test-not-quality-evaluation"

    def __init__(self, mode="ok"):
        self.mode = mode
        self.calls = 0

    def complete_json(self, task, payload):
        self.calls += 1
        if self.mode == "offline":
            raise LanguageModelError("unavailable")
        if task == "answer_knowledge":
            ss = payload["evidence"][0]["spans"][0]
            return {
                "claims": [
                    {
                        "text": "已经确定根因。" if self.mode == "unsupported" else ss["text"],
                        "support_span_ids": ["invented" if self.mode == "invented" else ss["id"]],
                    }
                ],
                "insufficient_evidence": False,
                "abstention_reason": "",
            }
        return {
            "sufficient": self.mode != "unsupported",
            "supported_claim_indexes": [0],
            "question_coverage": [
                {"requirement": "问题所问的事实", "covered": self.mode != "unsupported"}
            ],
        }


@pytest.fixture
def scenario():
    root = Path("evaluation/controlled_scenario_v2")
    data = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    doc = data["documents"][0]
    version = doc["version_id"]
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in data["chunks"] if c["version_id"] == version}
    nn = [GraphNode(**n) for n in graph["nodes"] if n["version_id"] == version]
    ee = [GraphEdge(**e) for e in graph["edges"] if e["version_id"] == version]

    class Graph:
        backend_name = "test_graph"

        def snapshot(self, *, active_version_ids, limit):
            return nn, ee

    names = {n.node_id: n.name for n in nn}
    rules = sorted(ee, key=lambda e: e.edge_id)[:3]
    query = f"从{names[rules[0].source_id]}到{names[rules[-1].target_id]}的因果路径是什么？"
    return chunks, Graph(), query


def service(scenario, text=False):
    chunks, graph, _ = scenario
    hits = [VectorMatch(next(iter(chunks.values())), 1.0)] if text else []
    return GatedKnowledgeQA(
        sparse=Store(hits),
        dense=Store(hits),
        graph=graph,
        catalog=lambda c: chunks,
        token_count=len,
    )


def test_complete_path_returned_with_source_proofs(scenario):
    r = service(scenario).retrieve(scenario[2], KnowledgeQueryContext())
    assert r["retrieval"]["graph_enabled"] and r["paths"]
    ids = {x["id"] for x in r["evidence"]}
    for p in r["paths"]:
        assert set(p["source_chunks"]) <= ids and p["hops"] in {2, 3}
        assert len(p["source_proofs"]) >= p["hops"]


def test_boundary_uses_text_without_graph(scenario):
    r = service(scenario, True).retrieve("是否一定能反推根因？", KnowledgeQueryContext())
    assert not r["retrieval"]["graph_enabled"] and r["evidence"] and not r["paths"]


def test_unknown_identifier_empty_and_no_search(scenario):
    s = service(scenario, True)
    r = s.answer("QB-99999999的因果路径？", KnowledgeQueryContext(), Model())
    assert r["insufficient_evidence"] and not r["evidence"]
    assert s.sparse.calls == s.dense.calls == 0


def test_graph_failure_preserves_text(scenario):
    s = service(scenario, True)

    class Broken:
        backend_name = "unavailable"

        def snapshot(self, **kw):
            raise OSError("do not expose this detail")

    s.graph = Broken()
    r = s.retrieve("机理路径", KnowledgeQueryContext())
    assert r["evidence"] and r["retrieval"]["reason"] == "graph_unavailable"
    assert "do not expose" not in json.dumps(r)


@pytest.mark.parametrize("mode", ["ok", "invented", "unsupported", "offline"])
def test_answer_citations_and_fail_closed(scenario, mode):
    s = service(scenario, True)
    r = s.answer("查看本资料事实。", KnowledgeQueryContext(), Model(mode))
    assert r["insufficient_evidence"] == (mode != "ok")
    if mode != "ok":
        assert not r["claims"]
    else:
        byid = {e["id"]: e for e in r["evidence"]}
        for claim in r["claims"]:
            for q in claim["citations"]:
                assert byid[q["chunk_id"]]["text"][q["start"] : q["end"]] == q["support_quote"]


def test_no_model_is_not_a_verified_answer(scenario):
    r = service(scenario, True).answer("查看事实", KnowledgeQueryContext(), None)
    assert r["insufficient_evidence"] and not r["claims"] and r["evidence"]


def test_stale_translation_rejected(scenario):
    c = next(iter(scenario[0].values()))
    with pytest.raises(ValueError):
        display_text(
            replace(
                c,
                metadata={
                    **c.metadata,
                    "display_content_zh": "错误译文",
                    "display_translation_source_sha256": "stale",
                },
            )
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = create_app(RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "qa.sqlite"))
    monkeypatch.setattr(
        qa_api, "make_model", lambda body, application: None if body.evidence_only else Model()
    )
    with TestClient(app) as client:
        application = app.state.application
        first = application.ingest_document.execute(
            display_name="中文资料.txt",
            content="压气机效率降低可能引起排气温度升高。必须结合检查证据判断。".encode(),
        )
        yield client, application, first


def test_http_answer_source_and_saved_session(client):
    c, a, _ = client
    session = a.conversation_sessions.create()
    r = c.post("/api/qa/answer", json={"question": "压气机效率降低", "session_id": session})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "evidence_ready"
    citation = body["claims"][0]["citations"][0]
    source = c.get("/api/qa/evidence/" + citation["chunk_id"])
    assert source.status_code == 200
    text = source.json()["text"]
    assert text[citation["start"] : citation["end"]] == citation["support_quote"]
    history = a.conversation_sessions.messages(session)
    assert len(history) == 2 and json.loads(history[-1].content)["kind"] == "grounded_qa"


def test_http_source_not_active_after_update(client):
    c, a, first = client
    body = c.post("/api/qa/answer", json={"question": "压气机效率", "evidence_only": True}).json()
    key = body["evidence"][0]["id"]
    a.ingest_document.execute(
        display_name="中文资料.txt",
        document_id=first.document_id,
        content="替换后的新版本说明。".encode(),
    )
    assert c.get("/api/qa/evidence/" + key).status_code == 404


def test_http_document_scope_and_missing_session(client):
    c, _, _ = client
    r = c.post(
        "/api/qa/answer",
        json={"question": "压气机效率", "knowledge_context": {"document_id": "missing"}},
    )
    assert r.status_code == 200 and r.json()["insufficient_evidence"] and not r.json()["evidence"]
    assert (
        c.post("/api/qa/answer", json={"question": "压气机", "session_id": "missing"}).status_code
        == 404
    )
    assert c.post("/api/qa/answer", json={"question": "  "}).status_code == 422


def test_http_evidence_only_has_no_claims(client):
    c, _, _ = client
    r = c.post("/api/qa/answer", json={"question": "压气机效率", "evidence_only": True}).json()
    assert r["evidence"] and not r["claims"] and r["generation"]["model_calls"] == 0


def test_http_strict_unknown_applicability_excluded(client):
    c, _, _ = client
    r = c.post(
        "/api/qa/answer",
        json={
            "question": "压气机效率",
            "knowledge_context": {"engine_model": "X", "require_known": True},
        },
    ).json()
    assert not r["evidence"] and r["insufficient_evidence"]


def test_relation_qualifier_conflict_closes_graph(scenario):
    _chunks, graph, query = scenario
    nn, ee = graph.snapshot(active_version_ids=frozenset(), limit=2000)
    revised = [
        replace(
            e, properties={**e.properties, "qualifiers": {"engine_models": [{"value": "engine-b"}]}}
        )
        for e in ee
    ]

    class Restricted:
        backend_name = "restricted"

        def snapshot(self, **kw):
            return nn, revised

    qa = service(scenario, True)
    qa.graph = Restricted()
    r = qa.retrieve(query, KnowledgeQueryContext(engine_model="engine-a"))
    assert not r["retrieval"]["graph_enabled"] and not r["paths"]


def test_whole_context_over_budget_does_not_call_model(scenario):
    qa = service(scenario, True)
    qa.tokens = lambda text: 100000
    model = Model()
    r = qa.answer("查看资料", KnowledgeQueryContext(), model)
    assert r["insufficient_evidence"] and model.calls == 0


def test_unicode_citation_offsets_are_source_codepoints(scenario):
    qa = service(scenario, True)
    c = VectorChunk("unicode", "doc", "v", "叶片💡损坏可能导致性能下降。", {})
    qa.catalog = lambda ctx: {c.chunk_id: c}
    qa.sparse = Store([VectorMatch(c, 1.0)])
    qa.dense = Store([VectorMatch(c, 1.0)])
    r = qa.answer("叶片损坏", KnowledgeQueryContext(), Model())
    quote = r["claims"][0]["citations"][0]
    assert c.content[quote["start"] : quote["end"]] == quote["support_quote"]


def test_http_complete_graph_path_and_source_view(client, scenario):
    from aerodiagnosis.adapters.persistence import create_vector_store
    from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
    from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
    from aerodiagnosis.ingestion.manifest import DocumentManifest, VersionStatus

    c, application, _ = client
    chunks, graph, query = scenario
    first = next(iter(chunks.values()))
    manifest = DocumentManifest(application.settings.database_path)
    manifest.register_document(first.document_id, "自建场景功能验收资料（规则构造）")
    manifest.create_version(
        document_id=first.document_id,
        version_id=first.version_id,
        parser_version="controlled-functional-fixture",
        content_hash=first.metadata["locator"]["coordinates"]["canonical_sha256"],
    )
    for state in (VersionStatus.PARSED, VersionStatus.INDEXING, VersionStatus.STAGED):
        manifest.transition(first.version_id, state)
    manifest.activate(first.version_id)
    SQLiteLexicalStore(application.settings.database_path).upsert(list(chunks.values()))
    create_vector_store(application.settings).upsert(list(chunks.values()))
    nn, ee = graph.snapshot(active_version_ids=frozenset(), limit=2000)
    store = SQLiteGraphStore(application.settings.database_path)
    store.upsert_nodes(nn)
    store.upsert_edges(ee)
    # Isolate graph participation; ranking parity is checked separately.
    c.app.state.knowledge_qa.sparse = Store()
    c.app.state.knowledge_qa.dense = Store()
    result = c.post(
        "/api/qa/answer",
        json={
            "question": query,
            "knowledge_context": {"document_id": first.document_id},
            "evidence_only": True,
        },
    )
    assert result.status_code == 200
    body = result.json()
    assert body["retrieval"]["graph_enabled"] and body["paths"]
    for path in body["paths"]:
        assert path["complete_in_context"]
        for key in path["source_chunks"]:
            assert c.get("/api/qa/evidence/" + key).status_code == 200
