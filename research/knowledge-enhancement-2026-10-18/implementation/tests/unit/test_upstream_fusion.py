from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aerodiagnosis.adapters.api.app import create_app
from aerodiagnosis.adapters.llm.research_client import ResearchClient
from aerodiagnosis.application.knowledge_applicability import assess_assertion_applicability
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ingestion.source_archive import SourceArchive


@pytest.mark.parametrize("scope", ["engine_models", "flight_phases"])
def test_explicit_assertion_conflict_cannot_be_overridden_by_generic_document(scope: str) -> None:
    context = KnowledgeQueryContext(engine_model="cfm56", flight_phase="cruise")
    props = {
        "qualifiers": {
            scope: [{"value": "leap" if scope == "engine_models" else "takeoff", "quote": "source"}]
        }
    }
    assert (
        assess_assertion_applicability(
            props, {"applicability": {"scope_status": "generic"}}, context
        )
        == "conflict"
    )


def test_original_document_download_requires_operator_and_correct_version(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path, database_path=tmp_path / "db.sqlite", operator_token="test-operator"
    )
    with TestClient(create_app(settings)) as http:
        body = {
            "display_name": "source.txt",
            "content_base64": base64.b64encode(
                b"Compressor deposits cause poor acceleration."
            ).decode(),
        }
        assert http.post("/api/documents/upload", json=body).status_code == 401
        response = http.post(
            "/api/documents/upload",
            json=body,
            headers={"X-AeroDiagnosis-Operator-Token": "test-operator"},
        )
        assert response.status_code == 201
        record = response.json()
        url = f"/api/documents/{record['document_id']}/versions/{record['version_id']}/source"
        assert http.get(url).status_code == 401
        allowed = http.get(url, headers={"X-AeroDiagnosis-Operator-Token": "test-operator"})
        assert allowed.status_code == 200 and allowed.content == base64.b64decode(
            body["content_base64"]
        )
        assert "attachment" in allowed.headers["content-disposition"]
        assert (
            http.get(
                url.replace(record["document_id"], "other"),
                headers={"X-AeroDiagnosis-Operator-Token": "test-operator"},
            ).status_code
            == 404
        )
        body["content_base64"] = "not base64!"
        assert (
            http.post(
                "/api/documents/upload",
                json=body,
                headers={"X-AeroDiagnosis-Operator-Token": "test-operator"},
            ).status_code
            == 422
        )


def test_source_archive_rejects_tampering_and_path_escape(tmp_path: Path) -> None:
    archive = SourceArchive(tmp_path)
    archive.store("a" * 64, "source.pdf", b"raw", b"text", {"docling.json": b"{}"})
    with pytest.raises(ValueError, match="identity"):
        archive.get("../outside")
    (tmp_path / ("a" * 64) / "original.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        archive.get("a" * 64)


@pytest.mark.skipif(importlib.util.find_spec("langextract") is None, reason="Upstream extra absent")
def test_real_langextract_sdk_maps_grounded_assertion_to_graph(tmp_path: Path) -> None:
    from aerodiagnosis.adapters.llm.langextract_graph import LangExtractGraphExtractor

    text = "Deposits may cause poor acceleration."
    assertion = {
        "subject": {"surface": "Deposits", "entity_type": "Cause"},
        "predicate": "CAUSES",
        "object": {"surface": "poor acceleration", "entity_type": "Symptom"},
        "polarity": "positive",
        "certainty": "hypothesis",
        "qualifiers": {"certainty_quote": "may"},
    }

    def transport(body):
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 100, "completion_tokens": 300},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "extractions": [
                                    {
                                        "assertion": text,
                                        "assertion_attributes": {
                                            "assertion_json": json.dumps(assertion)
                                        },
                                    }
                                ]
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(
        api_key="test-key", runtime=tmp_path / "ledger", cap_cny=5, transport=transport
    )
    extractor = LangExtractGraphExtractor(client, audit_root=tmp_path / "audit", stage="pilot")
    nodes, edges = extractor.extract(
        source_ref="document:d",
        version_id="a" * 64,
        chunks=[text],
        chunk_ids=["c" * 64],
        chunk_locators=[
            {
                "kind": "span",
                "coordinates": {
                    "start": 0,
                    "end": len(text),
                    "canonical_sha256": hashlib.sha256(text.encode()).hexdigest(),
                },
            }
        ],
    )
    assert len(nodes) == 2 and len(edges) == 1
    edge = edges[0]
    assert edge.relation == "CAUSES" and edge.properties["certainty"] == "hypothesis"
    assert edge.properties["evidence_text"] == text
    assert edge.properties["canonical_end"] == len(text)
    assert client.summary()["provider_call_count"] == 1


@pytest.mark.skipif(importlib.util.find_spec("docling") is None, reason="Upstream extra absent")
def test_real_docling_keeps_table_header_and_exact_snapshot_offsets() -> None:
    from aerodiagnosis.adapters.docling_parser import DoclingParser

    model = Path(__file__).resolve().parents[2] / ".runtime/models/multilingual-e5-small"
    if not model.exists():
        pytest.skip("Explicit local tokenizer absent")
    parsed = DoclingParser(".html", tokenizer_path=model).parse(
        b"<h1>Compressor</h1><p>Deposits may cause poor acceleration.</p>"
        b"<table><tr><th>Parameter</th><th>Trend</th></tr>"
        b"<tr><td>EGT</td><td>high</td></tr></table>"
    )
    assert b"Parameter" in parsed.source_artifacts["docling.json"]
    assert b"EGT" in parsed.canonical_content and b"Trend" in parsed.canonical_content
    assert any(c.locator.coordinates["contains_table"] for c in parsed.chunks)
    for chunk in parsed.chunks:
        start, end = chunk.locator.coordinates["start"], chunk.locator.coordinates["end"]
        assert parsed.canonical_content.decode()[start:end] == chunk.content
