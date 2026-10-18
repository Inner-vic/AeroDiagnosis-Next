from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aerodiagnosis.adapters.llm.chinese_document import ChineseDocumentDisplay
from aerodiagnosis.adapters.llm.research_client import ResearchClient
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import VectorChunk


def test_chinese_knowledge_is_searchable_without_changing_evidence_identity(tmp_path: Path) -> None:
    def transport(body):
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 30, "completion_tokens": 20},
            "choices": [
                {
                    "message": {"content": json.dumps({"text_zh": "积垢可能降低效率。"})},
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(
        api_key="test", runtime=tmp_path / "budget", cap_cny=5, transport=transport
    )
    chunk = VectorChunk("c", "d", "v", "Deposits may reduce efficiency.", {"evidence_id": "stable"})
    translator = ChineseDocumentDisplay(client, stage="pilot")
    result = translator.enrich([chunk])[0]
    assert result.chunk_id == chunk.chunk_id and result.content == chunk.content
    assert result.metadata["evidence_id"] == "stable"
    assert result.metadata["display_content_zh"] == "积垢可能降低效率。"
    assert (
        result.metadata["display_translation_source_sha256"]
        == hashlib.sha256(chunk.content.encode()).hexdigest()
    )
    assert translator.enrich([result])[0] == result
    assert client.summary()["provider_call_count"] == 1
    index = SQLiteLexicalStore(tmp_path / "index.sqlite")
    index.upsert([result])
    assert index.search("积垢")[0].chunk.chunk_id == "c"


def test_existing_chinese_material_does_not_call_external_translation(tmp_path: Path) -> None:
    client = ResearchClient(
        api_key="test",
        runtime=tmp_path / "budget",
        cap_cny=5,
        transport=lambda body: (_ for _ in ()).throw(AssertionError("unneeded call")),
    )
    chunk = VectorChunk("c", "d", "v", "积垢可能引起排气温度升高 EGT。", {})
    result = ChineseDocumentDisplay(client, stage="pilot").enrich([chunk])[0]
    assert result.metadata["display_content_zh"] == chunk.content
    assert client.summary()["provider_call_count"] == 0


def test_existing_wrong_technical_translation_is_repaired_before_indexing(tmp_path):
    calls = []

    def transport(body):
        calls.append(body)
        return {
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "choices": [
                {
                    "message": {"content": json.dumps({"text_zh": "低油位导致油泵气蚀。"})},
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(api_key="test", runtime=tmp_path, cap_cny=5, transport=transport)
    source = "Low oil level caused the pump to cavitate."
    chunk = VectorChunk("c", "d", "v", source, {"display_content_zh": "低油位导致油泵气塞。"})
    repaired = ChineseDocumentDisplay(client, stage="pilot").enrich([chunk])[0]
    assert repaired.content == source and repaired.chunk_id == "c"
    assert repaired.metadata["display_content_zh"] == "低油位导致油泵气蚀。"
    assert len(calls) == 1
