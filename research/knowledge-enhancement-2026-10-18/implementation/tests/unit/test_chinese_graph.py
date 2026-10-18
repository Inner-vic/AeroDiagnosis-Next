from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerodiagnosis.adapters.llm.chinese_graph import ChineseGraphDisplay
from aerodiagnosis.adapters.llm.research_client import ResearchCallError, ResearchClient
from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.ports.graph import GraphNode


def test_chinese_display_preserves_original_identity_and_exact_source(tmp_path: Path) -> None:
    def transport(body):
        payload = json.loads(body["messages"][1]["content"])
        response = (
            {"labels": {key: "压气机污染" for key in payload["items"]}}
            if isinstance(next(iter(payload["items"].values())), dict)
            else {"translations": {key: "压气机污染" for key in payload["items"]}}
        )
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 20, "completion_tokens": 50},
            "choices": [
                {
                    "message": {"content": json.dumps(response)},
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(
        api_key="test", runtime=tmp_path / "budget", cap_cny=5, transport=transport
    )
    original = GraphNode(
        "n",
        "Compressor contamination",
        "Cause",
        "Deposits may reduce efficiency.",
        "document:d",
        "v",
        {"evidence_text": "Deposits may reduce efficiency."},
    )
    translated = ChineseGraphDisplay(client, stage="pilot").enrich([original])[0]
    assert translated.node_id == original.node_id and translated.name == "压气机污染"
    assert translated.description == "压气机污染"
    assert translated.properties["evidence_text"] == original.properties["evidence_text"]
    assert translated.properties["name_zh"] == "压气机污染"
    assert translated.properties["display_translation"]["review_status"] == "machine_translation"
    database = SQLiteDatabase(tmp_path / "graph.sqlite")
    database.migrate()
    graph = SQLiteGraphStore(tmp_path / "graph.sqlite")
    graph.upsert_nodes([translated])
    assert graph.search_nodes("压气机")[0].node_id == "n"
    assert graph.search_nodes("压气机污染")[0].node_id == "n"


def test_wrong_translation_ids_cannot_be_assigned_to_other_entities(tmp_path: Path) -> None:
    def transport(body):
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "choices": [
                {
                    "message": {"content": '{"translations":{"wrong":"污染"}}'},
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(api_key="test", runtime=tmp_path, cap_cny=5, transport=transport)
    with pytest.raises(ResearchCallError, match="IDs"):
        ChineseGraphDisplay(client, stage="pilot").enrich(
            [GraphNode("n", "Deposits", "Cause", "", "document:d", "v")]
        )


def test_oversize_concept_is_rewritten_once_without_clipping_source(tmp_path: Path) -> None:
    calls = []

    def transport(body):
        calls.append(body)
        payload = json.loads(body["messages"][1]["content"])
        label = (
            "这是过长的整句名称需要重新给出准确而短的专业名词" if len(calls) == 1 else "压气机污染"
        )
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"labels": {k: label for k in payload["items"]}})
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(
        api_key="test", runtime=tmp_path / "budget", cap_cny=5, transport=transport
    )
    original = GraphNode("n", "Deposits", "Cause", "Deposits may reduce efficiency.", "d", "v")
    result = ChineseGraphDisplay(client, stage="pilot").compact_entities([original])[0]
    assert result.name == "压气机污染" and result.description == original.description
    assert result.node_id == original.node_id
    assert result.properties["concept_label_shortening_retry"] is True
    assert len(calls) == 2 and client.summary()["provider_call_count"] == 2


def test_persistent_oversize_concept_stops_after_one_billed_retry(tmp_path: Path) -> None:
    def transport(body):
        payload = json.loads(body["messages"][1]["content"])
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"labels": {k: "x" * 21 for k in payload["items"]}})
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(
        api_key="test", runtime=tmp_path / "budget", cap_cny=5, transport=transport
    )
    with pytest.raises(ResearchCallError, match="1-20"):
        ChineseGraphDisplay(client, stage="pilot").compact_entities(
            [GraphNode("n", "Deposits", "Cause", "", "d", "v")]
        )
    assert client.summary()["provider_call_count"] == 2


def test_code_list_label_keeps_full_part_numbers_in_description(tmp_path: Path) -> None:
    def transport(body):
        payload = json.loads(body["messages"][1]["content"])
        return {
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"labels": {k: "1668M37G02和1668M37G03" for k in payload["items"]}}
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    client = ResearchClient(api_key="test", runtime=tmp_path, cap_cny=5, transport=transport)
    original = GraphNode(
        "n",
        "1668M37G02 and 1668M37G03",
        "Parameter",
        "The part numbers are 1668M37G02 and 1668M37G03.",
        "d",
        "v",
    )
    result = ChineseGraphDisplay(client, stage="pilot").compact_entities([original])[0]
    assert result.name == "部件件号组合"
    assert result.description == original.description and "1668M37G03" in result.description
    assert result.node_id == original.node_id
