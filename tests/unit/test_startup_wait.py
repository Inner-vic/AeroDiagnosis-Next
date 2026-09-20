from __future__ import annotations

from aerodiagnosis.adapters.startup_wait import parse_bolt_uri, required_endpoints


def test_parse_bolt_uri_extracts_host_and_port() -> None:
    assert parse_bolt_uri("bolt://neo4j:7687") == ("neo4j", 7687)
    assert parse_bolt_uri("neo4j://neo4j:7687") == ("neo4j", 7687)


def test_required_endpoints_use_external_primary_environment(
    monkeypatch: object,
) -> None:
    env = {
        "AERODIAGNOSIS_VECTOR_BACKEND": "chroma_http",
        "AERODIAGNOSIS_GRAPH_BACKEND": "neo4j",
        "AERODIAGNOSIS_CHROMA_HOST": "chromadb",
        "AERODIAGNOSIS_CHROMA_PORT": "8000",
        "AERODIAGNOSIS_NEO4J_URI": "bolt://neo4j:7687",
    }
    monkeypatch.setattr("os.environ", env)

    endpoints = required_endpoints()

    assert ("chromadb", 8000) in endpoints
    assert ("neo4j", 7687) in endpoints
