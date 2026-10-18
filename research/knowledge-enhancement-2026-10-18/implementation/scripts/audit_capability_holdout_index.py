"""Independently verify frozen Chinese vectors and source payloads, without questions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.ports import GraphNode, VectorChunk
from aerodiagnosis.retrieval.graph_paths import _quote_valid, entity_identity

ROOT = Path("evaluation/retrieval_capability_holdout_v1_candidate")
OUT = Path(".runtime/holdout-capability-v1")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path = OUT / "index.sqlite"
    before = sha(path)
    corpus = read(ROOT / "corpus-final.json")
    assert before == corpus["reviewed_index_sha256"]
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    active = set(corpus["active_versions"])
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    maximum = 0.0
    count = 0
    tokens = {}
    sidecar_rows = 0
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("select * from vector_chunks order by chunk_id"):
            if row["version_id"] not in active:
                continue
            c = chunks[row["chunk_id"]]
            assert (row["document_id"], row["version_id"], row["content"]) == (
                c.document_id,
                c.version_id,
                c.content,
            )
            metadata = json.loads(row["metadata_json"])
            authoritative = json.loads(json.dumps(c.metadata))
            coords = authoritative.get("locator", {}).get("coordinates", {})
            if "canonical_sha256" in coords and "canonical_sha256" not in metadata.get(
                "locator", {}
            ).get("coordinates", {}):
                assert not c.document_id.startswith("cap1-")
                coords.pop("canonical_sha256")
                sidecar_rows += 1
            assert metadata == authoritative
            text = metadata["display_content_zh"]
            assert (
                metadata["display_translation_source_sha256"]
                == hashlib.sha256(c.content.encode()).hexdigest()
            )
            marker = metadata["vector_embedding_input"]
            assert marker == {
                "policy": "chinese_display@1",
                "provider": provider.name,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            stored = json.loads(row["embedding_json"])
            expected = provider.embed(text)
            assert len(stored) == len(expected) == 384
            error = max(abs(a - b) for a, b in zip(stored, expected, strict=True))
            maximum = max(maximum, error)
            assert error < 1e-7
            tokens[c.chunk_id] = len(provider.counter.encode(text).ids)
            count += 1
            if count % 200 == 0:
                print("Recomputed", count, "/", len(chunks), flush=True)
    assert count == len(chunks) and sha(path) == before
    graph = read(ROOT / "source-graph-final.json")
    proofs = 0
    for n in graph["nodes"]:
        node = GraphNode(**n)
        identity = entity_identity(node, chunks)
        assert identity.startswith("entity:")
        assert _quote_valid(
            chunks[n["properties"]["entity_link"]["chunk_id"]], n["properties"]["entity_link"]
        )
        proofs += 1
    for edge in graph["edges"]:
        for proof in [edge["properties"], *edge["properties"].get("supporting_evidence", [])]:
            assert _quote_valid(chunks[proof["chunk_id"]], proof)
            proofs += 1
    result = {
        "vectors_recomputed": count,
        "dimensions": 384,
        "max_abs_error": maximum,
        "all_source_payloads_and_chinese_provenance_equal": True,
        "old_metadata_canonical_hash_sidecar_rows": sidecar_rows,
        "graph_evidence_proofs_checked": proofs,
        "source_only_no_questions_read": True,
        "index_sha256_unchanged": True,
        "index_sha256": before,
        "maximum_chunk_tokens": max(tokens.values()),
        "chunks_larger_than_context4096": sum(v > 4096 for v in tokens.values()),
        "api_calls": 0,
    }
    (OUT / "index-independent-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
