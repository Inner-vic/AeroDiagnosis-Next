"""Compare fixed/structure chunk retrieval on a frozen real-source subset (candidate qrels)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder, SQLiteVectorStore
from aerodiagnosis.ingestion.parsers import PlainTextParser
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.metrics import ndcg_at_k
from aerodiagnosis.retrieval.reranker import LexicalReranker, RankedText

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def score(ranking: list[VectorMatch], qrels: dict[str, int]) -> dict[str, float]:
    relevant = {key for key, gain in qrels.items() if gain > 0}
    docs = unique([match.chunk.document_id for match in ranking])
    top_chunks = ranking[:5]
    top_docs_in_chunks = {match.chunk.document_id for match in top_chunks}
    first = next((i for i, doc in enumerate(docs, 1) if doc in relevant), None)
    return {
        "labeled_doc_recall_in_top5_chunks": len(top_docs_in_chunks & relevant) / len(relevant),
        "unique_documents_in_top5_chunks": float(len(top_docs_in_chunks)),
        "labeled_chunk_share_at5": sum(m.chunk.document_id in relevant for m in top_chunks) / 5,
        "labeled_doc_recall_at3": len(set(docs[:3]) & relevant) / len(relevant),
        "labeled_doc_hit_at1": float(bool(docs and docs[0] in relevant)),
        "labeled_doc_mrr": 1 / first if first else 0,
        "labeled_doc_ndcg_at3": ndcg_at_k(
            docs, {k: float(2**v - 1) for k, v in qrels.items()}, k=3
        ),
    }


class BM25:
    """Diagnostic offline comparator with the same tokenization; does not change production."""

    def __init__(self, chunks: list[VectorChunk]) -> None:
        self.chunks = chunks
        self.counts = [Counter(HashingEmbedder._tokens(chunk.content)) for chunk in chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.mean_length = sum(self.lengths) / len(self.lengths)
        self.df = Counter(token for counts in self.counts for token in counts)

    def search(self, query: str) -> list[VectorMatch]:
        terms = set(HashingEmbedder._tokens(query))
        output = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths, strict=True):
            total = 0.0
            for term in terms:
                freq = counts[term]
                if not freq:
                    continue
                idf = math.log(1 + (len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                total += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / self.mean_length))
            output.append(VectorMatch(chunk, total))
        return sorted(output, key=lambda match: (-match.score, match.chunk.chunk_id))


def run(snapshot_path: Path, queries_path: Path, output: Path) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    texts = {}
    cohorts = {}
    for doc in snapshot["documents"]:
        path = snapshot_path.parent / doc["processed_path"]
        if digest(path) != doc["sha256"]:
            raise ValueError(f"processed hash mismatch: {doc['document_id']}")
        for artifact in doc["artifacts"]:
            raw = snapshot_path.parent / "raw" / artifact["name"]
            if digest(raw) != artifact["sha256"] or raw.stat().st_size != artifact["bytes"]:
                raise ValueError(f"raw hash mismatch: {artifact['name']}")
        texts[doc["document_id"]] = path.read_text(encoding="utf-8")
        cohorts[doc["document_id"]] = doc["cohort"]
    queries = [
        json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line
    ]
    excluded = [
        dict(q, missing_source_ids=sorted(set(q["qrels"]) - set(texts)))
        for q in queries
        if set(q["qrels"]) - set(texts)
    ]
    eligible = [q for q in queries if not set(q["qrels"]) - set(texts)]
    if not eligible:
        raise ValueError("no queries with all labeled sources restored")
    rows = []
    inventory = []
    with tempfile.TemporaryDirectory(prefix="recall-comparison-") as directory:
        for strategy in ("fixed", "structure"):
            parser = PlainTextParser(strategy=strategy)
            chunks = []
            for doc_id, text in sorted(texts.items()):
                parsed = parser.parse(text.encode())
                for piece in parsed.chunks:
                    identity = hashlib.sha256(
                        f"{doc_id}:{strategy}:{piece.locator.canonical()}".encode()
                    ).hexdigest()
                    chunks.append(
                        VectorChunk(
                            identity,
                            doc_id,
                            strategy,
                            piece.content,
                            {"locator": piece.locator.canonical()},
                        )
                    )
            inventory.append(
                {
                    "strategy": strategy,
                    "chunks": len(chunks),
                    "parser_version": parser.parser_version,
                }
            )
            vector = SQLiteVectorStore(Path(directory) / f"{strategy}.db")
            vector.upsert(chunks)
            bm25 = BM25(chunks)
            reranker = LexicalReranker()
            for q in eligible:
                for backend in ("hashing256", "hashing256_lexical", "bm25_diagnostic"):
                    ranking = (
                        vector.search(q["query"], top_k=len(chunks))
                        if backend != "bm25_diagnostic"
                        else bm25.search(q["query"])
                    )
                    if backend == "hashing256_lexical":
                        # Production reranker reorders only the already retrieved Top K.
                        original = {match.chunk.chunk_id: match for match in ranking[:5]}
                        reranked = reranker.rerank(
                            q["query"],
                            [
                                RankedText(m.chunk.chunk_id, m.chunk.content, m.score)
                                for m in ranking[:5]
                            ],
                        )
                        ranking = [original[r.key] for r in reranked] + ranking[5:]
                    rows.append(
                        {
                            "strategy": strategy,
                            "backend": backend,
                            "query_id": q["query_id"],
                            "query": q["query"],
                            "qrels": q["qrels"],
                            "review_status": q["review_status"],
                            "cohort": "historical_exact"
                            if all(cohorts[d] == "historical_exact" for d in q["qrels"])
                            else "refreshed",
                            "metrics": score(ranking, q["qrels"]),
                            "unique_document_ranking": unique(
                                [m.chunk.document_id for m in ranking]
                            ),
                            "top_chunks": [
                                {
                                    "document_id": m.chunk.document_id,
                                    "chunk_id": m.chunk.chunk_id,
                                    "score": m.score,
                                    "locator": m.chunk.metadata["locator"],
                                    "excerpt": m.chunk.content[:240],
                                }
                                for m in ranking[:10]
                            ],
                        }
                    )
    summaries = []
    for strategy in ("fixed", "structure"):
        for backend in ("hashing256", "hashing256_lexical", "bm25_diagnostic"):
            for cohort in ("all_available", "historical_exact", "refreshed"):
                group = [
                    r
                    for r in rows
                    if r["strategy"] == strategy
                    and r["backend"] == backend
                    and (cohort == "all_available" or r["cohort"] == cohort)
                ]
                if group:
                    summaries.append(
                        {
                            "strategy": strategy,
                            "backend": backend,
                            "cohort": cohort,
                            "query_count": len(group),
                            "metrics": {
                                key: sum(r["metrics"][key] for r in group) / len(group)
                                for key in group[0]["metrics"]
                            },
                        }
                    )
    report = {
        "dataset_id": snapshot["dataset_id"],
        "snapshot_sha256": digest(snapshot_path),
        "queries_sha256": digest(queries_path),
        "runner_sha256": digest(Path(__file__)),
        "source_code_sha256": {
            name: digest(ROOT / name)
            for name in (
                "src/aerodiagnosis/ingestion/parsers.py",
                "src/aerodiagnosis/adapters/persistence/sqlite_vector.py",
                "src/aerodiagnosis/retrieval/reranker.py",
                "src/aerodiagnosis/retrieval/metrics.py",
            )
        },
        "cohort_definition": (
            "Query groups describe labeled source byte identity only; "
            "all groups search the same mixed historical/refreshed document pool."
        ),
        "embedding_identity": "hashing@256",
        "reranker_identity": "lexical@0.45",
        "bm25_parameters": {"k1": 1.2, "b": 0.75, "purpose": "offline diagnostic only"},
        "query_count": len(eligible),
        "excluded_query_count": len(excluded),
        "excluded_queries": excluded,
        "documents": sorted(texts),
        "inventory": inventory,
        "summaries": summaries,
        "query_results": rows,
        "limitations": [
            "All qrels remain machine_candidate and document-level, not evidence gold.",
            "Unjudged documents are not proven irrelevant; "
            "nDCG/MRR are provisional labeled signals.",
            "Missing-source queries are excluded, never scored as retrieval failures.",
            "Document ranks deduplicate full chunk ranking using each document's first occurrence.",
            "Top5 chunk recall counts each labeled document once; "
            "duplicate chunks cannot inflate recall.",
            "Refreshed inputs are a new snapshot; "
            "original qrels have not been independently reviewed.",
            "No model/weight tuning or production retrieval changes were performed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "evaluation/public_aero_corpus_v1/retrieval_queries.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.snapshot, args.queries, args.output)
    print(
        json.dumps(
            {
                "queries": report["query_count"],
                "excluded": report["excluded_query_count"],
                "summaries": report["summaries"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
