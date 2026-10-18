"""Observe the unchanged lexical KG extractor over a hash-locked academic corpus.

Only machine output coverage and exact source-location consistency are measured here.
A uniformly sampled, label-free packet supports a separate source-semantic review.
Full source text and extracted quotes are written only to the local runtime directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.domain import make_chunk_id, make_version_id
from aerodiagnosis.ingestion.parsers import PlainTextParser

STRATEGIES = ("fixed", "structure", "hierarchical")
SEED = 20261013
PARAMETERS = {"chunk_size": 800, "overlap": 100, "parent_budget": 3200, "child_budget": 800}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_sources(manifest_path: Path, repo: Path) -> list[dict[str, Any]]:
    """Fail closed on bytes, canonical blocks, or provenance fingerprint changes."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = []
    identifiers = set()
    for document in manifest["documents"]:
        docid = document["document_id"]
        if docid in identifiers:
            raise ValueError(f"Duplicate document ID: {docid}")
        identifiers.add(docid)
        content: dict[str, bytes] = {}
        for key, expected in (
            ("raw_path", "raw_sha256"),
            ("processed_path", "canonical_sha256"),
            ("blocks_path", "blocks_sha256"),
        ):
            path = Path(document[key])
            if not path.is_absolute():
                path = repo / path
            content[key] = path.read_bytes()
            if digest(content[key]) != document[expected]:
                raise ValueError(f"{docid}: {expected} mismatch")
        if len(content["raw_path"]) != document["raw_bytes"]:
            raise ValueError(f"{docid}: raw_bytes mismatch")
        text = content["processed_path"].decode("utf-8")
        blocks = [
            json.loads(line) for line in content["blocks_path"].decode("utf-8").splitlines() if line
        ]
        previous_end = 0
        for index, block in enumerate(blocks):
            start, end = block["canonical_start"], block["canonical_end"]
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and previous_end <= start < end <= len(text)
                and text[start:end] == block["text"]
            ):
                raise ValueError(f"{docid}: block {index} canonical span mismatch")
            if digest(block["text"].encode("utf-8")) != block["text_sha256"]:
                raise ValueError(f"{docid}: block {index} text fingerprint mismatch")
            if (
                block["canonical_sha256"] != document["canonical_sha256"]
                or block["raw_sha256"] != document["raw_sha256"]
            ):
                raise ValueError(f"{docid}: block {index} source fingerprint mismatch")
            if not isinstance(block["source_locator"], dict):
                raise ValueError(f"{docid}: block {index} missing source locator")
            previous_end = end
        sources.append({**document, "text": text, "blocks": blocks})
    return sources


def source_blocks(
    source: dict[str, Any], start: int, end: int
) -> tuple[list[dict[str, Any]], bool]:
    """Map an exact canonical span to all intersected raw-source block locators."""
    references = []
    covered = []
    for index, block in enumerate(source["blocks"]):
        left = max(start, block["canonical_start"])
        right = min(end, block["canonical_end"])
        if left < right:
            covered.append((left, right))
            references.append(
                {
                    "block_index": index,
                    "source_locator": block["source_locator"],
                    "canonical_start": left,
                    "canonical_end": right,
                    "block_text_sha256": block["text_sha256"],
                }
            )
    # Canonical separators between blocks may be whitespace. Content must be mapped.
    cursor = start
    complete = bool(references)
    for left, right in covered:
        if source["text"][cursor:left].strip():
            complete = False
        cursor = right
    if source["text"][cursor:end].strip():
        complete = False
    return references, complete


def structured_qualifiers(properties: dict[str, Any]) -> dict[str, bool]:
    """Count explicit structured fields, never infer them from free-text conditions."""
    qualifiers = properties.get("qualifiers")
    scope = qualifiers if isinstance(qualifiers, dict) else properties
    result = {}
    for name in ("engine_models", "flight_phases"):
        value = scope.get(name)
        result[name] = isinstance(value, (list, tuple)) and bool(value)
    revision = scope.get("revision")
    result["revision"] = isinstance(revision, str) and bool(revision.strip())
    return result


def claim_key(row: dict[str, Any], *, include_span: bool = True) -> tuple[Any, ...]:
    key = (
        row["document_id"],
        row["relation"],
        row["subject"]["semantic_key"],
        row["object"]["semantic_key"],
    )
    return (*key, row["canonical_start"], row["canonical_end"]) if include_span else key


def extract_source(
    source: dict[str, Any], strategy: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parser = PlainTextParser(strategy=strategy, **PARAMETERS)
    parsed = parser.parse(source["text"].encode("utf-8"))
    if parsed.canonical_content != source["text"].encode("utf-8"):
        raise ValueError(f"{source['document_id']}: parser changed frozen canonical text")
    version = make_version_id(
        source["document_id"], parsed.canonical_content, parser.parser_version
    )
    chunks = parsed.chunks
    nodes, edges = KnowledgeGraphExtractor().extract(
        source_ref=f"document:{source['document_id']}",
        version_id=version,
        chunks=tuple(c.content for c in chunks),
        chunk_ids=tuple(make_chunk_id(version, c.locator.canonical(), c.content) for c in chunks),
        chunk_locators=tuple(c.locator.canonical() for c in chunks),
    )
    by_id = {node.node_id: node for node in nodes}
    rows = []
    for edge in edges:
        properties = dict(edge.properties)
        chunk = chunks[properties["chunk_index"]]
        chunk_start = chunk.locator.coordinates["start"]
        chunk_end = chunk.locator.coordinates["end"]
        sentence_start, sentence_end = properties["sentence_start"], properties["sentence_end"]
        start, end = chunk_start + sentence_start, chunk_start + sentence_end
        evidence = properties["evidence_text"]
        valid = (
            0 <= sentence_start < sentence_end <= len(chunk.content)
            and 0 <= chunk_start <= start < end <= chunk_end <= len(source["text"])
            and source["text"][chunk_start:chunk_end] == chunk.content
            and chunk.content[sentence_start:sentence_end] == evidence
            and source["text"][start:end] == evidence
        )
        locators, mapped = source_blocks(source, start, end)
        entity = []
        for node_id in (edge.source_id, edge.target_id):
            node = by_id[node_id]
            entity.append(
                {
                    "node_id": node.node_id,
                    "label": node.name,
                    "kind": node.kind,
                    "semantic_key": node.properties["semantic_key"],
                    "semantic_id": node.properties["semantic_id"],
                }
            )
        rows.append(
            {
                "strategy": strategy,
                "document_id": source["document_id"],
                "family_id": source["family_id"],
                "source_ref": edge.source_ref,
                "source_url": source["source_url"],
                "version_id": version,
                "edge_id": edge.edge_id,
                "relation": edge.relation,
                "subject": entity[0],
                "object": entity[1],
                "canonical_start": start,
                "canonical_end": end,
                "canonical_sha256": source["canonical_sha256"],
                "raw_sha256": source["raw_sha256"],
                "evidence_text": evidence,
                "evidence_sha256": digest(evidence.encode("utf-8")),
                "offset_valid": valid,
                "source_locator_complete": mapped,
                "source_blocks": locators,
                "modality": properties.get("modality"),
                "polarity": properties.get("polarity"),
                "structured_qualifiers": structured_qualifiers(properties),
                "properties": properties,
            }
        )
    relations = Counter(row["relation"] for row in rows)
    diagnostic = [row for row in rows if row["relation"] != "HAS_COMPONENT"]
    summary = {
        "document_id": source["document_id"],
        "family_id": source["family_id"],
        "parser_identity": parsed.parser_version,
        "chunks": len(chunks),
        "nodes": len(nodes),
        "edges": len(rows),
        "unique_node_semantics": len({node.properties["semantic_key"] for node in nodes}),
        "unique_claims": len({claim_key(row) for row in rows}),
        "unique_source_relations": len({claim_key(row, include_span=False) for row in rows}),
        "diagnostic_edges": len(diagnostic),
        "unique_diagnostic_claims": len({claim_key(row) for row in diagnostic}),
        "relations": dict(sorted(relations.items())),
        "valid_offsets": sum(row["offset_valid"] for row in rows),
        "complete_source_locators": sum(row["source_locator_complete"] for row in rows),
        "edge_fingerprint": fingerprint(rows),
    }
    return summary, rows


def review_packet(rows: list[dict[str, Any]], size: int = 60) -> list[dict[str, Any]]:
    """Uniform sample from exact cross-strategy claim union; no assessment fields."""
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = claim_key(row)
        if key not in grouped:
            grouped[key] = {
                "document_id": row["document_id"],
                "family_id": row["family_id"],
                "source_ref": row["source_ref"],
                "relation": row["relation"],
                "subject": {k: row["subject"][k] for k in ("label", "kind", "semantic_key")},
                "object": {k: row["object"][k] for k in ("label", "kind", "semantic_key")},
                "canonical_start": row["canonical_start"],
                "canonical_end": row["canonical_end"],
                "canonical_sha256": row["canonical_sha256"],
                "raw_sha256": row["raw_sha256"],
                "evidence_sha256": row["evidence_sha256"],
                "source_blocks": row["source_blocks"],
                "strategies": [],
            }
        grouped[key]["strategies"].append(row["strategy"])
    population = [grouped[key] for key in sorted(grouped)]
    for row in population:
        row["strategies"] = sorted(set(row["strategies"]))
    sampled = random.Random(SEED).sample(population, min(size, len(population)))
    return [
        {"review_id": f"kg-observation-{index:03d}", **row} for index, row in enumerate(sampled, 1)
    ]


def build_observation(
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    all_rows = []
    for strategy in STRATEGIES:
        documents = []
        rows = []
        for source in sources:
            summary, extracted = extract_source(source, strategy)
            documents.append(summary)
            rows.extend(extracted)
        totals = {
            key: sum(doc[key] for doc in documents)
            for key in (
                "chunks",
                "nodes",
                "edges",
                "unique_claims",
                "unique_source_relations",
                "diagnostic_edges",
                "unique_diagnostic_claims",
                "valid_offsets",
                "complete_source_locators",
            )
        }
        totals["relations"] = dict(sorted(Counter(row["relation"] for row in rows).items()))
        totals["structured_qualifier_edges"] = {
            name: sum(row["structured_qualifiers"][name] for row in rows)
            for name in ("engine_models", "flight_phases", "revision")
        }
        totals["modality"] = dict(Counter(row["modality"] for row in rows))
        totals["polarity"] = dict(Counter(row["polarity"] for row in rows))
        summaries.append(
            {
                "strategy": strategy,
                "parser_identity": PlainTextParser(strategy=strategy, **PARAMETERS).parser_version,
                **totals,
                "edge_fingerprint": fingerprint(rows),
                "documents": documents,
            }
        )
        all_rows.extend(rows)
    return summaries, all_rows, review_packet(all_rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/aero_evidence_benchmark_v1/corpus_manifest.json"),
    )
    cli.add_argument(
        "--output", type=Path, default=Path(".runtime/experiments/academic-evidence-20261013/graph")
    )
    args = cli.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    output = (args.output if args.output.is_absolute() else repo / args.output).resolve()
    if not output.is_relative_to(repo / ".runtime"):
        raise ValueError(
            "Machine evidence output must stay under the repository .runtime directory"
        )
    sources = load_sources(manifest, repo)
    summaries, rows, packet = build_observation(sources)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "edges.jsonl", rows)
    write_jsonl(output / "review_packet.jsonl", packet)
    code_paths = [
        Path(__file__).resolve(),
        repo / "src/aerodiagnosis/application/knowledge_graph_extractor.py",
        repo / "src/aerodiagnosis/ingestion/parsers.py",
        repo / "src/aerodiagnosis/ingestion/structure.py",
    ]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    report = {
        "schema_version": 1,
        "experiment": "academic-evidence-graph-observation",
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "python_version": sys.version.split()[0],
        "manifest_sha256": digest(manifest.read_bytes()),
        "extractor_identity": KnowledgeGraphExtractor.identity,
        "code_sha256": {
            path.relative_to(repo).as_posix(): digest(path.read_bytes()) for path in code_paths
        },
        "parameters": PARAMETERS,
        "sources_validated": len(sources),
        "source_fingerprints": [
            {
                key: source[key]
                for key in ("document_id", "raw_sha256", "canonical_sha256", "blocks_sha256")
            }
            for source in sources
        ],
        "strategies": summaries,
        "review_sample": {
            "seed": SEED,
            "sampling": "uniform_without_replacement_from_cross_strategy_exact_claim_union",
            "population": len({claim_key(row) for row in rows}),
            "sample_size": len(packet),
            "packet_sha256": digest((output / "review_packet.jsonl").read_bytes()),
            "assessment": "not_assessed_in_this_script",
        },
        "interpretation": (
            "Machine extraction coverage and exact source-locator consistency only; "
            "not extraction precision/recall. HAS_COMPONENT denotes lexical component "
            "mentions under the extractor's generic Equipment node, not verified ownership "
            "by a particular engine. Free-text conditions are not structured applicability. "
            "Any semantic support estimate requires separate review and applies only to the "
            "random sampled output; it is not expert gold or full-graph accuracy."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    key: summary[key]
                    for key in (
                        "strategy",
                        "chunks",
                        "nodes",
                        "edges",
                        "unique_claims",
                        "diagnostic_edges",
                        "valid_offsets",
                    )
                }
                for summary in summaries
            ]
        )
    )


if __name__ == "__main__":
    main()
