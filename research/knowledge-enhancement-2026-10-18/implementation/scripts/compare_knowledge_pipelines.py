"""Offline fixed/structure and lexical@1/@2 comparison; candidate labels are not gold."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.ingestion.parsers import ParsedChunk, PlainTextParser
from aerodiagnosis.ports import GraphEdge, GraphNode

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "66e6a2420c5560220c0172a6266a896c1d36f295"
BASELINE_FILE = "src/aerodiagnosis/application/knowledge_graph_extractor.py"
STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "to",
        "and",
        "or",
        "for",
        "with",
        "on",
        "is",
        "are",
        "from",
        "by",
        "engine",
        "gas",
        "turbine",
        "can",
        "may",
        "approximately",
        "state",
        "parameters",
    ]
)
RELATION_MAPPING = {
    "caused": ("CAUSED_BY", "object", "subject"),
    "causes": ("CAUSED_BY", "object", "subject"),
    "may_cause": ("CAUSED_BY", "object", "subject"),
    "can_cause": ("CAUSED_BY", "object", "subject"),
}
# Explicit experiment mapping, not an extension to production extraction.
CONCEPT_ALIASES = {
    "symptom:egt_high": ("egt high", "high egt", "egt rise", "排气温度升高", "EGT 异常升高"),
    "symptom:vibration": ("vibration", "振动升高"),
    "cause:fouling": ("fouling", "积垢", "污染"),
    "cause:erosion": ("erosion", "侵蚀"),
    "cause:fod": ("fod", "外物损伤"),
    "cause:sensor_chain": ("sensor chain", "测量链异常"),
    "check:inspect": ("borescope", "孔探"),
    "check:verify_trend": ("trend review", "趋势复核"),
    "solution:wash": ("wash", "清洗"),
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_baseline(directory: Path) -> tuple[ModuleType, str]:
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{BASELINE_FILE}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    path = directory / "lexical_v1_frozen.py"
    path.write_bytes(source)
    return load_module(path, "aero_experiment_lexical_v1"), digest(source)


def inventory(manifest: Mapping[str, Any], runtime: Path) -> list[dict[str, Any]]:
    rows = []
    for document in manifest["documents"]:
        for artifact in document["artifacts"]:
            path = runtime / "raw" / artifact["name"]
            row = {
                "document_id": document["document_id"],
                "file": artifact["name"],
                "expected_sha256": artifact["sha256"],
                "expected_bytes": artifact["bytes"],
            }
            if not path.is_file():
                row["status"] = "missing"
            else:
                payload = path.read_bytes()
                row.update(actual_sha256=digest(payload), actual_bytes=len(payload))
                row["status"] = (
                    "verified"
                    if (len(payload) == artifact["bytes"] and digest(payload) == artifact["sha256"])
                    else "mismatch"
                )
            rows.append(row)
    return rows


def cut_metrics(text: str, chunks: Sequence[ParsedChunk]) -> dict[str, Any]:
    covered = bytearray(len(text))
    invalid = []
    records = []
    page_starts = [match.start() for match in re.finditer(r"\[PDF_PAGE \d+\]|\[PAGE_BREAK\]", text)]
    for index, chunk in enumerate(chunks):
        coords = chunk.locator.coordinates
        start, end = int(coords["start"]), int(coords["end"])
        valid = 0 <= start <= end <= len(text) and text[start:end] == chunk.content
        if valid:
            covered[start:end] = b"\x01" * (end - start)
        else:
            invalid.append(index)
        crossing = sum(start < boundary < end for boundary in page_starts)
        records.append(
            {
                "chunk_index": index,
                "start": start,
                "end": end,
                "characters": len(chunk.content),
                "sha256": digest(chunk.content.encode()),
                "source_located": valid,
                "page_boundary_count": crossing,
                "fallback_window": bool(coords.get("fallback_window", False)),
                "section_path": coords.get("section_path", []),
            }
        )
    non_space = sum(not char.isspace() for char in text)
    covered_non_space = sum(
        bool(covered[index]) for index, char in enumerate(text) if not char.isspace()
    )
    lengths = [len(chunk.content) for chunk in chunks]
    boundaries = sentence_spans(text)
    retained = sum(
        any(row["start"] <= start and end <= row["end"] for row in records)
        for start, end, _ in boundaries
    )
    lost = [
        {"start": start, "end": end, "text": value}
        for start, end, value in boundaries
        if not any(row["start"] <= start and end <= row["end"] for row in records)
    ]
    return {
        "chunk_count": len(chunks),
        "mean_characters": sum(lengths) / len(lengths) if lengths else 0,
        "max_characters": max(lengths, default=0),
        "total_index_characters": sum(lengths),
        "covered_non_whitespace": covered_non_space,
        "total_non_whitespace": non_space,
        "non_whitespace_coverage": covered_non_space / non_space if non_space else 1.0,
        "invalid_locator_indices": invalid,
        "heuristic_sentence_span_count": len(boundaries),
        "heuristic_sentence_spans_retained_whole": retained,
        "heuristic_sentence_retention": retained / len(boundaries) if boundaries else 1.0,
        "split_sentence_examples": lost[:5],
        "sentence_metric_note": "Regex spans only; no semantic completeness judgment.",
        "fallback_chunks": sum(row["fallback_window"] for row in records),
        "cross_page_marker_chunks": sum(row["page_boundary_count"] > 0 for row in records),
        "page_marker_count": len(page_starts),
        "chunks": records,
        "page_metric_scope": "PDF markers only; NASA reference TXT has no reliable page map.",
    }


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    result = []
    for match in re.finditer(r"[^。！？.!?\n]+[。！？.!?]?", text):  # noqa: RUF001
        raw = match.group()
        value = raw.strip()
        if value:
            start = match.start() + len(raw) - len(raw.lstrip())
            result.append((start, start + len(value), value))
    return result


def aliases(module: ModuleType) -> dict[tuple[str, str], tuple[str, ...]]:
    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for kind, attribute in (
        ("Component", "_COMPONENT_TERMS"),
        ("Symptom", "_SYMPTOM_TERMS"),
        ("Cause", "_CAUSE_TERMS"),
        ("Solution", "_SOLUTION_TERMS"),
    ):
        for terms in getattr(module, attribute).values():
            result[(kind, terms[0])] = terms
    return result


def claim_records(
    text: str,
    chunks: Sequence[ParsedChunk],
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    legacy_aliases: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    by_id = {node.node_id: node for node in nodes}
    records = []
    for edge in edges:
        source, target = by_id[edge.source_id], by_id[edge.target_id]
        row = {
            "edge_id": edge.edge_id,
            "source_ref": edge.source_ref,
            "version_id": edge.version_id,
            "confidence": edge.confidence,
            "confidence_interpretation": "Rule parameter, not calibrated probability",
            "relation": edge.relation,
            "source": {
                "name": source.name,
                "kind": source.kind,
                "semantic_key": source.properties.get("semantic_key"),
                "experiment_concept_keys": _concept(source.name, source.kind),
            },
            "target": {
                "name": target.name,
                "kind": target.kind,
                "semantic_key": target.properties.get("semantic_key"),
                "experiment_concept_keys": _concept(target.name, target.kind),
            },
            "properties": dict(edge.properties),
            "proofs": [],
        }
        if edge.properties.get("extraction") == "lexical@2":
            index = int(edge.properties["chunk_index"])
            chunk = chunks[index]
            start = int(chunk.locator.coordinates["start"]) + int(edge.properties["sentence_start"])
            end = int(chunk.locator.coordinates["start"]) + int(edge.properties["sentence_end"])
            value = edge.properties["evidence_text"]
            row["proof_method"] = "extractor_saved_sentence_span"
            row["proofs"] = [
                {
                    "start": start,
                    "end": end,
                    "text": value,
                    "source_located": text[start:end] == value,
                }
            ]
        else:
            row["proof_method"] = "posthoc_cooccurrence_not_original_provenance"
            terms_source = (legacy_aliases or {}).get((source.kind, source.name), ())
            terms_target = (legacy_aliases or {}).get((target.kind, target.name), ())
            for chunk in chunks:
                origin = int(chunk.locator.coordinates["start"])
                for start, end, sentence in sentence_spans(chunk.content):
                    folded = sentence.casefold()
                    if (
                        terms_source
                        and terms_target
                        and any(term.casefold() in folded for term in terms_source)
                        and any(term.casefold() in folded for term in terms_target)
                    ):
                        row["proofs"].append(
                            {
                                "start": origin + start,
                                "end": origin + end,
                                "text": sentence,
                                "source_located": text[origin + start : origin + end] == sentence,
                            }
                        )
        row["saved_provenance_located"] = edge.properties.get("extraction") == "lexical@2" and all(
            proof["source_located"] for proof in row["proofs"]
        )
        records.append(row)
    return records


def graph_metrics(
    records: Sequence[Mapping[str, Any]], nodes: Sequence[GraphNode]
) -> dict[str, Any]:
    triples = {
        (
            row["source"]["kind"],
            row["source"]["name"],
            row["relation"],
            row["target"]["kind"],
            row["target"]["name"],
        )
        for row in records
    }
    return {
        "node_count": len(nodes),
        "edge_count": len(records),
        "exact_label_triplet_count": len(triples),
        "relation_counts": dict(sorted(Counter(row["relation"] for row in records).items())),
        "saved_provenance_located_edges": sum(row["saved_provenance_located"] for row in records),
        "posthoc_reconstructable_edges": sum(
            bool(row["proofs"]) for row in records if row["proof_method"].startswith("posthoc")
        ),
        "note": "Triplet totals and located text are not relation correctness or recall.",
    }


def collision_metrics(origins: Mapping[str, set[tuple[str, str]]]) -> dict[str, int]:
    return {
        "distinct_ids": len(origins),
        "ids_shared_by_source_version_instances": sum(len(rows) > 1 for rows in origins.values()),
        "source_version_instances_lost_by_id_only_upsert": sum(
            len(rows) - 1 for rows in origins.values()
        ),
    }


def _concept(value: str, kind: str) -> list[str]:
    folded = value.casefold()
    return [
        key
        for key, terms in CONCEPT_ALIASES.items()
        if key.startswith(kind.casefold() + ":")
        and any(term.casefold() in folded for term in terms)
    ]


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold())) - STOP_WORDS


def candidate_audit(candidate: Mapping[str, Any], text: str | None) -> dict[str, Any]:
    row = {
        "candidate_id": candidate["candidate_id"],
        "review_status": candidate["review_status"],
        "source_document_id": candidate["source_document_id"],
        "original_candidate": dict(candidate),
        "is_gold": False,
        "pair_evaluable": False,
    }
    mapping = RELATION_MAPPING.get(candidate["predicate"])
    if mapping is None:
        row["mapping_status"] = "predicate_outside_current_ontology"
    else:
        relation, source_field, target_field = mapping
        source_kind, target_kind = (
            ("Symptom", "Cause") if relation == "CAUSED_BY" else ("Cause", "Check")
        )
        source_concepts = _concept(candidate[source_field], source_kind)
        target_concepts = _concept(candidate[target_field], target_kind)
        row["ontology_mapping"] = {
            "relation": relation,
            "source_field": source_field,
            "target_field": target_field,
            "source_concepts": source_concepts,
            "target_concepts": target_concepts,
            "qualifiers_preserved_in_candidate": True,
        }
        row["mapping_status"] = (
            "pair_mapped"
            if len(source_concepts) == len(target_concepts) == 1
            else ("relation_mapped_entities_outside_vocabulary_or_ambiguous")
        )
    if text is None:
        row["localization_status"] = "missing_document"
        row["evidence_candidates"] = []
        return row
    locator = candidate.get("source_locator", {})
    left, right = 0, len(text)
    if "page" in locator:
        marker = f"[PDF_PAGE {locator['page']}]"
        position = text.find(marker)
        if position >= 0:
            left = position + len(marker)
            match = re.search(r"\[PDF_PAGE \d+\]", text[left:])
            right = left + match.start() if match else len(text)
            row["localization_status"] = "page_marker_scope_located"
        else:
            row["localization_status"] = "page_not_available_in_canonical"
    else:
        section = str(locator.get("section", ""))
        matches = list(re.finditer(re.escape(section), text, re.IGNORECASE)) if section else []
        row["localization_status"] = (
            "section_phrase_occurrence_only" if matches else "section_not_located"
        )
        row["section_phrase_offsets"] = [match.start() for match in matches]
        # Section phrase occurrences do not establish a section boundary.
    expected = _tokens(candidate["subject"] + " " + candidate["object"])
    snippets = []
    for start, end, value in sentence_spans(text[left:right]):
        shared = expected & _tokens(value)
        if len(shared) >= 3:
            snippets.append(
                {
                    "start": left + start,
                    "end": left + end,
                    "text": value,
                    "source_located": text[left + start : left + end] == value,
                    "shared_tokens": sorted(shared),
                    "token_overlap": len(shared) / len(expected) if expected else 0,
                    "supports_relation": "unreviewed",
                }
            )
    snippets.sort(key=lambda item: (-item["token_overlap"], item["start"]))
    row["evidence_candidates"] = snippets[:3]
    # Page or keyword localization does not validate an unreviewed paraphrased relation.
    row["pair_evaluable"] = row["mapping_status"] == "pair_mapped" and bool(snippets)
    row["interpretation"] = (
        "Pair comparison only; keyword overlap is not semantic support or gold judgment."
    )
    return row


def compare_text(
    document_id: str,
    text: str,
    legacy: Any,
    legacy_aliases: Mapping[tuple[str, str], Sequence[str]],
    *,
    node_origins: dict[str, dict[str, set[tuple[str, str]]]],
    edge_origins: dict[str, dict[str, set[tuple[str, str]]]],
) -> dict[str, Any]:
    cuts = {
        strategy: PlainTextParser(strategy=strategy).parse(text.encode())
        for strategy in ("fixed", "structure")
    }
    assert all(parsed.canonical_content.decode() == text for parsed in cuts.values())
    row: dict[str, Any] = {
        "document_id": document_id,
        "canonical_sha256": digest(text.encode()),
        "canonical_characters": len(text),
        "cutting": {},
        "graphs": {},
    }
    for strategy, parsed in cuts.items():
        row["cutting"][strategy] = {
            "parser_version": parsed.parser_version,
            **cut_metrics(text, parsed.chunks),
        }
        for name, extractor in (("lexical_v1", legacy), ("lexical_v2", KnowledgeGraphExtractor())):
            policy = f"{strategy}/{name}"
            version = digest((document_id + "\0" + strategy + "\0" + text).encode())
            kwargs = {
                "source_ref": f"document:{document_id}",
                "version_id": version,
                "chunks": tuple(chunk.content for chunk in parsed.chunks),
            }
            if name == "lexical_v2":
                kwargs.update(
                    chunk_locators=tuple(chunk.locator.canonical() for chunk in parsed.chunks),
                    chunk_ids=tuple(
                        digest(f"{version}:{index}".encode()) for index in range(len(parsed.chunks))
                    ),
                )
            nodes, edges = extractor.extract(**kwargs)
            records = claim_records(
                text,
                parsed.chunks,
                nodes,
                edges,
                legacy_aliases=legacy_aliases if name == "lexical_v1" else None,
            )
            probe_nodes, probe_edges = extractor.extract(
                **{**kwargs, "version_id": version + "-revision-probe"}
            )
            row["graphs"][policy] = {
                **graph_metrics(records, nodes),
                "claims": records,
                "same_content_revision_identity_probe": {
                    "shared_node_ids": len(
                        {node.node_id for node in nodes} & {node.node_id for node in probe_nodes}
                    ),
                    "shared_edge_ids": len(
                        {edge.edge_id for edge in edges} & {edge.edge_id for edge in probe_edges}
                    ),
                    "note": "Synthetic version ID probe on real text; not a new manual revision.",
                },
            }
            for item in nodes:
                node_origins[policy].setdefault(item.node_id, set()).add((document_id, version))
            for item in edges:
                edge_origins[policy].setdefault(item.edge_id, set()).add((document_id, version))
    return row


def run(
    runtime: Path,
    output: Path,
    manifest_path: Path | None = None,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path or ROOT / "evaluation/public_aero_corpus_v1/source_manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    report: dict[str, Any] = {
        "schema_version": "1",
        "experiment": "knowledge_pipeline_comparison",
        "offline": True,
        "baseline_commit": BASELINE_COMMIT,
        "manifest_sha256": digest(manifest_payload),
        "current_source_sha256": {
            path: digest((ROOT / path).read_bytes())
            for path in (
                "src/aerodiagnosis/ingestion/parsers.py",
                "src/aerodiagnosis/application/knowledge_graph_extractor.py",
                "scripts/compare_knowledge_pipelines.py",
            )
        },
        "cutting_scope": "Canonical TXT adapters only; native PDF/OCR/layout strategy not tested.",
        "candidate_mapping_protocol": {
            "version": "candidate-exploration@1",
            "relations": RELATION_MAPPING,
            "concept_aliases": CONCEPT_ALIASES,
        },
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["version"],
        "raw_inventory": inventory(manifest, runtime),
        "interpretation": "Not gold or clinical accuracy; derived text, no OCR/layout test.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    source_lineage: dict[str, dict[str, Any]] = {}
    if snapshot_path is not None:
        snapshot_payload = snapshot_path.read_bytes()
        snapshot = json.loads(snapshot_payload)
        if snapshot["source_manifest_sha256"] != digest(manifest_payload):
            raise ValueError("snapshot source manifest hash mismatch")
        report["frozen_snapshot_sha256"] = digest(snapshot_payload)
        report["snapshot_dataset_id"] = snapshot["dataset_id"]
        report["unavailable_document_ids"] = snapshot.get("unavailable_document_ids", [])
        report["snapshot_inventory"] = []
        for document in snapshot["documents"]:
            document_id = document["document_id"]
            path = snapshot_path.parent / document["processed_path"]
            payload = path.read_bytes() if path.is_file() else b""
            text_status = (
                "verified"
                if payload and digest(payload) == document["sha256"]
                else "mismatch_or_missing"
            )
            artifact_rows = []
            for artifact in document["artifacts"]:
                raw = snapshot_path.parent / "raw" / artifact["name"]
                raw_payload = raw.read_bytes() if raw.is_file() else b""
                actual = digest(raw_payload)
                verified = (
                    bool(raw_payload)
                    and actual == artifact["sha256"]
                    and len(raw_payload) == artifact["bytes"]
                )
                artifact_rows.append(
                    {
                        **artifact,
                        "status": "verified_frozen" if verified else "mismatch_or_missing",
                        "historical_hash_match": actual == artifact["expected_sha256"],
                    }
                )
            report["snapshot_inventory"].append(
                {
                    "document_id": document_id,
                    "cohort": document["cohort"],
                    "text_status": text_status,
                    "artifacts": artifact_rows,
                }
            )
            texts[document_id] = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            source_lineage[document_id] = dict(document)
        if any(
            row["text_status"] != "verified"
            or any(artifact["status"] != "verified_frozen" for artifact in row["artifacts"])
            for row in report["snapshot_inventory"]
        ):
            report["status"] = "incomplete_frozen_snapshot"
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        report["cohort_counts"] = dict(
            Counter(document["cohort"] for document in snapshot["documents"])
        )
    else:
        if any(row["status"] != "verified" for row in report["raw_inventory"]):
            report["status"] = "incomplete_raw_assets"
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        builder = load_module(
            ROOT / "scripts/build_public_corpus.py", "aero_experiment_corpus_builder"
        )
        builder.MANIFEST_PATH = manifest_path
        report["build_report"] = builder.build(runtime, allow_download=False)
        for document in manifest["documents"]:
            document_id = document["document_id"]
            path = runtime / "processed" / (document_id.lower() + ".txt")
            texts[document_id] = path.read_text(encoding="utf-8")
            source_lineage[document_id] = {
                "cohort": "historical_exact",
                "processor": document["processor"],
            }
    node_origins: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(dict)
    edge_origins: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(dict)
    documents = []
    with tempfile.TemporaryDirectory(prefix="knowledge-baseline-", dir=output.parent) as temporary:
        module, source_hash = frozen_baseline(Path(temporary))
        report["baseline_source_sha256"] = source_hash
        legacy = module.KnowledgeGraphExtractor()
        legacy_aliases = aliases(module)
        for document_id, text in texts.items():
            row = compare_text(
                document_id,
                text,
                legacy,
                legacy_aliases,
                node_origins=node_origins,
                edge_origins=edge_origins,
            )
            row["source_lineage"] = source_lineage[document_id]
            documents.append(row)
    report["documents"] = documents
    report["identity_collision_risk"] = {
        policy: {
            "nodes": collision_metrics(origins),
            "edges": collision_metrics(edge_origins[policy]),
            "note": "Real source ID collisions; ID-only upsert simulation, not a DB migration.",
        }
        for policy, origins in node_origins.items()
    }
    candidate_path = ROOT / "evaluation/public_aero_corpus_v1/knowledge_candidates.jsonl"
    candidate_payload = candidate_path.read_bytes()
    candidate_rows = [
        json.loads(line) for line in candidate_payload.decode().splitlines() if line.strip()
    ]
    audits = [
        candidate_audit(candidate, texts.get(candidate["source_document_id"]))
        for candidate in candidate_rows
    ]
    doc_by_id = {document["document_id"]: document for document in documents}
    for audit in audits:
        if audit["pair_evaluable"]:
            mapping = audit["ontology_mapping"]
            audit["predicted_pair_presence"] = {}
            for policy, graph in doc_by_id[audit["source_document_id"]]["graphs"].items():
                audit["predicted_pair_presence"][policy] = any(
                    claim["relation"] == mapping["relation"]
                    and (
                        claim["source"]["semantic_key"] in mapping["source_concepts"]
                        or bool(
                            set(claim["source"]["experiment_concept_keys"])
                            & set(mapping["source_concepts"])
                        )
                    )
                    and (
                        claim["target"]["semantic_key"] in mapping["target_concepts"]
                        or bool(
                            set(claim["target"]["experiment_concept_keys"])
                            & set(mapping["target_concepts"])
                        )
                    )
                    for claim in graph["claims"]
                )
    report["candidate_audit"] = {
        "labels_sha256": digest(candidate_payload),
        "candidate_count": len(audits),
        "mapping_counts": dict(Counter(row["mapping_status"] for row in audits)),
        "localization_counts": dict(Counter(row["localization_status"] for row in audits)),
        "pair_evaluable_count": sum(row["pair_evaluable"] for row in audits),
        "is_gold": False,
        "items": audits,
    }
    report["failure_examples"] = [
        {
            "document_id": document["document_id"],
            "policy": policy,
            "failure": "legacy_edge_has_no_saved_provenance",
            "claim": claim,
        }
        for document in documents
        for policy, graph in document["graphs"].items()
        for claim in graph["claims"]
        if not claim["saved_provenance_located"]
    ][:20]
    report["status"] = "complete"
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    return report


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Compact, deterministic metrics; overlapping candidate dimensions stay separate."""
    documents = report.get("documents", [])

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        cuts: dict[str, Any] = {}
        graphs: dict[str, Any] = {}
        for strategy in ("fixed", "structure"):
            values = [row["cutting"][strategy] for row in rows]
            count = sum(value["chunk_count"] for value in values)
            retained = sum(value["heuristic_sentence_spans_retained_whole"] for value in values)
            total = sum(value["heuristic_sentence_span_count"] for value in values)
            indexed = sum(value["total_index_characters"] for value in values)
            covered = sum(value["covered_non_whitespace"] for value in values)
            whitespace_total = sum(value["total_non_whitespace"] for value in values)
            cuts[strategy] = {
                "chunk_count": count,
                "total_index_characters": indexed,
                "mean_characters": indexed / count if count else None,
                "max_characters": max((value["max_characters"] for value in values), default=0),
                "fallback_chunks": sum(value["fallback_chunks"] for value in values),
                "invalid_locator_count": sum(
                    len(value["invalid_locator_indices"]) for value in values
                ),
                "non_whitespace_coverage": covered / whitespace_total if whitespace_total else None,
                "heuristic_sentence_spans_retained_whole": retained,
                "heuristic_sentence_span_count": total,
                "heuristic_sentence_retention": retained / total if total else None,
                "cross_page_marker_chunks": sum(
                    value["cross_page_marker_chunks"] for value in values
                ),
            }
            for extractor in ("lexical_v1", "lexical_v2"):
                policy = strategy + "/" + extractor
                values = [row["graphs"][policy] for row in rows]
                relations: Counter[str] = Counter()
                triples = set()
                for value in values:
                    relations.update(value["relation_counts"])
                    for claim in value["claims"]:
                        triples.add(
                            (
                                claim["source"]["kind"],
                                claim["source"]["name"],
                                claim["relation"],
                                claim["target"]["kind"],
                                claim["target"]["name"],
                            )
                        )
                graphs[policy] = {
                    "edge_count": sum(value["edge_count"] for value in values),
                    "node_occurrence_count": sum(value["node_count"] for value in values),
                    "distinct_label_triplet_count": len(triples),
                    "sum_document_distinct_label_triplet_counts": sum(
                        value["exact_label_triplet_count"] for value in values
                    ),
                    "relation_counts": dict(relations),
                    "diagnostic_relation_count": sum(
                        relations[k] for k in ("CAUSED_BY", "CHECKED_BY", "MITIGATED_BY")
                    ),
                    "saved_provenance_located_edges": sum(
                        value["saved_provenance_located_edges"] for value in values
                    ),
                    "posthoc_reconstructable_edges": sum(
                        value["posthoc_reconstructable_edges"] for value in values
                    ),
                    "synthetic_revision_shared_node_ids": sum(
                        value["same_content_revision_identity_probe"]["shared_node_ids"]
                        for value in values
                    ),
                    "synthetic_revision_shared_edge_ids": sum(
                        value["same_content_revision_identity_probe"]["shared_edge_ids"]
                        for value in values
                    ),
                }
        return {
            "document_count": len(rows),
            "canonical_characters": sum(row["canonical_characters"] for row in rows),
            "cutting": cuts,
            "graphs": graphs,
        }

    examples = []
    for row in documents:
        structured = row["cutting"]["structure"]["chunks"]
        for failure in row["cutting"]["fixed"]["split_sentence_examples"][:1]:
            examples.append(
                {
                    "document_id": row["document_id"],
                    "kind": "fixed_split_regex_span",
                    **failure,
                    "retained_whole_by_structure": any(
                        chunk["start"] <= failure["start"] and chunk["end"] >= failure["end"]
                        for chunk in structured
                    ),
                }
            )
        for failure in row["cutting"]["structure"]["split_sentence_examples"][:1]:
            examples.append(
                {"document_id": row["document_id"], "kind": "structure_split_regex_span", **failure}
            )
        for claim in row["graphs"]["structure/lexical_v1"]["claims"]:
            if claim["relation"] == "CAUSED_BY":
                examples.append(
                    {
                        "document_id": row["document_id"],
                        "kind": "legacy_causal_claim_without_saved_evidence",
                        "source": claim["source"],
                        "relation": claim["relation"],
                        "target": claim["target"],
                        "proof_method": claim["proof_method"],
                        "posthoc_proofs": claim["proofs"][:2],
                        "note": (
                            "Posthoc cooccurrence localization is not original "
                            "provenance or causal support."
                        ),
                    }
                )
    audit = report.get("candidate_audit", {})
    items = audit.get("items", [])
    available = [item for item in items if item["localization_status"] != "missing_document"]
    exclusive = Counter(
        "missing_source"
        if item["localization_status"] == "missing_document"
        else "relation_unsupported"
        if item["mapping_status"] == "predicate_outside_current_ontology"
        else "entities_unsupported_or_ambiguous"
        if item["mapping_status"] == "relation_mapped_entities_outside_vocabulary_or_ambiguous"
        else "pair_evaluable"
        if item["pair_evaluable"]
        else "source_evidence_unlocalized"
        for item in items
    )
    return {
        "schema_version": 1,
        "status": report["status"],
        "baseline_commit": report["baseline_commit"],
        "baseline_source_sha256": report.get("baseline_source_sha256"),
        "current_source_sha256": report["current_source_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "frozen_snapshot_sha256": report.get("frozen_snapshot_sha256"),
        "cutting_scope": report["cutting_scope"],
        "total": aggregate(documents),
        "cohorts": {
            cohort: aggregate(
                [row for row in documents if row["source_lineage"]["cohort"] == cohort]
            )
            for cohort in sorted({row["source_lineage"]["cohort"] for row in documents})
        },
        "documents": [
            {
                "document_id": row["document_id"],
                "cohort": row["source_lineage"]["cohort"],
                "canonical_sha256": row["canonical_sha256"],
                **aggregate([row]),
            }
            for row in documents
        ],
        "unavailable_document_ids": report.get("unavailable_document_ids", []),
        "identity_collision_risk": report.get("identity_collision_risk", {}),
        "candidate_audit": {
            "candidate_count": audit.get("candidate_count", 0),
            "available_source_count": len(available),
            "exclusive_status_counts": dict(exclusive),
            "exclusive_priority": [
                "missing_source",
                "relation_unsupported",
                "entities_unsupported_or_ambiguous",
                "pair_evaluable",
                "source_evidence_unlocalized",
            ],
            "missing_source_count": len(items) - len(available),
            "mapping_counts_all": audit.get("mapping_counts", {}),
            "mapping_counts_available_sources": dict(
                Counter(item["mapping_status"] for item in available)
            ),
            "localization_counts": audit.get("localization_counts", {}),
            "pair_evaluable_count": audit.get("pair_evaluable_count", 0),
            "is_gold": False,
            "note": (
                "Mapping and source availability are overlapping dimensions, "
                "not additive failure counts. No gold accuracy/recall is inferred."
            ),
        },
        "source_located_examples": examples,
        "available_candidate_gaps": [
            {
                "candidate_id": item["candidate_id"],
                "document_id": item["source_document_id"],
                "mapping_status": item["mapping_status"],
                "predicate": item["original_candidate"]["predicate"],
                "subject": item["original_candidate"]["subject"],
                "object": item["original_candidate"]["object"],
                "localized_excerpt_candidates": item["evidence_candidates"][:1],
            }
            for item in available
        ],
        "evaluation_protocol_corrections": [
            "may_indicate is unsupported: a diagnostic indication is not a causal claim.",
            (
                "checks is unsupported: the existing candidate is troubleshooting->"
                "inspected target, not Cause->Check."
            ),
        ],
        "interpretation_limits": [
            "Canonical TXT comparison only; no native PDF parser, OCR or layout evaluation.",
            "Regex sentence retention is not semantic completeness.",
            (
                "fallback_window marks every chunk from a structural unit longer than 800 "
                "characters; sentence/newline-aware splits are included, so it is not "
                "a hard-cut failure rate."
            ),
            "Repeated claim edges are not additional distinct semantic facts.",
            "Exact span location proves provenance, not relation truth.",
            "Synthetic version probe measures ID separation, not actual revision applicability.",
            "Candidate ontology projection is exploratory, not gold or causal equivalence.",
            "No production extraction rules were tuned using these results.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root", type=Path, default=ROOT / ".runtime/datasets/public-aero-corpus-v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".runtime/experiments/knowledge-upgrade/pipeline_comparison.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Use an explicitly frozen subset/refreshed snapshot; verify its actual hashes",
    )
    parser.add_argument("--summary", type=Path, help="Optional compact reproducible JSON summary")
    args = parser.parse_args()
    report = run(args.runtime_root, args.output, snapshot_path=args.snapshot)
    if args.summary and report["status"] == "complete":
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summarize_report(report), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "documents": len(report.get("documents", [])),
            }
        )
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
