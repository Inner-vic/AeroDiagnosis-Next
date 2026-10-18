"""Compile auditable AI source-verified research annotations without claiming human gold.

Canonical offsets are Unicode code points, not bytes. Full source text stays local.
Blind packets intentionally omit author answers, evidence and graph verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


class AnnotationError(ValueError):
    """A release invariant was violated."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def annotation_hash(value: dict[str, Any]) -> str:
    return digest(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values),
        encoding="utf-8",
        newline="\n",
    )


def load_sources(catalog_path: Path) -> dict[str, dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    sources: dict[str, dict[str, Any]] = {}
    for raw in catalog["documents"]:
        item = dict(raw)
        docid = item["document_id"]
        if docid in sources:
            raise AnnotationError(f"Duplicate source: {docid}")
        for key in ("processed_path", "blocks_path", *(["raw_path"] if "raw_path" in item else [])):
            path = Path(item[key])
            if not path.is_absolute():
                path = catalog_path.resolve().parents[2] / path
            item[key] = str(path)
        data = Path(item["processed_path"]).read_bytes()
        if digest(data) != item["canonical_sha256"]:
            raise AnnotationError(f"Source hash mismatch: {docid}")
        if (
            item.get("blocks_sha256")
            and digest(Path(item["blocks_path"]).read_bytes()) != item["blocks_sha256"]
        ):
            raise AnnotationError(f"Source block fingerprint mismatch: {docid}")
        if (
            item.get("raw_path")
            and digest(Path(item["raw_path"]).read_bytes()) != item["raw_sha256"]
        ):
            raise AnnotationError(f"Raw source fingerprint mismatch: {docid}")
        item["text"] = data.decode("utf-8")
        item["blocks"] = read_jsonl(Path(item["blocks_path"]))
        for block in item["blocks"]:
            start, end = block["canonical_start"], block["canonical_end"]
            if item["text"][start:end] != block["text"]:
                raise AnnotationError(f"Block does not match canonical source: {docid}")
        sources[docid] = item
    return sources


def materialize_span(anchor: dict[str, Any], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    docid = anchor["document_id"]
    if docid not in sources:
        raise AnnotationError(f"Undeclared source: {docid}")
    source = sources[docid]
    index = anchor["block_index"]
    if not isinstance(index, int) or index < 0 or index >= len(source["blocks"]):
        raise AnnotationError(f"Invalid block index: {docid}:{index}")
    block = source["blocks"][index]
    needle = anchor["anchor_text"]
    if not needle.strip() or block["text"].count(needle) != 1:
        raise AnnotationError(f"Anchor must occur exactly once: {docid}:{index}: {needle[:60]!r}")
    start = block["canonical_start"] + block["text"].index(needle)
    end = start + len(needle)
    if source["text"][start:end] != needle:
        raise AnnotationError(f"Source span mismatch: {docid}:{index}")
    return {
        "document_id": docid,
        "canonical_start": start,
        "canonical_end": end,
        "text_sha256": digest(needle.encode()),
        "canonical_sha256": source["canonical_sha256"],
        "raw_sha256": source["raw_sha256"],
        "source_locator": block["source_locator"],
        "block_index": index,
    }


def blind_view(sample: dict[str, Any]) -> dict[str, Any]:
    keys = (
        ("sample_id", "query", "query_language", "scope")
        if "query" in sample
        else ("sample_id", "subject", "predicate", "object", "polarity", "certainty", "qualifiers")
    )
    return {key: sample[key] for key in keys if key in sample}


def tokens(text: str) -> list[str]:
    result: list[str] = []
    for word in re.findall(r"[^\W_]+", text.lower()):
        if re.search(r"[\u4e00-\u9fff]", word):
            result.extend(word[i : i + 2] for i in range(max(1, len(word) - 1)))
        elif len(word) > 1 and word not in {
            "the",
            "and",
            "for",
            "with",
            "was",
            "are",
            "that",
            "from",
            "what",
            "which",
            "does",
            "this",
            "were",
        }:
            result.append(word)
    return result


def blind_packets(
    samples: list[dict[str, Any]], sources: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """BM25 supplies an aid, not ground truth; reviewers can search every source."""
    corpus: list[tuple[str, int, Counter[str]]] = []
    df: Counter[str] = Counter()
    for docid, source in sources.items():
        for i, block in enumerate(source["blocks"]):
            counts = Counter(tokens(block["text"]))
            corpus.append((docid, i, counts))
            df.update(counts.keys())
    n = len(corpus)
    avglen = sum(sum(v.values()) for _, _, v in corpus) / max(1, n)
    result = []
    for sample in samples:
        view = blind_view(sample)
        text = sample.get(
            "query",
            " ".join(
                str(sample.get(k, "")) for k in ("subject", "predicate", "object", "qualifiers")
            ),
        )
        terms = set(tokens(text))
        ranked = []
        for docid, i, counts in corpus:
            length = sum(counts.values())
            score = sum(
                math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                * counts[t]
                * 2.2
                / (counts[t] + 1.2 * (0.25 + 0.75 * length / avglen))
                for t in terms
                if counts[t]
            )
            if score:
                ranked.append((score, docid, i))
        ranked.sort(key=lambda r: (-r[0], r[1], r[2]))
        selected = []
        per_doc: Counter[str] = Counter()
        for score, docid, i in ranked:
            if per_doc[docid] >= 3:
                continue
            selected.append(
                {"document_id": docid, "block_index": i, "lexical_pool_score": round(score, 6)}
            )
            per_doc[docid] += 1
            if len(selected) == 15:
                break
        result.append(
            {
                **view,
                "blind_sha256": annotation_hash(view),
                "search_aid": selected,
                "pool_is_exhaustive": False,
                "instructions": (
                    "Reconstruct from all sources; do not infer absence from this lexical pool."
                ),
            }
        )
    random.Random(20261013).shuffle(result)
    return result


def family_cluster_map(
    samples: list[dict[str, Any]], dependency_groups: list[list[str]] | None = None
) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for sample in samples:
        families = sample["family_ids"]
        for family in families:
            find(family)
        for family in families[1:]:
            a, b = find(families[0]), find(family)
            parent[max(a, b)] = min(a, b)
    for group in dependency_groups or []:
        for family in group:
            find(family)
        for family in group[1:]:
            a, b = find(group[0]), find(family)
            parent[max(a, b)] = min(a, b)
    return {family: find(family) for family in parent}


def compile_samples(
    raw: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    dev_families: set[str],
    dependency_groups: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    ids: set[str] = set()
    query_keys: set[str] = set()
    fact_splits: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for sample in raw:
        sid = sample["sample_id"]
        if sid in ids:
            raise AnnotationError(f"Duplicate sample id: {sid}")
        ids.add(sid)
        if "query" in sample:
            if sample["query_language"] not in {"en", "zh", "fr"}:
                raise AnnotationError(f"Undeclared query language: {sid}")
            if not sample["reference_answer"].strip():
                raise AnnotationError(f"Missing reference response: {sid}")
            variant = sample.get(
                "query_variant_type", "translation" if sample.get("paired_variant_of") else "native"
            )
            if variant not in {"native", "translation", "scenario_paraphrase"} or (
                (variant == "native") == bool(sample.get("paired_variant_of"))
            ):
                raise AnnotationError(f"Invalid query variant/parent: {sid}")
        else:
            graph_variant = sample.get("graph_variant_type", "native")
            if graph_variant not in {"native", "negation_stress"} or (
                (graph_variant == "native") == bool(sample.get("paired_claim_of"))
            ):
                raise AnnotationError(f"Invalid graph variant/parent: {sid}")
            if sample["predicate"] not in {
                "CAUSES",
                "CONTRIBUTES_TO",
                "INDICATES",
                "CHECK_WITH",
                "REQUIRES_ACTION",
                "APPLIES_TO",
                "HAS_COMPONENT",
            }:
                raise AnnotationError(f"Undeclared relation predicate: {sid}")
            if sample["polarity"] not in {"positive", "negative"} or sample["certainty"] not in {
                "finding",
                "hypothesis",
                "conditional",
                "reported",
            }:
                raise AnnotationError(f"Invalid qualified claim: {sid}")
        evidence = [materialize_span(a, sources) for a in sample["evidence"]]
        if len(
            {(e["document_id"], e["canonical_start"], e["canonical_end"]) for e in evidence}
        ) != len(evidence):
            raise AnnotationError(f"Duplicate spans: {sid}")
        if "query" in sample:
            qkey = " ".join(sample["query"].casefold().split())
            if qkey in query_keys:
                raise AnnotationError(f"Duplicate query: {sid}")
            query_keys.add(qkey)
            if sample["evidence_requirement"] not in {"all", "any"}:
                raise AnnotationError(f"Invalid evidence semantics: {sid}")
            if sample["answerability"] not in {"answerable", "insufficient_evidence"}:
                raise AnnotationError(f"Invalid answerability: {sid}")
            if sample["answerability"] == "answerable" and not evidence:
                raise AnnotationError(f"Answerable query has no evidence: {sid}")
            if sample["answerability"] == "insufficient_evidence":
                audit = sample.get("no_answer_audit", {})
                if (
                    evidence
                    or not audit.get("reason")
                    or not audit.get("search_terms")
                    or not audit.get("reviewed_document_ids")
                ):
                    raise AnnotationError(
                        f"Insufficiency needs explicit corpus search audit: {sid}"
                    )
                if set(audit["reviewed_document_ids"]) - sources.keys():
                    raise AnnotationError(f"Undeclared no-answer audit source: {sid}")
        elif sample["label"] not in {"supported", "unsupported"} or not evidence:
            raise AnnotationError(
                f"Graph claim requires verdict and contrastive source evidence: {sid}"
            )
        if (
            "query" not in sample
            and sample["label"] == "unsupported"
            and sample.get("unsupported_type") not in {"contradicted", "insufficient_support"}
        ):
            raise AnnotationError(
                f"Unsupported must distinguish contradiction from missing support: {sid}"
            )
        support_ids = [e["document_id"] for e in evidence]
        if not support_ids:
            support_ids = sample.get("evaluation_document_ids", [])
            if not support_ids:
                support_ids = [n["document_id"] for n in sample.get("hard_negatives", [])][:1]
            if not support_ids:
                raise AnnotationError(f"No evaluation family for insufficiency query: {sid}")
        families = sorted({sources[d]["family_id"] for d in support_ids})
        splits = {"dev" if f in dev_families else "test" for f in families}
        if len(splits) != 1:
            raise AnnotationError(f"Required evidence crosses dev/test families: {sid}: {families}")
        split = splits.pop()
        if "query" in sample:
            fact_id = sample["fact_id"]
            if fact_id in fact_splits and fact_splits[fact_id] != split:
                raise AnnotationError(f"Fact leaks across splits: {fact_id}")
            fact_splits[fact_id] = split
        portable = {k: v for k, v in sample.items() if k != "evidence"}
        portable.update(
            evidence=evidence,
            family_ids=families,
            split=split,
            annotation_sha256=annotation_hash(sample),
            review_status="ai_source_candidate",
            human_expert_reviewed=False,
        )
        output.append(portable)
    by_id = {s["sample_id"]: s for s in output}
    for sample in output:
        paired = sample.get("paired_variant_of")
        if paired and (
            paired not in by_id
            or by_id[paired].get("fact_id") != sample["fact_id"]
            or by_id[paired]["split"] != sample["split"]
        ):
            raise AnnotationError(f"Invalid translated pair: {sample['sample_id']}")
    for sample in output:
        paired_claim = sample.get("paired_claim_of")
        if not paired_claim:
            continue
        parent = by_id.get(paired_claim)
        if (
            not parent
            or "query" in parent
            or any(
                sample.get(key) != parent.get(key)
                for key in (
                    "subject",
                    "predicate",
                    "object",
                    "certainty",
                    "qualifiers",
                    "family_ids",
                    "split",
                    "claim_group_id",
                )
            )
            or sample["polarity"] == parent["polarity"]
        ):
            raise AnnotationError(f"Invalid paired negation claim: {sample['sample_id']}")
    clusters = family_cluster_map(output, dependency_groups)
    for sample in output:
        sample["statistical_cluster"] = clusters[sample["family_ids"][0]]
    return output


def apply_reviews(
    samples: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    reviews: list[dict[str, Any]],
    reconciliations: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    def indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out = {r["sample_id"]: r for r in rows}
        if len(out) != len(rows):
            raise AnnotationError("Duplicate review/reconciliation id")
        return out

    review_map, reconciliation_map = indexed(reviews), indexed(reconciliations)
    kept, excluded = [], []
    for sample in samples:
        sid = sample["sample_id"]
        if sid not in review_map or sid not in reconciliation_map:
            raise AnnotationError(f"Missing independent review: {sid}")
        review, reconciliation = review_map[sid], reconciliation_map[sid]
        raw = raw_by_id[sid]
        if review.get("reviewer") == raw["author"] or not review.get("reviewer"):
            raise AnnotationError(f"Self review is not independent: {sid}")
        if review.get("blind_sha256") != annotation_hash(blind_view(raw)) and review.get(
            "correction_verified_blind_sha256"
        ) != annotation_hash(blind_view(raw)):
            raise AnnotationError(f"Blind review no longer binds this query/claim: {sid}")
        if reconciliation.get("final_annotation_sha256") != sample["annotation_sha256"]:
            raise AnnotationError(f"Reconciliation does not bind final annotation: {sid}")
        if not reconciliation.get("reason"):
            raise AnnotationError(f"Reconciliation needs rationale: {sid}")
        if reconciliation["decision"] == "exclude":
            excluded.append(sid)
            continue
        if reconciliation["decision"] != "accept":
            raise AnnotationError(f"Invalid reconciliation decision: {sid}")
        if not review.get("reconstructed_answerability", review.get("reconstructed_label")):
            raise AnnotationError(f"Review missing reconstructed judgment: {sid}")
        if review.get("reconstructed_answerability") == "answerable" and not review.get("evidence"):
            raise AnnotationError(f"Source reconstruction needs evidence: {sid}")
        for anchor in review.get("evidence", []):
            materialize_span(anchor, sources)
        final = sample.get("answerability", sample.get("label"))
        verdict = review.get("reconstructed_answerability", review.get("reconstructed_label"))
        if final != verdict and reconciliation.get("disposition") != "adjudicated_disagreement":
            raise AnnotationError(f"Unresolved review disagreement: {sid}")
        kept.append(
            {
                **sample,
                "review_status": "ai_cross_source_verified",
                "reviewer": review["reviewer"],
                "reconciliation": reconciliation,
            }
        )
    return kept, excluded


def distribution(samples: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(s.get(key, "missing")) for s in samples).items()))


def build(
    catalog: Path,
    drafts: Path,
    protocol_path: Path,
    output: Path,
    *,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    sources = load_sources(catalog)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw = []
    for filename in (
        "engineering_queries.jsonl",
        "investigation_queries.jsonl",
        "engineering_graph.jsonl",
        "investigation_graph.jsonl",
        "graph_negation_stress.jsonl",
    ):
        path = drafts / filename
        if path.exists():
            raw.extend(read_jsonl(path))
    pre_excluded = [r["sample_id"] for r in protocol.get("pre_results_excluded_samples", [])]
    raw = [r for r in raw if r["sample_id"] not in pre_excluded]
    compiled = compile_samples(
        raw, sources, set(protocol["dev_families"]), protocol.get("source_dependency_groups", [])
    )
    excluded: list[str] = pre_excluded.copy()
    if review_dir:
        reviews = read_jsonl(review_dir / "engineering_reviews.jsonl") + read_jsonl(
            review_dir / "investigation_reviews.jsonl"
        )
        compiled, review_excluded = apply_reviews(
            compiled,
            {s["sample_id"]: s for s in raw},
            reviews,
            read_jsonl(review_dir / "reconciliation.jsonl"),
            sources,
        )
        excluded.extend(review_excluded)
    output.mkdir(parents=True, exist_ok=True)
    queries = [s for s in compiled if "query" in s]
    graph = [s for s in compiled if "query" not in s]
    write_jsonl(output / "queries.jsonl", queries)
    write_jsonl(output / "graph_claims.jsonl", graph)
    repo = protocol_path.resolve().parents[2]
    portable_sources = []
    for s in sources.values():
        keep = {
            k: s[k]
            for k in (
                "document_id",
                "logical_document_id",
                "family_id",
                "title",
                "publisher_id",
                "language",
                "source_type",
                "canonical_sha256",
                "source_url",
                "raw_sha256",
                "raw_bytes",
                "rights",
                "document_description",
                "artifact_description",
            )
        }
        for key in ("processed_path", "blocks_path"):
            keep[key] = Path(s[key]).resolve().relative_to(repo).as_posix()
        keep["blocks_sha256"] = digest(Path(s["blocks_path"]).read_bytes())
        raw_path = Path(s["processed_path"]).parents[2] / "raw" / s["artifact_description"]["name"]
        if digest(raw_path.read_bytes()) != s["raw_sha256"]:
            raise AnnotationError(f"Raw input fingerprint mismatch: {s['document_id']}")
        keep["raw_path"] = raw_path.resolve().relative_to(repo).as_posix()
        keep["canonical_character_count"] = len(s["text"])
        keep["split"] = "dev" if s["family_id"] in protocol["dev_families"] else "test"
        portable_sources.append(keep)
    write_json(
        output / "corpus_manifest.json",
        {
            "documents": portable_sources,
            "upstream_snapshot_sha256": protocol["upstream_snapshot_sha256"],
        },
    )
    audit = {
        "query_count": len(queries),
        "distinct_fact_count": len({s["fact_id"] for s in queries}),
        "graph_claim_count": len(graph),
        "graph_claim_group_count": len({g.get("claim_group_id", g["sample_id"]) for g in graph}),
        "graph_variants": distribution(graph, "graph_variant_type"),
        "graph_polarity_label": dict(
            sorted(Counter(f"{g['polarity']}:{g['label']}" for g in graph).items())
        ),
        "excluded_sample_ids": excluded,
        "query_split": distribution(queries, "split"),
        "task_type": distribution(queries, "task_type"),
        "query_language": distribution(queries, "query_language"),
        "engine_domain": distribution(queries, "engine_domain"),
        "query_variants": dict(
            sorted(
                Counter(
                    q.get(
                        "query_variant_type",
                        "translation" if q.get("paired_variant_of") else "native",
                    )
                    for q in queries
                ).items()
            )
        ),
        "native_query_count": sum(not q.get("paired_variant_of") for q in queries),
        "test_statistical_cluster_count": len(
            {q["statistical_cluster"] for q in queries if q["split"] == "test"}
        ),
        "answerability": distribution(queries, "answerability"),
        "graph_label": distribution(graph, "label"),
        "graph_certainty": distribution(graph, "certainty"),
        "evidence_span_count": sum(len(s["evidence"]) for s in compiled),
        "unique_evidence_span_count": len(
            {
                (e["document_id"], e["canonical_start"], e["canonical_end"])
                for s in compiled
                for e in s["evidence"]
            }
        ),
        "statistical_clusters": sorted({s["statistical_cluster"] for s in compiled}),
        "source_count": len(sources),
        "family_count": len({s["family_id"] for s in sources.values()}),
        "all_span_hashes_verified": True,
        "raw_and_block_fingerprints_verified": True,
        "human_expert_reviewed": False,
        "review_status": "ai_cross_source_verified" if review_dir else "ai_source_candidate",
        "publication_ready_expert_gold": False,
        "rank_qrels_exhaustive": False,
        "graph_extraction_gold": False,
        "limitations": [
            "Purposive sample, not representative fleet prevalence",
            "Incomplete alternative evidence judgments",
            "Same-model AI reviewers are not aviation experts",
            "Small number of independent source families",
            "Canonical PDF offsets do not prove visual table correctness",
        ],
    }
    write_json(output / "quality_audit.json", audit)
    if not review_dir:
        write_jsonl(output / "blind_review_packet.jsonl", blind_packets(raw, sources))
    fingerprints = {
        path.name: digest(path.read_bytes())
        for path in sorted(output.iterdir())
        if path.is_file()
        and path.suffix in {".json", ".jsonl", ".md"}
        and path.name != "release.json"
    }
    write_json(
        output / "release.json",
        {
            "benchmark_id": protocol["benchmark_id"],
            "protocol_sha256": digest(protocol_path.read_bytes()),
            "files": fingerprints,
            "builder_sha256": digest(Path(__file__).read_bytes()),
            "review_status": audit["review_status"],
            "human_expert_reviewed": False,
        },
    )
    return audit


def validate_release(output: Path, protocol_path: Path) -> dict[str, Any]:
    """Revalidate published annotations directly from portable offsets, without draft files."""
    release = json.loads((output / "release.json").read_text(encoding="utf-8"))
    if digest(protocol_path.read_bytes()) != release["protocol_sha256"]:
        raise AnnotationError("Protocol fingerprint mismatch")
    if digest(Path(__file__).read_bytes()) != release["builder_sha256"]:
        raise AnnotationError("Builder fingerprint mismatch")
    for name, expected in release["files"].items():
        if digest((output / name).read_bytes()) != expected:
            raise AnnotationError(f"Release file fingerprint mismatch: {name}")
    sources = load_sources(output / "corpus_manifest.json")
    samples = read_jsonl(output / "queries.jsonl") + read_jsonl(output / "graph_claims.jsonl")
    raw = []
    added = {
        "family_ids",
        "split",
        "annotation_sha256",
        "review_status",
        "human_expert_reviewed",
        "statistical_cluster",
        "reviewer",
        "reconciliation",
    }
    for sample in samples:
        original = {k: v for k, v in sample.items() if k not in added and k != "evidence"}
        anchors = []
        for span in sample["evidence"]:
            source = sources[span["document_id"]]
            start, end = span["canonical_start"], span["canonical_end"]
            if not 0 <= start < end <= len(source["text"]):
                raise AnnotationError("Invalid released span interval")
            if (
                span["canonical_sha256"] != source["canonical_sha256"]
                or span["raw_sha256"] != source["raw_sha256"]
            ):
                raise AnnotationError("Released source fingerprint mismatch")
            text = source["text"][start:end]
            if digest(text.encode()) != span["text_sha256"]:
                raise AnnotationError("Released span fingerprint mismatch")
            anchor = {
                "document_id": span["document_id"],
                "block_index": span["block_index"],
                "anchor_text": text,
            }
            if materialize_span(anchor, sources) != span:
                raise AnnotationError("Released span locator mismatch")
            anchors.append(anchor)
        original["evidence"] = anchors
        if annotation_hash(original) != sample["annotation_sha256"]:
            raise AnnotationError(
                f"Released annotation fingerprint mismatch: {sample['sample_id']}"
            )
        if sample["human_expert_reviewed"] is not False:
            raise AnnotationError("AI annotation cannot become expert gold")
        raw.append(original)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    replay = compile_samples(
        raw, sources, set(protocol["dev_families"]), protocol.get("source_dependency_groups", [])
    )
    for actual, expected in zip(samples, replay, strict=True):
        if actual["split"] != expected["split"] or actual["family_ids"] != expected["family_ids"]:
            raise AnnotationError("Released family/split mismatch")
    return {
        "validated_samples": len(samples),
        "validated_sources": len(sources),
        "all_fingerprints_match": True,
        "human_expert_reviewed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--drafts", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--validate-release", action="store_true")
    args = parser.parse_args()
    if args.validate_release:
        print(json.dumps(validate_release(args.output, args.protocol), indent=2))
        return
    if not args.catalog or not args.drafts:
        parser.error("--catalog and --drafts are required to compile")
    print(
        json.dumps(
            build(
                args.catalog, args.drafts, args.protocol, args.output, review_dir=args.review_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
