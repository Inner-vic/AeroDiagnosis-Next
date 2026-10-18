"""Materialize source-bound holdout candidates without ingesting gold into the app."""

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = ROOT / "evaluation/gas_path_holdout_v1_candidate"


def digest(v):
    return hashlib.sha256(v.encode("utf-8")).hexdigest()


if (r / "freeze.json").exists() or (r / "reconciliation.json").exists():
    raise RuntimeError(
        "Reviewed or frozen holdout cannot be regenerated in place; use a new version"
    )

m = json.loads((r / "source_manifest.json").read_text(encoding="utf-8"))
sources = {x["document_id"]: x for x in m["documents"]}
queries = []
blind = []
seen = set()
counts = collections.Counter()
for name in ["report_annotations.txt", "literature_annotations.txt"]:
    for line in (r / name).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        doc, indices, kind, q, a = line.split("|")
        source = sources[doc]
        assert source["status"].startswith("downloaded"), doc
        text = (r / "sources" / f"{doc}.txt").read_text(encoding="utf-8")
        assert digest(text) == source["canonical_sha256"]
        blocks = json.loads((r / "sources" / f"{doc}.blocks.json").read_text(encoding="utf-8"))
        ev = []
        for n in indices.split(","):
            block = blocks[int(n)]
            assert text[block["start"] : block["end"]] == block["text"], block["block_id"]
            ev.append(
                dict(
                    document_id=doc,
                    block_id=block["block_id"],
                    canonical_start=block["start"],
                    canonical_end=block["end"],
                    text_sha256=digest(block["text"]),
                    canonical_sha256=source["canonical_sha256"],
                    source_locator={"section": block["section"], "block_index": int(n)},
                    source_url=source["url"],
                )
            )
        full = f"依据资料「{doc}」：{q}"  # noqa: RUF001
        assert full not in seen
        seen.add(full)
        i = f"holdout-q{len(queries) + 1:03d}"
        stratum = (
            "investigation"
            if name.startswith("report")
            else ("teaching" if source["source_type"] == "teaching" else "research")
        )
        is_abstract = doc in ["sjtu-kd-deepgnn", "hkxb-diagnosis-review", "tjjs-onboard-review"]
        row = dict(
            sample_id=i,
            query=full,
            query_language="zh",
            reference_answer=a,
            minimal_facts=[
                dict(fact_key=i + "-answer", text=a, allowed_block_ids=[x["block_id"] for x in ev])
            ],
            evidence=ev,
            evidence_requirement="all",
            family_ids=[source["family_id"]],
            statistical_cluster=source["family_id"],
            fact_cluster=doc,
            task_type=kind,
            stratum=stratum,
            evidence_scope="saved_abstract_only" if is_abstract else "saved_source_text",
            answerability="insufficient_evidence"
            if kind == "insufficient_evidence"
            else "answerable",
            split="test_candidate",
            review_status="author_source_checked_pending_independent_review",
            human_expert_reviewed=False,
            human_source_reviewed=False,
            author="codex:root",
            prior_exposure="new source authoring; no system predictions or test tuning",
            scope={
                "source_revision": "captured 2026-10-14",
                "use": "source_scoped_knowledge_not_current_maintenance_authority",
            },
            scope_label="supporting_systems"
            if doc in ["aaib-n197dn", "ao-2015-124", "ao-2022-066", "a08c0108"]
            else "gas_path_or_diagnostic_method",
        )
        row["correlation_group"] = (
            "nasa-gas-path-health-management"
            if source["publisher"] == "NASA"
            else source["family_id"]
        )
        row["annotation_sha256"] = digest(json.dumps(row, ensure_ascii=False, sort_keys=True))
        queries.append(row)
        blind.append(
            {
                k: row[k]
                for k in [
                    "sample_id",
                    "query",
                    "query_language",
                    "family_ids",
                    "task_type",
                    "stratum",
                    "evidence_scope",
                    "scope",
                ]
            }
        )
        counts[stratum] += 1
assert len(queries) == 120, len(queries)
generation = [{k: x[k] for k in ["sample_id", "query", "query_language"]} for x in queries]
for name, rows in [
    ("queries.jsonl", queries),
    ("blind_queries.jsonl", blind),
    ("generation_queries.jsonl", generation),
]:
    (r / name).write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows),
        encoding="utf-8",
    )
used = {x["evidence"][0]["document_id"] for x in queries}
old = json.loads(
    (ROOT / "evaluation/public_aero_corpus_v2/sources-international.json").read_text(
        encoding="utf-8"
    )
)["documents"]
old_urls = {x["landing_url"].rstrip("/") for x in old}
overlap = [
    x["document_id"]
    for x in sources.values()
    if x["document_id"] in used and x["url"].rstrip("/") in old_urls
]
assert not overlap, overlap
for x in m["documents"]:
    x["candidate_used"] = x["document_id"] in used
    if not x["candidate_used"]:
        x["exclusion_reason"] = "reserve source; not counted as a test question source"
    if x["document_id"] in ["sjtu-kd-deepgnn", "hkxb-diagnosis-review", "tjjs-onboard-review"]:
        x["completeness"] = "abstract_and_metadata_not_full_paper"
    x.setdefault("rights_status", "public_access_not_full_redistribution_license_confirmation")
m.update(
    status="candidate_not_frozen",
    candidate_queries=120,
    selected_documents=len(used),
    statistical_clusters=len({x["statistical_cluster"] for x in queries}),
    strata=dict(counts),
    excluded_prior_landing_url_overlap=overlap,
    main_limitations=[
        "AI-authored; source-independent holdout does not establish model pretraining independence",
        "same source or causal fact is not an independent statistical sample",
        "15 research questions use abstracts only and must be reported separately",
        "no live knowledge ingestion or paid model evaluation has been run",
    ],
)
(r / "source_manifest.json").write_text(
    json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
)
receipt = dict(
    query_count=len(queries),
    documents=len(used),
    clusters=m["statistical_clusters"],
    conservative_correlation_groups=len({x["correlation_group"] for x in queries}),
    publishers=sorted({sources[i]["publisher"] for i in used}),
    strata=dict(counts),
    task_types=dict(collections.Counter(x["task_type"] for x in queries)),
    query_sha256=digest((r / "queries.jsonl").read_text(encoding="utf-8")),
    blind_query_sha256=digest((r / "blind_queries.jsonl").read_text(encoding="utf-8")),
    evidence_bindings=sum(len(x["evidence"]) for x in queries),
    source_offsets_valid=True,
    duplicate_questions=0,
    prior_source_url_overlaps=overlap,
    independent_review="pending",
    frozen=False,
    paid_calls=0,
)
(r / "validation.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False))
