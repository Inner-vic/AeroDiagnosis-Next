"""Probe real upstream LangExtract grounding against cached pilot model outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import langextract as lx
from langextract import data, resolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(
        Path("evaluation/aero_evidence_benchmark_v1/corpus_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    sources = {d["document_id"]: d for d in manifest["documents"]}
    ranges = {
        r["range_id"]: r
        for r in map(
            json.loads,
            Path("evaluation/gas_path_scope_v1/range_annotations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    rows = json.loads((args.run / "graph_outputs.json").read_text(encoding="utf-8"))
    aligner = resolver.Resolver()
    observations = []
    documents = []
    for row in rows:
        if row["method"] != "G2":
            continue
        span = ranges[row["task"]]["evidence"]
        source = sources[span["document_id"]]
        snapshot = Path(source["processed_path"]).read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != span["canonical_sha256"]:
            raise ValueError("Immutable source changed")
        text = snapshot.decode("utf-8")[span["canonical_start"] : span["canonical_end"]]
        extractions = []
        for index, candidate in enumerate(row["prediction"].get("claims", [])):
            extraction = data.Extraction(
                extraction_class="qualified_relation",
                extraction_text=candidate["evidence_quote"],
                attributes={
                    "predicate": candidate["predicate"],
                    "subject": candidate["subject"]["surface"],
                    "object": candidate["object"]["surface"],
                    "polarity": candidate["polarity"],
                    "certainty": candidate["certainty"],
                },
            )
            aligned = list(
                aligner.align(
                    [extraction],
                    source_text=text,
                    token_offset=0,
                    char_offset=0,
                    enable_fuzzy_alignment=False,
                    accept_match_lesser=False,
                )
            )
            result = aligned[0] if aligned else None
            interval = result.char_interval if result else None
            original = text[interval.start_pos : interval.end_pos] if interval else None
            exact = original == candidate["evidence_quote"]
            observations.append(
                {
                    "unit_id": row["task"],
                    "candidate_index": index,
                    "upstream_alignment_status": result.alignment_status.value
                    if result and result.alignment_status
                    else None,
                    "char_interval": {"start": interval.start_pos, "end": interval.end_pos}
                    if interval
                    else None,
                    "canonical_start": span["canonical_start"] + interval.start_pos
                    if interval
                    else None,
                    "canonical_end": span["canonical_start"] + interval.end_pos
                    if interval
                    else None,
                    "verbatim_original_match": exact,
                    "semantic_support": "not_established_by_grounding",
                    "entity_role_and_condition_support": "requires_separate_domain_validation",
                }
            )
            if result and exact:
                extractions.append(result)
        documents.append(
            data.AnnotatedDocument(document_id=row["task"], text=text, extractions=extractions)
        )
    source_bytes = Path(resolver.__file__).read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes
    ).hexdigest()
    summary = {
        "upstream": "google/langextract",
        "package_version": importlib.metadata.version("langextract"),
        "installed_resolver_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "installed_resolver_git_blob_sha1": blob,
        "github_reviewed_resolver_blob_sha1": "bf37e9a0dc73ab742068b04cf12a127c9252ad19",
        "installed_resolver_matches_reviewed_git_blob": blob
        == "bf37e9a0dc73ab742068b04cf12a127c9252ad19",
        "api_calls": 0,
        "additional_api_cost_cny": 0,
        "model_outputs": "unchanged cached first-pilot G2 predictions",
        "source_units": len(documents),
        "candidate_quotes": len(observations),
        "verbatim_aligned_quotes": sum(r["verbatim_original_match"] for r in observations),
        "production_adoption": False,
        "limits": "This probes library grounding only, not extraction F1 or relation correctness.",
        "observations": observations,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "source_script.py").write_bytes(Path(__file__).read_bytes())
    lx.io.save_annotated_documents(
        documents, output_name="grounded.jsonl", output_dir=str(args.output)
    )
    html = lx.visualize(str(args.output / "grounded.jsonl"))
    (args.output / "grounding-review.html").write_text(
        html.data if hasattr(html, "data") else html, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "package_version",
                    "source_units",
                    "candidate_quotes",
                    "verbatim_aligned_quotes",
                    "installed_resolver_matches_reviewed_git_blob",
                    "additional_api_cost_cny",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
