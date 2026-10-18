"""Freeze available public corpus inputs without silently accepting historical drift."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation/public_aero_corpus_v1/source_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(runtime: Path, *, allow_refreshed: bool = False) -> dict[str, Any]:
    if (runtime / "snapshot.json").exists():
        raise FileExistsError("snapshot already frozen; use a new runtime root")
    spec = importlib.util.spec_from_file_location(
        "corpus_builder", ROOT / "scripts/build_public_corpus.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    processed = runtime / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    documents = []
    unavailable = []
    for doc in manifest["documents"]:
        processor = doc["processor"]
        if processor == "ntrs_pdf_with_text_reference":
            inputs = [a for a in doc["artifacts"] if a["media_type"] == "text/plain"]
        elif processor in {"parallel_markup", "jats_xml"}:
            inputs = [a for a in doc["artifacts"] if a["media_type"] == "application/xml"]
        else:
            inputs = doc["artifacts"]
        records = []
        for artifact in inputs:
            path = runtime / "raw" / artifact["name"]
            if not path.is_file():
                break
            actual = sha(path)
            size = path.stat().st_size
            matches = actual == artifact["sha256"] and size == artifact["bytes"]
            if not matches and not allow_refreshed:
                break
            records.append(
                {
                    "name": artifact["name"],
                    "sha256": actual,
                    "bytes": size,
                    "expected_sha256": artifact["sha256"],
                    "expected_bytes": artifact["bytes"],
                    "url": artifact["url"],
                }
            )
        if len(records) != len(inputs):
            unavailable.append(doc["document_id"])
            continue
        path = runtime / "raw" / inputs[0]["name"]
        if processor == "ntrs_pdf_with_text_reference":
            text = builder._normalize_text(path.read_text(encoding="utf-8-sig"))
        elif processor == "pdf_pages":
            pages, _ = builder._pdf_pages(path)
            selected = doc.get("selected_pages", list(range(1, len(pages) + 1)))
            text = "\n\n".join(f"[PDF_PAGE {n}]\n{pages[n - 1]}" for n in selected)
        elif processor in {"parallel_markup", "jats_xml"}:
            text, _ = builder._xml_text(path, article_body_only=processor == "jats_xml")
        elif processor == "faa_sdr_csv":
            text, _ = builder._sdr_sample(doc, path, processed)
        else:
            raise ValueError(f"unsupported processor: {processor}")
        output = processed / (doc["document_id"].lower() + ".txt")
        output.write_text(text, encoding="utf-8")
        exact = all(
            a["sha256"] == a["expected_sha256"] and a["bytes"] == a["expected_bytes"]
            for a in records
        )
        documents.append(
            {
                "document_id": doc["document_id"],
                "processed_path": output.relative_to(runtime).as_posix(),
                "sha256": sha(output),
                "cohort": "historical_exact" if exact else "refreshed",
                "artifacts": records,
            }
        )
    if not documents:
        raise ValueError("no eligible real documents; restore raw artifacts first")
    snapshot = {
        "dataset_id": "public-aero-recovered-20261012",
        "source_manifest_sha256": sha(MANIFEST),
        "documents": documents,
        "unavailable_document_ids": unavailable,
        "allow_refreshed": allow_refreshed,
        "notes": [
            "Cohort uses canonical input artifacts only; auxiliary JSON/HTML do not enter text.",
            "Refreshed inputs form a new snapshot, not a reproduction of corpus v1.1.",
            "Original review statuses and labels are unchanged.",
        ],
    }
    (runtime / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--allow-refreshed",
        action="store_true",
        help="Explicitly freeze downloaded hash-drifted inputs in a separate cohort",
    )
    args = parser.parse_args()
    snapshot = materialize(args.runtime_root.resolve(), allow_refreshed=args.allow_refreshed)
    print(
        json.dumps(
            {
                "documents": len(snapshot["documents"]),
                "unavailable": snapshot["unavailable_document_ids"],
            }
        )
    )


if __name__ == "__main__":
    main()
