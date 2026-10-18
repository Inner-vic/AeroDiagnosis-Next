"""Resolve official landing summaries to full source PDFs; preserve both captures."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect_quality_sources import fetch

ROOT = Path("evaluation/strict_graph_holdout_v1_candidate")


def main():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf8"))
    targets = []
    for k in ["200702219", "200607815", "200606510", "aaib-g-nvsb"]:
        text = (ROOT / "sources" / (k + ".txt")).read_text(encoding="utf8")
        urls = re.findall(r"https?[^\s)<>]+\.pdf[^\s)<>]*", text, re.I)
        url = urls[0].replace(
            "assets.digital.cabinet-office.gov.uk", "assets.publishing.service.gov.uk"
        )
        agency = next(r["publisher"] for r in manifest["documents"] if r["document_id"] == k)
        targets.append((k, agency, url))

    def one(row):
        k, a, u = row
        try:
            fetch((k + "-full", a, u, "pdf"), ROOT)
            return dict(
                document_id=k,
                status="full_pdf_captured",
                full_pdf_document_id=k + "-full",
                full_pdf_url=u,
                raw_sha256=json.loads(
                    (ROOT / "sources" / (k + "-full.capture.json")).read_text(encoding="utf8")
                )["original_pdf_sha256"],
            )
        except Exception as e:
            return dict(document_id=k, status="full_pdf_failed", error_type=type(e).__name__)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, targets))
    for k, reg in [("aaib-g-flbc", "G-FLBC"), ("aaib-g-bosn", "G-BOSN")]:
        v = json.loads((ROOT / "sources" / (k + ".capture.json")).read_text(encoding="utf8"))
        pages = v["page_texts"]
        selected = [i for i, p in enumerate(pages) if reg in p[:550] and "CONTENTS" not in p[:2000]]
        selected = [i for i in selected if i > 2]
        text = "\n\n".join(pages[i] for i in selected)
        (ROOT / "sources" / (k + "-report-only.txt")).write_text(
            text, encoding="utf8", newline="\n"
        )
        results.append(
            dict(
                document_id=k,
                status="individual_report_pages_extracted",
                page_indices_zero_based=selected,
                derived_text_path="sources/" + k + "-report-only.txt",
                monthly_pdf_saved=True,
            )
        )
    (ROOT / "full-report-supplement.json").write_text(
        json.dumps(dict(source_only=True, items=results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
