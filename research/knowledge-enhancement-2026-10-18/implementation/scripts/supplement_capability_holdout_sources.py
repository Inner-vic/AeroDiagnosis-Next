# ruff: noqa: E501
"""Repair source-only event grouping and capture explicitly incorporated EASA ADs."""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

ROOT = Path("evaluation/retrieval_capability_holdout_v1_candidate")


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def get(url):
    if urllib.parse.urlparse(url).hostname != "ad.easa.europa.eu":
        raise ValueError("Unexpected source host")
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "AeroDiagnosis source-only research"}),
        timeout=45,
    ).read()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def supplement(row):
    text = (
        (ROOT / row["saved_source_path"]).read_text(encoding="utf8")
        if not str(row["saved_source_path"]).startswith("evaluation/")
        else Path(row["saved_source_path"]).read_text(encoding="utf8")
    )
    # Only directives explicitly incorporated in the required-action section, not arbitrary historical citations.
    actions = re.split(r"(?m)^\(g\)", text, maxsplit=1)[-1]
    found = re.findall(
        r"(?:EASA|European Union Aviation Safety Agency) AD\s+(\d{4})-\s*(\d{4})(R\d+)?", actions
    )
    if not found:
        return {"document_id": row["document_id"], "status": "no_easa_action_reference"}
    year, number, revision = found[0]
    number = year + "-" + number + revision
    base = ROOT / "sources" / ("easa-" + number)
    try:
        html = Path(str(base) + ".html")
        if not html.exists():
            html.write_bytes(get("https://ad.easa.europa.eu/ad/" + number))
        parser = Links()
        parser.feed(html.read_text(encoding="utf8"))
        choices = [
            urllib.parse.urljoin("https://ad.easa.europa.eu", x)
            for x in parser.links
            if "/blob/" in x and "EASA_AD_" in x and ".pdf" in x
        ]
        if len(choices) != 1:
            raise ValueError("Expected one official AD PDF")
        pdf = Path(str(base) + ".pdf")
        if not pdf.exists():
            pdf.write_bytes(get(choices[0]))
        reader = PdfReader(io.BytesIO(pdf.read_bytes()))
        pages = [p.extract_text(extraction_mode="layout") for p in reader.pages]
        # Preserve pages and natural text; do not infer content of inaccessible service bulletins.
        full = "\n\n".join(pages)
        Path(str(base) + ".full.txt").write_text(full, encoding="utf8", newline="\n")
        return {
            "document_id": row["document_id"],
            "status": "captured",
            "easa_id": number,
            "url": "https://ad.easa.europa.eu/ad/" + number,
            "pdf_url": choices[0],
            "pdf_path": pdf.as_posix(),
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "text_path": str(base) + ".full.txt",
            "canonical_text_sha256": digest(full),
            "page_count": len(pages),
            "characters": len(full),
            "text_pages": [{"page": i + 1, "text": t} for i, t in enumerate(pages)],
            "scope": "Separate EASA jurisdiction; FAA exceptions take precedence only for FAA questions; inaccessible SB details remain unknown.",
        }
    except Exception as e:
        return {
            "document_id": row["document_id"],
            "easa_id": number,
            "status": "failed",
            "safe_error": type(e).__name__ + ":" + str(e)[:150],
        }


def main():
    m = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf8"))
    review = json.loads((ROOT / "source-review-before-supplement.json").read_text(encoding="utf8"))
    items = {x["document_id"]: x for x in review["items"]}
    families = [
        ("PW1127GA-JM_IBR7_trigger_event", ["2026-10360", "2026-08106", "2025-03441"]),
        ("V2533-A5_HPT1_uncontained_trigger", ["2025-17066", "2024-03442"]),
        ("LEAP_iron_inclusion_material_family", ["2025-13660", "2025-11340", "2024-01166"]),
        ("Trent_IP8_HP3_air_tube_program", ["2026-11179", "2026-04448", "2023-28861"]),
        ("V2500_HPT1_AUSI_quality_program", ["2025-18469", "2025-12242"]),
    ]
    usable = []
    for row in m["sources"]:
        key = row["document_id"]
        if key not in items or items[key]["status"] == "reject":
            continue
        r = {
            **row,
            "source_quality_review": items[key],
            "source_family": row.get("source_cluster", key),
            "source_review_status": items[key]["status"],
            "limited_incorporation_by_reference": bool(items[key]["external_reference_limitation"]),
            "detailed_actions_present": "not_assumed_from_keyword_regex",
        }
        for name, members in families:
            if key in members:
                r["source_family"] = name
        # Renamed canonical hashes are decoded LF text hashes, distinct from file byte hashes.
        for field in ["full_source_path", "saved_source_path"]:
            if field in r:
                q = Path(r[field])
                q = q if q.exists() else ROOT / q
                r[field + "_raw_byte_sha256"] = hashlib.sha256(q.read_bytes()).hexdigest()
        usable.append(r)
    # Accepted self-contained sources first; remaining limited sources require actual incorporated text or restricted questions.
    usable.sort(
        key=lambda r: (
            r["source_review_status"] != "accept",
            not r.get("selected_for_scope_review", False),
            r["document_id"],
        )
    )
    chosen = []
    seen = set()
    for row in usable:
        if row["source_family"] in seen:
            continue
        chosen.append(row)
        seen.add(row["source_family"])
        if len(chosen) == 40:
            break
    assert len(chosen) == 40
    write(
        ROOT / "source-selection-after-review.json",
        {
            "source_selection_before_questions": True,
            "source_family_count": len(seen),
            "sources": chosen,
            "families": families,
            "rule": "Accepted sources prioritized, at most one source per reviewed event/technical family; no retrieval outcomes used.",
        },
    )
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(supplement, chosen))
    write(
        ROOT / "incorporated-easa-manifest.json",
        {"sources": results, "source_only": True, "not_all_external_SB_details_available": True},
    )
    print(
        "selected",
        len(chosen),
        "EASA",
        {s: sum(x["status"] == s for x in results) for s in {x["status"] for x in results}},
        flush=True,
    )


if __name__ == "__main__":
    main()
