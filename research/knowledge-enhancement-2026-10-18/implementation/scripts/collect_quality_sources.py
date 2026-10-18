"""Source-only public capture for a new study; no question or answer access."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sources():
    rows = []
    for identifier in [
        "a99w0234",
        "a06p0010",
        "a10p0388",
        "a10q0098",
        "a12o0074",
        "a11c0152",
        "a94h0001",
        "a15c0146",
        "a22p0023",
        "a19c0070",
    ]:
        year = "19" + identifier[1:3] if int(identifier[1:3]) > 80 else "20" + identifier[1:3]
        rows.append(
            (
                identifier,
                "TSB",
                f"https://www.tsb.gc.ca/eng/rapports-reports/aviation/{year}/{identifier}/{identifier}.html",
                "web",
            )
        )
    for identifier in ["200402819", "ao-2017-034", "200504188", "ao-2020-003"]:
        rows.append(
            (identifier, "ATSB", f"https://www.atsb.gov.au/investigations/{identifier}", "web")
        )
    rows.extend(
        [
            (
                "aaib-g-lmsa",
                "AAIB",
                "https://assets.publishing.service.gov.uk/media/6848390196e63bce58e4e6cd/ATR_42-500_G-LMSA_07-25.pdf",
                "pdf",
            ),
            (
                "aaib-oe-loa",
                "AAIB",
                "https://assets.publishing.service.gov.uk/media/5f21754ad3bf7f1b0fa79fc7/Airbus_A320-214_OE-LOA_09-20.pdf",
                "pdf",
            ),
            (
                "aaib-g-cgtc",
                "AAIB",
                "https://assets.publishing.service.gov.uk/media/612f8ea18fa8f50327b49b6f/BN2T-4S_Islander_G-CGTC_10-21.pdf",
                "pdf",
            ),
            (
                "aaib-g-leap",
                "AAIB",
                "https://www.gov.uk/aaib-reports/bn2t-islander-g-leap-27-august-2011",
                "landing_pdf",
            ),
        ]
    )
    for identifier in [
        "20100024368",
        "20110014208",
        "20080031176",
        "20030065943",
        "20130001700",
        "20050215692",
    ]:
        rows.append(
            (
                "nasa-" + identifier,
                "NASA",
                f"https://ntrs.nasa.gov/citations/{identifier}",
                "abstract",
            )
        )
    return rows


def fetch(row, out):
    identifier, publisher, url, kind = row
    folder = out / "sources"
    capture = folder / (identifier + ".capture.json")
    if capture.exists():
        packet = json.loads(capture.read_text(encoding="utf-8"))
        text = packet["data"]["content"]
    elif kind in {"pdf", "landing_pdf"}:
        if kind == "landing_pdf":
            body = urllib.request.urlopen(url, timeout=40).read()
            soup = BeautifulSoup(body, "html.parser")
            candidates = [
                a["href"]
                for a in soup.find_all("a", href=True)
                if a["href"].lower().endswith(".pdf") and "leap" in a["href"].lower()
            ]
            if not candidates:
                raise ValueError("No report PDF")
            url = candidates[0].replace(
                "assets.digital.cabinet-office.gov.uk", "assets.publishing.service.gov.uk"
            )
        raw = urllib.request.urlopen(url, timeout=60).read()
        if not raw.startswith(b"%PDF"):
            raise ValueError("Not PDF bytes")
        (folder / (identifier + ".pdf")).write_bytes(raw)
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        packet = {
            "data": {"content": text, "url": url, "title": identifier},
            "original_pdf_sha256": hashlib.sha256(raw).hexdigest(),
            "page_texts": pages,
        }
        save(capture, packet)
    else:
        proc = subprocess.run(
            [shutil.which("ept"), "webfetch", "--url", url, "--format", "markdown"],
            capture_output=True,
            timeout=160,
            encoding="utf-8",
            errors="replace",
        )
        packet = json.loads(proc.stdout)
        text = packet.get("data", {}).get("content", "")
        save(capture, packet)
    if len(text) < 500 or "\ufffd" in text:
        raise ValueError("Missing/bad source text")
    (folder / (identifier + ".txt")).write_text(text, encoding="utf-8")
    blocks = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S):
        blocks.append(
            {
                "block_id": f"{identifier}-b{len(blocks):04d}",
                "start": m.start(),
                "end": m.end(),
                "text": m.group(),
                "section": "",
            }
        )
    save(folder / (identifier + ".blocks.json"), blocks)
    result = {
        "document_id": identifier,
        "publisher": publisher,
        "url": url,
        "family_id": identifier,
        "source_type": "research_paper" if publisher == "NASA" else "accident_report",
        "source_language": "en",
        "completeness": "publisher_abstract_only"
        if kind == "abstract"
        else (
            "pdf_bytes_saved_derived_text_figures_not_verified"
            if kind in {"pdf", "landing_pdf"}
            else "saved_official_source_body_not_pdf_verified"
        ),
        "title": packet.get("data", {}).get("title", identifier),
        "canonical_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "captured_at": "2026-10-14",
        "candidate_used": False,
        "source_layer": "summary_auxiliary" if kind == "abstract" else "source_text",
        "scope_label": "gas_path_or_directly_related_engine_diagnostics",
        "capture_kind": kind,
        "characters": len(text),
        "status": "captured_pending_scope_review",
    }
    print(identifier, len(text), flush=True)
    return result


def main():
    out = Path("evaluation/gas_path_holdout_v3_candidate")
    out.mkdir(exist_ok=True)
    (out / "sources").mkdir(exist_ok=True)
    if (out / "freeze.json").exists():
        raise ValueError("Frozen sources")
    old_ids = set()
    for path in Path("evaluation").glob("*/source_manifest.json"):
        if path.parent == out:
            continue
        old_ids.update(
            re.sub(r"^(?:tsb-|nasa-)", "", r.get("document_id", "").lower())
            for r in json.loads(path.read_text(encoding="utf-8")).get("documents", [])
        )
    seeds = sources()
    assert not any(re.sub(r"^(?:tsb-|nasa-)", "", r[0]) in old_ids for r in seeds)

    def safe(row):
        try:
            return fetch(row, out)
        except Exception as exc:
            print("Capture failed", row[0], type(exc).__name__, flush=True)
            return {
                "document_id": row[0],
                "url": row[2],
                "candidate_used": False,
                "status": "capture_failed",
                "error_type": type(exc).__name__,
            }

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(safe, seeds))
    save(
        out / "source_manifest.json",
        {
            "schema": "new-source-study-v3@1",
            "source_only": True,
            "llm_calls": 0,
            "documents": results,
            "prior_test_source_overlap": [],
            "status": "captured_not_annotated_not_ingested",
        },
    )


if __name__ == "__main__":
    main()
