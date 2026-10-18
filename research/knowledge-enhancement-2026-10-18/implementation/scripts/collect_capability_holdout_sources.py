# ruff: noqa: E501 -- Source selection prose and bounded regular expressions.
"""Source-only public FAA corpus discovery. No questions, model or paid provider calls."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("evaluation/retrieval_capability_holdout_v1_candidate")
CUTOFF = "2026-10-15"


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fetch(url):
    if urllib.parse.urlparse(url).hostname not in {"www.federalregister.gov", "www.govinfo.gov"}:
        raise ValueError("Only official public source hosts")
    request = urllib.request.Request(
        url, headers={"User-Agent": "AeroDiagnosis source-only research"}
    )
    return urllib.request.urlopen(request, timeout=45).read().decode("utf-8-sig")


def previous():
    ids = set()
    urls = set()

    def scan(value):
        if isinstance(value, dict):
            for key, v in value.items():
                if isinstance(v, str):
                    if key in {"document_id", "event_id", "source_id", "original_source_id"}:
                        ids.add(v.lower())
                    if key in {"url", "source_url", "resolved_url", "html_url"} and v.startswith(
                        "http"
                    ):
                        urls.add(v)
                elif isinstance(v, (dict, list)):
                    scan(v)
        elif isinstance(value, list):
            for v in value:
                scan(v)

    files = []
    for p in sorted(Path("evaluation").rglob("source_manifest.json")):
        if OUT in p.parents:
            continue
        scan(json.loads(p.read_text(encoding="utf-8")))
        files.append(dict(path=p.as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return ids, urls, files


def capture(metadata):
    key = metadata["document_number"]
    folder = OUT / "sources"
    target = folder / (key + ".metadata.json")
    try:
        full = (
            json.loads(target.read_text(encoding="utf-8"))
            if target.exists()
            else json.loads(
                fetch(
                    metadata.get("json_url")
                    or "https://www.federalregister.gov/api/v1/documents/" + key + ".json"
                )
            )
        )
        save(target, full)
        if full["type"] != "Rule" or "correction" in full["title"].lower():
            return dict(document_id=key, status="excluded_not_original_final_rule")
        path = folder / (key + ".full.txt")
        text = path.read_text(encoding="utf-8") if path.exists() else fetch(full["raw_text_url"])
        path.write_text(text, encoding="utf-8")
        # Normalize only CRLF, never invent or stitch sentences.
        text = text.replace("\r\n", "\n")
        path.write_text(text, encoding="utf-8")
        start = re.search(r"(?m)^\(c\) Applicability\s*$", text)
        if not start:
            return dict(document_id=key, status="excluded_no_anchored_applicability")
        last = re.search(
            r"(?m)^\([h-z]\) (?:Alternative Methods|Related Information|Additional Information|Material Incorporated|Incorporation by Reference)",
            text[start.start() :],
        )
        end = start.start() + last.start() if last else min(len(text), start.start() + 10000)
        selected = text[start.start() : end].rstrip()
        if not 1000 <= len(selected) <= 12000:
            return dict(
                document_id=key, status="excluded_technical_length", characters=len(selected)
            )
        unsafe = re.search(r"(?ms)^\(e\) Unsafe Condition\s*\n(.*?)(?=^\([f-z]\))", selected)
        actions = re.search(r"(?ms)^\(g\) .*?\n(.*)", selected)
        if not unsafe or not actions:
            return dict(document_id=key, status="excluded_missing_technical_clauses")
        technical = unsafe.group(1)
        if not re.search(r"compressor|turbine|blade|vane|combust|fuel|seal", technical, re.I):
            return dict(document_id=key, status="excluded_outside_engine_fault_scope")
        # Incorporation-only requirements cannot support detailed repair threshold questions.
        detailed_actions = bool(
            re.search(
                r"(?:within|before|after|exceed|cycles|flight hours|inspect|remove|replace|installation)",
                actions.group(1),
                re.I,
            )
        )
        project = [v for v in full.get("docket_ids", []) if v.startswith("Project Identifier")]
        related = sorted(set(re.findall(r"(?i)\bAD\s+(20\d{2}-\d{2}-\d{2})\b", text)))
        saved = folder / (key + ".selected.txt")
        saved.write_text(selected, encoding="utf-8")
        return dict(
            document_id=key,
            source_agency="FAA/FederalRegister",
            source_type="historical_final_airworthiness_directive",
            title=full["title"],
            url=full["html_url"],
            publication_date=full["publication_date"],
            effective_on=full.get("effective_on"),
            status="source_candidate_pending_independent_scope_review",
            candidate_used=False,
            questions_authored=False,
            full_text_path=path.relative_to(OUT).as_posix(),
            full_text_sha256=digest(text),
            saved_source_path=saved.relative_to(OUT).as_posix(),
            saved_source_sha256=digest(selected),
            saved_characters=len(selected),
            full_characters=len(text),
            selection_segments=[
                dict(
                    full_start=start.start(),
                    full_end=start.start() + len(selected),
                    saved_start=0,
                    saved_end=len(selected),
                )
            ],
            project_identifiers=project,
            related_ad_ids=related,
            detailed_actions_present=detailed_actions,
            limited_incorporation_by_reference=not detailed_actions,
            language="en",
            expert_reviewed=False,
            limitation="Saved clauses only; incorporated EASA/manufacturer documents not captured; do not invent absent thresholds or treat historical rule as current legal advice",
        )
    except Exception as error:
        return dict(document_id=key, status="capture_failed", error_type=type(error).__name__)


def main():
    if (OUT / "source-freeze.json").exists():
        raise ValueError("Source selection frozen")
    OUT.mkdir(exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    old_ids, old_urls, old_files = previous()
    save(
        OUT / "previous-source-exclusions.json",
        dict(ids=sorted(old_ids), urls=sorted(old_urls), manifests=old_files),
    )
    candidates = []
    for page in range(1, 5):
        query = urllib.parse.urlencode(
            {
                "per_page": 100,
                "page": page,
                "order": "newest",
                "conditions[agencies][]": "federal-aviation-administration",
                "conditions[type][]": "RULE",
                "conditions[term]": "Airworthiness Directives Engines",
                "conditions[publication_date][lte]": CUTOFF,
                "conditions[publication_date][gte]": "2024-01-01",
            }
        )
        path = OUT / (f"discovery-page{page}.json")
        packet = (
            json.loads(path.read_text())
            if path.exists()
            else json.loads(fetch("https://www.federalregister.gov/api/v1/documents.json?" + query))
        )
        save(path, packet)
        for row in packet.get("results", []):
            title = row["title"].lower()
            if "engine" not in title or "airworthiness" not in title or "correction" in title:
                continue
            if re.search(r"lycoming|continental|austro|rotax|piston", title):
                continue
            if row["document_number"].lower() in old_ids or row["html_url"] in old_urls:
                continue
            candidates.append(row)
        if len(candidates) >= 90 or not packet.get("next_page_url"):
            break
    save(
        OUT / "discovery-policy.json",
        dict(
            created_at_utc=datetime.now(UTC).isoformat(),
            question_access=False,
            api_calls=0,
            selection="Official FAA original final rules published2024..cutoff; turbine engine fault clauses, natural applicability-to-actions span; no performance-based selection",
            desired_source_clusters=40,
            questions_target_before_necessity_review=240,
            candidates=len(candidates),
            paid_api_cap_cny=36,
        ),
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(capture, candidates))
    # Group revisions sharing a project or a cited AD; conservative connected source families.
    parents = {
        r["document_id"]: r["document_id"]
        for r in rows
        if r["status"].startswith("source_candidate")
    }

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    seen = {}
    for row in rows:
        if row["document_id"] not in parents:
            continue
        keys = row["project_identifiers"] + ["AD:" + v for v in row["related_ad_ids"]]
        for key in keys:
            if key in seen:
                parents[find(row["document_id"])] = find(seen[key])
            else:
                seen[key] = row["document_id"]
    chosen = []
    groups = set()
    for row in rows:
        if row["document_id"] not in parents:
            continue
        row["source_cluster"] = find(row["document_id"])
        if row["source_cluster"] not in groups and len(chosen) < 40:
            groups.add(row["source_cluster"])
            row["selected_for_scope_review"] = True
            chosen.append(row)
        else:
            row["selected_for_scope_review"] = False
    save(
        OUT / "source_manifest.json",
        dict(
            schema="capability-holdout-source-candidates@1",
            created_at_utc=datetime.now(UTC).isoformat(),
            status="source_only_pending_scope_review_translation_graph_and_question_freeze",
            sources=rows,
            selected_candidates=len(chosen),
            selected_clusters=len(groups),
            selected_characters=sum(r["saved_characters"] for r in chosen),
            questions_authored=0,
            independent_predictions=0,
            paid_api_calls=0,
            distribution="New FAA final-rule maintenance-document retrieval; differs from accident-report development; must report domain shift separately",
        ),
    )
    print(
        json.dumps(
            {
                "captured_candidates": len(rows),
                "usable_candidates": len(parents),
                "selected_clusters": len(groups),
                "selected_characters": sum(r["saved_characters"] for r in chosen),
                "paid_api_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
