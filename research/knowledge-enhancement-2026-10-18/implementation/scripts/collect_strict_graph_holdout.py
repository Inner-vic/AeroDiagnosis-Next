"""Capture bounded new official investigation sources; no questions or retrieval scores."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from collect_quality_sources import fetch

ROOT = Path("evaluation/strict_graph_holdout_v1_candidate")


def seeds():
    rows = []
    for k in ["ao-2016-101", "ao-2015-094", "ao-2019-046", "200702219", "200607815", "200606510"]:
        rows.append((k, "ATSB", "https://www.atsb.gov.au/investigations/" + k, "web"))
    for k in [
        "a95q0118",
        "a07c0082",
        "a24w0066",
        "a93w0204",
        "a12p0134",
        "a12q0029",
        "a98w0192",
        "a97c0236",
        "a97p0044",
        "a98p0022",
        "a10c0159",
    ]:
        y = ("19" if int(k[1:3]) > 80 else "20") + k[1:3]
        rows.append(
            (
                k,
                "TSB",
                f"https://www.tsb.gc.ca/eng/rapports-reports/aviation/{y}/{k}/{k}.html",
                "web",
            )
        )
    rows.extend(
        [
            (
                "bea-f-hcic",
                "BEA",
                "https://bea.aero/fileadmin/uploads/tx_elydbrapports/f-ic130722.en.pdf",
                "pdf",
            ),
            (
                "bea2019-0585",
                "BEA",
                "https://bea.aero/fileadmin/user_upload/BEA2019-0585.en.pdf",
                "pdf",
            ),
            (
                "aaib-g-nvsb",
                "AAIB",
                "https://www.gov.uk/aaib-reports/de-havilland-canada-dhc-8-series-311-g-nvsb-9-august-2005",
                "web",
            ),
            (
                "aaib-g-flbc",
                "AAIB",
                "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/523449/AAIB_Bulletin_6-2016.pdf",
                "pdf",
            ),
            (
                "aaib-g-bosn",
                "AAIB",
                "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1089043/AAIB_Bulletin_1-2022.pdf",
                "pdf",
            ),
            ("taic-ao-2015-001", "TAIC", "https://taic.org.nz/inquiry/ao-2015-001", "web"),
            ("taic-ao-2024-006", "TAIC", "https://taic.org.nz/inquiry/ao-2024-006", "web"),
        ]
    )
    return rows


def main():
    (ROOT / "sources").mkdir(exist_ok=True)
    old = set()

    def scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {
                    "document_id",
                    "original_source_id",
                    "event_id",
                    "source_id",
                } and isinstance(v, str):
                    old.add(v.casefold())
                elif isinstance(v, (dict, list)):
                    scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)

    oldfiles = []
    for name in ["source_manifest.json", "sources.json", "source-catalog.json"]:
        for p in Path("evaluation").rglob(name):
            if ROOT not in p.parents:
                scan(json.loads(p.read_text(encoding="utf8")))
                oldfiles.append(p.as_posix())
    rows = seeds()
    assert len(rows) <= 40 and all(r[0] not in old for r in rows)

    def one(row):
        try:
            r = fetch(row, ROOT)
            r["captured_at"] = datetime.now(UTC).isoformat()
            return r
        except Exception as e:
            print("Capture failure", row[0], type(e).__name__, flush=True)
            return dict(
                document_id=row[0],
                publisher=row[1],
                url=row[2],
                status="capture_failed",
                candidate_used=False,
                error_type=type(e).__name__,
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(one, rows))
    (ROOT / "source_manifest.json").write_text(
        json.dumps(
            dict(
                source_only=True,
                questions_authored=False,
                api_calls=0,
                prior_manifest_paths=oldfiles,
                documents=result,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(
        "Captured",
        sum(r["status"] != "capture_failed" for r in result),
        "/",
        len(result),
        flush=True,
    )


if __name__ == "__main__":
    main()
