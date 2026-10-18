"""Capture public sources without gold, predictions, translation or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def capture(row, out):
    target = out / "sources" / (row["document_id"] + ".capture.json")
    if target.exists():
        packet = json.loads(target.read_text(encoding="utf-8-sig"))
    else:
        proc = subprocess.run(
            [shutil.which("ept"), "webfetch", "--url", row["url"], "--format", "markdown"],
            capture_output=True,
            timeout=150,
            encoding="utf-8",
            errors="replace",
        )
        try:
            packet = json.loads(proc.stdout)
        except ValueError:
            return {**row, "candidate_used": False, "status": "capture_failed"}
        save(target, packet)
    data = packet.get("data", {})
    text = data.get("content", "")
    if len(text) < 500 or "\ufffd" in text:
        return {**row, "candidate_used": False, "status": "capture_missing_or_bad_text"}
    (out / "sources" / (row["document_id"] + ".txt")).write_text(text, encoding="utf-8")
    blocks = []
    section = ""
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S):
        value = m.group()
        if value.startswith("#"):
            section = value.splitlines()[0].lstrip("# ")
        blocks.append(
            {
                "block_id": row["document_id"] + f"-b{len(blocks):04d}",
                "start": m.start(),
                "end": m.end(),
                "text": value,
                "section": section,
            }
        )
    save(out / "sources" / (row["document_id"] + ".blocks.json"), blocks)
    result = {
        **row,
        "title": data.get("title", ""),
        "canonical_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "captured_at": "2026-10-14",
        "status": "downloaded_pending_scope_review",
        "candidate_used": False,
        "characters": len(text),
        "blocks": len(blocks),
        "resolved_url": data.get("url", row["url"]),
    }
    print(
        json.dumps(
            {k: result[k] for k in ["document_id", "status", "characters", "title"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seeds", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "freeze.json").exists():
        raise ValueError("Frozen sources cannot be changed")
    (args.output / "sources").mkdir(exist_ok=True)
    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))["documents"]
    old = json.loads(
        Path("evaluation/gas_path_holdout_v1_candidate/source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    old_ids = {r["document_id"] for r in old["documents"] if r.get("candidate_used")}
    if any(r["document_id"] in old_ids for r in seeds):
        raise ValueError("Prior adopted source identity overlap")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: capture(row, args.output), seeds))
    save(
        args.output / "source_manifest.json",
        {
            "schema": "followup-independent-sources@1",
            "source_only": True,
            "llm_calls": 0,
            "documents": rows,
            "prior_test_adopted_source_overlap": [],
            "status": "captured_not_adopted_not_annotated",
        },
    )


if __name__ == "__main__":
    main()
