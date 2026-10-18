"""Audit source-only provider provenance for cached v5 Chinese translations (no API)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--index", type=Path, required=True)
    args = cli.parse_args()
    root = Path.cwd()
    runtime = root / ".runtime/thesis-research-provider-v1"
    chunks = {p.stem: read(p) for p in (args.index / "translated").glob("*.json")}
    review = {i["chunk_id"]: i for i in read(args.dataset / "translation-review.json")["items"]}
    candidates = defaultdict(list)
    correction_path = root / ".runtime/holdout-v5/provider-key-correction.json"
    correction = read(correction_path) if correction_path.exists() else {}
    with sqlite3.connect(runtime / "calls.sqlite3") as db:
        rows = db.execute("SELECT request_sha,receipt FROM calls WHERE state='settled'").fetchall()
    for request_sha, raw_receipt in rows:
        receipt = json.loads(raw_receipt)
        task = receipt.get("task", "")
        if not (
            task.startswith("text-hybrid-v5") or task == "knowledge-chinese-semantic-repair-v2"
        ):
            continue
        path = runtime / "cache" / (request_sha + ".json")
        if not path.exists():
            continue
        cached = read(path)
        payload = json.loads(cached["request"]["messages"][1]["content"])
        outputs = cached["value"]
        pairs = []
        if "chunks" in payload:
            translations = {}
            for value in outputs.get("translations", []):
                key = value["key"]
                if request_sha == correction.get("request_sha256") and key == correction.get(
                    "provider_key"
                ):
                    key = correction["correct_chunk_id"]
                translations[key] = value["text_zh"]
            aliases = {}
            if "source_batch_id" in payload:
                aliases = read(
                    args.index / "request-key-maps" / (payload["source_batch_id"] + ".json")
                )
            for source in payload["chunks"]:
                if source["key"] in translations:
                    pairs.append(
                        (
                            aliases.get(source["key"], source["key"]),
                            source["source_text"],
                            translations[source["key"]],
                        )
                    )
        elif "source_text" in payload and isinstance(outputs.get("text_zh"), str):
            pairs.extend(
                (key, payload["source_text"], outputs["text_zh"])
                for key, chunk in chunks.items()
                if chunk["content"] == payload["source_text"]
            )
        for key, source, translation in pairs:
            if key not in chunks or chunks[key]["content"] != source:
                continue
            candidates[key].append(
                {
                    "request_sha256": request_sha,
                    "call_id": receipt["call_id"],
                    "task": task,
                    "model": receipt["requested_model"],
                    "source_sha256": sha(source),
                    "provider_translation_sha256": sha(translation),
                }
            )
    items = []
    missing = []
    for key, chunk in sorted(chunks.items()):
        original = review.get(key, {}).get(
            "original_translation", chunk["metadata"]["display_content_zh"]
        )
        matching = [x for x in candidates[key] if x["provider_translation_sha256"] == sha(original)]
        if not matching:
            missing.append(key)
        items.append(
            {
                "chunk_id": key,
                "source_sha256": sha(chunk["content"]),
                "review_input_translation_sha256": sha(original),
                "final_translation_sha256": sha(chunk["metadata"]["display_content_zh"]),
                "reviewed": key in review,
                "provider_candidates": matching,
                "source_verified_repair": bool(review.get(key, {}).get("replacement_translation")),
            }
        )
    result = {
        "source_only": True,
        "questions_read": False,
        "api_calls": 0,
        "chunks": len(chunks),
        "all_chunks_provider_provenance_matched": not missing,
        "missing": missing,
        "provider_models": dict(
            Counter(x["provider_candidates"][0]["model"] for x in items if x["provider_candidates"])
        ),
        "items": items,
    }
    (args.index / "translation-provenance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
