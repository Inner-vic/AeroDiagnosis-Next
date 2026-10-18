# ruff: noqa: E501
"""One bounded semantic repair of a cached translation with empty/missing entries."""

import argparse
import json

from prepare_capability_holdout import client
from prepare_strict_graph_holdout import OUT, ROOT, read, save, sha


def repair(index):
    target = OUT / "translation-batches" / (str(index).zfill(3) + ".json")
    assert not target.exists()
    data = read(ROOT / "source-catalog.json")
    batches = []
    for c in data["chunks"]:
        if not batches or sum(len(x["content"]) for x in batches[-1]) + len(c["content"]) > 3200:
            batches.append([])
        batches[-1].append(c)
    batch = batches[index]
    model = client()
    receipts = [
        r for r in model.summary()["receipts"] if r.get("task") == "capability-holdout-translate-v1"
    ]
    failed = receipts[-1]
    cached = read(model.runtime / "cache" / (failed["request_sha256"] + ".json"))
    values = cached["value"].get("translations", cached["value"].get("items", []))
    mapping = {str(v["key"]): v.get("text_zh", v.get("text")) for v in values}
    keys = set(map(str, range(len(batch))))
    assert set(mapping) <= keys
    missing = [
        k
        for k in sorted(keys, key=int)
        if not isinstance(mapping.get(k), str) or not mapping[k].strip()
    ]
    assert missing and len(missing) <= 5
    context = sorted(
        {j for k in missing for j in range(max(0, int(k) - 1), min(len(batch), int(k) + 2))}
    )
    reply = model.complete(
        task="strict-holdout-translation-empty-entry-repair-v1",
        stage="supplemental_review",
        max_tokens=2048,
        instruction="Translate ONLY requested keys fully to Chinese. Every requested key must have a NONEMPTY text_zh. These are source-preserving consecutive fragments of a sentence: use neighboring context for meaning, but keep each translated fragment aligned with its exact input. Do not omit fragments because incomplete. Preserve uncertainty, numbers and conditions. Return translations:[{key,text_zh}]. Context-only keys must not be returned.",
        payload={
            "requested_keys": missing,
            "items": [
                {
                    "key": str(i),
                    "original": batch[i]["content"],
                    "context_only": str(i) not in missing,
                }
                for i in context
            ],
        },
    )
    fixed = {str(x["key"]): x["text_zh"] for x in reply.value["translations"]}
    assert set(fixed) == set(missing) and all(
        isinstance(t, str) and t.strip() for t in fixed.values()
    )
    mapping.update(fixed)
    save(
        target,
        {
            "request_source_sha256": sha(json.dumps(batch, ensure_ascii=False, sort_keys=True)),
            "translations": [
                {"chunk_id": c["chunk_id"], "text_zh": mapping[str(j)]} for j, c in enumerate(batch)
            ],
            "receipt": cached["receipt"],
            "semantic_repair_receipt": reply.receipt,
            "repaired_keys": missing,
            "repair_reason": "Provider returned empty Chinese text for explicit source sentence fragments; original bad cache retained unchanged",
            "original_cached_request_sha256": failed["request_sha256"],
        },
    )
    print("Repaired batch", index, "keys", missing, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("index", type=int)
    a = p.parse_args()
    repair(a.index)
