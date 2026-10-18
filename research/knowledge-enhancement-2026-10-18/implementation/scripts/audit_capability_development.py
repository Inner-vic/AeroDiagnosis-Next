"""Recompute every rank, evidence score and candidate/context budget without APIs."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_capability_development import DATA, OUT, read, save
from tokenizers import Tokenizer


def main():
    protocol = read(OUT / "protocol.json")
    for name, digest in protocol["file_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    v5 = read(Path("evaluation/text_hybrid_holdout_v5_candidate/freeze.json"))
    expected_index = v5["aligned_index_sha256"]
    for index in (Path(".runtime/holdout-v5/text-index/chinese.sqlite"), OUT / "index.sqlite"):
        assert hashlib.sha256(index.read_bytes()).hexdigest() == expected_index, str(index)
    freeze = read(DATA / "development-freeze.json")
    for name, digest in freeze["file_sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (DATA / "questions-final.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    chunks = {c["chunk_id"]: c for c in read(DATA / "corpus.json")["chunks"]}
    rows = list(
        map(json.loads, (OUT / "predictions.jsonl").read_text(encoding="utf-8").splitlines())
    )
    assert len(rows) == 60 * len(protocol["methods"])
    assert len({(r["sample_id"], r["method"]) for r in rows}) == len(rows)
    counter = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    counter.no_padding()
    counter.no_truncation()
    maximum_candidates = maximum_tokens = 0
    for row in rows:
        q = questions[row["sample_id"]]
        retrieval = row["retrieval"]
        assert sum(x["requested"] for x in retrieval["trace"]) == 150
        assert all(x["returned"] <= x["requested"] for x in retrieval["trace"])
        actual = sum(x["returned"] for x in retrieval["trace"])
        maximum_candidates = max(maximum_candidates, actual)
        scores = defaultdict(float)
        denominator = sum(r["weight"] for r in retrieval["routes"]) / 61
        for route in retrieval["routes"]:
            assert len(route["chunk_ids"]) == len(set(route["chunk_ids"]))
            for rank, key in enumerate(route["chunk_ids"], 1):
                assert key in chunks
                scores[key] += route["weight"] / (60 + rank) / denominator
        expected = sorted(scores, key=lambda key: (-scores[key], key))[:50]
        assert expected == [x[0] for x in retrieval["post_fusion_pool"]]
        assert all(abs(scores[x[0]] - x[1]) < 1e-12 for x in retrieval["post_fusion_pool"])
        hit_chunks = [chunks[h["chunk_id"]] for h in retrieval["hits"]]
        hit_ids = [c["chunk_id"] for c in hit_chunks]
        assert len(hit_ids) == len(set(hit_ids))
        assert all(key in expected for key in hit_ids)
        assert [expected.index(key) for key in hit_ids] == sorted(
            expected.index(key) for key in hit_ids
        )
        assert all(abs(h["score"] - scores[h["chunk_id"]]) < 1e-12 for h in retrieval["hits"])
        assert max(Counter(c["document_id"] for c in hit_chunks).values(), default=0) <= 5
        tokens = sum(
            len(counter.encode(c["metadata"].get("display_content_zh", c["content"])).ids)
            for c in hit_chunks
        )
        assert tokens == retrieval["context_tokens"] and tokens <= 4096
        maximum_tokens = max(maximum_tokens, tokens)
        units = {u["unit_id"]: u for u in q["support_units"]}

        def covered(k, units=units, hit_chunks=hit_chunks, q=q):
            present = set()
            for key, u in units.items():
                ranges = []
                for c in hit_chunks[:k]:
                    if c["document_id"] + ":" + c["version_id"] != u["source_key"]:
                        continue
                    coordinate = c["metadata"]["locator"]["coordinates"]
                    ranges.append((coordinate["start"], coordinate["end"]))
                if all(
                    any(lo <= pos < hi for lo, hi in ranges)
                    for pos in range(u["start"], u["end"])
                    if not u["quote"][pos - u["start"]].isspace()
                ):
                    present.add(key)
            full = any(set(s) <= present for s in q["acceptable_support_sets"])
            fraction = max(len(set(s) & present) / len(s) for s in q["acceptable_support_sets"])
            return full, fraction

        for k in (5, 10):
            full, fraction = covered(k)
            assert full == row["complete" + str(k)]
            if k == 5:
                assert abs(fraction - row["coverage5"]) < 1e-12
        first = next((k for k in range(1, len(hit_chunks) + 1) if covered(k)[0]), None)
        assert abs((1 / first if first else 0) - row["mrr10"]) < 1e-12
    result = dict(
        predictions_checked=len(rows),
        questions_checked=len(questions),
        source_candidate_maximum=maximum_candidates,
        context_token_maximum=maximum_tokens,
        all_weighted_rrf_rankings_recomputed=True,
        all_evidence_metrics_independently_recomputed=True,
        all_freeze_and_code_hashes_valid=True,
        original_and_research_indexes_equal_frozen_v5=True,
        index_sha256=expected_index,
        api_calls=0,
    )
    save(OUT / "audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
