"""Publish queries and original sources only; never expose author answers/support labels."""

from prepare_capability_holdout import OUT, ROOT, read, save, sha


def main():
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    docs = {d["document_id"]: d for d in data["documents"]}
    rows = []
    mapping = {}
    for f in sorted((OUT / "authoring").glob("*.json")):
        r = read(f)
        source = []
        for key, identifier in r["chunk_key_map"].items():
            c = chunks[identifier]
            d = docs[c["document_id"]]
            source.append(
                {
                    "key": key,
                    "chunk_id": identifier,
                    "document_id": d["document_id"],
                    "agency": d["agency"],
                    "content": c["content"],
                    "source_quality_restriction": c["metadata"].get(
                        "source_annotation_restriction"
                    ),
                }
            )
        qs = []
        for i, q in enumerate(r["questions"]):
            identifier = sha(r["family"] + ":" + str(i))[:16]
            qs.append({"blind_id": identifier, "query": q["query"]})
            mapping[identifier] = {
                "family": r["family"],
                "author_file": f.as_posix(),
                "question_index": i,
                "planned_kind": q["kind"],
            }
        rows.append({"family": r["family"], "source_chunks": source, "questions": qs})
    save(
        OUT / "blind-question-review-pack.json",
        {
            "reference_answers_hidden": True,
            "gold_supports_hidden": True,
            "method_and_prediction_hidden": True,
            "families": rows,
        },
    )
    save(OUT / "blind-question-private-map.json", mapping)
    print("Blind query pack", len(rows), "families", len(mapping), "questions", flush=True)


if __name__ == "__main__":
    main()
