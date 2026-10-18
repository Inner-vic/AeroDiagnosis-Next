"""Prebuild a persistent source catalog; unchanged active documents are skipped."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DocumentMetadata
from aerodiagnosis.ingestion import DocumentManifest


def run(catalog_path: Path, profile_root: Path) -> dict[str, Any]:
    settings = RuntimeSettings.from_env(profile_root)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    receipt_path = settings.runtime_dir / "prebuild-receipt.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"documents": []}
    prior = {item["document_id"]: item for item in previous["documents"]}
    manifest = DocumentManifest(settings.database_path)
    application = None
    receipt: dict[str, Any] = {"catalog": str(catalog_path), "status": "running", "documents": []}
    profile = {
        "quality_policy": "source-rows@3:source-roles@1:grounded-source@1",
        "parser": settings.knowledge_parser_backend,
        "extractor": settings.knowledge_extractor_backend,
        "chinese_display": settings.knowledge_graph_chinese_display,
        "chunking": settings.knowledge_chunking,
        "llm_model": settings.default_llm_model,
        "parent_budget": settings.knowledge_parent_budget,
        "child_budget": settings.knowledge_child_budget,
        "tokenizer_sha256": hashlib.sha256(
            (settings.knowledge_tokenizer_path / "tokenizer.json").read_bytes()
        ).hexdigest()
        if settings.knowledge_tokenizer_path
        else None,
    }
    for row in catalog["documents"]:
        source = (catalog_path.parent / row["path"]).resolve(strict=True)
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Catalog source hash mismatch: " + row["document_id"])
        metadata = DocumentMetadata.model_validate(row.get("metadata", {}))
        fingerprint = hashlib.sha256(
            json.dumps(
                {"content": digest, "metadata": metadata.model_dump(), "profile": profile},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        old = prior.get(row["document_id"], {})
        version = manifest.get_version(old["version_id"]) if old.get("version_id") else None
        if old.get("fingerprint") == fingerprint and version and version.status.value == "active":
            item = {**old, "reused": True}
        else:
            if application is None:
                application = bootstrap(settings)
            result = application.ingest_document.execute(
                display_name=row.get("display_name", source.name),
                content=content,
                document_id=row["document_id"],
                metadata=metadata,
            )
            item = {
                **asdict(result),
                "status": result.status.value,
                "fingerprint": fingerprint,
                "source_sha256": digest,
                "derivation": row.get("derivation"),
            }
        receipt["documents"].append(item)
        settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "document_id": item["document_id"],
                    "chunks": item["chunk_count"],
                    "reused": item["reused"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    receipt["status"] = "complete"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.catalog, args.profile_root)


if __name__ == "__main__":
    main()
