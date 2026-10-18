"""Build an isolated prepared holdout with bounded LLM calls and resumable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aerodiagnosis.adapters.llm.chinese_document import ChineseDocumentDisplay
from aerodiagnosis.adapters.llm.chinese_graph import (
    ChineseDisplayGraphExtractor,
    ChineseGraphDisplay,
)
from aerodiagnosis.adapters.llm.langextract_graph import LangExtractGraphExtractor
from aerodiagnosis.adapters.llm.research_client import (
    BudgetExceeded,
    CompletionStructureError,
    ResearchClient,
)
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DocumentMetadata, EvidenceLocator
from aerodiagnosis.ingestion import DocumentManifest, ParserRegistry
from aerodiagnosis.ingestion.parsers import ParsedChunk, ParsedContext, ParsedDocument
from aerodiagnosis.ingestion.service import DocumentIngestionService


def save(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RetryJSONClient(ResearchClient):
    def complete(self, **kwargs: Any) -> Any:
        try:
            return super().complete(**kwargs)
        except CompletionStructureError:
            # Only known-billed malformed JSON receives one retry; no transport retry.
            kwargs["instruction"] += (
                " The previous response was malformed JSON. Escape backslashes and control "
                "characters inside all string values and return valid JSON without markdown. "
                "Use Chinese quotation marks inside text_zh; never unescaped ASCII quotes. "
                "Do not use unescaped newline characters inside string values. "
                "The entire response must be exactly one JSON object, ending at its closing "
                "brace: absolutely no XML or HTML suffix such as </final> or </br>. "
                "Encode paragraph breaks as JSON newline escapes exactly once."
            )
            return super().complete(**kwargs)


class PreparedParser:
    def __init__(self, prepared: Path, rows: list[dict[str, Any]], attempt: int = 1) -> None:
        self.attempt = attempt
        self.prepared = prepared
        self.rows = {x["sha256"]: x for x in rows}

    def parse(self, content: bytes) -> ParsedDocument:
        row = self.rows[hashlib.sha256(content).hexdigest()]
        data = json.loads((self.prepared / row["parsed_path"]).read_text(encoding="utf-8"))
        chunks = []
        for c in data["chunks"]:
            loc = EvidenceLocator(**c["locator"])
            parent = (
                ParsedContext(c["parent"]["content"], EvidenceLocator(**c["parent"]["locator"]))
                if c["parent"]
                else None
            )
            chunks.append(ParsedChunk(c["content"], loc, parent))
        folder = (self.prepared / row["parsed_path"]).parent
        return ParsedDocument(
            parser_version=(
                data["parser_version"]
                + ":holdout-source-body@1"
                + (f":processing-attempt-{self.attempt}" if self.attempt > 1 else "")
            ),
            media_type=data["media_type"],
            canonical_content=data["canonical_content"].encode(),
            chunks=tuple(chunks),
            source_artifacts={
                "docling.json": (folder / "docling.json").read_bytes(),
                "capture-provenance.json": json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "original_source_id",
                            "source_capture_sha256",
                            "completeness",
                            "source_family",
                        ]
                    },
                    ensure_ascii=False,
                ).encode(),
            },
        )


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prepared", type=Path, default=Path(".runtime/holdout-v1/prepared"))
    cli.add_argument("--profile", type=Path, default=Path(".runtime/holdout-v1/profile"))
    cli.add_argument("--workers", type=int, default=3, choices=[1, 2, 3, 4])
    args = cli.parse_args()
    root = Path.cwd().resolve()
    settings = RuntimeSettings.from_env(args.profile.resolve())
    if (
        settings.knowledge_budget_runtime.resolve()
        != (root / ".runtime/thesis-research-provider-v1").resolve()
    ):
        raise ValueError("Only the existing canonical ledger is allowed")
    live = RuntimeSettings.from_env(root / ".runtime/fusion/profile")
    if settings.database_path.resolve() == live.database_path.resolve():
        raise ValueError("Holdout must use a separate database")
    if (
        settings.knowledge_api_cap_cny != 20
        or settings.knowledge_extraction_stage != "corpus_graph"
    ):
        raise ValueError(
            "This build requires the authorized cumulative cap 20 and corpus_graph stage"
        )
    catalog = json.loads((args.prepared / "catalog.json").read_text(encoding="utf-8"))
    if catalog["source_only"] is not True:
        raise ValueError("Source-only preparation required")
    coverage = json.loads((args.prepared / "evidence-coverage.json").read_text(encoding="utf-8"))
    if coverage["all_bound_evidence_blocks_survive"] != coverage["questions"]:
        raise ValueError("Unresolved preparation coverage")
    # Bootstrap migrations once before concurrent source-specific ingestion.
    app = bootstrap(settings)
    template = app.ingest_document._ingestion
    parser = ParserRegistry(parsers={".html": PreparedParser(args.prepared, catalog["documents"])})
    manifest = DocumentManifest(settings.database_path)
    receipt_path = settings.runtime_dir / "prebuild-receipt.json"
    records = (
        json.loads(receipt_path.read_text(encoding="utf-8")).get("documents", [])
        if receipt_path.exists()
        else []
    )
    by = {x["document_id"]: x for x in records}
    lock = threading.Lock()
    stop = threading.Event()

    def build(row: dict[str, Any]) -> dict[str, Any]:
        prior = by.get(row["document_id"], {})
        version = manifest.get_version(prior["version_id"]) if prior.get("version_id") else None
        if prior.get("status") == "active" and version and version.status.value == "active":
            return {**prior, "reused": True}
        if stop.is_set():
            return {"document_id": row["document_id"], "status": "not_started"}
        assert settings.default_llm_api_key is not None and settings.default_llm_model is not None
        client = RetryJSONClient(
            api_key=settings.default_llm_api_key.get_secret_value(),
            runtime=settings.knowledge_budget_runtime,
            model=settings.default_llm_model,
            cap_cny=settings.knowledge_api_cap_cny,
        )
        extractor = ChineseDisplayGraphExtractor(
            LangExtractGraphExtractor(
                client,
                audit_root=settings.runtime_dir / "extraction-audits",
                stage=settings.knowledge_extraction_stage,
            ),
            ChineseGraphDisplay(client, stage=settings.knowledge_extraction_stage),
        )
        service = DocumentIngestionService(
            manifest,
            template._vector_store,
            graph_store=template._graph_store,
            graph_extractor=extractor,
            parsers=parser,
            source_archive=template._source_archive,
            context_store=template._contexts,
            chinese_chunks=ChineseDocumentDisplay(
                client, stage=settings.knowledge_extraction_stage
            ).enrich,
        )
        metadata = DocumentMetadata.model_validate(row["metadata"])
        # Retry identity describes processing, never a fabricated source revision.
        attempts = int(prior.get("attempt", 0)) + 1
        if prior.get("status") == "failed":
            service._parsers = ParserRegistry(
                parsers={".html": PreparedParser(args.prepared, catalog["documents"], attempts)}
            )
        content = (args.prepared / row["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Prepared source changed")
        try:
            result = service.ingest(
                display_name=row["display_name"],
                content=content,
                document_id=row["document_id"],
                metadata=metadata,
            )
            return {
                **asdict(result),
                "status": result.status.value,
                "original_source_id": row["original_source_id"],
                "attempt": attempts,
                "canonical_sha256": row["canonical_sha256"],
            }
        except Exception as exc:
            if isinstance(exc, BudgetExceeded) or "budget" in str(exc).lower():
                stop.set()
            return {
                "document_id": row["document_id"],
                "original_source_id": row["original_source_id"],
                "status": "failed",
                "attempt": attempts,
                "error_type": type(exc).__name__,
                "reason": str(exc)[:300],
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build, row): row for row in catalog["documents"]}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                by[result["document_id"]] = result
                save(
                    receipt_path,
                    {
                        "status": "running",
                        "corpus": "holdout-v1",
                        "cumulative_cap_cny": 20,
                        "documents": list(by.values()),
                    },
                )
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ["evidence_ids"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    success = sum(x.get("status") == "active" for x in by.values())
    save(
        receipt_path,
        {
            "status": "complete" if success == len(catalog["documents"]) else "incomplete",
            "corpus": "holdout-v1",
            "cumulative_cap_cny": 20,
            "active_documents": success,
            "documents": list(by.values()),
        },
    )
    print("Active documents", success, "/", len(catalog["documents"]), flush=True)


if __name__ == "__main__":
    main()
