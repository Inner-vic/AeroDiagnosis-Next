"""Local command-line entry points for AeroDiagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ingestion import IngestionResult


def ingest_file(
    path: Path,
    *,
    document_id: str | None = None,
    settings: RuntimeSettings | None = None,
) -> IngestionResult:
    """Ingest one local file without retaining or modifying the source file."""

    source = path.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"source is not a regular file: {source}")
    application = bootstrap(settings)
    return application.ingest_document.execute(
        display_name=source.name,
        content=source.read_bytes(),
        document_id=document_id,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerodiagnosis-ingest",
        description="Index one local TXT, Markdown, or CSV document into the v3 runtime.",
    )
    parser.add_argument("path", type=Path, help="path to the source document")
    parser.add_argument(
        "--document-id",
        help="logical document ID to update; omit to create a new document",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        result = ingest_file(args.path, document_id=args.document_id)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
