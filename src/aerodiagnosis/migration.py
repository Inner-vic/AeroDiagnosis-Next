"""Recoverable, idempotent migration of legacy uploaded source documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".csv"})


@dataclass(frozen=True, slots=True)
class MigrationFailure:
    source: str
    error: str


@dataclass(frozen=True, slots=True)
class LegacyMigrationReport:
    source_root: str
    apply: bool
    candidates: int
    imported: int
    reused: int
    skipped: int
    total_bytes: int
    backup_path: str | None
    failures: tuple[MigrationFailure, ...]


class LegacyUploadMigrator:
    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._database = SQLiteDatabase(settings.database_path)

    @staticmethod
    def _files(source_root: Path) -> tuple[Path, ...]:
        root = source_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"legacy source is not a directory: {root}")
        files = []
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            files.append(resolved)
        return tuple(sorted(files, key=lambda item: str(item).casefold()))

    @staticmethod
    def _source_key(relative_path: str) -> str:
        return hashlib.sha256(relative_path.casefold().encode()).hexdigest()

    def _backup(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self._settings.runtime_dir / "backups" / f"before-v2-migration-{stamp}.db"
        return self._database.backup_to(destination)

    def migrate(self, source_root: Path, *, apply: bool = False) -> LegacyMigrationReport:
        root = source_root.resolve(strict=True)
        files = self._files(root)
        total_bytes = sum(path.stat().st_size for path in files)
        if not apply:
            return LegacyMigrationReport(
                source_root=str(root),
                apply=False,
                candidates=len(files),
                imported=0,
                reused=0,
                skipped=0,
                total_bytes=total_bytes,
                backup_path=None,
                failures=(),
            )

        application = bootstrap(self._settings)
        backup_path = self._backup()
        imported = reused = skipped = 0
        failures: list[MigrationFailure] = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            source_key = self._source_key(relative)
            content = path.read_bytes()
            source_hash = hashlib.sha256(content).hexdigest()
            with self._database.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM legacy_migration_ledger WHERE source_key = ?",
                    (source_key,),
                ).fetchone()
            already_imported = (
                row is not None
                and row["status"] == "imported"
                and row["source_hash"] == source_hash
            )
            if already_imported:
                reused += 1
                continue
            document_id = str(row["document_id"]) if row is not None else str(uuid.uuid4())
            try:
                result = application.ingest_document.execute(
                    display_name=path.name,
                    content=content,
                    document_id=document_id,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(MigrationFailure(source=relative, error=str(exc)))
                self._record(
                    source_key=source_key,
                    source_path=relative,
                    source_hash=source_hash,
                    document_id=document_id,
                    version_id=None,
                    status="failed",
                    error=str(exc),
                )
                skipped += 1
                continue
            self._record(
                source_key=source_key,
                source_path=relative,
                source_hash=source_hash,
                document_id=result.document_id,
                version_id=result.version_id,
                status="imported",
                error=None,
            )
            imported += 1
        return LegacyMigrationReport(
            source_root=str(root),
            apply=True,
            candidates=len(files),
            imported=imported,
            reused=reused,
            skipped=skipped,
            total_bytes=total_bytes,
            backup_path=str(backup_path),
            failures=tuple(failures),
        )

    def _record(
        self,
        *,
        source_key: str,
        source_path: str,
        source_hash: str,
        document_id: str,
        version_id: str | None,
        status: str,
        error: str | None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO legacy_migration_ledger (
                    source_key, source_path, source_hash, document_id, version_id,
                    status, error, migrated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    source_path=excluded.source_path,
                    source_hash=excluded.source_hash,
                    document_id=excluded.document_id,
                    version_id=excluded.version_id,
                    status=excluded.status,
                    error=excluded.error,
                    migrated_at=excluded.migrated_at
                """,
                (
                    source_key,
                    source_path,
                    source_hash,
                    document_id,
                    version_id,
                    status,
                    error,
                    now,
                ),
            )
            connection.commit()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerodiagnosis-migrate-v2",
        description="Preview or import supported legacy upload files into the v3 runtime.",
    )
    parser.add_argument("source", type=Path, help="legacy upload directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create a database backup and apply the import; default is read-only preview",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        report = LegacyUploadMigrator(RuntimeSettings.from_env()).migrate(
            args.source,
            apply=args.apply,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0 if not report.failures else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
