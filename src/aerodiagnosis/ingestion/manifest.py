"""SQLite-backed document manifest and guarded version transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase


class VersionStatus(StrEnum):
    RECEIVED = "received"
    PARSED = "parsed"
    INDEXING = "indexing"
    STAGED = "staged"
    ACTIVE = "active"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DELETING = "deleting"
    DELETED = "deleted"


_TRANSITIONS = {
    VersionStatus.RECEIVED: {VersionStatus.PARSED, VersionStatus.FAILED},
    VersionStatus.PARSED: {VersionStatus.INDEXING, VersionStatus.FAILED},
    VersionStatus.INDEXING: {VersionStatus.STAGED, VersionStatus.FAILED},
    VersionStatus.STAGED: {VersionStatus.SUPERSEDED, VersionStatus.FAILED},
    VersionStatus.ACTIVE: set(),
    VersionStatus.FAILED: {VersionStatus.DELETING},
    VersionStatus.SUPERSEDED: {VersionStatus.DELETING},
    VersionStatus.DELETING: {VersionStatus.DELETED},
    VersionStatus.DELETED: set(),
}


@dataclass(frozen=True, slots=True)
class VersionRecord:
    version_id: str
    document_id: str
    revision: int
    parser_version: str
    content_hash: str
    status: VersionStatus


@dataclass(frozen=True, slots=True)
class DocumentCatalogRecord:
    document_id: str
    display_name: str
    active_version_id: str | None
    desired_revision: int
    created_at: str
    updated_at: str
    versions: tuple[VersionRecord, ...]


class DocumentManifest:
    """Authority for logical documents and visible immutable versions."""

    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    def register_document(self, document_id: str, display_name: str) -> None:
        if not document_id.strip() or not display_name.strip():
            raise ValueError("document_id and display_name must not be empty")
        now = datetime.now(UTC).isoformat()
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, display_name, desired_revision, created_at, updated_at
                ) VALUES (?, ?, 0, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    updated_at=excluded.updated_at
                """,
                (document_id, display_name, now, now),
            )
            connection.commit()

    def create_version(
        self,
        *,
        document_id: str,
        version_id: str,
        parser_version: str,
        content_hash: str,
    ) -> VersionRecord:
        if len(content_hash) != 64:
            raise ValueError("content_hash must be a SHA-256 hex digest")
        now = datetime.now(UTC).isoformat()
        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT desired_revision FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown document: {document_id}")
            revision = int(row["desired_revision"]) + 1
            connection.execute(
                """
                INSERT INTO document_versions (
                    version_id, document_id, revision, parser_version, content_hash,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    document_id,
                    revision,
                    parser_version,
                    content_hash,
                    VersionStatus.RECEIVED,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE documents SET desired_revision = ?, updated_at = ? WHERE document_id = ?",
                (revision, now, document_id),
            )
            connection.commit()
        return VersionRecord(
            version_id=version_id,
            document_id=document_id,
            revision=revision,
            parser_version=parser_version,
            content_hash=content_hash,
            status=VersionStatus.RECEIVED,
        )

    def transition(self, version_id: str, target: VersionStatus) -> VersionRecord:
        """Apply a simple transition; activation is only available through ``activate``."""

        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown document version: {version_id}")
            current = VersionStatus(row["status"])
            if target not in _TRANSITIONS[current]:
                raise ValueError(f"invalid version transition: {current} -> {target}")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE document_versions SET status = ?, updated_at = ? WHERE version_id = ?",
                (target, now, version_id),
            )
            connection.commit()
        return VersionRecord(
            version_id=version_id,
            document_id=row["document_id"],
            revision=int(row["revision"]),
            parser_version=row["parser_version"],
            content_hash=row["content_hash"],
            status=target,
        )

    def activate(self, version_id: str) -> VersionRecord:
        """Publish only the latest staged revision and supersede the old active one."""

        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT version.*, document.desired_revision, document.active_version_id
                FROM document_versions AS version
                JOIN documents AS document USING(document_id)
                WHERE version.version_id = ?
                """,
                (version_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown document version: {version_id}")
            if VersionStatus(row["status"]) is not VersionStatus.STAGED:
                raise ValueError("only a staged version can be activated")
            if int(row["revision"]) != int(row["desired_revision"]):
                raise ValueError("stale candidate cannot replace the desired revision")
            now = datetime.now(UTC).isoformat()
            previous_id = row["active_version_id"]
            if previous_id:
                connection.execute(
                    """
                    UPDATE document_versions SET status = ?, updated_at = ?
                    WHERE version_id = ? AND status = ?
                    """,
                    (VersionStatus.SUPERSEDED, now, previous_id, VersionStatus.ACTIVE),
                )
            connection.execute(
                "UPDATE document_versions SET status = ?, updated_at = ? WHERE version_id = ?",
                (VersionStatus.ACTIVE, now, version_id),
            )
            connection.execute(
                """
                UPDATE documents SET active_version_id = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (version_id, now, row["document_id"]),
            )
            connection.commit()
        return VersionRecord(
            version_id=version_id,
            document_id=row["document_id"],
            revision=int(row["revision"]),
            parser_version=row["parser_version"],
            content_hash=row["content_hash"],
            status=VersionStatus.ACTIVE,
        )

    def active_version_ids(self) -> frozenset[str]:
        with self._database.connect() as connection:
            rows = connection.execute(
                "SELECT active_version_id FROM documents WHERE active_version_id IS NOT NULL"
            ).fetchall()
        return frozenset(str(row["active_version_id"]) for row in rows)

    def get_version(self, version_id: str) -> VersionRecord | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
        if row is None:
            return None
        return VersionRecord(
            version_id=row["version_id"],
            document_id=row["document_id"],
            revision=int(row["revision"]),
            parser_version=row["parser_version"],
            content_hash=row["content_hash"],
            status=VersionStatus(row["status"]),
        )

    def counts(self) -> tuple[int, int]:
        with self._database.connect() as connection:
            documents = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
            versions = int(
                connection.execute("SELECT count(*) FROM document_versions").fetchone()[0]
            )
        return documents, versions

    def list_documents(self) -> tuple[DocumentCatalogRecord, ...]:
        with self._database.connect() as connection:
            documents = connection.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC, document_id"
            ).fetchall()
            versions = connection.execute(
                "SELECT * FROM document_versions ORDER BY document_id, revision DESC"
            ).fetchall()
        by_document: dict[str, list[VersionRecord]] = {}
        for row in versions:
            by_document.setdefault(str(row["document_id"]), []).append(
                VersionRecord(
                    version_id=row["version_id"],
                    document_id=row["document_id"],
                    revision=int(row["revision"]),
                    parser_version=row["parser_version"],
                    content_hash=row["content_hash"],
                    status=VersionStatus(row["status"]),
                )
            )
        return tuple(
            DocumentCatalogRecord(
                document_id=row["document_id"],
                display_name=row["display_name"],
                active_version_id=row["active_version_id"],
                desired_revision=int(row["desired_revision"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                versions=tuple(by_document.get(str(row["document_id"]), [])),
            )
            for row in documents
        )
