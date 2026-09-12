"""Embedded versioned case store with deterministic lexical retrieval."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.ports import CaseMatch, CaseRecord

from .sqlite import SQLiteDatabase


def _tokens(value: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", normalized)
    return frozenset(words)


class SQLiteCaseStore:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    @property
    def backend_name(self) -> str:
        return "sqlite_cases"

    def upsert(self, case: CaseRecord) -> None:
        if not case.case_id.strip() or not case.summary.strip() or case.version < 1:
            raise ValueError("case identity, version and summary must be valid")
        now = datetime.now(UTC).isoformat()
        payload = json.dumps(
            {"summary": case.summary, "attributes": dict(case.attributes)},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO cases (case_id, version, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    version=excluded.version,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                WHERE excluded.version >= cases.version
                """,
                (case.case_id, case.version, payload, now, now),
            )
            connection.commit()

    def search(self, query: str, *, limit: int = 5) -> list[CaseMatch]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        with self._database.connect() as connection:
            rows = connection.execute("SELECT * FROM cases").fetchall()
        matches = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            summary = str(payload["summary"])
            case_tokens = _tokens(summary)
            overlap = query_tokens & case_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens | case_tokens)
            matches.append(
                CaseMatch(
                    case=CaseRecord(
                        case_id=row["case_id"],
                        version=int(row["version"]),
                        summary=summary,
                        attributes=payload.get("attributes", {}),
                    ),
                    score=score,
                )
            )
        matches.sort(key=lambda match: (-match.score, match.case.case_id))
        return matches[:limit]

    def count(self) -> int:
        with self._database.connect() as connection:
            return int(connection.execute("SELECT count(*) FROM cases").fetchone()[0])

    def list_cases(self, *, limit: int = 100, offset: int = 0) -> list[CaseRecord]:
        if limit < 1 or offset < 0:
            raise ValueError("case pagination must be positive")
        with self._database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases ORDER BY updated_at DESC, case_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        records = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            records.append(
                CaseRecord(
                    case_id=row["case_id"],
                    version=int(row["version"]),
                    summary=str(payload["summary"]),
                    attributes=payload.get("attributes", {}),
                )
            )
        return records
