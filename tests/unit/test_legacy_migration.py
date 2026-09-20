from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite import LATEST_SCHEMA_VERSION, SQLiteDatabase
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.migration import LegacyUploadMigrator


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        runtime_dir=tmp_path / "runtime",
        database_path=tmp_path / "runtime" / "data" / "app.db",
    )


def test_legacy_migration_previews_then_backs_up_and_imports_idempotently(tmp_path: Path) -> None:
    source = tmp_path / "legacy-uploads"
    source.mkdir()
    manual = source / "manual.txt"
    manual.write_text("EGT rise indicates compressor degradation.", encoding="utf-8")
    (source / "ignored.pdf").write_bytes(b"not supported")
    settings = _settings(tmp_path)
    migrator = LegacyUploadMigrator(settings)

    preview = migrator.migrate(source)

    assert preview.apply is False
    assert preview.candidates == 1
    assert preview.backup_path is None
    assert not settings.database_path.exists()
    assert manual.exists()

    first = migrator.migrate(source, apply=True)
    second = migrator.migrate(source, apply=True)

    assert first.imported == 1
    assert first.reused == 0
    assert first.backup_path is not None
    assert Path(first.backup_path).is_file()
    assert second.imported == 0
    assert second.reused == 1
    assert manual.read_text(encoding="utf-8").startswith("EGT rise")
    status = bootstrap(settings).get_runtime_status.execute()
    assert status.document_count == 1
    assert status.version_count == 1


def test_legacy_migration_tracks_failed_files_and_schema_backup_guard(tmp_path: Path) -> None:
    source = tmp_path / "legacy-uploads"
    source.mkdir()
    (source / "empty.txt").write_bytes(b"")
    settings = _settings(tmp_path)
    result = LegacyUploadMigrator(settings).migrate(source, apply=True)

    assert result.skipped == 1
    assert len(result.failures) == 1
    database = SQLiteDatabase(settings.database_path)
    assert database.schema_version() == LATEST_SCHEMA_VERSION
    try:
        database.backup_to(settings.database_path)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:  # pragma: no cover - guards destructive regression
        raise AssertionError("live database must never be its own backup target")
