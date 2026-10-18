"""Archive exact frozen algorithm, input and SQLite knowledge bytes for replay."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

from evaluate_quality_holdout import read, save

from aerodiagnosis.config import RuntimeSettings


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    out = root / ".runtime/experiments/holdout-v3-final-20261014"
    frozen = read(dataset / "freeze.json")
    archive = out / "frozen-code-and-inputs"
    archive.mkdir(exist_ok=True)
    copied = []
    files = {
        **frozen["code_sha256_bytes"],
        **{
            "evaluation/gas_path_holdout_v3_candidate/" + f: sha
            for f, sha in frozen["file_sha256_bytes"].items()
        },
    }
    protocol = read(dataset / "protocol.json")
    files.update({"scripts/" + f: sha for f, sha in protocol["analysis_helpers_sha256"].items()})
    for name in [
        "pyproject.toml",
        "uv.lock",
        "evaluation/gas_path_holdout_v3_candidate/development-code-contract.json",
    ]:
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name, sha in files.items():
        source = root / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen file mismatch: " + name)
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise ValueError("Existing immutable archive mismatch")
        if not target.exists():
            shutil.copyfile(source, target)
        copied.append({"path": name, "sha256": sha})
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v3/profile")
    backup = out / "frozen-knowledge.sqlite"
    if not backup.exists():
        with sqlite3.connect(settings.database_path) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
    save(
        out / "frozen-archive-manifest.json",
        {
            "files": copied,
            "sqlite_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
            "snapshot_sha256": frozen["snapshot_sha256"],
            "model_artifacts_sha256": frozen["configuration"]["trained_artifact_sha256"],
            "source_captures": (
                "evaluation/gas_path_holdout_v3_candidate/.sources "
                "and .runtime/holdout-v3/prepared-v5"
            ),
            "secrets_copied": False,
        },
    )
    print("Frozen files archived", len(copied), "knowledge database saved")


if __name__ == "__main__":
    main()
