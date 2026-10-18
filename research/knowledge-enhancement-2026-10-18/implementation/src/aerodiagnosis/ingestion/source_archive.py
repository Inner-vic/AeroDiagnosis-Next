"""Immutable, application-owned source artifacts for citation downloads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class SourceArchive:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _directory(self, version_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", version_id):
            raise ValueError("Invalid version identity")
        return self.root / version_id

    def store(
        self,
        version_id: str,
        display_name: str,
        raw: bytes,
        canonical: bytes,
        artifacts: dict[str, bytes],
    ) -> None:
        directory = self._directory(version_id)
        directory.mkdir(parents=True, exist_ok=True)
        files = {"original.bin": raw, "canonical.txt": canonical, **artifacts}
        for name, data in files.items():
            if Path(name).name != name:
                raise ValueError("Invalid artifact filename")
            path = directory / name
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("Immutable source artifact mismatch")
            else:
                with path.open("xb") as stream:
                    stream.write(data)
        metadata = {
            "display_name": display_name,
            "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        path = directory / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["sha256"] != metadata["sha256"]:
                raise ValueError("Immutable artifact manifest mismatch")
        else:
            path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, version_id: str, kind: str = "original") -> tuple[Path, str]:
        names = {
            "original": "original.bin",
            "canonical": "canonical.txt",
            "structure": "docling.json",
        }
        if kind not in names:
            raise ValueError("Unknown source artifact")
        directory = self._directory(version_id)
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        path = directory / names[kind]
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"][names[kind]]:
            raise ValueError("Source artifact integrity failure")
        return path, metadata["display_name"] if kind == "original" else names[kind]
