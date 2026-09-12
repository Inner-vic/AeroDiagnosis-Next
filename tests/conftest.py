"""Test fixtures that also work under restrictive Windows directory ACL policies."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Create a private test directory without tempfile's Windows 0700 ACL edge case."""

    root = Path(os.environ.get("AERODIAGNOSIS_TEST_TMP", ".test-tmp")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)
