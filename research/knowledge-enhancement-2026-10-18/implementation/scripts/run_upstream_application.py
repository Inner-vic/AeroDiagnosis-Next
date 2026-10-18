"""Launch an explicit upstream-fusion profile without changing native defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from aerodiagnosis.adapters.api.app import create_app
from aerodiagnosis.config import RuntimeSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    args = parser.parse_args()
    settings = RuntimeSettings.from_env(project_root=args.profile_root)
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
