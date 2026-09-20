from __future__ import annotations

import os

from aerodiagnosis.adapters.startup_wait import wait_for_external_services


def main() -> None:
    wait_for_external_services()
    os.execvp(
        "python",
        [
            "python",
            "-m",
            "uvicorn",
            "aerodiagnosis.adapters.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--no-server-header",
        ],
    )


if __name__ == "__main__":
    main()
