"""Incrementally replicate the SQLite source of truth to Chroma and Neo4j."""

from __future__ import annotations

import argparse
import json
import time

from aerodiagnosis.adapters.persistence import create_external_sync
from aerodiagnosis.config import RuntimeSettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forever",
        action="store_true",
        help="continue polling the durable outbox until the process is stopped",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--no-backfill", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    settings = RuntimeSettings.from_env()
    settings.prepare()
    sync = create_external_sync(settings)
    if not sync.streams:
        parser.error("configure chroma_http and/or neo4j before running external sync")
    backfilled = 0 if args.no_backfill else sync.enqueue_backfill()

    try:
        while True:
            report = sync.flush(limit=settings.sync_batch_size)
            print(
                json.dumps(
                    {
                        "streams": sync.streams,
                        "backfilled": backfilled,
                        "claimed": report.claimed,
                        "applied": report.applied,
                        "failed": report.failed,
                        "pending": report.pending,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            backfilled = 0
            if not args.forever:
                return 0 if report.failed == 0 else 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
