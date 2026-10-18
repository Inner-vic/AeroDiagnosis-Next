"""Exact rank-prefix cache for cross-policy replay over one immutable experiment index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class FrozenCandidateCache:
    def __init__(self, base, route, directory):
        self.base = base
        self.route = route
        self.directory = Path(directory)
        self.key = None
        self.matches = []
        self.last_trace = {}
        self.physical_calls = 0
        self.loaded_candidates = 0

    @property
    def backend_name(self):
        return self.base.backend_name + ":frozen_query_rank_cache@1"

    def search(self, query, *, top_k=5, active_version_ids=None):
        if not query.strip() or not 1 <= top_k <= 150 or not active_version_ids:
            raise ValueError("Frozen cache requires query, active versions and prefix<=150")
        key = (query, tuple(sorted(active_version_ids)))
        if key != self.key:
            self.matches = self.base.search(query, top_k=150, active_version_ids=active_version_ids)
            if len(self.matches) > 150:
                raise ValueError("Raw source exceeded150 prefetch bound")
            self.key = key
            self.last_trace = dict(getattr(self.base, "last_trace", {}))
            self.physical_calls += 1
            self.loaded_candidates += len(self.matches)
            self.directory.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256(repr(key).encode()).hexdigest()
            (self.directory / (self.route + "-" + identity + ".json")).write_text(
                json.dumps(
                    {
                        "route": self.route,
                        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                        "active_versions": list(key[1]),
                        "raw_requested": 150,
                        "hits": [
                            {"chunk_id": m.chunk.chunk_id, "score": m.score} for m in self.matches
                        ],
                        "graph_trace": self.last_trace,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf8",
            )
        return self.matches[:top_k]
