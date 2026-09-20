from __future__ import annotations

import os
import socket
import time


def parse_bolt_uri(uri: str) -> tuple[str, int]:
    value = uri.rsplit("://", 1)[-1]
    host, port_text = value.rsplit(":", 1)
    return host, int(port_text)


def required_endpoints() -> list[tuple[str, int]]:
    endpoints: list[tuple[str, int]] = []
    if os.environ.get("AERODIAGNOSIS_VECTOR_BACKEND") == "chroma_http":
        endpoints.append(
            (
                os.environ.get("AERODIAGNOSIS_CHROMA_HOST", "chromadb"),
                int(os.environ.get("AERODIAGNOSIS_CHROMA_PORT", "8000")),
            )
        )
    if os.environ.get("AERODIAGNOSIS_GRAPH_BACKEND") == "neo4j":
        host, port = parse_bolt_uri(
            os.environ.get("AERODIAGNOSIS_NEO4J_URI", "bolt://neo4j:7687")
        )
        endpoints.append((host, port))
    return endpoints


def wait_for_tcp(host: str, port: int, *, timeout_seconds: float = 120.0) -> float:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return timeout_seconds - max(0.0, deadline - time.monotonic())
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def wait_for_external_services() -> None:
    for host, port in required_endpoints():
        wait_for_tcp(host, port)
