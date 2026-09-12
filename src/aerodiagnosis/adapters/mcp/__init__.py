"""MCP inbound adapter exposing read-only application capabilities."""

from .server import create_server, main

__all__ = ["create_server", "main"]
