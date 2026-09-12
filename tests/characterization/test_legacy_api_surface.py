"""Freeze the old public surface while endpoints move behind the new package."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_MAIN = PROJECT_ROOT / "code" / "python" / "api" / "main.py"


def _routes() -> set[tuple[str, str]]:
    tree = ast.parse(LEGACY_MAIN.read_text(encoding="utf-8"))
    routes: set[tuple[str, str]] = set()
    for node in tree.body:
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            owner = decorator.func.value
            if not isinstance(owner, ast.Name) or owner.id != "app" or not decorator.args:
                continue
            path = decorator.args[0]
            if isinstance(path, ast.Constant) and isinstance(path.value, str):
                routes.add((decorator.func.attr.upper(), path.value))
    return routes


def test_legacy_api_has_the_observed_route_count() -> None:
    assert len(_routes()) == 21


def test_legacy_core_routes_remain_available_during_migration() -> None:
    routes = _routes()

    assert ("POST", "/api/ingest/upload") in routes
    assert ("POST", "/api/qa/ask") in routes
    assert ("GET", "/api/admin/graph") in routes
    assert ("GET", "/api/cases/list") in routes
    assert ("GET", "/api/health") in routes
