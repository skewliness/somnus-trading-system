from __future__ import annotations

import ast
from pathlib import Path


CLASSIFICATION_CALLS = {
    "classify_exception",
    "log_classified_exception",
    "record_caught_exception",
}


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            names.add(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            names.add(item.func.attr)
    return names


def test_every_broad_v6_execution_exception_is_classified() -> None:
    source_path = Path(__file__).parents[1] / "trading_system" / "v6_execution.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    broad_handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "Exception"
    ]
    missing = [
        handler.lineno
        for handler in broad_handlers
        if not (_call_names(handler) & CLASSIFICATION_CALLS)
    ]

    assert broad_handlers, "expected V6 execution to contain guarded compatibility boundaries"
    assert missing == [], f"broad exception handlers without structured classification: {missing}"
