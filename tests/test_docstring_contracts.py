"""Docstring coverage checks for maintained cyPredict source modules."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATHS = [
    *sorted((Path("cyPredict") / "core").glob("*.py")),
    Path("cyPredict/config.py"),
    Path("cyPredict/results.py"),
    Path("cyPredict/logging_utils.py"),
]


def test_core_classes_and_functions_have_docstrings():
    """Every maintained class/function should have a non-empty docstring."""
    missing = []

    for path in SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    missing.append(f"{path}:{node.lineno}:{node.name}")

    assert missing == []
