# SPDX-License-Identifier: Apache-2.0
"""Guard the test suite's async marker contract."""

from __future__ import annotations

import ast
from pathlib import Path


def _is_pytest_mark_asyncio(node: ast.AST) -> bool:
    """Return True when a decorator is exactly pytest.mark.asyncio."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "asyncio"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def test_async_tests_use_anyio_markers():
    """The suite should not depend on pytest-asyncio anywhere."""
    offenders: list[str] = []
    tests_dir = Path(__file__).parent

    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if any(
                    _is_pytest_mark_asyncio(decorator)
                    for decorator in node.decorator_list
                ):
                    offenders.append(f"{path.name}:{node.lineno}")

    assert (
        offenders == []
    ), "Found pytest.mark.asyncio after the suite migrated to AnyIO: " + ", ".join(
        offenders
    )
