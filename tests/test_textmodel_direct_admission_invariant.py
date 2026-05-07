# SPDX-License-Identifier: Apache-2.0
"""
Track the admission-control invariant for serialized TextModel-direct routes.

Issue #495 documents a P0 that bit a downstream operator: text-only MLLM
requests bypassed the MLLM scheduler and entered a serialized TextModel-direct
generation path guarded by a single async lock with a 120-second wait bound.
Concurrent agent traffic piled up behind that lock instead of receiving a
fast retryable admission result, producing memory pressure and request stalls.

Current ``main`` does not have that exact bug — text-only MLLM requests route
through ``MLLMScheduler``, and ``SimpleEngine._generation_lock`` exists only
to serialize Metal command-buffer access, not as a request-admission gate.

This module pins the invariant so that any future change reintroducing a
serialized TextModel-direct route without fail-fast admission breaks the test
suite. Acceptance criteria mirror the issue:

- A concurrent request behind an occupied serialized TextModel-direct route
  must fail fast rather than waiting for minutes.
- The error must be machine-readable, e.g. ``text_generation_busy``, with
  HTTP 503.
- Tests must prove no long-waiter pileup occurs.
- Comments must not imply the serialized route proves scheduler batching or
  queue absorption.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from vllm_mlx.engine.simple import SimpleEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_ENGINE_PATH = REPO_ROOT / "vllm_mlx" / "engine" / "simple.py"


def _load_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


# ---------------------------------------------------------------------------
# Static surface checks
# ---------------------------------------------------------------------------


def test_no_textmodel_direct_class_exists():
    """No class or function with a ``TextModelDirect``-style name should exist
    anywhere in the package. Reintroducing a route by that name without
    fail-fast admission is the exact regression issue #495 wants to prevent.
    """
    forbidden = re.compile(r"\bTextModelDirect[A-Za-z_]*\b")
    package_root = REPO_ROOT / "vllm_mlx"
    matches: list[str] = []
    for py_file in package_root.rglob("*.py"):
        text = py_file.read_text()
        for m in forbidden.finditer(text):
            matches.append(f"{py_file.relative_to(REPO_ROOT)}: {m.group(0)}")
    assert matches == [], (
        "Found TextModelDirect-style identifier in upstream code. Per issue "
        "#495 this route must not be revived without fail-fast admission. "
        f"Matches: {matches}"
    )


def test_generation_lock_is_documented_as_metal_only():
    """``SimpleEngine._generation_lock`` must remain documented as Metal
    command-buffer serialization only. If a future change repurposes it as a
    wait-mode admission gate, the documenting comment in ``__init__`` should
    be updated, and that update will surface here so the invariant from
    issue #495 is re-evaluated.
    """
    src = SIMPLE_ENGINE_PATH.read_text()
    # Find the lock declaration and the comment block immediately above it.
    lines = src.splitlines()
    decl_idx = None
    for i, line in enumerate(lines):
        if "self._generation_lock = asyncio.Lock()" in line:
            decl_idx = i
            break
    assert decl_idx is not None, "lock declaration not found"

    # Walk backwards collecting contiguous comment lines.
    comment_lines: list[str] = []
    j = decl_idx - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        comment_lines.append(lines[j].lstrip())
        j -= 1
    comment_block = " ".join(reversed(comment_lines))

    assert (
        "Metal" in comment_block
        or "command-buffer" in comment_block
        or "command buffer" in comment_block
    ), (
        f"_generation_lock comment block must document its purpose as Metal "
        f"command-buffer serialization. Got: {comment_block!r}"
    )
    assert "#495" in comment_block, (
        "_generation_lock comment must reference issue #495 so future readers "
        "see the admission-control invariant before repurposing the lock."
    )


def test_no_long_wait_for_in_simple_engine_text_paths():
    """No call site in ``SimpleEngine`` should wrap a lock acquire in
    ``asyncio.wait_for(... , timeout=T)`` with ``T >= 5`` seconds. Such a
    pattern would reintroduce the wait-mode admission queue called out in
    issue #495 (the original route used a 120-second wait bound).

    Short timeouts (e.g. for graceful shutdown probes) are fine; the test
    flags only the long-waiter pattern.
    """
    tree = _load_module_ast(SIMPLE_ENGINE_PATH)
    offending: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `asyncio.wait_for(...)` and `wait_for(...)`.
        target = None
        if isinstance(func, ast.Attribute) and func.attr == "wait_for":
            target = func
        elif isinstance(func, ast.Name) and func.id == "wait_for":
            target = func
        if target is None:
            continue

        # Pull out timeout (positional arg index 1, or keyword).
        timeout_node = None
        if len(node.args) >= 2:
            timeout_node = node.args[1]
        for kw in node.keywords:
            if kw.arg == "timeout":
                timeout_node = kw.value

        if timeout_node is None:
            continue
        # Static-only: flag obvious numeric constants >= 5s.
        if isinstance(timeout_node, ast.Constant) and isinstance(
            timeout_node.value, (int, float)
        ):
            if timeout_node.value >= 5:
                offending.append(
                    f"line {node.lineno}: wait_for(..., timeout={timeout_node.value})"
                )

    assert offending == [], (
        "Found asyncio.wait_for with a long timeout in vllm_mlx/engine/simple.py. "
        "Issue #495 asks that any serialized TextModel-direct route fail fast "
        f"rather than wait. Offending sites: {offending}"
    )


def test_simple_engine_does_not_expose_admission_queue_attribute():
    """``SimpleEngine`` must not carry an attribute that names an admission
    queue, such as ``_text_admission_queue`` or ``_text_admission_lock``,
    without an accompanying fail-fast contract. The presence of such an
    attribute is a strong signal that someone added a wait-mode admission
    path, which is exactly the shape #495 warns against.
    """
    src = inspect.getsource(SimpleEngine)
    forbidden_names = (
        "_text_admission_queue",
        "_text_admission_lock",
        "_text_direct_lock",
        "_textmodel_direct_lock",
    )
    found = [name for name in forbidden_names if name in src]
    assert found == [], (
        "SimpleEngine declares attribute(s) that look like a serialized "
        "TextModel-direct admission queue, which issue #495 forbids without "
        f"fail-fast admission semantics: {found}"
    )


# ---------------------------------------------------------------------------
# Future-proofing: if a route ever lands, its error must be machine-readable
# ---------------------------------------------------------------------------


def test_text_generation_busy_error_if_present_is_machine_readable():
    """If a future change adds a ``text_generation_busy`` error class or HTTP
    response, it must surface as HTTP 503 with a machine-readable code, per
    the acceptance criteria in issue #495. We import the package and look
    for an exception or constant by that name; if absent (today's state),
    the test simply passes — the moment someone adds it, this test
    materialises a real assertion against its shape.
    """
    import vllm_mlx
    import pkgutil

    candidates: list[tuple[str, object]] = []
    for finder, name, ispkg in pkgutil.walk_packages(
        vllm_mlx.__path__, prefix="vllm_mlx."
    ):
        try:
            mod = __import__(name, fromlist=["*"])
        except Exception:
            continue
        for attr_name in dir(mod):
            lowered = attr_name.lower()
            if "text_generation_busy" in lowered or "textgenerationbusy" in lowered:
                candidates.append((f"{name}.{attr_name}", getattr(mod, attr_name)))

    if not candidates:
        pytest.skip(
            "No text_generation_busy admission error defined yet — invariant "
            "is preserved by the absence of a serialized TextModel-direct "
            "route. The moment one is added, this test should be tightened "
            "to assert HTTP 503 + machine-readable code."
        )

    # If something does exist, the asserted contract:
    #   - It carries an HTTP-compatible status_code attribute set to 503, or
    #   - It defines/uses the literal "text_generation_busy" as a code/key.
    for label, obj in candidates:
        # Class-like: must declare a 503 status.
        if isinstance(obj, type):
            status = getattr(obj, "status_code", None)
            assert status == 503, (
                f"{label}: text-generation-busy error must surface as HTTP 503 "
                f"per issue #495 (got {status!r})"
            )
        # String-like: must be the canonical code.
        elif isinstance(obj, str):
            assert obj == "text_generation_busy", (
                f"{label}: text-generation-busy code must be the literal "
                f"'text_generation_busy' (got {obj!r})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
