# SPDX-License-Identifier: Apache-2.0
"""Portable coverage for the clean-exit policy (vllm_mlx.shutdown).

The policy decides whether the serve process may skip CPython finalization at
the end of lifespan shutdown (the MLX thread-local teardown segfault
workaround). The decision module imports no engine code, so these tests run
in the standard no-MLX CI job on every platform. The os._exit/lifespan wiring
itself — driving the real server through startup and shutdown — remains in
tests/test_shutdown_exit.py, which needs MLX and runs on Apple Silicon.

vllm_mlx.server and vllm_mlx.cli cannot be imported without MLX, so the tests
that pin their call sites read the source instead: an AST assertion fails the
same way a broken import would, without needing the platform.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_mlx.shutdown import (
    clean_exit_enabled,
    exit_without_finalizing,
    should_exit_without_finalizing,
)

PACKAGE = Path(__file__).resolve().parent.parent / "vllm_mlx"


# ---------------------------------------------------------------------------
# The decision predicate
# ---------------------------------------------------------------------------


def test_clean_shutdown_with_optin_exits():
    assert should_exit_without_finalizing(None, None, True, env={}) is True


def test_primary_failure_never_exits():
    """A failed serving phase must propagate and exit non-zero."""
    assert (
        should_exit_without_finalizing(RuntimeError("boom"), None, True, env={})
        is False
    )


def test_cleanup_failure_never_exits():
    """A failed cleanup phase is a failed shutdown, not a success."""
    assert (
        should_exit_without_finalizing(None, RuntimeError("boom"), True, env={})
        is False
    )


def test_both_failures_never_exit():
    assert (
        should_exit_without_finalizing(
            RuntimeError("a"), RuntimeError("b"), True, env={}
        )
        is False
    )


def test_env_disable_keeps_normal_finalization():
    """VLLM_MLX_CLEAN_EXIT=0 restores ordinary interpreter teardown."""
    env = {"VLLM_MLX_CLEAN_EXIT": "0"}
    assert should_exit_without_finalizing(None, None, True, env=env) is False
    assert clean_exit_enabled(env) is False


def test_env_default_is_enabled():
    assert clean_exit_enabled({}) is True
    assert clean_exit_enabled({"VLLM_MLX_CLEAN_EXIT": "1"}) is True


def test_embedded_use_never_exits():
    """Library/embedded use never opts in: exiting the host process is not

    ours to do. The opt-in flag is threaded through explicitly, so with it
    false the decision is false no matter how clean the shutdown was.
    """
    assert should_exit_without_finalizing(None, None, False, env={}) is False
    assert (
        should_exit_without_finalizing(
            None, None, False, env={"VLLM_MLX_CLEAN_EXIT": "1"}
        )
        is False
    )


# ---------------------------------------------------------------------------
# The exit mechanics (os._exit monkeypatched — nothing here kills pytest)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_exit(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr("vllm_mlx.shutdown.os._exit", calls.append)
    return calls


def test_exit_forwards_status_and_flushes(fake_exit, monkeypatch):
    monkeypatch.delenv("VLLM_MLX_CLEAN_EXIT", raising=False)
    exit_without_finalizing(3)
    assert fake_exit == [3]


def test_exit_default_status_is_zero(fake_exit, monkeypatch):
    monkeypatch.delenv("VLLM_MLX_CLEAN_EXIT", raising=False)
    exit_without_finalizing()
    assert fake_exit == [0]


def test_exit_respects_env_disable(fake_exit, monkeypatch):
    monkeypatch.setenv("VLLM_MLX_CLEAN_EXIT", "0")
    exit_without_finalizing(0)
    assert fake_exit == []


def test_exit_survives_temp_cleanup_failure(fake_exit, monkeypatch):
    """The temp-file sweep is best-effort; its failure must not stop the exit."""
    import sys
    import types

    monkeypatch.delenv("VLLM_MLX_CLEAN_EXIT", raising=False)
    broken = types.ModuleType("vllm_mlx.models.mllm")

    def _raise():
        raise OSError("disk went away")

    broken.cleanup_all_temp_files = _raise
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.mllm", broken)
    exit_without_finalizing(0)
    assert fake_exit == [0]


# ---------------------------------------------------------------------------
# Call-site wiring (source-level: server/cli import MLX, so importing them is
# exactly what this file must not do)
# ---------------------------------------------------------------------------


def _assignments_to(tree: ast.AST, attr: str) -> list[ast.Assign]:
    """Assign statements whose target is ``<something>.attr`` or ``attr``."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = None
            if isinstance(target, ast.Attribute):
                name = target.attr
            elif isinstance(target, ast.Name):
                name = target.id
            if name == attr:
                found.append(node)
    return found


def test_module_default_is_no_exit():
    """server.py must default the opt-in flag to False for embedded use."""
    tree = ast.parse((PACKAGE / "server.py").read_text())
    defaults = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_exit_process_after_shutdown"
    ]
    assert defaults, "module-level _exit_process_after_shutdown declaration missing"
    assert isinstance(defaults[0].value, ast.Constant)
    assert defaults[0].value.value is False


def test_both_launcher_call_sites_opt_in():
    """Both serve entry points — `vllm-mlx serve` (cli.py) and
    `python -m vllm_mlx.server` (server.py main) — must set the flag True
    before uvicorn.run, or the guard silently never fires on that path.
    """
    for module in ("cli.py", "server.py"):
        tree = ast.parse((PACKAGE / module).read_text())
        assigns = _assignments_to(tree, "_exit_process_after_shutdown")
        true_assigns = [
            a
            for a in assigns
            if isinstance(a.value, ast.Constant) and a.value.value is True
        ]
        assert (
            true_assigns
        ), f"{module}: no `_exit_process_after_shutdown = True` call site"


def test_lifespan_consults_the_shared_predicate():
    """The guard in server.py must route through vllm_mlx.shutdown, so the
    behavior standard CI verifies is the behavior the server runs.
    """
    source = (PACKAGE / "server.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "should_exit_without_finalizing"
    ]
    assert calls, "lifespan does not call should_exit_without_finalizing"
    args = calls[0].args
    assert (
        len(args) == 3
    ), "predicate must receive primary_exc, cleanup_exc, opt-in flag"
