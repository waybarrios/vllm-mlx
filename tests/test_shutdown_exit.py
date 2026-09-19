# SPDX-License-Identifier: Apache-2.0
"""Lifecycle tests for the clean-exit guard.

The guard skips CPython finalization at the end of lifespan shutdown, because
MLX's thread-local compile cache deallocates Python objects from a dying thread
and segfaults. Skipping finalization is only defensible when the shutdown
itself succeeded: if it exits on a failed cleanup, a broken shutdown is
reported as a clean one and the exception that would have said otherwise is
never raised. These tests pin that boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

# vllm_mlx.server and vllm_mlx.cli both import mlx transitively, so these run
# where MLX does — Apple Silicon — like the project's other server-level tests.
# The ubuntu CI job installs no MLX and selects tests explicitly, so this file
# skips there rather than failing.
pytest.importorskip("mlx")


@pytest.fixture
def server(monkeypatch):
    """vllm_mlx.server with every optional subsystem disabled.

    Startup is entirely conditional on these globals, so nulling them makes
    lifespan cheap enough to drive in a unit test — no model, no event loop
    state, no disk.
    """
    import vllm_mlx.server as srv

    for name in (
        "_engine",
        "_mcp_manager",
        "_model_manager",
        "_lifecycle_task",
        "_registry_idle_reaper_task",
        "_residency_manager",
        "_default_model_key",
        "_warm_prompts_path",
    ):
        monkeypatch.setattr(srv, name, None, raising=False)
    monkeypatch.setattr(srv, "_lifespan_active", False, raising=False)
    return srv


@pytest.fixture
def exit_calls(monkeypatch):
    """Record calls to the exit helper instead of ending the test process.

    The lifespan guard calls vllm_mlx.shutdown.exit_without_finalizing (the
    cli name is a kept alias for other callers), so that is what gets patched:
    patching the alias would let the real os._exit(0) end the pytest process
    mid-run — silently, since the status is 0.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        "vllm_mlx.server.exit_without_finalizing",
        lambda *args, **kwargs: calls.append(args),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_mlx.shutdown.exit_without_finalizing",
        lambda *args, **kwargs: calls.append(args),
    )
    return calls


async def _run_lifespan(srv):
    # lifespan is a bare async generator; FastAPI is what wraps it. Do the same
    # here rather than reaching into the app.
    async with asynccontextmanager(srv.lifespan)(object()):
        pass


def test_clean_shutdown_skips_finalization(server, exit_calls, monkeypatch):
    """The whole point: a shutdown that raised nothing exits early."""
    monkeypatch.setattr(server, "_exit_process_after_shutdown", True, raising=False)

    asyncio.run(_run_lifespan(server))

    assert len(exit_calls) == 1


def test_cleanup_failure_is_raised_not_swallowed(server, exit_calls, monkeypatch):
    """A failed cleanup must surface, not be reported as a clean exit.

    This is the regression that matters. Exiting inside the finally block on a
    cleanup failure makes the `raise cleanup_exc` below it unreachable, so a
    cache-save or engine-stop failure would leave the process claiming success.
    """
    monkeypatch.setattr(server, "_exit_process_after_shutdown", True, raising=False)
    monkeypatch.setattr(server, "_engine", MagicMock(), raising=False)

    def boom():
        raise RuntimeError("cache save failed")

    monkeypatch.setattr(server, "_save_prefix_cache_to_disk", boom, raising=False)

    with pytest.raises(RuntimeError, match="cache save failed"):
        asyncio.run(_run_lifespan(server))

    assert exit_calls == [], "must not exit early when cleanup failed"


def test_serving_failure_is_raised_not_swallowed(server, exit_calls, monkeypatch):
    """An exception from the serving phase must propagate too."""
    monkeypatch.setattr(server, "_exit_process_after_shutdown", True, raising=False)

    async def run_and_fail():
        async with asynccontextmanager(server.lifespan)(object()):
            raise RuntimeError("request handling exploded")

    with pytest.raises(RuntimeError, match="request handling exploded"):
        asyncio.run(run_and_fail())

    assert exit_calls == [], "must not exit early when serving failed"


def test_library_use_never_exits(server, exit_calls, monkeypatch):
    """Importing the app must not give it the right to end the host process."""
    monkeypatch.setattr(server, "_exit_process_after_shutdown", False, raising=False)

    asyncio.run(_run_lifespan(server))

    assert exit_calls == []


def test_flag_defaults_off():
    """The default has to be off, or embedding vllm_mlx exits its host."""
    source = inspect.getsource(__import__("vllm_mlx.server", fromlist=["x"]))
    assert "_exit_process_after_shutdown: bool = False" in source


def test_both_launchers_arm_the_flag():
    """Serving from either entry point should get the same behaviour."""
    import vllm_mlx.cli as cli
    import vllm_mlx.server as srv

    assert "_exit_process_after_shutdown = True" in inspect.getsource(cli.serve_command)
    assert "_exit_process_after_shutdown = True" in inspect.getsource(srv.main)


def test_env_var_disables_the_guard(monkeypatch):
    """VLLM_MLX_CLEAN_EXIT=0 restores normal finalization for debugging."""
    import vllm_mlx.cli as cli

    monkeypatch.setenv("VLLM_MLX_CLEAN_EXIT", "0")
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))

    cli._exit_without_finalizing()

    assert exits == [], "with the guard disabled the process must finalize normally"


def test_exit_runs_owned_atexit_work_first(monkeypatch):
    """Skipping finalization skips atexit, so our own handler runs explicitly.

    Temp files from decoded video and rendered document pages are the only
    atexit work this package owns; leaking them on every shutdown would fill
    the temp directory.
    """
    import vllm_mlx.cli as cli

    monkeypatch.delenv("VLLM_MLX_CLEAN_EXIT", raising=False)
    cleaned: list[bool] = []
    monkeypatch.setattr(
        "vllm_mlx.models.mllm.cleanup_all_temp_files",
        lambda: cleaned.append(True),
    )
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))

    cli._exit_without_finalizing()

    assert cleaned == [True], "temp files must be cleaned before exiting"
    assert exits == [0]


def test_exit_survives_cleanup_errors(monkeypatch):
    """A failure in our own cleanup must not stop the exit it precedes."""
    import vllm_mlx.cli as cli

    monkeypatch.delenv("VLLM_MLX_CLEAN_EXIT", raising=False)

    def boom():
        raise OSError("temp dir vanished")

    monkeypatch.setattr("vllm_mlx.models.mllm.cleanup_all_temp_files", boom)
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))

    cli._exit_without_finalizing()

    assert exits == [0]
