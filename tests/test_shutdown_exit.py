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
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
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
    # A server that bound its port; tests of the failed-bind path override it.
    monkeypatch.setattr(srv, "_uvicorn_server", SimpleNamespace(started=True))
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


def test_failed_bind_shutdown_does_not_exit_early(server, exit_calls, monkeypatch):
    """Uvicorn shuts the lifespan down after a failed bind, then exits 3.

    The bind error never reaches the app, so this shutdown raises nothing.
    Exiting early here would turn uvicorn's startup-failure status into 0.
    """
    monkeypatch.setattr(server, "_exit_process_after_shutdown", True, raising=False)
    monkeypatch.setattr(server, "_uvicorn_server", SimpleNamespace(started=False))

    asyncio.run(_run_lifespan(server))

    assert exit_calls == [], "must not exit early before the listener started"


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


def test_both_launchers_serve_through_run_uvicorn():
    """Serving from either entry point should get the same behaviour."""
    import vllm_mlx.cli as cli
    import vllm_mlx.server as srv

    assert "server.run_uvicorn(" in inspect.getsource(cli.serve_command)
    assert "run_uvicorn(" in inspect.getsource(srv.main)


class _FakeUvicornServer:
    """Stands in for uvicorn.Server; ``run`` reports what the globals were."""

    def __init__(self, started: bool, raises: BaseException | None = None):
        self.started = started
        self._raises = raises
        self.seen: dict = {}

    def __call__(self, config):
        self.config = config
        return self

    def run(self):
        import vllm_mlx.server as srv

        self.seen = {
            "opt_in": srv._exit_process_after_shutdown,
            "server": srv._uvicorn_server,
        }
        if self._raises is not None:
            raise self._raises


def test_run_uvicorn_arms_the_exit_only_while_running(monkeypatch):
    import vllm_mlx.server as srv

    fake = _FakeUvicornServer(started=True)
    monkeypatch.setattr(srv.uvicorn, "Server", fake)

    srv.run_uvicorn(srv.app, host="127.0.0.1", port=0)

    assert fake.seen == {"opt_in": True, "server": fake}
    # Process-wide state: left on, the next lifespan shutdown in this
    # interpreter would call the real os._exit(0).
    assert srv._exit_process_after_shutdown is False
    assert srv._uvicorn_server is None


def test_run_uvicorn_keeps_uvicorns_startup_failure_status(monkeypatch):
    """A server that never started exits 3, exactly as uvicorn.run does."""
    import vllm_mlx.server as srv

    monkeypatch.setattr(srv.uvicorn, "Server", _FakeUvicornServer(started=False))

    with pytest.raises(SystemExit) as excinfo:
        srv.run_uvicorn(srv.app, host="127.0.0.1", port=0)

    assert excinfo.value.code == 3
    assert srv._exit_process_after_shutdown is False


def test_run_uvicorn_resets_state_when_run_raises(monkeypatch):
    import vllm_mlx.server as srv

    monkeypatch.setattr(
        srv.uvicorn,
        "Server",
        _FakeUvicornServer(started=False, raises=OSError("address in use")),
    )

    with pytest.raises(OSError, match="address in use"):
        srv.run_uvicorn(srv.app, host="127.0.0.1", port=0)

    assert srv._exit_process_after_shutdown is False
    assert srv._uvicorn_server is None


# ---------------------------------------------------------------------------
# Real launchers, real uvicorn, in a subprocess. Mocking uvicorn.run cannot
# reproduce the ordering that matters: on a failed bind uvicorn runs lifespan
# shutdown *before* it exits with its startup-failure status.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _launcher_cmd(launcher: str, model_dir: Path, port: int) -> list[str]:
    common = ["--host", "127.0.0.1", "--port", str(port), "--lazy-load-model"]
    if launcher == "cli":
        return [sys.executable, "-m", "vllm_mlx.cli", "serve", str(model_dir)] + common
    return [sys.executable, "-m", "vllm_mlx.server", "--model", str(model_dir)] + common


@pytest.fixture
def lazy_model_dir(tmp_path):
    """A local model path; --lazy-load-model never reads past config.json."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "llama"}')
    return model_dir


def _launcher_env() -> dict:
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    env.pop("VLLM_MLX_CLEAN_EXIT", None)  # the workaround's default: enabled
    return env


@pytest.mark.parametrize("launcher", ["cli", "server"])
def test_occupied_port_exits_with_startup_failure(launcher, lazy_model_dir):
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]

        result = subprocess.run(
            _launcher_cmd(launcher, lazy_model_dir, port),
            cwd=REPO_ROOT,
            env=_launcher_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

    output = result.stdout + result.stderr
    assert "address already in use" in output.lower(), output[-2000:]
    assert result.returncode == 3, output[-2000:]


@pytest.mark.parametrize("launcher", ["cli", "server"])
def test_started_server_still_exits_early_and_cleanly(launcher, lazy_model_dir):
    """The other half: a server that listened still takes the early exit."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    proc = subprocess.Popen(
        _launcher_cmd(launcher, lazy_model_dir, port),
        cwd=REPO_ROOT,
        env=_launcher_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.25)
        assert proc.poll() is None, "server exited before it was reachable"
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, output[-2000:]
    # Logged by the lifespan just before exit_without_finalizing().
    assert ":Shutdown complete" in output, output[-2000:]


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
