# SPDX-License-Identifier: Apache-2.0
"""Tests for cache control endpoints."""

import asyncio
import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_cache_stats_includes_engine_cache(monkeypatch):
    import vllm_mlx.server as server

    fake_utils = types.ModuleType("mlx_vlm.utils")
    fake_utils.get_multimodal_kv_cache_stats = lambda: {"entries": 1}
    fake_utils.get_pixel_values_cache_stats = lambda: {"entries": 2}
    fake_utils.get_pil_cache_stats = lambda: {"entries": 3}

    class DummyEngine:
        def get_cache_stats(self):
            return {"prefix_cache": {"hits": 7, "misses": 2}}

    original_engine = server._engine
    original_api_key = server._api_key
    original_module = sys.modules.get("mlx_vlm.utils")
    try:
        server._engine = DummyEngine()
        server._api_key = None
        sys.modules["mlx_vlm.utils"] = fake_utils
        client = TestClient(server.app)

        response = client.get("/v1/cache/stats")
        assert response.status_code == 200
        assert response.json()["engine_cache"] == {
            "prefix_cache": {"hits": 7, "misses": 2}
        }
    finally:
        server._engine = original_engine
        server._api_key = original_api_key
        if original_module is not None:
            sys.modules["mlx_vlm.utils"] = original_module
        else:
            sys.modules.pop("mlx_vlm.utils", None)


@pytest.fixture
def cache_route(monkeypatch):
    """Exercise real HTTP handlers; replace only backend/cache boundaries."""
    import mlx.core as mx

    import vllm_mlx.server as server

    events = []
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("flush"))
    utils = types.ModuleType("mlx_vlm.utils")
    utils.clear_multimodal_kv_cache = lambda: events.append("multimodal")
    utils.clear_pixel_values_cache = lambda: events.append("pixel")
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_warm_prompts_path", None)
    return server, TestClient(server.app), events


def _engine_with_clear(events, method, mode):
    """Owner and legacy operations finish only after the async yield."""

    def clear():
        events.append("engine")
        return {"prefix_cache": True}

    async def async_clear():
        await asyncio.sleep(0)
        return clear()

    def wrong_fallback():
        raise AssertionError("owner helper must take precedence")

    engine = types.SimpleNamespace(model=object(), requests={"live": object()})
    setattr(engine, method, clear if mode == "sync" else async_clear)
    if mode == "owner":
        setattr(engine, method, wrong_fallback)
        setattr(engine, f"_{method}_on_owner", async_clear)
    return engine


@pytest.mark.parametrize("mode", ["owner", "sync", "awaitable"])
@pytest.mark.parametrize("rewarm", [False, True])
def test_prefix_clear_flushes_after_owner_release_before_rewarm(
    cache_route, monkeypatch, mode, rewarm
):
    server, client, events = cache_route
    engine = _engine_with_clear(events, "clear_prefix_cache", mode)
    model, requests = engine.model, engine.requests.copy()
    monkeypatch.setattr(server, "_engine", engine)
    if rewarm:
        import vllm_mlx.prompt_warmup as warmup

        engine.stream_chat = object()
        monkeypatch.setattr(server, "_warm_prompts_path", "unused.json")
        monkeypatch.setattr(warmup, "load_warmup_file", lambda _: [])

        async def warm(*args):
            events.append("rewarm")
            return {"count": 0, "skipped": 0, "elapsed_ms": 0}

        monkeypatch.setattr(warmup, "warm_prefix_cache", warm)

    response = client.delete("/v1/cache/prefix")

    assert response.status_code == 200
    assert response.json() == {"status": "cleared", "rewarm_scheduled": rewarm}
    assert events == ["engine", "flush"] + (["rewarm"] if rewarm else [])
    assert server._engine is engine
    assert engine.model is model
    assert engine.requests == requests


@pytest.mark.parametrize("state", ["no_engine", "unsupported", "error"])
def test_prefix_clear_does_not_flush_when_no_references_were_cleared(
    cache_route, monkeypatch, state
):
    server, client, events = cache_route
    if state != "no_engine":
        engine = types.SimpleNamespace()
        if state == "error":

            async def fail():
                raise RuntimeError("owner failed")

            engine.clear_prefix_cache = fail
        monkeypatch.setattr(server, "_engine", engine)

    response = client.delete("/v1/cache/prefix")

    expected = (
        {"status": "no_engine"}
        if state == "no_engine"
        else {"status": "not_supported", "rewarm_scheduled": False}
    )
    assert response.status_code == 200
    assert response.json() == expected
    assert events == []


@pytest.mark.parametrize("mode", ["owner", "sync", "awaitable"])
@pytest.mark.parametrize("optional_vlm", [True, False])
def test_all_cache_clear_flushes_last_including_optional_import_fallback(
    cache_route, monkeypatch, mode, optional_vlm
):
    server, client, events = cache_route
    engine = _engine_with_clear(events, "clear_runtime_caches", mode)
    model, requests = engine.model, engine.requests.copy()
    monkeypatch.setattr(server, "_engine", engine)
    if not optional_vlm:
        monkeypatch.setitem(sys.modules, "mlx_vlm.utils", None)

    response = client.delete("/v1/cache")

    expected = {"status": "cleared", "engine_cache": {"prefix_cache": True}}
    if optional_vlm:
        expected["caches"] = ["multimodal_kv", "pixel_values", "pil_image"]
    else:
        expected["error"] = "Cache clear not available (mlx_vlm not loaded)"
    assert response.status_code == 200
    assert response.json() == expected
    assert events == ["engine"] + (["multimodal", "pixel"] if optional_vlm else []) + [
        "flush"
    ]
    assert server._engine is engine
    assert engine.model is model
    assert engine.requests == requests


def test_all_cache_clear_flushes_global_caches_without_engine(cache_route):
    _, client, events = cache_route
    response = client.delete("/v1/cache")
    assert response.status_code == 200
    assert response.json()["engine_cache"] is None
    assert events == ["multimodal", "pixel", "flush"]


def test_all_cache_clear_does_not_load_unused_optional_backend(
    cache_route, monkeypatch
):
    server, _, events = cache_route
    cache = {"entry": object()}
    monkeypatch.setattr(
        server, "_engine", types.SimpleNamespace(clear_runtime_caches=cache.clear)
    )
    monkeypatch.delitem(sys.modules, "mlx_vlm.utils")
    original_import = builtins.__import__

    def forbid_unused_backend(name, *args, **kwargs):
        if name == "mlx_vlm.utils":
            raise RuntimeError("unused optional backend must not be loaded")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_unused_backend)
    client = TestClient(server.app, raise_server_exceptions=False)

    response = client.delete("/v1/cache")

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "engine_cache": None,
        "error": "Cache clear not available (mlx_vlm not loaded)",
    }
    assert cache == {}
    assert "mlx_vlm.utils" not in sys.modules
    assert events == ["flush"]


def test_all_cache_clear_keeps_engine_error_and_releases_global_caches(
    cache_route, monkeypatch
):
    server, client, events = cache_route

    def fail():
        events.append("engine-error")
        raise RuntimeError("owner failed")

    monkeypatch.setattr(
        server, "_engine", types.SimpleNamespace(clear_runtime_caches=fail)
    )
    response = client.delete("/v1/cache")
    assert response.status_code == 200
    assert response.json()["engine_cache"] == {"error": "owner failed"}
    assert events == ["engine-error", "multimodal", "pixel", "flush"]


def test_all_cache_clear_flushes_released_subset_if_global_clear_fails(
    cache_route, monkeypatch
):
    server, _, events = cache_route

    def fail():
        events.append("pixel-error")
        raise RuntimeError("global cache failure")

    monkeypatch.setattr(sys.modules["mlx_vlm.utils"], "clear_pixel_values_cache", fail)
    client = TestClient(server.app, raise_server_exceptions=False)
    assert client.delete("/v1/cache").status_code == 500
    assert events == ["multimodal", "pixel-error", "flush"]


@pytest.mark.parametrize("route", ["/v1/cache", "/v1/cache/prefix"])
def test_allocator_flush_failure_is_logged_without_changing_response(
    cache_route, monkeypatch, caplog, route
):
    import mlx.core as mx

    server, client, events = cache_route
    monkeypatch.setattr(
        server, "_engine", _engine_with_clear(events, "clear_prefix_cache", "owner")
    )

    def fail():
        raise RuntimeError("allocator unavailable")

    monkeypatch.setattr(mx, "clear_cache", fail)
    response = client.delete(route)
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"
    assert "allocator unavailable" in caplog.text


@pytest.mark.parametrize("route", ["/v1/cache", "/v1/cache/prefix"])
def test_optional_mlx_absence_preserves_clear_response(cache_route, monkeypatch, route):
    server, client, events = cache_route
    monkeypatch.setattr(
        server, "_engine", _engine_with_clear(events, "clear_prefix_cache", "owner")
    )
    original_import = builtins.__import__

    def import_without_mlx(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("MLX not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mlx)
    response = client.delete(route)
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"
    assert "flush" not in events


@pytest.mark.parametrize("route", ["/v1/cache", "/v1/cache/prefix"])
def test_cache_clear_auth_failure_does_not_release_anything(
    cache_route, monkeypatch, route
):
    server, client, events = cache_route
    monkeypatch.setattr(server, "_api_key", "required-test-key")
    response = client.delete(route)
    assert response.status_code == 401
    assert events == []


@pytest.mark.anyio
async def test_cache_release_benchmark_without_barrier_does_not_synchronize(
    monkeypatch,
):
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cache_release.py"
    )
    spec = importlib.util.spec_from_file_location("cache_release_no_barrier", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    def unexpected_sync(*args):
        raise AssertionError("zero-request observation must not add a stream barrier")

    mx = types.SimpleNamespace(synchronize=unexpected_sync)
    engine = types.SimpleNamespace(
        get_stats=lambda: {"num_running": 0, "num_waiting": 0}
    )
    await benchmark._quiesce(mx, engine, barrier=False)


@pytest.mark.anyio
async def test_cache_release_benchmark_registry_uses_manager_with_two_residents(
    monkeypatch,
):
    import mlx.core as mx

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cache_release.py"
    )
    spec = importlib.util.spec_from_file_location("cache_release_registry", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    args = types.SimpleNamespace(synthetic_registry=True, cache_limit_mib=1024)
    manager, group = await benchmark._load_registry(args, mx)
    try:
        assert set(manager._loaded) == {"synthetic-a", "synthetic-b"}
        assert manager.has_model("unloaded")
        for engine in group.engines.values():
            engine.fill(mx, 16 * benchmark.MIB)
        assert benchmark._entry_count(group.get_cache_stats()) == 2
        result = await manager.clear_caches(prefix_only=True)
        assert all(item["status"] == "cleared" for item in result.values())
        assert benchmark._entry_count(group.get_cache_stats()) == 0
        assert set(manager._loaded) == {"synthetic-a", "synthetic-b"}
    finally:
        await manager.shutdown()


def test_clear_cache_clears_engine_managed_runtime_caches(monkeypatch):
    import vllm_mlx.server as server

    calls = {"multimodal": 0, "pixel": 0, "engine": 0}
    fake_utils = types.ModuleType("mlx_vlm.utils")

    def clear_multimodal():
        calls["multimodal"] += 1

    def clear_pixel():
        calls["pixel"] += 1

    fake_utils.clear_multimodal_kv_cache = clear_multimodal
    fake_utils.clear_pixel_values_cache = clear_pixel

    class DummyEngine:
        def clear_runtime_caches(self):
            calls["engine"] += 1
            return {"prefix_cache": True}

    original_engine = server._engine
    original_api_key = server._api_key
    original_module = sys.modules.get("mlx_vlm.utils")
    try:
        server._engine = DummyEngine()
        server._api_key = None
        sys.modules["mlx_vlm.utils"] = fake_utils
        client = TestClient(server.app)

        response = client.delete("/v1/cache")
        assert response.status_code == 200
        assert response.json()["engine_cache"] == {"prefix_cache": True}
        assert calls == {"multimodal": 1, "pixel": 1, "engine": 1}
    finally:
        server._engine = original_engine
        server._api_key = original_api_key
        if original_module is not None:
            sys.modules["mlx_vlm.utils"] = original_module
        else:
            sys.modules.pop("mlx_vlm.utils", None)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "finish_reason", "completion_tokens", "must_fail"),
    [
        ("", "error", 0, True),
        ("partial", "error", 1, True),
        ("", "stop", 0, False),  # Valid immediate EOS is not a failed request.
        ("OK", "stop", 1, False),
        ("OK", "length", 16, False),
    ],
)
async def test_cache_release_benchmark_rejects_terminal_generation_errors(
    capsys, text, finish_reason, completion_tokens, must_fail
):
    """A failed fill or post-clear liveness request cannot earn acceptance."""
    from vllm_mlx.engine.base import GenerationOutput

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cache_release.py"
    )
    spec = importlib.util.spec_from_file_location("cache_release_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class OutputEngine:
        async def stream_chat(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Reply with OK."}]
            assert kwargs == {"max_tokens": 16, "temperature": 0}
            yield GenerationOutput(text="", finished=False, finish_reason=None)
            yield GenerationOutput(
                text=text,
                prompt_tokens=64,
                completion_tokens=completion_tokens,
                finished=True,
                finish_reason=finish_reason,
            )

    if must_fail:
        with pytest.raises(RuntimeError, match="generation"):
            await benchmark._generate(OutputEngine(), "Reply with OK.")
        assert capsys.readouterr().out == ""
    else:
        await benchmark._generate(OutputEngine(), "Reply with OK.")
        assert '"event": "generation"' in capsys.readouterr().out
