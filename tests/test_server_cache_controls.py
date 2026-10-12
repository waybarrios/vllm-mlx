# SPDX-License-Identifier: Apache-2.0
"""Tests for cache control endpoints."""

import asyncio
import sys
import types

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


class _RuntimeOnlyEngine:
    """Minimal registry engine without a prefix cache; records cache clears."""

    def __init__(self, name, calls):
        self.name = name
        self._calls = calls

    async def start(self):
        return None

    async def stop(self):
        return None

    def clear_runtime_caches(self):
        self._calls.append(("runtime", self.name))
        return {"prefix_cache": True}


class _CacheEngine(_RuntimeOnlyEngine):
    def clear_prefix_cache(self):
        self._calls.append(("prefix", self.name))


def _registry_manager(tmp_path, names, calls, *, no_prefix=(), loaded=None):
    """Build a real ModelManager with ``loaded`` (default: all) engines resident."""
    from vllm_mlx.model_registry import (
        ContentionPolicy,
        ModelManager,
        RegisteredModel,
        RegistryManagerConfig,
        RegistryServeDefaults,
    )
    from vllm_mlx.utils.download import DownloadConfig

    registry = {}
    for name in names:
        source = tmp_path / name
        source.mkdir()
        registry[name] = RegisteredModel(
            name=name, source=str(source), estimated_memory_bytes=1024**3
        )
    manager = ModelManager(
        RegistryManagerConfig(
            memory_budget_bytes=len(names) * 1024**3,
            idle_unload_seconds=0.0,
            policy=ContentionPolicy(strategy="wait_then_fail", wait_timeout_s=1.0),
        ),
        registry,
        RegistryServeDefaults(
            continuous_batching=False,
            force_mllm=False,
            enable_mtp=False,
            prefill_step_size=2048,
            specprefill_enabled=False,
            specprefill_threshold=8192,
            specprefill_keep_pct=0.3,
            specprefill_backbone_pct=0.0,
            specprefill_draft_model=None,
            prefix_trie_cache=False,
            prefix_trie_cache_size=32,
            prefix_trie_cache_memory_mb=None,
            stream_interval=1,
            gpu_memory_utilization=0.9,
            scheduler_config=None,
            max_tokens=32768,
            download_config=DownloadConfig(),
        ),
        engine_factory=lambda config: (
            _RuntimeOnlyEngine if config.entry.name in no_prefix else _CacheEngine
        )(config.entry.name, calls),
    )

    async def _load():
        for name in names if loaded is None else loaded:
            lease = await manager.acquire(name)
            await lease.release()

    asyncio.run(_load())
    return manager


@pytest.fixture
def registry_client(monkeypatch):
    import vllm_mlx.server as server

    calls = {"multimodal": 0, "pixel": 0}
    fake_utils = types.ModuleType("mlx_vlm.utils")

    def clear_multimodal():
        calls["multimodal"] += 1

    def clear_pixel():
        calls["pixel"] += 1

    fake_utils.clear_multimodal_kv_cache = clear_multimodal
    fake_utils.clear_pixel_values_cache = clear_pixel
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_utils)
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_api_key", None)

    def _use(manager):
        monkeypatch.setattr(server, "_model_manager", manager)
        return TestClient(server.app)

    return _use, calls


def test_clear_cache_registry_mode_clears_every_loaded_engine(
    tmp_path, registry_client
):
    use, global_calls = registry_client
    calls = []
    client = use(_registry_manager(tmp_path, ["alpha", "beta"], calls))

    response = client.delete("/v1/cache")

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "engine_caches": {
            "alpha": {"prefix_cache": True},
            "beta": {"prefix_cache": True},
        },
        "caches": ["multimodal_kv", "pixel_values", "pil_image"],
    }
    assert sorted(calls) == [("runtime", "alpha"), ("runtime", "beta")]
    assert global_calls == {"multimodal": 1, "pixel": 1}


def test_clear_cache_registry_mode_targets_one_engine(tmp_path, registry_client):
    use, global_calls = registry_client
    calls = []
    client = use(_registry_manager(tmp_path, ["alpha", "beta"], calls))

    response = client.delete("/v1/cache", params={"model": "beta"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "engine_caches": {"beta": {"prefix_cache": True}},
    }
    assert calls == [("runtime", "beta")]
    # Process-wide mlx-vlm caches are shared by every model; a targeted
    # clear must not wipe them.
    assert global_calls == {"multimodal": 0, "pixel": 0}


def test_clear_prefix_cache_registry_mode_clears_every_loaded_engine(
    tmp_path, registry_client
):
    use, _ = registry_client
    calls = []
    client = use(_registry_manager(tmp_path, ["alpha", "beta"], calls))

    response = client.delete("/v1/cache/prefix")

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "rewarm_scheduled": False,
        "models": {"alpha": {"status": "cleared"}, "beta": {"status": "cleared"}},
    }
    assert sorted(calls) == [("prefix", "alpha"), ("prefix", "beta")]

    calls.clear()
    response = client.delete("/v1/cache/prefix", params={"model": "alpha"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "cleared",
        "rewarm_scheduled": False,
        "models": {"alpha": {"status": "cleared"}},
    }
    assert calls == [("prefix", "alpha")]


def test_clear_prefix_cache_registry_mode_reports_partial_clear(
    tmp_path, registry_client
):
    use, _ = registry_client
    calls = []
    client = use(
        _registry_manager(tmp_path, ["alpha", "beta"], calls, no_prefix={"beta"})
    )

    response = client.delete("/v1/cache/prefix")

    assert response.status_code == 200
    assert response.json() == {
        "status": "partial",
        "rewarm_scheduled": False,
        "models": {
            "alpha": {"status": "cleared"},
            "beta": {"status": "not_supported"},
        },
    }
    assert calls == [("prefix", "alpha")]


@pytest.mark.parametrize("path", ["/v1/cache", "/v1/cache/prefix"])
def test_cache_clear_registry_mode_rejects_unknown_or_unloaded_target(
    tmp_path, registry_client, path
):
    use, global_calls = registry_client
    calls = []
    client = use(
        _registry_manager(tmp_path, ["alpha", "beta"], calls, loaded=["alpha"])
    )

    unknown = client.delete(path, params={"model": "nope"})
    unloaded = client.delete(path, params={"model": "beta"})

    assert unknown.status_code == 404
    assert "does not exist" in unknown.json()["detail"]
    assert unloaded.status_code == 404
    assert "not loaded" in unloaded.json()["detail"]
    assert calls == []
    assert global_calls == {"multimodal": 0, "pixel": 0}


def test_clear_prefix_cache_registry_mode_with_nothing_loaded(
    tmp_path, registry_client
):
    use, _ = registry_client
    calls = []
    client = use(_registry_manager(tmp_path, ["alpha"], calls, loaded=[]))

    response = client.delete("/v1/cache/prefix")

    assert response.status_code == 200
    assert response.json() == {
        "status": "no_engine",
        "rewarm_scheduled": False,
        "models": {},
    }


def test_clear_prefix_cache_single_model_mode_unchanged(monkeypatch):
    import vllm_mlx.server as server

    calls = []
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_engine", _CacheEngine("solo", calls))
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_warm_prompts_path", None)
    client = TestClient(server.app)

    response = client.delete("/v1/cache/prefix")

    assert response.status_code == 200
    assert response.json() == {"status": "cleared", "rewarm_scheduled": False}
    assert calls == [("prefix", "solo")]
