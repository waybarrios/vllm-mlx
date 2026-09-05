# SPDX-License-Identifier: Apache-2.0
"""Tests for cache control endpoints."""

import sys
import types

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


def test_clear_prefix_cache_routes_through_model_manager_in_registry_mode(monkeypatch):
    """Regression: in registry mode (--models-config) the module-global
    _engine is never populated (_sync_engine_from_residency only syncs from
    _residency_manager, which is None in registry mode), so the endpoint
    must resolve the live engine via _model_manager.get_metrics_engine() --
    the same pattern /metrics already uses -- instead of silently returning
    {"status": "no_engine"} while _engine stays None.
    """
    import vllm_mlx.server as server

    calls = {"cleared": 0}

    class DummyEngine:
        def clear_prefix_cache(self):
            calls["cleared"] += 1

    class FakeModelManager:
        def __init__(self, engine):
            self._engine = engine

        def get_metrics_engine(self):
            return self._engine

    original_engine = server._engine
    original_model_manager = server._model_manager
    original_api_key = server._api_key
    try:
        server._engine = None  # registry mode: global engine stays unset
        server._model_manager = FakeModelManager(DummyEngine())
        server._api_key = None
        client = TestClient(server.app)

        response = client.delete("/v1/cache/prefix")

        assert response.status_code == 200
        assert response.json()["status"] == "cleared"
        assert calls["cleared"] == 1
    finally:
        server._engine = original_engine
        server._model_manager = original_model_manager
        server._api_key = original_api_key


def test_clear_cache_routes_through_model_manager_in_registry_mode(monkeypatch):
    """Same registry-mode fix as clear_prefix_cache, for DELETE /v1/cache."""
    import vllm_mlx.server as server

    calls = {"cleared": 0}
    fake_utils = types.ModuleType("mlx_vlm.utils")
    fake_utils.clear_multimodal_kv_cache = lambda: None
    fake_utils.clear_pixel_values_cache = lambda: None

    class DummyEngine:
        def clear_runtime_caches(self):
            calls["cleared"] += 1
            return {"prefix_cache": True}

    class FakeModelManager:
        def __init__(self, engine):
            self._engine = engine

        def get_metrics_engine(self):
            return self._engine

    original_engine = server._engine
    original_model_manager = server._model_manager
    original_api_key = server._api_key
    original_module = sys.modules.get("mlx_vlm.utils")
    try:
        server._engine = None  # registry mode: global engine stays unset
        server._model_manager = FakeModelManager(DummyEngine())
        server._api_key = None
        sys.modules["mlx_vlm.utils"] = fake_utils
        client = TestClient(server.app)

        response = client.delete("/v1/cache")

        assert response.status_code == 200
        assert response.json()["engine_cache"] == {"prefix_cache": True}
        assert calls["cleared"] == 1
    finally:
        server._engine = original_engine
        server._model_manager = original_model_manager
        server._api_key = original_api_key
        if original_module is not None:
            sys.modules["mlx_vlm.utils"] = original_module
        else:
            sys.modules.pop("mlx_vlm.utils", None)
