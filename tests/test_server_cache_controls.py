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
