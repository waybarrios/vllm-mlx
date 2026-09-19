# SPDX-License-Identifier: Apache-2.0
"""Test that BatchedEngine and EngineCore route cache operations to the MLX owner thread."""

import threading
from concurrent.futures import ThreadPoolExecutor
import pytest

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.engine_core import EngineConfig, EngineCore
from vllm_mlx.mlx_executor import MLXExecutor


def test_cache_operations_run_on_mlx_executor_thread():
    """When callers call load/save/clear cache from MainThread or random threads,

    they MUST execute on the MLX owner thread where model was loaded.
    """
    engine = object.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._loaded = True

    # Dedicated worker
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    try:
        executor = MLXExecutor(worker=worker, owns_worker=False)
        worker_tid = executor.owner_thread_id
        main_tid = threading.get_ident()
        assert worker_tid != main_tid

        engine._mlx_executor = executor

        # Create dummy core engine
        core = object.__new__(EngineCore)
        core.config = EngineConfig()
        core._mlx_executor = executor

        # Track execution threads
        run_threads = {}

        class FakeScheduler:
            def load_cache_from_disk(self, cache_dir):
                run_threads["load"] = threading.get_ident()
                return 42

            def save_cache_to_disk(self, cache_dir):
                run_threads["save"] = threading.get_ident()
                return True

            def clear_runtime_caches(self):
                run_threads["clear_runtime"] = threading.get_ident()
                return {"cleared": True}

            def clear_prefix_cache(self):
                run_threads["clear_prefix"] = threading.get_ident()

        core.scheduler = FakeScheduler()
        engine._engine = core

        # 1. Call from MainThread
        loaded = engine.load_cache_from_disk("/tmp/test-cache")
        assert loaded == 42
        assert run_threads["load"] == worker_tid, "load_cache_from_disk must run on worker thread"

        saved = engine.save_cache_to_disk("/tmp/test-cache")
        assert saved is True
        assert run_threads["save"] == worker_tid, "save_cache_to_disk must run on worker thread"

        cleared_runtime = engine.clear_runtime_caches()
        assert cleared_runtime == {"cleared": True}
        assert run_threads["clear_runtime"] == worker_tid, "clear_runtime_caches must run on worker thread"

        engine.clear_prefix_cache()
        assert run_threads["clear_prefix"] == worker_tid, "clear_prefix_cache must run on worker thread"

    finally:
        worker.shutdown(wait=True)
