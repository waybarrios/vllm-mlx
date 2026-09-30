# SPDX-License-Identifier: Apache-2.0
"""Portable coverage for request admission on the MLX owner thread."""

import asyncio
import importlib
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def engine_core_module(monkeypatch):
    """Import engine_core with lightweight substitutes for MLX-only modules."""
    fake_mlx = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mx

    fake_scheduler = types.ModuleType("vllm_mlx.scheduler")

    class SchedulerConfig:
        pass

    fake_scheduler.Scheduler = object
    fake_scheduler.SchedulerConfig = SchedulerConfig

    fake_registry = types.ModuleType("vllm_mlx.model_registry")
    fake_registry.get_registry = lambda: None

    fake_streams = types.ModuleType("vllm_mlx.mlx_streams")
    fake_streams.bind_generation_streams = lambda: None

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "vllm_mlx.scheduler", fake_scheduler)
    monkeypatch.setitem(sys.modules, "vllm_mlx.model_registry", fake_registry)
    monkeypatch.setitem(sys.modules, "vllm_mlx.mlx_streams", fake_streams)
    monkeypatch.delitem(sys.modules, "vllm_mlx.engine_core", raising=False)

    return importlib.import_module("vllm_mlx.engine_core")


@pytest.fixture
def batched_module(monkeypatch):
    """Import the batched engine without requiring a usable MLX runtime."""
    fake_mlx = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.clear_cache = lambda: None
    fake_mlx.core = fake_mx

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.delitem(sys.modules, "vllm_mlx.engine.batched", raising=False)

    module = importlib.import_module("vllm_mlx.engine.batched")
    return module, fake_mx


@pytest.mark.anyio
async def test_add_request_runs_cache_lookup_on_supplied_model_worker(
    engine_core_module,
):
    """Cache reconstruction during admission must use the model owner thread."""
    engine = object.__new__(engine_core_module.EngineCore)
    engine.config = engine_core_module.EngineConfig(stream_interval=1)
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._request_event = asyncio.Event()

    add_threads = []

    class FakeScheduler:
        def add_request(self, request):
            add_threads.append(threading.get_ident())

    engine.scheduler = FakeScheduler()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    engine._external_generation_worker = worker
    owner_thread = worker.submit(threading.get_ident).result(timeout=5)

    try:
        request_id = await engine.add_request([1, 2, 3], request_id="request-1")
    finally:
        worker.shutdown(wait=True)

    assert request_id == "request-1"
    assert add_threads == [owner_thread]


@pytest.mark.anyio
async def test_cancelled_add_request_waits_for_admission_and_aborts(
    engine_core_module,
):
    """Cancellation must not leave an admitted request without an owner."""
    engine = object.__new__(engine_core_module.EngineCore)
    engine.config = engine_core_module.EngineConfig(stream_interval=1)
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._request_event = asyncio.Event()

    admission_started = threading.Event()
    release_admission = threading.Event()
    aborted = []
    removed = []

    class FakeScheduler:
        def add_request(self, request):
            admission_started.set()
            assert release_admission.wait(timeout=5)

        def abort_request(self, request_id):
            aborted.append(request_id)
            return True

        def remove_finished_request(self, request_id):
            removed.append(request_id)

    engine.scheduler = FakeScheduler()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    engine._external_generation_worker = worker

    task = asyncio.create_task(
        engine.add_request([1, 2, 3], request_id="cancelled-request")
    )
    try:
        assert await asyncio.to_thread(admission_started.wait, 5)
        task.cancel()
        release_admission.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_admission.set()
        worker.shutdown(wait=True)

    assert aborted == ["cancelled-request"]
    assert removed == []
    assert "cancelled-request" not in engine._output_collectors
    assert "cancelled-request" not in engine._stream_states
    assert "cancelled-request" not in engine._finished_events


@pytest.mark.anyio
async def test_cancelled_start_releases_prepared_model_before_reraising(
    batched_module,
):
    """Drained startup cancellation must not interrupt its own cleanup."""
    module, _ = batched_module
    prepare_started = threading.Event()
    release_prepare = threading.Event()

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    engine._generation_thread_id = None
    engine._loaded = False
    engine._model = None
    engine._tokenizer = None
    engine._processor = None
    engine._engine = None
    engine._mllm_scheduler = None
    engine._mllm_instance = None
    engine._is_mllm = False

    def prepare_for_start():
        prepare_started.set()
        assert release_prepare.wait(timeout=5)
        engine._model = object()
        engine._tokenizer = object()

    async def unexpected_start_llm():
        raise AssertionError("_start_llm should not run after cancellation")

    engine.prepare_for_start = prepare_for_start
    engine._start_llm = unexpected_start_llm

    start_task = asyncio.create_task(engine.start())
    try:
        assert await asyncio.to_thread(prepare_started.wait, 5)
        start_task.cancel()
        release_prepare.set()

        with pytest.raises(asyncio.CancelledError):
            await start_task

        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._generation_executor is None
    finally:
        release_prepare.set()
        await engine.stop()


@pytest.mark.anyio
async def test_stop_clears_mlx_cache_on_model_owner_thread(batched_module):
    """The final MLX cache flush must happen before its owner worker exits."""
    module, fake_mx = batched_module
    clear_threads = []
    fake_mx.clear_cache = lambda: clear_threads.append(threading.get_ident())

    class FakeCore:
        def close(self):
            pass

    class FakeAsyncEngine:
        engine = FakeCore()

        async def stop(self):
            pass

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    engine._engine = FakeAsyncEngine()
    engine._mllm_scheduler = None
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = None
    engine._mllm_instance = None
    engine._loaded = True

    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result(timeout=5)

    await engine.stop()

    assert clear_threads == [owner_thread]


@pytest.mark.anyio
async def test_stop_failure_preserves_generation_worker_for_retry(batched_module):
    """A retryable cleanup error must retain the engine and its owner."""
    module, _ = batched_module
    close_threads: list[int] = []

    class FakeCore:
        def close(self):
            close_threads.append(threading.get_ident())

    class FailingOnceAsyncEngine:
        engine = FakeCore()
        stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("engine cleanup failed")

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    async_engine = FailingOnceAsyncEngine()
    engine._engine = async_engine
    engine._mllm_scheduler = None
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = None
    engine._mllm_instance = None
    engine._loaded = True
    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result(timeout=5)

    with pytest.raises(RuntimeError, match="engine cleanup failed"):
        await engine.stop()

    assert close_threads == []
    assert engine._engine is async_engine
    assert engine._loaded is True
    assert worker.submit(threading.get_ident).result(timeout=5) == owner_thread

    await engine.stop()

    assert close_threads == [owner_thread]
    assert engine._engine is None
    assert engine._generation_executor is None


@pytest.mark.anyio
async def test_cancelled_stop_drains_owner_cleanup_before_retry(
    batched_module, monkeypatch
):
    """Cancellation cannot skip a queued owner-thread close operation."""
    module, _ = batched_module
    close_threads: list[int] = []
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    close_submitted = asyncio.Event()

    class FakeCore:
        closed = False

        def close(self):
            if self.closed:
                return
            self.closed = True
            close_threads.append(threading.get_ident())

    class FakeAsyncEngine:
        engine = FakeCore()

        async def stop(self):
            pass

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    async_engine = FakeAsyncEngine()
    engine._engine = async_engine
    engine._mllm_scheduler = None
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = None
    engine._mllm_instance = None
    engine._loaded = True
    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result(timeout=5)

    def block_worker():
        blocker_started.set()
        assert release_blocker.wait(timeout=5)

    worker.submit(block_worker)
    assert await asyncio.to_thread(blocker_started.wait, 5)

    loop = asyncio.get_running_loop()
    original_run_in_executor = loop.run_in_executor

    def track_close_submission(executor, operation, *args):
        future = original_run_in_executor(executor, operation, *args)
        if operation == async_engine.engine.close:
            close_submitted.set()
        return future

    monkeypatch.setattr(loop, "run_in_executor", track_close_submission)

    stop_task = asyncio.create_task(engine.stop())
    await asyncio.wait_for(close_submitted.wait(), timeout=5)
    stop_task.cancel()
    release_blocker.set()

    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert close_threads == [owner_thread]
    assert engine._engine is async_engine
    assert worker.submit(threading.get_ident).result(timeout=5) == owner_thread

    await engine.stop()
    assert engine._engine is None
    assert engine._generation_executor is None


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("load_cache_from_disk", 1), ("save_cache_to_disk", True)],
)
def test_cache_persistence_methods_remain_synchronous(
    batched_module, operation, expected
):
    """Direct BatchedEngine callers keep the established concrete return types."""
    module, _ = batched_module

    class FakePrefixCache:
        def load_from_disk(self, cache_dir):
            return 1

        def save_to_disk(self, cache_dir):
            return True

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    engine._engine = None
    engine._mllm_scheduler = types.SimpleNamespace(
        batch_generator=types.SimpleNamespace(prefix_cache=FakePrefixCache()),
        _ensure_batch_generator=lambda: None,
    )

    assert getattr(engine, operation)("cache") == expected


@pytest.mark.parametrize(
    ("operation", "args", "expected"),
    [
        ("load_cache_from_disk", ("cache",), 1),
        ("save_cache_to_disk", ("cache",), True),
        ("clear_runtime_caches", (), {"memory_aware_cache": True}),
        ("clear_prefix_cache", (), None),
    ],
)
def test_synchronous_cache_methods_use_model_owner_thread(
    batched_module, operation, args, expected
):
    """The compatible synchronous API must still respect MLX ownership."""
    module, _ = batched_module
    operation_threads = []

    class FakeAsyncEngine:
        def load_cache_from_disk(self, cache_dir):
            operation_threads.append(threading.get_ident())
            return 1

        def save_cache_to_disk(self, cache_dir):
            operation_threads.append(threading.get_ident())
            return True

        def clear_runtime_caches(self):
            operation_threads.append(threading.get_ident())
            return {"memory_aware_cache": True}

        def clear_prefix_cache(self):
            operation_threads.append(threading.get_ident())

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    engine._engine = FakeAsyncEngine()
    engine._mllm_scheduler = None
    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result(timeout=5)

    try:
        result = getattr(engine, operation)(*args)
    finally:
        worker.shutdown(wait=True)
        engine._generation_executor = None

    assert result == expected
    assert operation_threads == [owner_thread]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("_clear_runtime_caches_on_owner", {"memory_aware_cache": True}),
        ("_clear_prefix_cache_on_owner", None),
    ],
)
async def test_cache_mutations_run_on_model_owner_thread(
    batched_module, operation, expected
):
    """Dropping cached MLX arrays must stay on the text model owner thread."""
    module, _ = batched_module
    mutation_threads = []

    class FakeAsyncEngine:
        def clear_runtime_caches(self):
            mutation_threads.append(threading.get_ident())
            return {"memory_aware_cache": True}

        def clear_prefix_cache(self):
            mutation_threads.append(threading.get_ident())

    engine = object.__new__(module.BatchedEngine)
    engine._generation_executor = None
    engine._engine = FakeAsyncEngine()
    engine._mllm_scheduler = None

    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result(timeout=5)

    try:
        result = await getattr(engine, operation)()
    finally:
        worker.shutdown(wait=True)
        engine._generation_executor = None

    assert result == expected
    assert mutation_threads == [owner_thread]
