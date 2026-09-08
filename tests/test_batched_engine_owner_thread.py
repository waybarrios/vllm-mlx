# SPDX-License-Identifier: Apache-2.0
"""BatchedEngine must load the model on the thread that later steps it.

MLX streams exist only in the thread that created them, and ``BatchGenerator``
captures ``generation_stream`` into ``self._stream`` when it is built. So a
model loaded on one thread cannot be stepped from another: the first batched
step raises "There is no Stream(gpu, N) in current thread".

That failure does not look like a stream problem from the outside. ``EngineCore``
catches it, falls back to stepping on the model thread, hits the same error
there, and — because that fallback fires only once — then spins on the error.
What an operator sees is ``running=1``, the step counter climbing into the
millions, and not one token emitted.

These tests record threads rather than touching MLX, so they run anywhere.
Fakes follow the convention in ``test_engine_core_thread_streams.py``.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


async def _noop(marker: list) -> None:
    marker.append(True)


class _SchedulerOutput:
    outputs = []
    finished_request_ids = []


class _FakeScheduler:
    batch_generator = SimpleNamespace(_partial=None)

    def __init__(self, engine, steps: int = 3):
        self._engine = engine
        self._steps = steps
        self.calls = 0
        self.step_threads: list[int] = []

    def has_requests(self):
        return self.calls < self._steps

    def step(self):
        self.step_threads.append(threading.get_ident())
        self.calls += 1
        if self.calls == self._steps:
            self._engine._running = False
        return _SchedulerOutput()

    def _close_batch_generator(self):
        pass

    def close_ssd_tier(self):
        pass


def _bare_engine_core(monkeypatch, worker=None):
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=0, stream_interval=1)
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    if worker is not None:
        engine._external_generation_worker = worker
    engine.scheduler = _FakeScheduler(engine)
    monkeypatch.setattr(
        "vllm_mlx.engine_core.bind_generation_streams", lambda *a, **k: None
    )
    return engine


def _run_executors_deterministically(monkeypatch):
    """Drive executor callbacks synchronously while retaining worker identity."""
    loop = asyncio.get_running_loop()

    def run_in_executor(executor, operation, *args):
        future = loop.create_future()
        try:
            if executor is None:
                result = operation(*args)
            else:
                result = executor.submit(operation, *args).result(timeout=5)
            future.set_result(result)
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(loop, "run_in_executor", run_in_executor)


def test_generation_worker_is_one_reused_thread():
    """The worker has to be stable: it owns the loaded model for the process."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None

    worker = engine._generation_worker()
    assert worker is engine._generation_worker(), "executor must be reused"

    names = {worker.submit(threading.get_ident).result() for _ in range(5)}
    assert len(names) == 1, f"stepping spread over several threads: {names}"
    worker.shutdown(wait=True)


def test_worker_thread_is_named_for_the_engine_loop():
    """engine_core's own fallback pool uses this prefix; keep them consistent."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    worker = engine._generation_worker()
    name = worker.submit(lambda: threading.current_thread().name).result()
    assert name.startswith("engine-core"), name
    worker.shutdown(wait=True)


@pytest.mark.anyio
async def test_model_load_and_stepping_share_one_thread(monkeypatch):
    """The regression: load on thread A, step on thread B.

    Loading inline on the event loop (issue #407) did not fix this, because
    stepping runs on engine_core's worker, not on the loop.
    """
    from vllm_mlx.engine.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._loaded = False
    engine._model = None
    engine._model_name = "fake"
    engine._is_mllm = False

    load_threads: list[int] = []

    def prepare_for_start():
        load_threads.append(threading.get_ident())
        engine._model = object()

    engine.prepare_for_start = prepare_for_start

    # Drive the engine's real start(). Replicating its load call in the test
    # body instead would leave the code under test unexercised: reverting
    # start() to the unpinned pool would still go green.
    started_llm = []
    engine._start_llm = lambda: _noop(started_llm)

    await asyncio.wait_for(engine.start(), timeout=10)
    assert started_llm, "start() never reached engine startup"

    # Step exactly as the engine loop does: on the worker start() loaded on.
    core = _bare_engine_core(monkeypatch, worker=engine._generation_worker())
    await asyncio.wait_for(core._engine_loop(), timeout=5)

    engine._generation_executor.shutdown(wait=True)

    assert load_threads, "the model was never loaded"
    assert core.scheduler.step_threads, "the scheduler never stepped"
    assert set(load_threads) == set(core.scheduler.step_threads), (
        f"load ran on {load_threads}, stepping on {core.scheduler.step_threads} — "
        "BatchGenerator would carry a stream the stepping thread does not have"
    )
    assert threading.get_ident() not in load_threads, "load must leave the event loop"


@pytest.mark.anyio
async def test_mllm_load_and_step_share_the_event_loop_thread(monkeypatch):
    """The other half of the invariant, and the one easy to break.

    MLLM never reaches AsyncEngineCore: ``_start_mllm`` drives MLLMScheduler,
    whose ``_process_loop`` calls ``step()`` on the event loop with no executor
    hop. Pinning the load to the generation worker regardless of path would put
    the model on one stream owner and the stepping on another — the same
    failure this PR fixes, inverted, and on the configuration that actually
    serves MLLM traffic.
    """
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._loaded = False
    engine._model = None
    engine._model_name = "fake"
    engine._is_mllm = True

    assert engine._model_load_executor() is None, "MLLM must not pin to a worker"

    load_threads: list[int] = []

    def prepare_for_start(self):
        load_threads.append(threading.get_ident())
        self._model = object()

    # Patch the class, not the instance: _uses_default_prepare_for_start
    # compares against the class function, and an instance attribute would
    # route this down the "overridden prepare may block" branch instead of the
    # real default one.
    monkeypatch.setattr(BatchedEngine, "prepare_for_start", prepare_for_start)
    started: list[bool] = []
    engine._start_mllm = lambda: _noop(started)

    await asyncio.wait_for(engine.start(), timeout=10)
    assert started, "start() never reached MLLM startup"

    # Now step, the way MLLMScheduler actually does.
    scheduler = object.__new__(MLLMScheduler)
    scheduler._running = True
    step_threads: list[int] = []

    class _FakeBatchGenerator:
        _partial = None

        def close(self):
            pass

    scheduler.batch_generator = _FakeBatchGenerator()
    scheduler.has_requests = lambda: len(step_threads) < 3

    def step():
        step_threads.append(threading.get_ident())
        if len(step_threads) == 3:
            scheduler._running = False

    scheduler.step = step
    monkeypatch.setattr(
        "vllm_mlx.mllm_scheduler.bind_generation_streams", lambda *a, **k: None
    )
    await asyncio.wait_for(scheduler._process_loop(), timeout=5)

    assert load_threads and step_threads
    assert set(load_threads) == set(step_threads), (
        f"MLLM loaded on {load_threads} but steps on {step_threads} — "
        "the model would carry a stream the stepping thread does not have"
    )
    assert load_threads == [
        threading.get_ident()
    ], "MLLM must load on the event loop, which is where MLLMScheduler steps"
    assert (
        engine._generation_executor is None
    ), "MLLM must not spin up a generation worker it never uses"


@pytest.mark.anyio
async def test_residency_manager_honours_the_per_path_load_thread():
    """ResidencyManager has its own start path and must not override this."""
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.lifecycle import ResidencyManager, ResidentModel

    async def load_thread_for(is_mllm: bool) -> int:
        engine = object.__new__(BatchedEngine)
        engine._generation_executor = None
        engine._is_mllm = is_mllm
        seen: list[int] = []
        engine.prepare_for_start = lambda: seen.append(threading.get_ident())

        manager = object.__new__(ResidencyManager)
        manager._lock = asyncio.Lock()
        resident = object.__new__(ResidentModel)
        resident._prepare_task = None
        await manager._prepare_engine_start(resident, engine)

        if engine._generation_executor is not None:
            engine._generation_executor.shutdown(wait=True)
        return seen[0]

    loop_thread = threading.get_ident()
    assert (
        await load_thread_for(is_mllm=True) == loop_thread
    ), "MLLM loaded off the event loop, where MLLMScheduler cannot step it"
    assert (
        await load_thread_for(is_mllm=False) != loop_thread
    ), "non-MLLM must load on the generation worker that EngineCore steps on"


@pytest.mark.anyio
async def test_supplied_worker_outlives_the_engine_loop(monkeypatch):
    """The caller's worker owns the loaded model; the loop must not close it.

    Shutting it down here would take the model's streams with it while
    BatchedEngine still holds the model, and the next start() would find a dead
    executor.
    """
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    core = _bare_engine_core(monkeypatch, worker=worker)

    await asyncio.wait_for(core._engine_loop(), timeout=5)

    assert worker.submit(lambda: "alive").result() == "alive"
    worker.shutdown(wait=True)


@pytest.mark.anyio
async def test_engine_core_stop_cleanup_stays_on_supplied_worker(monkeypatch):
    """Cancellation, its safety net, and repeated stop keep one owner."""
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    core = _bare_engine_core(monkeypatch, worker=worker)
    core._running = False
    core._task = None
    core._request_event = None
    close_threads: list[int] = []

    class StopScheduler(_FakeScheduler):
        def has_requests(self):
            return False

        def _close_batch_generator(self):
            close_threads.append(threading.get_ident())

    core.scheduler = StopScheduler(core)
    _run_executors_deterministically(monkeypatch)

    try:
        await core.start()
        await asyncio.sleep(0)
        loop_task = core._task

        await core.stop()
        await core.stop()

        owner_thread = worker.submit(threading.get_ident).result()
    finally:
        worker.shutdown(wait=True)

    assert loop_task is not None and loop_task.cancelled()
    assert close_threads == [owner_thread, owner_thread, owner_thread]


@pytest.mark.anyio
async def test_loop_still_cleans_up_a_worker_it_created(monkeypatch):
    """Without a supplied worker the loop owns its pool and must retire it."""
    core = _bare_engine_core(monkeypatch, worker=None)

    await asyncio.wait_for(core._engine_loop(), timeout=5)

    assert core.scheduler.step_threads, "the scheduler never stepped"
    assert threading.get_ident() not in core.scheduler.step_threads
    # The loop's own pool is local, so observe it through its thread instead:
    # a shut-down single-worker pool has let its thread exit.
    stepped_on = core.scheduler.step_threads[0]
    assert stepped_on not in {
        t.ident for t in threading.enumerate()
    }, "the loop created its own worker and left it running"


@pytest.mark.anyio
async def test_supplied_model_worker_is_not_replaced_by_stream_fallback(monkeypatch):
    """A caller-supplied worker owns the model and cannot be abandoned."""
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    core = _bare_engine_core(monkeypatch, worker=worker)

    class StreamErrorScheduler(_FakeScheduler):
        def step(self):
            self.step_threads.append(threading.get_ident())
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("There is no Stream(gpu, 0) in current thread")
            core._running = False
            return _SchedulerOutput()

        def _recover_from_cache_error(self):
            pass

        def _reschedule_running_requests(self):
            pass

    core.scheduler = StreamErrorScheduler(core)
    loop = asyncio.get_running_loop()
    executor_calls: list[str] = []

    def run_in_executor(executor, operation, *args):
        future = loop.create_future()
        executor_calls.append(getattr(operation, "__name__", type(operation).__name__))
        try:
            future.set_result(operation(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(loop, "run_in_executor", run_in_executor)
    try:
        await asyncio.wait_for(core._engine_loop(), timeout=2)
    finally:
        worker.shutdown(wait=True)

    assert executor_calls.count("_step_on_worker") == 2
    assert "_recover_stream_thread_error_on_worker" not in executor_calls


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "expected"),
    [("load_cache_from_disk", 1), ("save_cache_to_disk", True)],
)
async def test_cache_io_uses_the_model_generation_worker(operation, expected):
    """Persisted MLX cache state must stay on the model owner thread."""
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.engine_core import AsyncEngineCore, EngineCore

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._mllm_scheduler = None
    operation_threads: list[int] = []

    class FakeScheduler:
        def load_cache_from_disk(self, cache_dir):
            operation_threads.append(threading.get_ident())
            return 1

        def save_cache_to_disk(self, cache_dir):
            operation_threads.append(threading.get_ident())
            return True

    worker = engine._generation_worker()
    core = object.__new__(EngineCore)
    core.scheduler = FakeScheduler()
    core._external_generation_worker = worker
    async_engine = object.__new__(AsyncEngineCore)
    async_engine.engine = core
    engine._engine = async_engine

    try:
        result = await getattr(engine, operation)("cache")
        owner_threads = {thread.ident for thread in worker._threads}
    finally:
        worker.shutdown(wait=True)
        engine._generation_executor = None

    assert result == expected
    assert operation_threads == list(owner_threads)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "expected"),
    [("load_cache_from_disk", 1), ("save_cache_to_disk", True)],
)
async def test_mllm_cache_io_stays_on_the_model_event_loop(operation, expected):
    """MLLM cache state follows MLLMScheduler instead of the text worker."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._engine = None
    operation_threads: list[int] = []

    class FakePrefixCache:
        def load_from_disk(self, cache_dir):
            operation_threads.append(threading.get_ident())
            return 1

        def save_to_disk(self, cache_dir):
            operation_threads.append(threading.get_ident())
            return True

    prefix_cache = FakePrefixCache()
    engine._mllm_scheduler = SimpleNamespace(
        batch_generator=SimpleNamespace(prefix_cache=prefix_cache),
        _ensure_batch_generator=lambda: None,
    )

    result = await getattr(engine, operation)("cache")

    assert result == expected
    assert operation_threads == [threading.get_ident()]
    assert engine._generation_executor is None


@pytest.mark.anyio
async def test_stop_retires_the_generation_worker():
    """The model and its streams live on that thread; both go with it."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._engine = None
    engine._scheduler = None
    engine._mllm_scheduler = None
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = None
    engine._mllm_instance = None
    engine._loaded = True

    worker = engine._generation_worker()
    await engine.stop()

    assert engine._generation_executor is None
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        worker.submit(lambda: None)


@pytest.mark.anyio
async def test_stop_closes_core_on_generation_worker_and_is_repeatable(monkeypatch):
    """Deep reset runs on the model owner before that worker is retired."""
    from vllm_mlx.engine.batched import BatchedEngine

    close_threads: list[int] = []

    class FakeCore:
        def close(self):
            close_threads.append(threading.get_ident())

    class FakeAsyncEngine:
        engine = FakeCore()

        async def stop(self):
            pass

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._engine = FakeAsyncEngine()
    engine._mllm_scheduler = None
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = None
    engine._mllm_instance = None
    engine._loaded = True
    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result()
    _run_executors_deterministically(monkeypatch)

    await engine.stop()
    await engine.stop()

    assert close_threads == [owner_thread]
    assert engine._generation_executor is None


@pytest.mark.anyio
async def test_stop_retires_generation_worker_when_engine_cleanup_fails(monkeypatch):
    """A cleanup error must not leave the model-owning executor alive."""
    from vllm_mlx.engine.batched import BatchedEngine

    close_threads: list[int] = []

    class FakeCore:
        def close(self):
            close_threads.append(threading.get_ident())

    class FailingAsyncEngine:
        engine = FakeCore()

        async def stop(self):
            raise RuntimeError("engine cleanup failed")

    engine = object.__new__(BatchedEngine)
    engine._generation_executor = None
    engine._engine = FailingAsyncEngine()
    engine._mllm_scheduler = None
    engine._model = object()
    engine._tokenizer = object()
    engine._processor = None
    engine._mllm_instance = None
    engine._loaded = True
    worker = engine._generation_worker()
    owner_thread = worker.submit(threading.get_ident).result()
    _run_executors_deterministically(monkeypatch)

    with pytest.raises(RuntimeError, match="engine cleanup failed"):
        await engine.stop()

    assert close_threads == [owner_thread]
    assert engine._generation_executor is None
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        worker.submit(lambda: None)


def test_async_engine_core_passes_the_worker_through():
    """AsyncEngineCore is what BatchedEngine actually constructs."""
    from vllm_mlx.engine_core import AsyncEngineCore

    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    try:
        async_core = AsyncEngineCore(
            model=object(), tokenizer=object(), generation_worker=worker
        )
        assert async_core.engine._external_generation_worker is worker
    finally:
        worker.shutdown(wait=True)
