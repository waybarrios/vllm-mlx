# SPDX-License-Identifier: Apache-2.0
"""SimpleEngine must run model load and every generation route on one thread.

MLX streams exist only in the thread that created them, and an array with
pending primitives carries the stream those primitives were built on. So a
model loaded on one thread cannot be driven from another: generation dies in
``mx.eval`` with "There is no Stream(gpu, N) in current thread".

These tests record the thread of each call rather than touching MLX, so they
run anywhere. Fakes follow the convention in ``test_gemma4_mllm_patch.py``.
"""

import asyncio
import sys
import threading
import time
import types
from typing import Any

import pytest


def _install_mlx_stubs() -> None:
    """Stub mlx.core only when the real one is missing.

    The stub must carry a real __spec__: transformers probes packages with
    importlib.util.find_spec, which raises ValueError on a spec-less module and
    would break every other test sharing this pytest process.
    """
    try:  # pragma: no cover - depends on the environment
        import mlx.core  # noqa: F401

        return
    except Exception:
        pass

    import importlib.machinery

    core = types.ModuleType("mlx.core")
    core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)

    class _Stream:
        def __init__(self, idx: int = 0) -> None:
            self.idx = idx

    class _Array:
        """Must be a class, not a factory function.

        The engine annotates parameters as ``mx.array | None``, and PEP 604
        unions are evaluated at definition time, so a plain function here makes
        importing the module raise "unsupported operand type(s) for |".
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    core.Stream = _Stream
    core.array = _Array
    core.default_device = lambda: "gpu"
    core.new_stream = lambda *a, **k: _Stream()
    core.set_default_stream = lambda *a, **k: None
    core.clear_cache = lambda: None
    core.metal = types.SimpleNamespace(
        get_active_memory=lambda: 0,
        get_peak_memory=lambda: 0,
        get_cache_memory=lambda: 0,
    )
    mlx = types.ModuleType("mlx")
    mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None, is_package=True)
    mlx.__path__ = []
    mlx.core = core
    sys.modules.setdefault("mlx", mlx)
    sys.modules["mlx.core"] = core


_install_mlx_stubs()


class _Chunk:
    def __init__(self, text: str, finish_reason: str | None = None) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.prompt_tokens = 1
        self.finished = finish_reason is not None


class _ThreadRecordingModel:
    """Records the thread each MLX-touching call runs on."""

    def __init__(
        self,
        threads: dict[str, list[str]],
        *,
        chunks: int = 3,
        chunk_gate: threading.Event | None = None,
    ) -> None:
        self._threads = threads
        self._chunks = chunks
        self._chunk_gate = chunk_gate
        self.closed = threading.Event()
        self.tokenizer = types.SimpleNamespace(encode=lambda s: [0] * len(s.split()))
        self.model = object()

    def _record(self, key: str) -> None:
        # Name plus ident: a fresh ThreadPoolExecutor restarts its numbering, so
        # two different threads both answer to "simple-generate_0".
        current = threading.current_thread()
        self._threads.setdefault(key, []).append(f"{current.name}/{current.ident}")

    def load(self) -> None:
        self._record("load")

    def stream_generate(self, **kwargs: Any):
        # Recorded per chunk: the generator body runs where it is *pumped*,
        # not where it is constructed, which is exactly the bug.
        try:
            for i in range(self._chunks):
                if self._chunk_gate is not None:
                    # Stands in for a slow decode: one chunk blocks the worker
                    # until the test lets it go.
                    self._chunk_gate.wait(timeout=10.0)
                self._record("stream_generate")
                yield _Chunk(f"tok{i}", "stop" if i == self._chunks - 1 else None)
        finally:
            self._record("close")
            self.closed.set()

    def generate(self, **kwargs: Any) -> str:
        self._record("generate")
        return "done"


@pytest.fixture()
def engine_module():
    from vllm_mlx.engine import simple

    return simple


def _make_engine(engine_module, model=None):
    """Build a SimpleEngine with just enough state to drive its real routes.

    ``object.__new__`` keeps this off the model-loading path while still
    exercising the engine's own code rather than a copy of it in the test.
    """
    from contextlib import asynccontextmanager

    engine = object.__new__(engine_module.SimpleEngine)
    engine._generation_executor = None
    engine._generation_streams_bound = False
    engine._pre_bind_generation_streams = None
    engine._worker_generation_stream = None
    engine._stopping = False
    engine._draining_executors = []
    engine._generation_users = 0
    engine._generation_idle = asyncio.Event()
    engine._generation_idle.set()
    engine._generation_abort_hooks = {}
    engine._active_requests = {}
    engine._loaded = True
    engine._is_mllm = False
    engine._draft_model = None
    engine._model = model
    engine._text_model = None
    engine._text_tokenizer = None
    engine._system_kv_cache = {}
    engine._system_kv_cache_stats = {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
    }
    engine._supports_system_kv_cache = False
    engine._prefix_trie_cache = None
    engine._prefix_trie_cache_lock = threading.Lock()
    engine._prefix_trie_cache_stats = {
        "lookups": 0,
        "hits": 0,
        "misses": 0,
        "inserts": 0,
        "skips": 0,
        "tokens_saved": 0,
    }

    @asynccontextmanager
    async def _slot(request_id: str):
        yield

    engine._acquire_generation_slot = _slot
    return engine


def test_generation_worker_is_a_single_thread(engine_module):
    """The pinned executor must hand out one thread, and always the same one."""
    engine = _make_engine(engine_module)

    worker = engine._generation_worker()
    assert worker is engine._generation_worker(), "executor must be reused"

    names = set()
    for _ in range(5):
        names.add(worker.submit(lambda: threading.current_thread().name).result())
    assert len(names) == 1, f"generation spread over several threads: {names}"
    worker.shutdown(wait=True)


def test_load_and_generation_share_one_thread(engine_module):
    """Regression: load on thread A + generation on thread B is the bug.

    Before the fix, ``prepare_for_start`` ran on the event loop thread while
    the non-stream route hopped to ``asyncio.to_thread``, so the two never
    agreed and every request after a mode switch failed.

    The streaming half drives ``_stream_generate_impl`` itself. Re-implementing
    the pump in the test body would leave the engine's own pumping code — the
    ``_next_chunk`` closure, the ``_STREAM_DONE`` protocol, the close routing —
    untested on the no-MLX CI job, so reverting it to iterate on the loop thread
    would still go green.
    """
    threads: dict[str, list[str]] = {}
    model = _ThreadRecordingModel(threads)
    engine = _make_engine(engine_module, model)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        # load, as start() now does: on the generation worker
        await loop.run_in_executor(engine._generation_worker(), model.load)
        # non-stream route, through the engine's own serialized helper
        await engine._run_blocking_serialized(model.generate)
        # stream route, through the engine's own pump
        async for _ in engine._stream_generate_impl(prompt="hi", max_tokens=8):
            pass

    asyncio.run(scenario())
    engine._generation_executor.shutdown(wait=True)

    observed = {t for calls in threads.values() for t in calls}
    assert threads["load"], "load was never recorded"
    assert threads["generate"], "non-stream generation was never recorded"
    assert threads["stream_generate"], "stream generation was never recorded"
    assert threads["close"], "generator cleanup was never recorded"
    assert len(observed) == 1, (
        "load and generation must share one thread, saw: "
        f"{ {k: sorted(set(v)) for k, v in threads.items()} }"
    )
    assert not any(
        t.startswith("MainThread") for t in observed
    ), "MLX work must leave the event loop thread"


def test_stream_done_sentinel_is_distinct(engine_module):
    """StopIteration cannot cross a thread boundary, hence the sentinel."""
    sentinel = engine_module._STREAM_DONE
    assert sentinel is not None
    assert next(iter([]), sentinel) is sentinel
    assert next(iter([_Chunk("x")]), sentinel) is not sentinel


def test_stop_does_not_block_the_event_loop(engine_module):
    """stop() must stay bounded while the worker is inside a generation.

    ``shutdown(wait=True)`` on the loop thread froze health checks, timers and
    cancellation for as long as the request ran — up to ``max_tokens`` of decode
    with the server default of 32768.
    """
    threads: dict[str, list[str]] = {}
    gate = threading.Event()
    model = _ThreadRecordingModel(threads, chunks=64, chunk_gate=gate)
    engine = _make_engine(engine_module, model)

    async def scenario() -> None:
        stream = engine._stream_generate_impl(prompt="hi", max_tokens=64)

        # One chunk through, then the worker blocks on the gate mid-generation.
        gate.set()
        await stream.__anext__()
        gate.clear()
        pump = asyncio.ensure_future(_drain(stream))
        await asyncio.sleep(0)

        ticks = 0

        async def _heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        # asyncio.wait_for cannot catch this: a synchronous block holds the loop
        # thread, so wait_for's own timeout callback cannot fire either and it
        # returns successfully once the block finally clears. Measuring the
        # loop's own progress is the only thing that sees it.
        beat = asyncio.ensure_future(_heartbeat())
        started = time.monotonic()
        await engine.stop()
        elapsed = time.monotonic() - started
        beat.cancel()

        assert elapsed < 2.0, (
            f"stop() held on to the loop for {elapsed:.1f}s while the worker was "
            "mid-generation"
        )
        assert ticks > 0, "the event loop made no progress while stop() ran"

        gate.set()
        await asyncio.wait_for(pump, timeout=5.0)

    async def _drain(stream) -> list:
        return [chunk async for chunk in stream]

    engine.STOP_DRAIN_TIMEOUT_S = 0.2
    asyncio.run(scenario())


def test_pump_crossing_stop_ends_cleanly(engine_module):
    """A stream that outlives stop() must not leak the executor's RuntimeError.

    The pump captures the worker at entry, so a submit landing after
    ``shutdown()`` used to raise ``cannot schedule new futures after shutdown``
    in the middle of a client response.
    """
    threads: dict[str, list[str]] = {}
    model = _ThreadRecordingModel(threads, chunks=64)
    engine = _make_engine(engine_module, model)
    # Nothing is pulling the stream while stop() runs, so the drain cannot
    # resolve and has to time out. Keep that short here; the point of the test
    # is what happens after.
    engine.STOP_DRAIN_TIMEOUT_S = 0.2

    async def scenario() -> list:
        stream = engine._stream_generate_impl(prompt="hi", max_tokens=64)
        first = await stream.__anext__()
        assert first.new_text == "tok0"

        await engine.stop()

        return [chunk async for chunk in stream]

    rest = asyncio.run(scenario())

    assert rest, "the stream ended without a terminating chunk"
    assert rest[-1].finished is True
    assert rest[-1].finish_reason == "abort", (
        "a stream cut short by stop() must report abort, "
        f"saw {rest[-1].finish_reason!r}"
    )
    assert model.closed.is_set(), "the generator was never closed"
    assert threads["close"] == threads["stream_generate"][:1], (
        "cleanup must run on the thread that owns the generator, saw "
        f"{threads['close']} vs {threads['stream_generate'][:1]}"
    )


def test_submit_after_stop_raises_engine_stopped(engine_module):
    """A raw executor RuntimeError is not something callers can act on."""
    from vllm_mlx.engine.base import EngineStopped

    engine = _make_engine(engine_module)

    async def scenario() -> None:
        worker = engine._generation_worker()
        await engine.stop()
        with pytest.raises(EngineStopped):
            await engine._submit_to_generation_worker(worker, lambda: "late")

    asyncio.run(scenario())


def test_submit_to_a_dead_executor_is_translated(engine_module):
    """The executor's own RuntimeError must not reach callers verbatim.

    Distinct from the check above, which never reaches ``submit`` at all: here
    the engine still believes the worker is current, so the error comes out of
    ``ThreadPoolExecutor`` itself as "cannot schedule new futures after
    shutdown".
    """
    from vllm_mlx.engine.base import EngineStopped

    engine = _make_engine(engine_module)

    async def scenario() -> None:
        worker = engine._generation_worker()
        worker.shutdown(wait=True)
        assert engine._generation_worker_is_live(worker), "setup: still current"

        with pytest.raises(EngineStopped) as excinfo:
            await engine._submit_to_generation_worker(worker, lambda: "late")
        assert isinstance(excinfo.value.__cause__, RuntimeError)

        # Unrelated RuntimeErrors from the work itself must pass through.
        engine._generation_executor = None
        fresh = engine._generation_worker()

        def _boom():
            raise RuntimeError("model exploded")

        with pytest.raises(RuntimeError, match="model exploded"):
            await engine._submit_to_generation_worker(fresh, _boom)
        fresh.shutdown(wait=True)

    asyncio.run(scenario())


def test_lifecycle_prepare_runs_on_the_engine_generation_worker(engine_module):
    """ResidencyManager must load through the engine's pinned worker.

    The manager has its own start path, so an engine loaded through it would
    otherwise land on asyncio's default pool while generation stayed pinned —
    the same mismatch, reached by a different route.
    """
    from vllm_mlx.lifecycle import ResidencyManager, ResidentModel

    threads: dict[str, list[str]] = {}
    model = _ThreadRecordingModel(threads)
    engine = _make_engine(engine_module, model)
    engine.prepare_for_start = model.load

    async def scenario() -> None:
        manager = object.__new__(ResidencyManager)
        manager._lock = asyncio.Lock()
        resident = object.__new__(ResidentModel)
        resident._prepare_task = None

        await manager._prepare_engine_start(resident, engine)

    asyncio.run(scenario())

    # Probe the worker for its identity before retiring it.
    worker = engine._generation_worker()
    worker_thread = worker.submit(
        lambda: f"{threading.current_thread().name}/{threading.get_ident()}"
    ).result()
    worker.shutdown(wait=True)

    assert threads["load"], "prepare_for_start never ran"
    loaded_on = threads["load"][0]
    assert not loaded_on.startswith(
        "MainThread"
    ), f"model load stayed on the event loop thread: {loaded_on}"
    assert (
        loaded_on == worker_thread
    ), f"model load landed on {loaded_on}, generation runs on {worker_thread}"


def test_restart_after_stop_mid_stream_uses_a_fresh_worker(engine_module):
    """stop() then start() must hand generation a new, exclusively-owned thread.

    The detached worker may still be finishing a generation, and two threads
    inside Metal at once is what the pinning is there to prevent — so the new
    thread has to join the old one before it loads anything.
    """
    threads: dict[str, list[str]] = {}
    gate = threading.Event()
    slow = _ThreadRecordingModel(threads, chunks=64, chunk_gate=gate)
    engine = _make_engine(engine_module, slow)

    async def scenario() -> None:
        stream = engine._stream_generate_impl(prompt="hi", max_tokens=64)
        gate.set()
        await stream.__anext__()
        gate.clear()
        pump = asyncio.ensure_future(_drain(stream))
        await asyncio.sleep(0)

        engine.STOP_DRAIN_TIMEOUT_S = 0.2
        await asyncio.wait_for(engine.stop(), timeout=1.0)
        assert engine._draining_executors, "a busy worker must be parked, not lost"
        detached = engine._draining_executors[0]

        gate.set()
        await asyncio.wait_for(pump, timeout=5.0)

        # Restart: a new worker, whose first job is to wait the old one out.
        engine._stopping = False
        engine._loaded = True
        fresh = _ThreadRecordingModel(threads)
        engine._model = fresh
        worker = engine._generation_worker()
        assert worker is not detached, "stop() must not hand back the dead worker"
        assert not engine._draining_executors, "the parked worker must be claimed"

        async for _ in engine._stream_generate_impl(prompt="again", max_tokens=8):
            pass
        engine._generation_executor.shutdown(wait=True)

    async def _drain(stream) -> list:
        return [chunk async for chunk in stream]

    asyncio.run(scenario())

    names = sorted(set(threads["stream_generate"]))
    assert len(names) == 2, f"restart must not reuse the old thread, saw {names}"
    assert not any(t.startswith("MainThread") for t in names)


def test_new_worker_never_binds_the_retired_workers_stream(engine_module, monkeypatch):
    """A restarted engine must allocate its own stream, not inherit the old one.

    An MLX stream exists only in the thread that created it, so pointing a fresh
    worker at the retired worker's stream puts that thread's default on a stream
    it cannot enter, and every array it builds afterwards dies with "There is no
    Stream(gpu, N) in current thread".

    The engine is driven through its real stop/restart path; the binding helper
    is faked so the check runs on machines without MLX, matching the rest of
    this module.
    """
    owner_of: dict[int, int] = {}
    cross_thread: list[tuple[int, int | None, int]] = []
    issued = [0]

    def fake_bind(stream=None):
        me = threading.get_ident()
        if stream is not None:
            if owner_of.get(stream) != me:
                cross_thread.append((stream, owner_of.get(stream), me))
            return stream
        issued[0] += 1
        owner_of[issued[0]] = me
        return issued[0]

    monkeypatch.setattr(engine_module, "_bind_worker_generation_streams", fake_bind)

    threads: dict[str, list[str]] = {}
    engine = _make_engine(engine_module, _ThreadRecordingModel(threads))

    async def scenario() -> None:
        async for _ in engine._stream_generate_impl(prompt="hi", max_tokens=4):
            pass
        first_stream = engine._worker_generation_stream
        assert first_stream is not None, "the first generation must bind a stream"

        # A second route on the same worker re-points at the stream already
        # allocated there. Allocating per request would leak one stream per
        # request, which is how the "Stream(gpu, 32)" numbering got that high.
        async for _ in engine._stream_generate_impl(prompt="more", max_tokens=4):
            pass
        assert engine._worker_generation_stream == first_stream
        assert issued[0] == 1, f"one stream per worker, allocated {issued[0]}"

        engine.STOP_DRAIN_TIMEOUT_S = 0.2
        await asyncio.wait_for(engine.stop(), timeout=2.0)
        assert (
            engine._worker_generation_stream is None
        ), "the retired worker's stream must not survive stop()"

        engine._stopping = False
        engine._loaded = True
        engine._model = _ThreadRecordingModel(threads)
        async for _ in engine._stream_generate_impl(prompt="again", max_tokens=4):
            pass
        assert engine._worker_generation_stream != first_stream, (
            "the fresh worker must allocate its own stream, "
            f"but reused {first_stream}"
        )
        engine._generation_executor.shutdown(wait=True)

    asyncio.run(scenario())

    assert not cross_thread, (
        "a thread bound a stream created by another thread: "
        f"{cross_thread} (stream, owner_ident, binder_ident)"
    )
