# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for MLX stream/thread ownership in engine loops."""

import asyncio
import threading
from types import SimpleNamespace

import pytest


class _SchedulerOutput:
    outputs = []
    finished_request_ids = []


@pytest.mark.anyio
async def test_engine_core_runs_all_scheduler_steps_on_one_worker_thread(monkeypatch):
    """Continuous batching must not bounce MLX steps across threads."""
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=0, stream_interval=1)
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}

    main_thread = threading.get_ident()
    bind_threads: list[int] = []

    class FakeScheduler:
        batch_generator = SimpleNamespace(_partial=None)

        def __init__(self):
            self.calls = 0
            self.step_threads: list[int] = []
            self.close_threads: list[int] = []

        def has_requests(self):
            return self.calls < 3

        def step(self):
            self.step_threads.append(threading.get_ident())
            self.calls += 1
            if self.calls == 3:
                engine._running = False
            return _SchedulerOutput()

        def _close_batch_generator(self):
            self.close_threads.append(threading.get_ident())

    scheduler = FakeScheduler()
    engine.scheduler = scheduler

    def bind_streams():
        bind_threads.append(threading.get_ident())

    monkeypatch.setattr("vllm_mlx.engine_core.bind_generation_streams", bind_streams)

    await asyncio.wait_for(engine._engine_loop(), timeout=2)

    assert scheduler.step_threads
    assert len(set(scheduler.step_threads)) == 1
    assert scheduler.step_threads[0] != main_thread
    assert bind_threads == [scheduler.step_threads[0]]
    assert scheduler.close_threads == [scheduler.step_threads[0]]


@pytest.mark.anyio
async def test_mllm_scheduler_runs_steps_on_model_load_thread(monkeypatch):
    """MLLM keeps generation on the event-loop thread that loaded the model."""
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    scheduler = object.__new__(MLLMScheduler)
    scheduler._running = True

    main_thread = threading.get_ident()
    bind_threads: list[int] = []
    step_threads: list[int] = []
    close_threads: list[int] = []

    class FakeBatchGenerator:
        _partial = None

        def close(self):
            close_threads.append(threading.get_ident())

    scheduler.batch_generator = FakeBatchGenerator()

    def has_requests():
        return len(step_threads) < 3

    def step():
        step_threads.append(threading.get_ident())
        if len(step_threads) == 3:
            scheduler._running = False

    def bind_streams():
        bind_threads.append(threading.get_ident())

    scheduler.has_requests = has_requests
    scheduler.step = step
    monkeypatch.setattr("vllm_mlx.mllm_scheduler.bind_generation_streams", bind_streams)

    await asyncio.wait_for(scheduler._process_loop(), timeout=2)

    assert step_threads
    assert len(set(step_threads)) == 1
    assert step_threads[0] == main_thread
    assert bind_threads == [main_thread]
    assert close_threads == []
