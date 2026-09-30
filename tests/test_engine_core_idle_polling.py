# SPDX-License-Identifier: Apache-2.0
"""Regression tests for EngineCore idle polling behavior."""

import asyncio
from types import SimpleNamespace

import pytest


class _IdleScheduler:
    def has_requests(self):
        return False

    def _close_batch_generator(self):
        pass

    def close_ssd_tier(self):
        pass


class _RequestCapturingScheduler:
    def __init__(self):
        self.requests = []

    def add_request(self, request):
        self.requests.append(request)


class _ParkingEvent(asyncio.Event):
    """Signals once the engine has reached its event-backed idle wait."""

    def __init__(self):
        super().__init__()
        self.wait_started = asyncio.Event()

    async def wait(self):
        self.wait_started.set()
        return await super().wait()


@pytest.mark.anyio
async def test_engine_loop_uses_idle_interval_when_scheduler_is_empty(monkeypatch):
    """An empty scheduler should use its configured idle wait."""
    import vllm_mlx.engine_core as engine_core
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=0.25)
    engine.scheduler = _IdleScheduler()
    engine._running = True
    engine._request_event = None
    engine._steps_executed = 0

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        engine._running = False

    monkeypatch.setattr(engine_core.asyncio, "sleep", fake_sleep)

    await engine._engine_loop()

    assert sleeps == [0.25]


@pytest.mark.anyio
async def test_engine_loop_wakes_promptly_for_request_added_while_parked(monkeypatch):
    """A request must wake the real long-timeout idle loop before it expires."""
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=10)
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._request_event = _ParkingEvent()
    engine._running = True
    engine._steps_executed = 0

    class _WakeableScheduler(_RequestCapturingScheduler):
        def __init__(self):
            super().__init__()
            self.step_started = asyncio.Event()

        def has_requests(self):
            return bool(self.requests)

        def step(self):
            self.step_started.set()
            engine._running = False
            return SimpleNamespace(outputs=[], finished_request_ids=[])

        def _close_batch_generator(self):
            pass

        def close_ssd_tier(self):
            pass

    engine.scheduler = _WakeableScheduler()
    monkeypatch.setattr("vllm_mlx.engine_core.bind_generation_streams", lambda: None)

    engine_task = asyncio.create_task(engine._engine_loop())
    await asyncio.wait_for(engine._request_event.wait_started.wait(), timeout=0.5)

    started = asyncio.get_running_loop().time()
    request_id = await engine.add_request("hello", request_id="req-1")
    await asyncio.wait_for(engine.scheduler.step_started.wait(), timeout=0.5)
    elapsed = asyncio.get_running_loop().time() - started
    await engine_task

    assert request_id == "req-1"
    assert len(engine.scheduler.requests) == 1
    assert elapsed < 0.5


def test_engine_config_keeps_existing_positional_argument_mapping():
    """Appending fields must not remap the established public constructor."""
    from vllm_mlx.engine_core import EngineConfig

    config = EngineConfig("model", None, 0.001, 4, 0.8)

    assert config.step_interval == 0.001
    assert config.stream_interval == 4
    assert config.gpu_memory_utilization == 0.8
