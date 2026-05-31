# SPDX-License-Identifier: Apache-2.0
"""Regression tests for EngineCore idle polling behavior."""

import pytest


class _IdleScheduler:
    def has_requests(self):
        return False

    def _close_batch_generator(self):
        pass


class _RequestCapturingScheduler:
    def __init__(self):
        self.requests = []

    def add_request(self, request):
        self.requests.append(request)


@pytest.mark.anyio
async def test_engine_loop_uses_idle_interval_when_scheduler_is_empty(monkeypatch):
    """An empty scheduler should not keep polling at the active 1ms interval."""
    import vllm_mlx.engine_core as engine_core
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=0.001, idle_step_interval=0.25)
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
async def test_add_request_wakes_idle_engine_loop():
    """Adding work should wake the idle loop instead of waiting for timeout."""
    import asyncio

    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig()
    engine.scheduler = _RequestCapturingScheduler()
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._request_event = asyncio.Event()

    request_id = await engine.add_request("hello", request_id="req-1")

    assert request_id == "req-1"
    assert len(engine.scheduler.requests) == 1
    assert engine._request_event.is_set()
