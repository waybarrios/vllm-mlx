# SPDX-License-Identifier: Apache-2.0
"""Test that EngineCore watchdog aborts stranded requests when batch generator desyncs."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vllm_mlx.engine_core import EngineConfig, EngineCore
from vllm_mlx.mlx_executor import MLXExecutor
from vllm_mlx.output_collector import RequestOutputCollector, RequestStreamState
from vllm_mlx.request import RequestOutput


class _EmptySchedulerOutput:
    outputs = []
    finished_request_ids = []


@pytest.mark.anyio
async def test_watchdog_aborts_stranded_requests_after_max_empty_steps(monkeypatch):
    """If running requests exist but step() produces 0 outputs for 1000 steps,

    watchdog must abort them and notify events instead of spinning infinitely.
    """
    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=0, stream_interval=1)
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {"req-1": RequestOutputCollector(aggregate=True)}
    engine._stream_states = {"req-1": RequestStreamState(stream_interval=1)}
    finished_event = asyncio.Event()
    engine._finished_events = {"req-1": finished_event}

    aborted_calls = []

    class FakeScheduler:
        running = {"req-1": object()}
        calls = 0

        def has_requests(self):
            # Keep returning True until aborted
            return bool(self.running)

        def step(self):
            self.calls += 1
            return _EmptySchedulerOutput()

        def _recover_from_generation_error(self):
            aborted_calls.append("req-1")
            self.running.clear()
            return {"req-1"}

        def _close_batch_generator(self):
            pass

        def close_ssd_tier(self):
            pass

    engine.scheduler = FakeScheduler()
    engine._mlx_executor = MLXExecutor(inline=True)

    monkeypatch.setattr("vllm_mlx.engine_core.bind_generation_streams", lambda *a, **k: None)

    # Run the engine loop; it should hit 1000 empty steps, trigger watchdog,
    # abort req-1, and then has_requests() becomes False, which we can then stop
    async def stop_after_watchdog():
        await finished_event.wait()
        engine._running = False

    stop_task = asyncio.create_task(stop_after_watchdog())
    loop_task = asyncio.create_task(engine._engine_loop())

    await asyncio.wait_for(asyncio.gather(stop_task, loop_task), timeout=5)

    assert len(aborted_calls) == 1
    assert "req-1" in aborted_calls
    collector = engine._output_collectors["req-1"]
    out = collector.get_nowait()
    assert out is not None
    assert out.finished is True
    assert out.finish_reason == "error"
