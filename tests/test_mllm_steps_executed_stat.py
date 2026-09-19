# SPDX-License-Identifier: Apache-2.0
"""Deterministic coverage for #746: vllm_mlx_engine_steps_executed is never
populated for MLLM-routed models. Also covers two PR #749 review threads:
a step that raises must not be counted (mirrors AsyncEngineCore's
increment-after-success placement, engine_core.py), and the
scheduler->engine->gauge chain must have CI coverage on Linux, not just
Apple Silicon.

The original #746 gap was three-layered:
- ``MLLMScheduler.get_stats()`` never produced a ``steps_executed`` key.
- ``BatchedEngine.get_stats()``'s MLLM promotion allowlist didn't forward
  that key to the top-level stats dict ``metrics.py`` reads even if it did.
- ``metrics.py``'s ``_update_engine_gauges`` already read it correctly once
  produced -- no change needed there, but it still needed a test.

TestMLLMSchedulerStepsExecuted and TestBatchedEngineStepsExecutedPromotion
lock in the first two with fakes -- no real model/generation -- mirroring
the fake/fixture style used in test_mllm_continuous_batching.py.
``vllm_mlx.mllm_scheduler`` and ``vllm_mlx.engine.batched`` both hard-import
``mlx.core`` at module scope. Real MLX is used where available; where it
isn't (e.g. the Linux CI matrix), ``tests._mlx_stub.install_if_unavailable``
stubs it so this import still succeeds -- the counter/promotion logic under
test is pure Python bookkeeping either way, so it's meaningful coverage in
both cases. Kept in its own CI step, separate from files with their own
real ``except ImportError`` mlx-optional handling -- see _mlx_stub.py's
docstring for why that matters.

TestMetricsEngineStepsExecutedGauge locks in the third: it calls
``vllm_mlx.metrics.MetricsCollector`` directly, needing neither the mlx
stub above nor a real prometheus_client registry -- see its own docstring
for why it isn't in test_metrics.py instead.
"""

from tests import _mlx_stub

_mlx_stub.install_if_unavailable()

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

pytestmark = pytest.mark.skipif(
    not HAS_MLX, reason="mlx.core not importable (even as a stub)"
)


class TestMLLMSchedulerStepsExecuted:
    """MLLMScheduler.get_stats() must report a step counter that increments
    once per step(), mirroring AsyncEngineCore._steps_executed
    (engine_core.py) for the plain-LLM path.
    """

    def _make_scheduler(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        class FakeTokenizer:
            eos_token_id = None

            def encode(self, text):
                return [1, 2, 3]

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        class FakeModel:
            config = None

        scheduler = MLLMScheduler(model=FakeModel(), processor=FakeProcessor())
        # step() unconditionally calls _schedule_waiting(), which lazily
        # builds a real MLLMBatchGenerator from the (fake) model/processor.
        # The counter contract under test doesn't depend on that machinery,
        # so replace it with a no-op to keep this deterministic and fast.
        scheduler._schedule_waiting = lambda: []
        return scheduler

    def test_get_stats_reports_steps_executed(self):
        scheduler = self._make_scheduler()
        assert scheduler.get_stats()["steps_executed"] == 0

    def test_step_increments_steps_executed(self):
        scheduler = self._make_scheduler()

        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 1

        scheduler.step()
        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 3

    def test_step_that_raises_is_not_counted(self):
        """Regression for the PR #749 review: step() used to increment
        _steps_executed before any operation that can raise (schedule_waiting,
        the batch generator's forward pass, response processing), so a step
        that failed partway through -- and whose requests get terminated by
        _fail_requests_after_step_error rather than retried, since retrying a
        partially mutated batch is unsafe -- was still counted as executed.
        The counter must only advance on the successful-return path, mirroring
        AsyncEngineCore (engine_core.py), which increments only after
        self.scheduler.step() returns.
        """
        scheduler = self._make_scheduler()

        class ExplodingBatchGenerator:
            def process_pending_removals(self):
                pass

            def next(self):
                raise RuntimeError("simulated forward-pass failure")

        scheduler.batch_generator = ExplodingBatchGenerator()
        # Any truthy value satisfies step()'s `self.batch_generator is not
        # None and self.running` gate to reach next() -- get_stats() (which
        # would need a fully request-shaped fake here) is deliberately not
        # called in this test; the counter is checked directly below instead.
        scheduler.running = {"req-exploding": object()}

        with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
            scheduler.step()

        assert scheduler._steps_executed == 0

        # The counter isn't left in a broken state by the earlier exception --
        # a subsequent step that actually completes still counts normally.
        scheduler.batch_generator = None
        scheduler.running = {}
        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 1


class TestBatchedEngineStepsExecutedPromotion:
    """BatchedEngine.get_stats() must promote steps_executed from the MLLM
    scheduler's stats dict into the top-level stats dict metrics.py reads.
    """

    def _make_engine(self, mllm_stats):
        from vllm_mlx.engine.batched import BatchedEngine

        class FakeMLLMScheduler:
            def get_stats(self):
                return mllm_stats

        # get_stats() only touches a handful of plain attributes; bypass
        # BatchedEngine.__init__ (which loads a real model) and set just
        # those directly.
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = FakeMLLMScheduler()
        engine._engine = None
        engine._model_name = "fake-mllm-model"
        engine._created_at = 0.0
        engine._is_mllm = True
        engine._loaded = True
        engine._stream_interval = 1
        engine._mllm_draft_model = None
        return engine

    def test_promotes_steps_executed_to_top_level(self):
        engine = self._make_engine({"steps_executed": 42, "num_waiting": 0})

        stats = engine.get_stats()

        assert stats["steps_executed"] == 42

    def test_omits_steps_executed_when_scheduler_lacks_it(self):
        engine = self._make_engine({"num_waiting": 0})

        stats = engine.get_stats()

        assert "steps_executed" not in stats


class TestMetricsEngineStepsExecutedGauge:
    """get_stats()["steps_executed"] must reach the exported
    vllm_mlx_engine_steps_executed gauge (metrics.py:_update_engine_gauges).

    This was originally an assertion inside test_metrics.py's HTTP-layer
    test (a FastAPI TestClient hitting the real /metrics endpoint). Moved
    here per PR #749's review: that file's import chain needs uvicorn and
    prometheus-client in addition to mlx.core, and the Linux CI job installs
    none of the three -- mlx-stubbing them the same way test_mllm_scheduler
    tests are stubbed broke fixture setup with
    ``ModuleNotFoundError: No module named 'uvicorn'``.

    vllm_mlx.metrics has no mlx dependency at all, so this needs no stub.
    Calling MetricsCollector._update_engine_gauges() directly (instead of
    going through render_metrics()/the HTTP layer) also means it needs no
    real prometheus_client registry: a defaultdict(MagicMock) stands in for
    the ``_prom`` dict of real Counter/Gauge/Histogram objects
    _init_prometheus() would normally build, so every ``self._prom[key]``
    access _update_engine_gauges makes for gauges this test doesn't care
    about (engine_type, cache_type, metal_memory_bytes, ...) just resolves
    to a fresh, harmless mock -- only "engine_steps_executed" is inspected.
    """

    def test_steps_executed_reaches_the_gauge(self):
        from collections import defaultdict
        from unittest.mock import MagicMock

        from vllm_mlx.metrics import MetricsCollector

        class FakeEngine:
            def get_stats(self):
                return {"steps_executed": 7, "num_waiting": 0}

        collector = MetricsCollector.__new__(MetricsCollector)
        collector._prom = defaultdict(MagicMock)

        collector._update_engine_gauges(engine=FakeEngine(), mcp_manager=None)

        collector._prom["engine_steps_executed"].set.assert_called_once_with(7)
