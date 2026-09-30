# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #759.

``MLLMScheduler.clear_runtime_caches()`` / ``.reset()`` used to reference a
nonexistent ``self.vision_cache`` instead of the real generator-owned
``self.batch_generator.vision_cache``, raising ``AttributeError`` (caught and
masked as a soft error by the server). These tests build a bare scheduler via
``__new__`` (no model/processor needed) with a fake batch generator, so they
run without any model weights.
"""

from collections import deque

from vllm_mlx.mllm_scheduler import MLLMScheduler


class FakeCache:
    def __init__(self):
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class FakeBatchGenerator:
    def __init__(self, vision_cache, prefix_cache, call_order=None):
        self.vision_cache = vision_cache
        self.prefix_cache = prefix_cache
        self._call_order = call_order if call_order is not None else []
        self.closed = False

    def close(self) -> None:
        self._call_order.append("close")
        self.closed = True


def _bare_scheduler() -> MLLMScheduler:
    """Build an MLLMScheduler without running __init__ (no model needed)."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.running = {}
    scheduler.finished_req_ids = set()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler._detokenizer_pool = {}
    scheduler.batch_generator = None
    return scheduler


def test_clear_runtime_caches_clears_both_vision_and_prefix_caches():
    scheduler = _bare_scheduler()
    vision_cache = FakeCache()
    prefix_cache = FakeCache()
    scheduler.batch_generator = FakeBatchGenerator(vision_cache, prefix_cache)

    cleared = scheduler.clear_runtime_caches()

    assert cleared == {"vision_cache": True, "prefix_cache": True}
    assert vision_cache.clear_calls == 1
    assert prefix_cache.clear_calls == 1


def test_reset_clears_vision_cache_before_generator_close():
    scheduler = _bare_scheduler()
    call_order: list = []
    vision_cache = FakeCache()

    real_clear = vision_cache.clear

    def tracked_clear():
        call_order.append("vision_clear")
        real_clear()

    vision_cache.clear = tracked_clear
    prefix_cache = FakeCache()
    generator = FakeBatchGenerator(vision_cache, prefix_cache, call_order=call_order)
    scheduler.batch_generator = generator

    # Must not raise AttributeError on a nonexistent scheduler.vision_cache.
    scheduler.reset()

    assert vision_cache.clear_calls == 1
    assert call_order == ["vision_clear", "close"]
    assert scheduler.batch_generator is None
