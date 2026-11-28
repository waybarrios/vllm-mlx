# SPDX-License-Identifier: Apache-2.0
"""Clearing the prefix cache must reach the scheduler and free Metal memory.

Regression coverage for issue #793:

* ``AsyncEngineCore`` had no ``clear_prefix_cache``, so the ``hasattr`` check
  in ``BatchedEngine`` and ``DELETE /v1/cache/prefix`` silently skipped the
  clear while still answering ``{"status": "cleared"}``.
* Even when entries were dropped, their buffers stayed in MLX's retained
  buffer pool, so the process footprint never shrank.
"""

import pytest

from vllm_mlx import scheduler as scheduler_module
from vllm_mlx.engine_core import AsyncEngineCore, EngineCore
from vllm_mlx.scheduler import Scheduler


class _RecordingCache:
    def __init__(self, events, name):
        self._events = events
        self._name = name

    def clear(self):
        self._events.append(f"{self._name}.clear")


@pytest.fixture
def events(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        scheduler_module.mx, "clear_cache", lambda: recorded.append("mx.clear_cache")
    )
    return recorded


@pytest.mark.parametrize(
    "method",
    [
        "clear_prefix_cache",
        "clear_runtime_caches",
        "save_cache_to_disk",
        "load_cache_from_disk",
    ],
)
def test_async_engine_core_exposes_cache_methods_the_server_probes(method):
    """The server and BatchedEngine find these with hasattr; keep them real."""
    assert callable(getattr(AsyncEngineCore, method, None))
    assert callable(getattr(EngineCore, method, None))


def test_async_engine_core_clear_prefix_cache_delegates_to_engine_core():
    calls = []

    class FakeEngineCore:
        def clear_prefix_cache(self):
            calls.append("clear_prefix_cache")

    core = object.__new__(AsyncEngineCore)
    core.engine = FakeEngineCore()

    core.clear_prefix_cache()

    assert calls == ["clear_prefix_cache"]


def _bare_scheduler(memory_aware=None, prefix=None, block_aware=None):
    sched = object.__new__(Scheduler)
    sched.memory_aware_cache = memory_aware
    sched.prefix_cache = prefix
    sched.block_aware_cache = block_aware
    return sched


@pytest.mark.parametrize("which", ["memory_aware", "prefix"])
def test_clear_prefix_cache_releases_buffers_after_dropping_entries(events, which):
    sched = _bare_scheduler(**{which: _RecordingCache(events, which)})

    sched.clear_prefix_cache()

    assert events == [f"{which}.clear", "mx.clear_cache"]


def test_clear_runtime_caches_releases_buffers_once_after_all_clears(events):
    sched = _bare_scheduler(
        memory_aware=_RecordingCache(events, "memory_aware"),
        prefix=_RecordingCache(events, "prefix"),
        block_aware=_RecordingCache(events, "block_aware"),
    )

    cleared = sched.clear_runtime_caches()

    assert cleared == {
        "paged_cache": True,
        "memory_aware_cache": True,
        "prefix_cache": True,
    }
    assert events == [
        "block_aware.clear",
        "memory_aware.clear",
        "prefix.clear",
        "mx.clear_cache",
    ]


def test_clear_runtime_caches_without_caches_does_not_touch_allocator(events):
    sched = _bare_scheduler()

    cleared = sched.clear_runtime_caches()

    assert not any(cleared.values())
    assert events == []
