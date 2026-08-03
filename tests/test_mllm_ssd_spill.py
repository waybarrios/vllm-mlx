# SPDX-License-Identifier: Apache-2.0
"""Tests that the SSD cold tier is wired onto the MLLM prefix cache.

`--ssd-cache-dir` used to be a silent no-op on the MLLM path (Qwen3.5 et al.)
because the SSD tier was only attached to the standard Scheduler's
MemoryAwarePrefixCache.  These tests assert the MLLM scheduler now builds an
SSDCacheTier and calls set_ssd_tier() on the generator's prefix cache when the
flag is set, and does NOT when it is unset.  Model-free: the batch generator,
sampler and SSD tier are all faked.
"""

import sys
import types
from types import SimpleNamespace

from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig


def _install_fakes(monkeypatch, prefix_cache_obj):
    """Patch the heavy collaborators _ensure_batch_generator pulls in."""
    import vllm_mlx.mllm_scheduler as sched_mod

    set_calls = []
    tier_instances = []

    class FakeGenerator:
        def __init__(self, *a, **kw):
            self.prefix_cache = prefix_cache_obj
            self.language_model = SimpleNamespace(mtp=None)

    class FakeTier:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.reconciled = False
            tier_instances.append(self)

        def start_writer(self):
            self.started = True

        def reconcile(self):
            self.reconciled = True

    class FakeTierConfig:
        def __init__(self, cache_dir=None, max_size_gb=10.0):
            self.cache_dir = cache_dir
            self.max_size_gb = max_size_gb

    if prefix_cache_obj is not None:
        monkeypatch.setattr(
            prefix_cache_obj,
            "set_ssd_tier",
            lambda tier: set_calls.append(tier),
            raising=False,
        )

    monkeypatch.setattr(sched_mod, "MLLMBatchGenerator", FakeGenerator)

    # make_sampler and MemoryCacheConfig are imported lazily inside the method.
    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.make_sampler = lambda **kw: (lambda x: x)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    fake_ssd = types.ModuleType("vllm_mlx.ssd_cache")
    fake_ssd.SSDCacheTier = FakeTier
    fake_ssd.SSDCacheConfig = FakeTierConfig
    monkeypatch.setitem(sys.modules, "vllm_mlx.ssd_cache", fake_ssd)

    return set_calls, tier_instances


def _bare_scheduler(config):
    """Construct an MLLMScheduler without running its heavy __init__."""
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.config = config
    sched.model = SimpleNamespace()
    sched.processor = SimpleNamespace()
    sched.mm_processor = SimpleNamespace()
    sched.stop_tokens = set()
    sched.batch_generator = None
    return sched


def test_ssd_tier_attached_when_dir_set(monkeypatch, tmp_path):
    prefix_cache = SimpleNamespace()
    set_calls, tiers = _install_fakes(monkeypatch, prefix_cache)

    cfg = MLLMSchedulerConfig(ssd_cache_dir=str(tmp_path), ssd_cache_max_gb=7.0)
    sched = _bare_scheduler(cfg)
    sched._ensure_batch_generator()

    assert len(tiers) == 1
    assert tiers[0].started and tiers[0].reconciled
    assert tiers[0].config.cache_dir == str(tmp_path)
    assert tiers[0].config.max_size_gb == 7.0
    assert len(set_calls) == 1
    assert set_calls[0] is tiers[0]
    assert sched._ssd_tier is tiers[0]


def test_no_tier_when_dir_unset(monkeypatch):
    prefix_cache = SimpleNamespace()
    set_calls, tiers = _install_fakes(monkeypatch, prefix_cache)

    cfg = MLLMSchedulerConfig(ssd_cache_dir=None)
    sched = _bare_scheduler(cfg)
    sched._ensure_batch_generator()

    assert tiers == []
    assert set_calls == []
    assert sched._ssd_tier is None


def test_no_tier_when_prefix_cache_absent(monkeypatch, tmp_path):
    # Prefix caching disabled → generator.prefix_cache is None → no SSD tier.
    set_calls, tiers = _install_fakes(monkeypatch, None)

    cfg = MLLMSchedulerConfig(ssd_cache_dir=str(tmp_path))
    sched = _bare_scheduler(cfg)
    sched._ensure_batch_generator()

    assert tiers == []
    assert sched._ssd_tier is None
