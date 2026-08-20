# SPDX-License-Identifier: Apache-2.0
"""Tests for registry-backed multi-model serving."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

import pytest

from vllm_mlx.engine.base import BaseEngine, GenerationOutput
from vllm_mlx.model_registry import (
    ContentionPolicy,
    ModelManager,
    RegisteredModel,
    RegistryManagerConfig,
    RegistryServeDefaults,
    ResolvedModelConfig,
    build_memory_budget_report,
    load_registry_config,
    log_memory_budget_report,
)
from vllm_mlx.utils.download import DownloadConfig


class FakeEngine(BaseEngine):
    """Small test double for model lifecycle behaviour."""

    def __init__(
        self, config: ResolvedModelConfig, start_gate: asyncio.Event | None = None
    ):
        self._config = config
        self._start_gate = start_gate
        self.started = 0
        self.stopped = 0

    @property
    def model_name(self) -> str:
        return self._config.resolved_source

    @property
    def is_mllm(self) -> bool:
        return False

    @property
    def tokenizer(self) -> Any:
        return None

    async def start(self) -> None:
        if self._start_gate is not None:
            await self._start_gate.wait()
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def generate(self, *args, **kwargs) -> GenerationOutput:
        return GenerationOutput(text="ok")

    async def stream_generate(self, *args, **kwargs):
        yield GenerationOutput(text="ok", new_text="ok", finished=True)

    async def chat(self, *args, **kwargs) -> GenerationOutput:
        return GenerationOutput(text="ok")

    async def stream_chat(self, *args, **kwargs):
        yield GenerationOutput(text="ok", new_text="ok", finished=True)


def _defaults() -> RegistryServeDefaults:
    return RegistryServeDefaults(
        continuous_batching=False,
        force_mllm=False,
        enable_mtp=False,
        prefill_step_size=2048,
        specprefill_enabled=False,
        specprefill_threshold=8192,
        specprefill_keep_pct=0.3,
        specprefill_backbone_pct=0.0,
        specprefill_draft_model=None,
        prefix_trie_cache=False,
        prefix_trie_cache_size=32,
        prefix_trie_cache_memory_mb=None,
        stream_interval=1,
        gpu_memory_utilization=0.9,
        scheduler_config=None,
        max_tokens=32768,
        download_config=DownloadConfig(),
    )


def _manager_config(
    *,
    budget_gb: float,
    strategy: str = "wait_then_fail",
    wait_timeout_s: float | None = 1.0,
    preempt_after_s: float | None = None,
) -> RegistryManagerConfig:
    return RegistryManagerConfig(
        memory_budget_bytes=int(budget_gb * (1024**3)),
        policy=ContentionPolicy(
            strategy=strategy,
            wait_timeout_s=wait_timeout_s,
            preempt_after_s=preempt_after_s,
        ),
    )


def _registry(tmp_path: Path, sizes_gb: dict[str, float]) -> dict[str, RegisteredModel]:
    registry = {}
    for name, size_gb in sizes_gb.items():
        source = tmp_path / name
        source.mkdir()
        registry[name] = RegisteredModel(
            name=name,
            source=str(source),
            estimated_memory_bytes=int(size_gb * (1024**3)),
        )
    return registry


def _write_registry_config(tmp_path: Path, manager_yaml: str) -> Path:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(f"""
manager:
{manager_yaml}
models:
  - name: test
    path: /tmp/test-model
""".strip())
    return config_path


def test_cli_memory_budget_overrides_yaml_value(tmp_path):
    config_path = _write_registry_config(tmp_path, "  memory_budget_gb: 4")

    manager, _ = load_registry_config(
        config_path,
        _defaults(),
        memory_budget_gb=7.5,
    )

    assert manager.memory_budget_bytes == int(7.5 * (1024**3))


def test_omitting_cli_memory_budget_preserves_yaml_value(tmp_path):
    config_path = _write_registry_config(tmp_path, "  memory_budget: 2048mb")

    manager, _ = load_registry_config(config_path, _defaults())

    assert manager.memory_budget_bytes == 2048 * (1024**2)


def test_cli_memory_budget_works_without_yaml_manager_budget(tmp_path):
    config_path = _write_registry_config(tmp_path, "  contention_policy: {}")

    manager, _ = load_registry_config(
        config_path,
        _defaults(),
        memory_budget_gb=3.25,
    )

    assert manager.memory_budget_bytes == int(3.25 * (1024**3))


def test_cli_memory_budget_takes_precedence_over_invalid_yaml_value(tmp_path):
    config_path = _write_registry_config(tmp_path, "  memory_budget_gb: invalid")

    manager, _ = load_registry_config(
        config_path,
        _defaults(),
        memory_budget_gb=2.5,
    )

    assert manager.memory_budget_bytes == int(2.5 * (1024**3))


@pytest.mark.parametrize("raw_value", [".nan", ".inf", "0", "-0.1"])
def test_registry_rejects_invalid_manager_memory_budget(tmp_path, raw_value):
    config_path = _write_registry_config(tmp_path, f"  memory_budget_gb: {raw_value}")

    with pytest.raises(ValueError, match="must be a positive finite number"):
        load_registry_config(config_path, _defaults())


@pytest.mark.parametrize("raw_value", [".nan", ".inf", "0", "-0.1", "1.1"])
def test_registry_rejects_invalid_per_entry_gpu_memory_utilization(tmp_path, raw_value):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(f"""
manager:
  memory_budget_gb: 8
models:
  - name: alpha
    path: /tmp/alpha
    gpu_memory_utilization: {raw_value}
""")

    with pytest.raises(
        ValueError,
        match=(
            r"models-config entry 'alpha' gpu_memory_utilization must be finite "
            r"and within \(0, 1\]"
        ),
    ):
        load_registry_config(config_path, _defaults())


def test_acquire_shares_single_inflight_load(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 4})
        gate = asyncio.Event()
        created: list[FakeEngine] = []

        def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
            engine = FakeEngine(config, start_gate=gate)
            created.append(engine)
            return engine

        manager = ModelManager(
            _manager_config(budget_gb=8),
            registry,
            _defaults(),
            engine_factory=engine_factory,
        )

        first = asyncio.create_task(manager.acquire("alpha"))
        await asyncio.sleep(0.05)  # give _resolve_source thread time to return
        second = asyncio.create_task(manager.acquire("alpha"))
        await asyncio.sleep(0)

        assert len(created) == 1
        gate.set()

        lease_a = await first
        lease_b = await second
        assert lease_a.engine is lease_b.engine
        assert created[0].started == 1

        await lease_a.release()
        await lease_b.release()

    asyncio.run(_run())


def test_idle_lru_eviction_preserves_budget(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 4, "beta": 4, "gamma": 5})
        created: dict[str, FakeEngine] = {}

        def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
            engine = FakeEngine(config)
            created[config.entry.name] = engine
            return engine

        manager = ModelManager(
            _manager_config(budget_gb=9),
            registry,
            _defaults(),
            engine_factory=engine_factory,
        )

        lease = await manager.acquire("alpha")
        await lease.release()
        await asyncio.sleep(0.01)

        lease = await manager.acquire("beta")
        await lease.release()
        await asyncio.sleep(0.01)

        lease = await manager.acquire("gamma")
        await lease.release()

        assert "alpha" not in manager._loaded
        assert "beta" in manager._loaded
        assert "gamma" in manager._loaded
        assert created["alpha"].stopped == 1
        assert created["beta"].stopped == 0

    asyncio.run(_run())


def test_preempt_policy_cancels_active_request_and_loads_waiting_model(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 8, "beta": 8})
        created: dict[str, FakeEngine] = {}

        def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
            engine = FakeEngine(config)
            created[config.entry.name] = engine
            return engine

        manager = ModelManager(
            _manager_config(
                budget_gb=10,
                strategy="preempt",
                wait_timeout_s=2.0,
            ),
            registry,
            _defaults(),
            engine_factory=engine_factory,
        )

        acquired = asyncio.Event()
        cancelled = asyncio.Event()

        async def hold_alpha() -> None:
            lease = await manager.acquire("alpha")
            acquired.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                await lease.release()

        active_task = asyncio.create_task(hold_alpha())
        await acquired.wait()

        beta_lease = await manager.acquire("beta")
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        await beta_lease.release()

        with pytest.raises(asyncio.CancelledError):
            await active_task

        assert "beta" in manager._loaded
        assert "alpha" not in manager._loaded
        assert created["alpha"].stopped == 1
        assert created["beta"].started == 1

    asyncio.run(_run())


def test_non_local_registry_entry_requires_explicit_memory_estimate():
    async def _run():
        registry = {
            "remote": RegisteredModel(
                name="remote",
                source="mlx-community/some-remote-model",
            )
        }
        manager = ModelManager(
            _manager_config(budget_gb=8),
            registry,
            _defaults(),
            engine_factory=lambda config: FakeEngine(config),
        )

        with pytest.raises(ValueError, match="estimated_memory_gb"):
            await manager.acquire("remote")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Memory budget vs Metal allocation ceiling reconciliation (issue #627)
# ---------------------------------------------------------------------------


GB = 1024**3


def _defaults_with(**overrides: Any) -> RegistryServeDefaults:
    base = _defaults()
    return dataclasses.replace(base, **overrides)


def test_budget_report_flags_budget_above_allocation_ceiling(tmp_path):
    """A weights budget larger than gpu_memory_utilization x device RAM is a conflict."""
    report = build_memory_budget_report(
        _manager_config(budget_gb=68),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )

    assert report.allocation_ceiling_bytes == 64 * GB
    assert report.exceeds_ceiling


def test_budget_report_accepts_budget_under_ceiling(tmp_path):
    report = build_memory_budget_report(
        _manager_config(budget_gb=60),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )

    # Guard against passing vacuously: a ceiling must actually exist here.
    assert report.allocation_ceiling_bytes == 64 * GB
    assert not report.exceeds_ceiling


def test_cache_limit_is_reported_per_engine_not_subtracted_once(tmp_path):
    """--cache-memory-mb is a per-engine maximum, so it never moves the fit-check.

    It is cloned into each resident continuous-batching engine, so it is neither
    a single process-wide reservation nor a subtractable bound.
    """
    from vllm_mlx.scheduler import SchedulerConfig

    registry = _registry(tmp_path, {"alpha": 32})
    ceiling_gb = 64  # 0.5 x 128 GB

    without_cache = build_memory_budget_report(
        _manager_config(budget_gb=ceiling_gb - 1),
        registry,
        _defaults_with(
            gpu_memory_utilization=0.5,
            continuous_batching=True,
            scheduler_config=SchedulerConfig(),
        ),
        device_working_set_bytes=128 * GB,
    )
    assert without_cache.per_engine_cache_limit_bytes is None
    assert without_cache.per_engine_cache_percent == pytest.approx(0.20)
    assert not without_cache.exceeds_ceiling

    with_cache = build_memory_budget_report(
        _manager_config(budget_gb=ceiling_gb - 1),
        registry,
        _defaults_with(
            gpu_memory_utilization=0.5,
            continuous_batching=True,
            scheduler_config=SchedulerConfig(cache_memory_mb=20480),
        ),
        device_working_set_bytes=128 * GB,
    )
    assert with_cache.per_engine_cache_limit_bytes == 20 * GB
    # The budget still fits: the cache limit does not shrink the weight capacity.
    assert not with_cache.exceeds_ceiling
    assert not with_cache.cache_limit_exceeds_ceiling


def test_cache_limit_ignored_for_simple_mode_entries(tmp_path):
    """SimpleEngine never receives scheduler_config, so the cap does not apply."""
    from vllm_mlx.scheduler import SchedulerConfig

    report = build_memory_budget_report(
        _manager_config(budget_gb=10),
        _registry(tmp_path, {"alpha": 8}),
        _defaults_with(
            continuous_batching=False,
            scheduler_config=SchedulerConfig(cache_memory_mb=20480),
        ),
        device_working_set_bytes=128 * GB,
    )

    assert report.continuous_batching_entries == 0
    assert report.total_entries == 1
    assert report.per_engine_cache_limit_bytes is None
    assert report.per_engine_cache_percent is None


def test_cache_limit_ignored_when_paged_cache_supersedes_it(tmp_path):
    """cache_memory_mb only binds for the memory-aware prefix cache."""
    from vllm_mlx.scheduler import SchedulerConfig

    report = build_memory_budget_report(
        _manager_config(budget_gb=10),
        _registry(tmp_path, {"alpha": 8}),
        _defaults_with(
            continuous_batching=True,
            scheduler_config=SchedulerConfig(
                cache_memory_mb=20480, use_paged_cache=True
            ),
        ),
        device_working_set_bytes=128 * GB,
    )

    assert report.continuous_batching_entries == 1
    assert report.per_engine_cache_limit_bytes is None


def test_cache_limit_above_ceiling_warns_without_negative_numbers(tmp_path, caplog):
    """Regression: an 8 GiB ceiling with a 20 GiB cache cap must not report -12 GiB."""
    from vllm_mlx.scheduler import SchedulerConfig

    report = build_memory_budget_report(
        _manager_config(budget_gb=4),
        _registry(tmp_path, {"alpha": 2}),
        _defaults_with(
            gpu_memory_utilization=0.5,
            continuous_batching=True,
            scheduler_config=SchedulerConfig(cache_memory_mb=20480),
        ),
        device_working_set_bytes=16 * GB,
    )

    assert report.allocation_ceiling_bytes == 8 * GB
    assert report.cache_limit_exceeds_ceiling
    # The weights budget itself still fits under the ceiling.
    assert not report.exceeds_ceiling

    with caplog.at_level(logging.INFO, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(report)

    assert "is at or above the Metal allocation ceiling" in caplog.text
    # No negative quantity may ever reach the operator (the -12 GiB regression).
    assert not re.search(r"-\d", caplog.text)


def test_budget_report_uses_tightest_per_entry_utilization(tmp_path):
    """A per-model override lowers the process-wide ceiling once that model loads."""
    registry = _registry(tmp_path, {"alpha": 8, "beta": 8})
    registry["beta"] = dataclasses.replace(registry["beta"], gpu_memory_utilization=0.4)

    report = build_memory_budget_report(
        _manager_config(budget_gb=60),
        registry,
        _defaults_with(gpu_memory_utilization=0.9, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )

    assert report.gpu_memory_utilization == pytest.approx(0.4)
    assert "beta" in report.gpu_memory_utilization_source
    assert report.allocation_ceiling_bytes == int(0.4 * 128 * GB)
    assert report.exceeds_ceiling


def test_no_ceiling_attributed_when_no_entry_installs_one(tmp_path):
    """Only BatchedEngine calls mx.set_memory_limit, so an all-simple registry
    must not attribute a ceiling to --gpu-memory-utilization."""
    report = build_memory_budget_report(
        _manager_config(budget_gb=10_000),
        _registry(tmp_path, {"alpha": 8, "beta": 8}),
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=False),
        device_working_set_bytes=128 * GB,
    )

    assert report.continuous_batching_entries == 0
    assert report.gpu_memory_utilization is None
    assert report.gpu_memory_utilization_source is None
    assert report.allocation_ceiling_bytes is None
    # A wildly oversized budget must not warn against a ceiling nothing installs.
    assert not report.exceeds_ceiling


def test_simple_mode_entry_override_is_not_a_ceiling_candidate(tmp_path):
    """A lower override on a simple-mode entry is inert and must be ignored."""
    registry = _registry(tmp_path, {"batched": 8, "simple": 8})
    registry["simple"] = dataclasses.replace(
        registry["simple"], continuous_batching=False, gpu_memory_utilization=0.1
    )
    registry["batched"] = dataclasses.replace(
        registry["batched"], continuous_batching=True
    )

    report = build_memory_budget_report(
        _manager_config(budget_gb=10),
        registry,
        _defaults_with(gpu_memory_utilization=0.8, continuous_batching=False),
        device_working_set_bytes=128 * GB,
    )

    assert report.continuous_batching_entries == 1
    assert report.gpu_memory_utilization == pytest.approx(0.8)
    assert report.gpu_memory_utilization_source == "serve default"
    assert report.allocation_ceiling_bytes == int(0.8 * 128 * GB)


def test_serve_default_does_not_win_when_every_batched_entry_overrides(tmp_path):
    """The default only competes when some batched entry actually inherits it."""
    registry = _registry(tmp_path, {"alpha": 8, "beta": 8})
    registry["alpha"] = dataclasses.replace(
        registry["alpha"], gpu_memory_utilization=0.7
    )
    registry["beta"] = dataclasses.replace(registry["beta"], gpu_memory_utilization=0.9)

    report = build_memory_budget_report(
        _manager_config(budget_gb=10),
        registry,
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )

    # 0.5 is never installed by anything, so the tightest installed value wins.
    assert report.gpu_memory_utilization == pytest.approx(0.7)
    assert "alpha" in report.gpu_memory_utilization_source


def test_tied_utilization_is_attributed_to_the_serve_default(tmp_path):
    """On a tie the default is the broader cause, so name it rather than an entry."""
    registry = _registry(tmp_path, {"alpha": 8, "beta": 8})
    registry["alpha"] = dataclasses.replace(
        registry["alpha"], gpu_memory_utilization=0.5
    )

    report = build_memory_budget_report(
        _manager_config(budget_gb=10),
        registry,
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )

    assert report.gpu_memory_utilization == pytest.approx(0.5)
    assert report.gpu_memory_utilization_source == "serve default"


def test_log_says_why_when_no_entry_installs_a_ceiling(tmp_path, caplog):
    report = build_memory_budget_report(
        _manager_config(budget_gb=10_000),
        _registry(tmp_path, {"alpha": 8}),
        _defaults_with(continuous_batching=False),
        device_working_set_bytes=128 * GB,
    )

    with caplog.at_level(logging.INFO, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(report)

    assert "no continuous-batching entries" in caplog.text
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]


def test_budget_report_ignores_cache_limit_in_the_fit_check(tmp_path):
    """The deterministic check compares process-wide quantities only."""
    from vllm_mlx.scheduler import SchedulerConfig

    registry = _registry(tmp_path, {"alpha": 32})
    args = dict(gpu_memory_utilization=0.5, continuous_batching=True)

    bare = build_memory_budget_report(
        _manager_config(budget_gb=63),
        registry,
        _defaults_with(**args, scheduler_config=SchedulerConfig()),
        device_working_set_bytes=128 * GB,
    )
    capped = build_memory_budget_report(
        _manager_config(budget_gb=63),
        registry,
        _defaults_with(**args, scheduler_config=SchedulerConfig(cache_memory_mb=20480)),
        device_working_set_bytes=128 * GB,
    )

    assert bare.exceeds_ceiling == capped.exceeds_ceiling is False


def test_budget_report_is_inconclusive_without_device_memory(tmp_path, monkeypatch):
    """Non-Metal hosts get no false warning — the ceiling simply is not knowable."""
    monkeypatch.setattr(
        "vllm_mlx.model_registry._device_working_set_bytes", lambda: None
    )
    report = build_memory_budget_report(
        _manager_config(budget_gb=10_000),
        _registry(tmp_path, {"alpha": 8}),
        _defaults_with(continuous_batching=True),
    )

    # A utilization IS attributable here; only the device working set is unknown.
    assert report.gpu_memory_utilization is not None
    assert report.allocation_ceiling_bytes is None
    assert not report.exceeds_ceiling
    assert not report.cache_limit_exceeds_ceiling


def test_log_memory_budget_report_warns_only_on_conflict(tmp_path, caplog):
    over = build_memory_budget_report(
        _manager_config(budget_gb=68),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(over)
    assert "exceeds the Metal allocation ceiling" in caplog.text
    assert "64.0 GB" in caplog.text

    caplog.clear()
    under = build_memory_budget_report(
        _manager_config(budget_gb=40),
        _registry(tmp_path, {"beta": 32}),
        _defaults_with(gpu_memory_utilization=0.5, continuous_batching=True),
        device_working_set_bytes=128 * GB,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(under)
    assert caplog.text == ""


def test_log_memory_budget_report_reports_ceiling_and_cache(tmp_path, caplog):
    from vllm_mlx.scheduler import SchedulerConfig

    report = build_memory_budget_report(
        _manager_config(budget_gb=40),
        _registry(tmp_path, {"alpha": 32, "beta": 4}),
        _defaults_with(
            gpu_memory_utilization=0.5,
            continuous_batching=True,
            scheduler_config=SchedulerConfig(cache_memory_mb=20480),
        ),
        device_working_set_bytes=128 * GB,
    )
    with caplog.at_level(logging.INFO, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(report)

    assert "Registry memory budget: 40.0 GB" in caplog.text
    assert "Metal allocation ceiling 64.0 GB" in caplog.text
    assert "20.0 GB per continuous-batching engine" in caplog.text
    assert "2 of 2 entries" in caplog.text


def test_log_memory_budget_report_says_so_when_ceiling_unknown(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(
        "vllm_mlx.model_registry._device_working_set_bytes", lambda: None
    )
    report = build_memory_budget_report(
        _manager_config(budget_gb=68),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(continuous_batching=True),
    )

    with caplog.at_level(logging.INFO, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(report)

    assert "no Metal allocation ceiling to reconcile it with" in caplog.text
    assert "MLX cannot report a device working set" in caplog.text
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
