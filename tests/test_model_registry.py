# SPDX-License-Identifier: Apache-2.0
"""Tests for registry-backed multi-model serving."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
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
        _defaults_with(gpu_memory_utilization=0.5),
        device_working_set_bytes=128 * GB,
    )

    assert report.allocation_ceiling_bytes == 64 * GB
    assert report.weights_headroom_bytes == 64 * GB
    assert report.exceeds_ceiling


def test_budget_report_accepts_budget_under_ceiling(tmp_path):
    report = build_memory_budget_report(
        _manager_config(budget_gb=60),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(gpu_memory_utilization=0.5),
        device_working_set_bytes=128 * GB,
    )

    assert not report.exceeds_ceiling


def test_budget_report_subtracts_explicit_cache_reservation(tmp_path):
    """--cache-memory-mb is carved out of the ceiling before the weights fit-check."""
    from vllm_mlx.scheduler import SchedulerConfig

    registry = _registry(tmp_path, {"alpha": 32})
    ceiling_gb = 64  # 0.5 x 128 GB

    fits = build_memory_budget_report(
        _manager_config(budget_gb=ceiling_gb - 1),
        registry,
        _defaults_with(gpu_memory_utilization=0.5, scheduler_config=SchedulerConfig()),
        device_working_set_bytes=128 * GB,
    )
    assert fits.cache_reservation_bytes is None
    assert fits.cache_reservation_percent == pytest.approx(0.20)
    assert not fits.exceeds_ceiling

    reserved = build_memory_budget_report(
        _manager_config(budget_gb=ceiling_gb - 1),
        registry,
        _defaults_with(
            gpu_memory_utilization=0.5,
            scheduler_config=SchedulerConfig(cache_memory_mb=20480),
        ),
        device_working_set_bytes=128 * GB,
    )
    assert reserved.cache_reservation_bytes == 20 * GB
    assert reserved.weights_headroom_bytes == (ceiling_gb - 20) * GB
    assert reserved.exceeds_ceiling


def test_budget_report_uses_tightest_per_entry_utilization(tmp_path):
    """A per-model override lowers the process-wide ceiling once that model loads."""
    registry = _registry(tmp_path, {"alpha": 8, "beta": 8})
    registry["beta"] = dataclasses.replace(registry["beta"], gpu_memory_utilization=0.4)

    report = build_memory_budget_report(
        _manager_config(budget_gb=60),
        registry,
        _defaults_with(gpu_memory_utilization=0.9),
        device_working_set_bytes=128 * GB,
    )

    assert report.gpu_memory_utilization == pytest.approx(0.4)
    assert "beta" in report.gpu_memory_utilization_source
    assert report.allocation_ceiling_bytes == int(0.4 * 128 * GB)
    assert report.exceeds_ceiling


def test_budget_report_is_inconclusive_without_device_memory(tmp_path, monkeypatch):
    """Non-Metal hosts get no false warning — the ceiling simply is not knowable."""
    monkeypatch.setattr(
        "vllm_mlx.model_registry._device_working_set_bytes", lambda: None
    )
    report = build_memory_budget_report(
        _manager_config(budget_gb=10_000),
        _registry(tmp_path, {"alpha": 8}),
        _defaults(),
    )

    assert report.allocation_ceiling_bytes is None
    assert report.weights_headroom_bytes is None
    assert not report.exceeds_ceiling


def test_log_memory_budget_report_warns_only_on_conflict(tmp_path, caplog):
    over = build_memory_budget_report(
        _manager_config(budget_gb=68),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(gpu_memory_utilization=0.5),
        device_working_set_bytes=128 * GB,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(over)
    assert "exceeds the memory actually allocatable" in caplog.text
    assert "64.0 GB" in caplog.text

    caplog.clear()
    under = build_memory_budget_report(
        _manager_config(budget_gb=40),
        _registry(tmp_path, {"beta": 32}),
        _defaults_with(gpu_memory_utilization=0.5),
        device_working_set_bytes=128 * GB,
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(under)
    assert caplog.text == ""


def test_log_memory_budget_report_reports_ceiling_and_cache(tmp_path, caplog):
    from vllm_mlx.scheduler import SchedulerConfig

    report = build_memory_budget_report(
        _manager_config(budget_gb=40),
        _registry(tmp_path, {"alpha": 32}),
        _defaults_with(
            gpu_memory_utilization=0.5,
            scheduler_config=SchedulerConfig(cache_memory_mb=20480),
        ),
        device_working_set_bytes=128 * GB,
    )
    with caplog.at_level(logging.INFO, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(report)

    assert "Registry memory budget: 40.0 GB" in caplog.text
    assert "Metal allocation ceiling 64.0 GB" in caplog.text
    assert "20.0 GB (--cache-memory-mb)" in caplog.text


def test_log_memory_budget_report_says_so_when_ceiling_unknown(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(
        "vllm_mlx.model_registry._device_working_set_bytes", lambda: None
    )
    report = build_memory_budget_report(
        _manager_config(budget_gb=68),
        _registry(tmp_path, {"alpha": 32}),
        _defaults(),
    )

    with caplog.at_level(logging.INFO, logger="vllm_mlx.model_registry"):
        log_memory_budget_report(report)

    assert "cannot be reconciled" in caplog.text
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
