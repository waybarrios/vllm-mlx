# SPDX-License-Identifier: Apache-2.0
"""Tests for registry-backed multi-model serving."""

from __future__ import annotations

import asyncio
import time
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
    idle_unload_seconds: float = 0.0,
) -> RegistryManagerConfig:
    return RegistryManagerConfig(
        memory_budget_bytes=int(budget_gb * (1024**3)),
        idle_unload_seconds=idle_unload_seconds,
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


def test_unload_idle_unloads_stale_models_and_skips_busy_ones(tmp_path):
    """Idle-timeout unload must fire without any competing model ever being
    requested, unlike memory-budget eviction which only triggers on acquire().
    """

    async def _run():
        registry = _registry(tmp_path, {"alpha": 4, "beta": 4})
        created: dict[str, FakeEngine] = {}

        def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
            engine = FakeEngine(config)
            created[config.entry.name] = engine
            return engine

        manager = ModelManager(
            _manager_config(budget_gb=16, idle_unload_seconds=999),
            registry,
            _defaults(),
            engine_factory=engine_factory,
        )

        alpha_lease = await manager.acquire("alpha")
        await alpha_lease.release()

        # beta stays busy (never released) so it must survive the sweep.
        await manager.acquire("beta")

        manager._loaded["alpha"].last_used_at = time.time() - 1000
        manager._loaded["beta"].last_used_at = time.time() - 1000

        unloaded = await manager.unload_idle()

        assert unloaded == ["alpha"]
        assert "alpha" not in manager._loaded
        assert "beta" in manager._loaded
        assert created["alpha"].stopped == 1
        assert created["beta"].stopped == 0

    asyncio.run(_run())


def test_unload_idle_noop_when_disabled(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 4})
        manager = ModelManager(
            _manager_config(budget_gb=16, idle_unload_seconds=0.0),
            registry,
            _defaults(),
            engine_factory=lambda config: FakeEngine(config),
        )

        lease = await manager.acquire("alpha")
        await lease.release()
        manager._loaded["alpha"].last_used_at = time.time() - 100000

        unloaded = await manager.unload_idle()

        assert unloaded == []
        assert "alpha" in manager._loaded

    asyncio.run(_run())


def test_unload_idle_leaves_fresh_models_alone(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 4})
        manager = ModelManager(
            _manager_config(budget_gb=16, idle_unload_seconds=999),
            registry,
            _defaults(),
            engine_factory=lambda config: FakeEngine(config),
        )

        lease = await manager.acquire("alpha")
        await lease.release()

        unloaded = await manager.unload_idle()

        assert unloaded == []
        assert "alpha" in manager._loaded

    asyncio.run(_run())


def test_run_idle_reaper_unloads_after_timeout(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 4})
        created: dict[str, FakeEngine] = {}

        def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
            engine = FakeEngine(config)
            created[config.entry.name] = engine
            return engine

        manager = ModelManager(
            _manager_config(budget_gb=16, idle_unload_seconds=0.2),
            registry,
            _defaults(),
            engine_factory=engine_factory,
        )

        lease = await manager.acquire("alpha")
        await lease.release()

        reaper = asyncio.create_task(manager.run_idle_reaper())
        try:
            for _ in range(50):
                if "alpha" not in manager._loaded:
                    break
                await asyncio.sleep(0.05)
        finally:
            reaper.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reaper

        assert "alpha" not in manager._loaded
        assert created["alpha"].stopped == 1

    asyncio.run(_run())


def test_run_idle_reaper_returns_immediately_when_disabled(tmp_path):
    async def _run():
        registry = _registry(tmp_path, {"alpha": 4})
        manager = ModelManager(
            _manager_config(budget_gb=16, idle_unload_seconds=0.0),
            registry,
            _defaults(),
            engine_factory=lambda config: FakeEngine(config),
        )

        await asyncio.wait_for(manager.run_idle_reaper(), timeout=1.0)

    asyncio.run(_run())


class TestLoadRegistryConfigIdleUnload:
    """YAML-level wiring for the manager.idle_unload_seconds knob."""

    def _write_config(self, tmp_path: Path, manager_extra: str = "") -> Path:
        model_dir = tmp_path / "alpha"
        model_dir.mkdir()
        config_path = tmp_path / "models.yaml"
        config_path.write_text(f"""
manager:
  memory_budget_gb: 16
{manager_extra}
models:
  - name: alpha
    path: {model_dir}
    estimated_memory_gb: 4
""")
        return config_path

    def test_explicit_yaml_value_wins(self, tmp_path):
        from vllm_mlx.model_registry import load_registry_config

        config_path = self._write_config(tmp_path, "  idle_unload_seconds: 120\n")
        defaults = _defaults()
        defaults = type(defaults)(
            **{**defaults.__dict__, "auto_unload_idle_seconds": 300.0}
        )

        manager_config, _ = load_registry_config(config_path, defaults)

        assert manager_config.idle_unload_seconds == 120.0

    def test_falls_back_to_cli_default_when_unset(self, tmp_path):
        from vllm_mlx.model_registry import load_registry_config

        config_path = self._write_config(tmp_path)
        defaults = _defaults()
        defaults = type(defaults)(
            **{**defaults.__dict__, "auto_unload_idle_seconds": 300.0}
        )

        manager_config, _ = load_registry_config(config_path, defaults)

        assert manager_config.idle_unload_seconds == 300.0

    def test_defaults_to_disabled(self, tmp_path):
        from vllm_mlx.model_registry import load_registry_config

        config_path = self._write_config(tmp_path)

        manager_config, _ = load_registry_config(config_path, _defaults())

        assert manager_config.idle_unload_seconds == 0.0
