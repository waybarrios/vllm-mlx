# SPDX-License-Identifier: Apache-2.0
"""Tests for registry-backed multi-model serving."""

from __future__ import annotations

import asyncio
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
