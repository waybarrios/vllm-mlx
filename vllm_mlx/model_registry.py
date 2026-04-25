# SPDX-License-Identifier: Apache-2.0
"""
Registry-backed multi-model serving with memory-budget eviction.

The registry maps OpenAI-compatible ``model`` names to concrete local paths or
declared HuggingFace IDs. Models are loaded lazily, optionally preloaded, and
evicted according to a memory-budget policy with configurable wait/fail/preempt
behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .api.utils import is_mllm_model
from .engine.base import BaseEngine
from .engine.batched import BatchedEngine
from .engine.simple import SimpleEngine
from .scheduler import SchedulerConfig
from .utils.download import DownloadConfig, ensure_model_downloaded

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class ModelOwnershipError(RuntimeError):
    """Raised when an EngineCore attempts to use a model already in use."""


class _ModelOwnershipRegistry:
    """Process-local model ownership guard used by EngineCore."""

    def __init__(self) -> None:
        self._owners: dict[int, str] = {}

    def acquire(
        self,
        *,
        model: Any,
        engine: Any,
        engine_id: str,
        force: bool = True,
    ) -> None:
        key = id(model)
        owner = self._owners.get(key)
        if owner is not None and owner != engine_id and not force:
            raise ModelOwnershipError(
                f"Model is already owned by engine {owner}; "
                f"engine {engine_id} cannot acquire it"
            )
        self._owners[key] = engine_id

    def release(self, model: Any, engine_id: str) -> None:
        key = id(model)
        owner = self._owners.get(key)
        if owner == engine_id:
            self._owners.pop(key, None)

    def is_owned(self, model: Any) -> tuple[bool, str | None]:
        key = id(model)
        owner = self._owners.get(key)
        if owner is not None:
            return (True, owner)
        return (False, None)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._owners),
            "active_owners": len(self._owners),
        }


_ownership_registry = _ModelOwnershipRegistry()


def get_registry() -> _ModelOwnershipRegistry:
    """Return the global model ownership registry used by EngineCore."""
    return _ownership_registry


# ============================================================================
# Registry-backed multi-model serving types
# ============================================================================

ContentionStrategy = Literal[
    "fail",
    "wait",
    "preempt",
    "wait_then_fail",
    "wait_then_preempt",
]

EngineFactory = Callable[["ResolvedModelConfig"], BaseEngine]


@dataclass(frozen=True)
class RegistryServeDefaults:
    """Global serve defaults inherited by registry entries."""

    continuous_batching: bool
    force_mllm: bool
    enable_mtp: bool
    prefill_step_size: int
    specprefill_enabled: bool
    specprefill_threshold: int
    specprefill_keep_pct: float
    specprefill_draft_model: str | None
    stream_interval: int
    gpu_memory_utilization: float
    scheduler_config: SchedulerConfig | None
    max_tokens: int
    download_config: DownloadConfig


@dataclass(frozen=True)
class ContentionPolicy:
    """Policy used when a new model cannot fit inside the memory budget."""

    strategy: ContentionStrategy = "wait_then_fail"
    wait_timeout_s: float | None = 30.0
    preempt_after_s: float | None = None


@dataclass(frozen=True)
class RegistryManagerConfig:
    """Global registry manager configuration."""

    memory_budget_bytes: int
    policy: ContentionPolicy


@dataclass(frozen=True)
class RegisteredModel:
    """One configured model entry."""

    name: str
    source: str
    preload: bool = False
    continuous_batching: bool | None = None
    force_mllm: bool | None = None
    enable_mtp: bool | None = None
    prefill_step_size: int | None = None
    specprefill_enabled: bool | None = None
    specprefill_threshold: int | None = None
    specprefill_keep_pct: float | None = None
    specprefill_draft_model: str | None = None
    stream_interval: int | None = None
    gpu_memory_utilization: float | None = None
    estimated_memory_bytes: int | None = None


@dataclass(frozen=True)
class ResolvedModelConfig:
    """Effective configuration for a loaded model."""

    entry: RegisteredModel
    resolved_source: str
    continuous_batching: bool
    force_mllm: bool
    enable_mtp: bool
    prefill_step_size: int
    specprefill_enabled: bool
    specprefill_threshold: int
    specprefill_keep_pct: float
    specprefill_draft_model: str | None
    stream_interval: int
    gpu_memory_utilization: float
    scheduler_config: SchedulerConfig | None
    estimated_memory_bytes: int


@dataclass
class LoadedModel:
    """Runtime state for a loaded engine."""

    config: ResolvedModelConfig
    engine: BaseEngine
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    active_requests: int = 0
    active_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    preempting: bool = False


@dataclass
class PendingLoad:
    """A reserved model load in progress."""

    model_name: str
    required_bytes: int
    future: asyncio.Future[LoadedModel]


@dataclass
class ModelLease:
    """Active lease for a loaded model."""

    manager: "ModelManager | None"
    model_name: str
    engine: BaseEngine
    release_cb: Callable[[], Awaitable[None]]

    async def release(self) -> None:
        if self.manager is None:
            return
        manager = self.manager
        self.manager = None
        await self.release_cb()

    async def __aenter__(self) -> "ModelLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


def _clone_scheduler_config(config: SchedulerConfig | None) -> SchedulerConfig | None:
    """Clone a SchedulerConfig so per-model overrides do not mutate globals."""
    if config is None:
        return None
    return SchedulerConfig(**vars(config))


def _parse_memory_budget_bytes(value: Any) -> int:
    """Parse a memory budget from bytes, MB, or GB."""
    if value is None:
        raise ValueError("models-config manager.memory_budget_gb is required")
    if isinstance(value, (int, float)):
        return int(float(value) * (1024**3))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw.endswith("gb"):
            return int(float(raw[:-2]) * (1024**3))
        if raw.endswith("mb"):
            return int(float(raw[:-2]) * (1024**2))
        if raw.endswith("b"):
            return int(float(raw[:-1]))
        return int(float(raw) * (1024**3))
    raise TypeError(f"Unsupported memory budget value: {value!r}")


def _safe_available_memory_bytes() -> int:
    """Best-effort available system memory."""
    if psutil is None:  # pragma: no cover - fallback only
        return 0
    return int(psutil.virtual_memory().available)


def _estimate_model_bytes_from_source(source: str) -> int:
    """Estimate model footprint from local artifact size when possible."""
    path = Path(source)
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size if path.suffix in {".safetensors", ".gguf"} else 0

    total = 0
    for pattern in ("*.safetensors", "*.gguf"):
        for fp in path.rglob(pattern):
            try:
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def load_registry_config(
    config_path: str | os.PathLike[str],
    defaults: RegistryServeDefaults,
) -> tuple[RegistryManagerConfig, dict[str, RegisteredModel]]:
    """Load and validate the models registry YAML file."""
    import yaml  # lazy: only needed when a registry config is provided

    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models-config must define a non-empty 'models' list")

    manager_raw = raw.get("manager") or {}
    policy_raw = manager_raw.get("contention_policy") or {}
    policy = ContentionPolicy(
        strategy=policy_raw.get("strategy", "wait_then_fail"),
        wait_timeout_s=(
            float(policy_raw["wait_timeout_s"])
            if policy_raw.get("wait_timeout_s") is not None
            else 30.0
        ),
        preempt_after_s=(
            float(policy_raw["preempt_after_s"])
            if policy_raw.get("preempt_after_s") is not None
            else None
        ),
    )
    if policy.strategy not in {
        "fail",
        "wait",
        "preempt",
        "wait_then_fail",
        "wait_then_preempt",
    }:
        raise ValueError(f"Unsupported contention strategy: {policy.strategy}")

    manager = RegistryManagerConfig(
        memory_budget_bytes=_parse_memory_budget_bytes(
            manager_raw.get("memory_budget_gb", manager_raw.get("memory_budget"))
        ),
        policy=policy,
    )

    registry: dict[str, RegisteredModel] = {}
    for item in models:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid model entry: {item!r}")
        name = item.get("name")
        source = item.get("path") or item.get("source") or item.get("model")
        if not name or not source:
            raise ValueError(
                f"Each model entry must define 'name' and one of 'path'/'source'/'model': {item!r}"
            )
        if name in registry:
            raise ValueError(f"Duplicate model name in registry: {name}")

        estimated = item.get("estimated_memory_gb")
        estimated_bytes = (
            int(float(estimated) * (1024**3)) if estimated is not None else None
        )

        registry[name] = RegisteredModel(
            name=name,
            source=str(source),
            preload=bool(item.get("preload", False)),
            continuous_batching=item.get("continuous_batching"),
            force_mllm=item.get("mllm"),
            enable_mtp=item.get("enable_mtp"),
            prefill_step_size=item.get("prefill_step_size"),
            specprefill_enabled=item.get("specprefill"),
            specprefill_threshold=item.get("specprefill_threshold"),
            specprefill_keep_pct=item.get("specprefill_keep_pct"),
            specprefill_draft_model=item.get("specprefill_draft_model"),
            stream_interval=item.get("stream_interval"),
            gpu_memory_utilization=item.get("gpu_memory_utilization"),
            estimated_memory_bytes=estimated_bytes,
        )

    return manager, registry


class ModelManager:
    """Registry-backed model manager with lazy load and memory-budget eviction."""

    def __init__(
        self,
        manager_config: RegistryManagerConfig,
        registry: dict[str, RegisteredModel],
        defaults: RegistryServeDefaults,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._config = manager_config
        self._registry = registry
        self._defaults = defaults
        self._engine_factory = engine_factory
        self._loaded: dict[str, LoadedModel] = {}
        self._loading: dict[str, PendingLoad] = {}
        self._unloading: dict[str, LoadedModel] = {}
        self._condition = asyncio.Condition()
        self._shutting_down = False

    @property
    def memory_budget_bytes(self) -> int:
        return self._config.memory_budget_bytes

    @property
    def registered_model_names(self) -> list[str]:
        """Return sorted list of all registered model names."""
        return sorted(self._registry.keys())

    def has_model(self, model_name: str) -> bool:
        return model_name in self._registry

    def list_models(self) -> list[dict[str, Any]]:
        """Return registry state for /v1/models."""
        data = []
        for name, entry in self._registry.items():
            loaded = self._loaded.get(name)
            unloading = self._unloading.get(name)
            loading = self._loading.get(name)
            state = "unloaded"
            if loaded is not None:
                state = "preempting" if loaded.preempting else "loaded"
            elif loading is not None:
                state = "loading"
            elif unloading is not None:
                state = "unloading"

            estimated = (
                loaded.config.estimated_memory_bytes
                if loaded is not None
                else (
                    unloading.config.estimated_memory_bytes
                    if unloading is not None
                    else (
                        loading.required_bytes
                        if loading is not None
                        else self._resolve_estimated_bytes(entry, entry.source)
                    )
                )
            )
            data.append(
                {
                    "id": name,
                    "status": state,
                    "loaded": loaded is not None,
                    "owned_by": "vllm-mlx",
                    "source": entry.source,
                    "memory_gb": round(estimated / (1024**3), 2) if estimated else None,
                }
            )
        return data

    async def preload(self) -> None:
        """Preload any entries marked preload=true."""
        for entry in self._registry.values():
            if entry.preload:
                lease = await self.acquire(entry.name)
                await lease.release()

    async def shutdown(self) -> None:
        """Stop and unload every loaded engine."""
        pending: list[asyncio.Future[LoadedModel]] = []
        unloads: list[LoadedModel] = []
        cancel_tasks: set[asyncio.Task[Any]] = set()

        async with self._condition:
            self._shutting_down = True
            pending = [item.future for item in self._loading.values()]
            for loaded in self._loaded.values():
                loaded.preempting = True
                cancel_tasks.update(loaded.active_tasks)
            for name in list(self._loaded.keys()):
                if self._loaded[name].active_requests == 0:
                    unloads.append(self._begin_unload_locked(name))
            self._condition.notify_all()

        for task in cancel_tasks:
            task.cancel()
        await self._run_unloads(unloads)

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        remaining: list[LoadedModel] = []
        async with self._condition:
            for name in list(self._loaded.keys()):
                if self._loaded[name].active_requests == 0:
                    remaining.append(self._begin_unload_locked(name))
            self._condition.notify_all()

        await self._run_unloads(remaining)

    async def acquire(self, model_name: str) -> ModelLease:
        """Acquire a lease for a configured model."""
        if model_name not in self._registry:
            raise KeyError(model_name)

        start = time.monotonic()
        while True:
            load: PendingLoad | None = None
            unloads: list[LoadedModel] = []
            cancel_tasks: set[asyncio.Task[Any]] = set()
            same_model_future: asyncio.Future[LoadedModel] | None = None
            wait_timeout: float | None = None

            async with self._condition:
                if self._shutting_down:
                    raise RuntimeError("Model manager is shutting down")

                claimed = self._claim_loaded_locked(model_name)
                if claimed is not None:
                    return claimed

                same_model_future = self._loading.get(model_name, None)
                if same_model_future is not None:
                    same_model_future = same_model_future.future
                elif model_name in self._unloading:
                    wait_timeout = self._remaining_wait_timeout(start)
                else:
                    entry = self._registry[model_name]
                    required_bytes = self._resolve_estimated_bytes(entry, entry.source)
                    unloads = self._collect_idle_unloads_locked(
                        model_name, required_bytes
                    )
                    if not unloads and self._can_reserve_locked(required_bytes):
                        load = self._reserve_load_locked(model_name, required_bytes)
                    elif not unloads:
                        cancel_tasks = self._maybe_preempt_locked(
                            model_name=model_name,
                            required_bytes=required_bytes,
                            start=start,
                        )
                        if not cancel_tasks and not self._should_wait_locked(start):
                            raise RuntimeError(
                                f"Cannot load '{model_name}' within memory budget "
                                f"({self._config.memory_budget_bytes / (1024**3):.1f} GB)"
                            )
                        wait_timeout = self._remaining_wait_timeout(start)

            if unloads:
                await self._run_unloads(unloads)
                continue

            if cancel_tasks:
                for task in cancel_tasks:
                    task.cancel()
                timeout = self._remaining_wait_timeout(start)
                await self._wait_for_change(timeout)
                continue

            if load is not None:
                loaded = await self._execute_load(load)
                async with self._condition:
                    claimed = self._claim_loaded_locked(
                        model_name, loaded_override=loaded
                    )
                    if claimed is not None:
                        return claimed
                continue

            if same_model_future is not None:
                loaded = await same_model_future
                async with self._condition:
                    claimed = self._claim_loaded_locked(
                        model_name, loaded_override=loaded
                    )
                    if claimed is not None:
                        return claimed
                continue

            await self._wait_for_change(wait_timeout)

    async def release(self, model_name: str) -> None:
        """Release a previously acquired model lease."""
        unload: LoadedModel | None = None

        async with self._condition:
            loaded = self._loaded.get(model_name)
            if loaded is None:
                return

            loaded.active_requests = max(0, loaded.active_requests - 1)
            loaded.last_used_at = time.time()
            task = asyncio.current_task()
            if task is not None:
                loaded.active_tasks.discard(task)

            if loaded.preempting and loaded.active_requests == 0:
                unload = self._begin_unload_locked(model_name)

            self._condition.notify_all()

        if unload is not None:
            await self._run_unloads([unload])

    def _claim_loaded_locked(
        self,
        model_name: str,
        *,
        loaded_override: LoadedModel | None = None,
    ) -> ModelLease | None:
        loaded = loaded_override or self._loaded.get(model_name)
        if loaded is None:
            return None

        if loaded_override is not None and model_name not in self._loaded:
            self._loaded[model_name] = loaded

        if loaded.preempting:
            return None

        loaded.active_requests += 1
        loaded.last_used_at = time.time()
        task = asyncio.current_task()
        if task is not None:
            loaded.active_tasks.add(task)

        async def _release() -> None:
            await self.release(model_name)

        return ModelLease(
            manager=self,
            model_name=model_name,
            engine=loaded.engine,
            release_cb=_release,
        )

    async def _execute_load(self, pending: PendingLoad) -> LoadedModel:
        """Instantiate a reserved model load outside the manager lock."""
        entry = self._registry[pending.model_name]
        loaded: LoadedModel | None = None
        unload_after_load: LoadedModel | None = None

        try:
            resolved_source = await self._resolve_source(entry)
            loaded = await self._instantiate_model(entry, resolved_source)
        except Exception as exc:
            async with self._condition:
                current = self._loading.pop(pending.model_name, None)
                if current is pending and not current.future.done():
                    current.future.set_exception(exc)
                self._condition.notify_all()
            raise

        async with self._condition:
            current = self._loading.pop(pending.model_name, None)
            if current is not pending:
                unload_after_load = loaded
            elif self._shutting_down:
                unload_after_load = loaded
                if not current.future.done():
                    current.future.set_exception(
                        RuntimeError("Model manager is shutting down")
                    )
            else:
                self._loaded[pending.model_name] = loaded
                if not current.future.done():
                    current.future.set_result(loaded)
            self._condition.notify_all()

        if unload_after_load is not None:
            await unload_after_load.engine.stop()
            raise RuntimeError("Model load was aborted before it became available")

        return loaded

    async def _wait_for_change(self, timeout: float | None) -> None:
        async with self._condition:
            if timeout is None:
                await self._condition.wait()
                return
            if timeout <= 0:
                raise RuntimeError("Timed out waiting for model capacity")
            await asyncio.wait_for(self._condition.wait(), timeout=timeout)

    async def _run_unloads(self, unloads: list[LoadedModel]) -> None:
        for loaded in unloads:
            try:
                await loaded.engine.stop()
            finally:
                async with self._condition:
                    self._unloading.pop(loaded.config.entry.name, None)
                    self._condition.notify_all()

    def _reserve_load_locked(self, model_name: str, required_bytes: int) -> PendingLoad:
        future: asyncio.Future[LoadedModel] = asyncio.get_running_loop().create_future()
        pending = PendingLoad(
            model_name=model_name,
            required_bytes=required_bytes,
            future=future,
        )
        self._loading[model_name] = pending
        return pending

    def _begin_unload_locked(self, model_name: str) -> LoadedModel:
        loaded = self._loaded.pop(model_name)
        self._unloading[model_name] = loaded
        return loaded

    def _collect_idle_unloads_locked(
        self, requested_model: str, required_bytes: int
    ) -> list[LoadedModel]:
        selected: list[LoadedModel] = []
        projected_bytes = self._committed_bytes_locked()
        candidates = sorted(
            (
                loaded
                for name, loaded in self._loaded.items()
                if name != requested_model and loaded.active_requests == 0
            ),
            key=lambda item: item.last_used_at,
        )

        for loaded in candidates:
            if projected_bytes + required_bytes <= self._config.memory_budget_bytes:
                break
            selected.append(self._begin_unload_locked(loaded.config.entry.name))
            projected_bytes -= loaded.config.estimated_memory_bytes

        return selected

    def _maybe_preempt_locked(
        self,
        *,
        model_name: str,
        required_bytes: int,
        start: float,
    ) -> set[asyncio.Task[Any]]:
        if not self._should_preempt_locked(start):
            return set()

        cancel_tasks: set[asyncio.Task[Any]] = set()
        projected_bytes = self._committed_bytes_locked()
        candidates = sorted(
            (
                loaded
                for name, loaded in self._loaded.items()
                if name != model_name and loaded.active_requests > 0
            ),
            key=lambda item: item.last_used_at,
        )

        for loaded in candidates:
            if projected_bytes + required_bytes <= self._config.memory_budget_bytes:
                break
            if loaded.preempting:
                projected_bytes -= loaded.config.estimated_memory_bytes
                continue
            loaded.preempting = True
            cancel_tasks.update(loaded.active_tasks)
            projected_bytes -= loaded.config.estimated_memory_bytes

        if cancel_tasks:
            self._condition.notify_all()
        return cancel_tasks

    def _should_wait_locked(self, start: float) -> bool:
        strategy = self._config.policy.strategy
        if strategy == "fail":
            return False
        timeout = self._remaining_wait_timeout(start)
        return timeout is None or timeout > 0

    def _should_preempt_locked(self, start: float) -> bool:
        policy = self._config.policy
        elapsed = time.monotonic() - start
        if policy.strategy == "preempt":
            return True
        if policy.strategy != "wait_then_preempt":
            return False
        trigger = policy.preempt_after_s if policy.preempt_after_s is not None else 0.0
        return elapsed >= trigger

    def _remaining_wait_timeout(self, start: float) -> float | None:
        timeout = self._config.policy.wait_timeout_s
        if timeout is None or timeout <= 0:
            return None
        return max(timeout - (time.monotonic() - start), 0.0)

    def _can_reserve_locked(self, required_bytes: int) -> bool:
        return (
            self._committed_bytes_locked() + required_bytes
            <= self._config.memory_budget_bytes
        )

    def _committed_bytes_locked(self) -> int:
        loaded_bytes = sum(
            loaded.config.estimated_memory_bytes for loaded in self._loaded.values()
        )
        loading_bytes = sum(item.required_bytes for item in self._loading.values())
        unloading_bytes = sum(
            loaded.config.estimated_memory_bytes for loaded in self._unloading.values()
        )
        return loaded_bytes + loading_bytes + unloading_bytes

    async def _instantiate_model(
        self, entry: RegisteredModel, resolved_source: str
    ) -> LoadedModel:
        config = self._resolve_model_config(entry, resolved_source)

        if self._engine_factory is not None:
            engine = self._engine_factory(config)
        elif config.continuous_batching:
            engine = BatchedEngine(
                model_name=config.resolved_source,
                scheduler_config=config.scheduler_config,
                stream_interval=config.stream_interval,
                force_mllm=config.force_mllm,
                gpu_memory_utilization=config.gpu_memory_utilization,
            )
        else:
            engine = SimpleEngine(
                model_name=config.resolved_source,
                force_mllm=config.force_mllm,
                mtp=config.enable_mtp,
                prefill_step_size=config.prefill_step_size,
                specprefill_enabled=config.specprefill_enabled,
                specprefill_threshold=config.specprefill_threshold,
                specprefill_keep_pct=config.specprefill_keep_pct,
                specprefill_draft_model=config.specprefill_draft_model,
            )

        await engine.start()
        return LoadedModel(config=config, engine=engine)

    async def _resolve_source(self, entry: RegisteredModel) -> str:
        return await asyncio.to_thread(self._resolve_source_sync, entry)

    def _resolve_source_sync(self, entry: RegisteredModel) -> str:
        source = entry.source
        if Path(source).exists():
            return source
        downloaded = ensure_model_downloaded(
            source,
            config=self._defaults.download_config,
            is_mllm=is_mllm_model(source) or bool(entry.force_mllm),
        )
        return str(downloaded)

    def _resolve_estimated_bytes(
        self, entry: RegisteredModel, resolved_source: str
    ) -> int:
        if entry.estimated_memory_bytes is not None:
            return entry.estimated_memory_bytes
        estimated = _estimate_model_bytes_from_source(resolved_source)
        if estimated > 0:
            return estimated
        source_path = Path(resolved_source)
        if not source_path.exists():
            raise ValueError(
                "models-config entry "
                f"'{entry.name}' uses non-local source '{entry.source}' without "
                "estimated_memory_gb. Registry-backed loading requires an explicit "
                "memory estimate for non-local models so eviction remains deterministic."
            )

        available = _safe_available_memory_bytes()
        if available > 0:
            logger.warning(
                "Falling back to a coarse memory estimate for registry entry '%s' "
                "because no weight files were found under %s; set estimated_memory_gb "
                "explicitly for deterministic eviction.",
                entry.name,
                resolved_source,
            )
            return max(available // 8, 1)

        raise ValueError(
            "Cannot estimate memory for registry entry "
            f"'{entry.name}' from '{resolved_source}'. Set estimated_memory_gb "
            "explicitly in the models config."
        )

    def _resolve_model_config(
        self, entry: RegisteredModel, resolved_source: str
    ) -> ResolvedModelConfig:
        scheduler_config = _clone_scheduler_config(self._defaults.scheduler_config)

        continuous_batching = (
            entry.continuous_batching
            if entry.continuous_batching is not None
            else self._defaults.continuous_batching
        )
        force_mllm = (
            entry.force_mllm
            if entry.force_mllm is not None
            else self._defaults.force_mllm
        )
        enable_mtp = (
            entry.enable_mtp
            if entry.enable_mtp is not None
            else self._defaults.enable_mtp
        )
        prefill_step_size = (
            entry.prefill_step_size
            if entry.prefill_step_size is not None
            else self._defaults.prefill_step_size
        )
        specprefill_enabled = (
            entry.specprefill_enabled
            if entry.specprefill_enabled is not None
            else self._defaults.specprefill_enabled
        )
        specprefill_threshold = (
            entry.specprefill_threshold
            if entry.specprefill_threshold is not None
            else self._defaults.specprefill_threshold
        )
        specprefill_keep_pct = (
            entry.specprefill_keep_pct
            if entry.specprefill_keep_pct is not None
            else self._defaults.specprefill_keep_pct
        )
        specprefill_draft_model = (
            entry.specprefill_draft_model
            if entry.specprefill_draft_model is not None
            else self._defaults.specprefill_draft_model
        )
        stream_interval = (
            entry.stream_interval
            if entry.stream_interval is not None
            else self._defaults.stream_interval
        )
        gpu_memory_utilization = (
            entry.gpu_memory_utilization
            if entry.gpu_memory_utilization is not None
            else self._defaults.gpu_memory_utilization
        )
        estimated_memory_bytes = self._resolve_estimated_bytes(entry, resolved_source)

        return ResolvedModelConfig(
            entry=entry,
            resolved_source=resolved_source,
            continuous_batching=continuous_batching,
            force_mllm=force_mllm,
            enable_mtp=enable_mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_draft_model=specprefill_draft_model,
            stream_interval=stream_interval,
            gpu_memory_utilization=gpu_memory_utilization,
            scheduler_config=scheduler_config,
            estimated_memory_bytes=estimated_memory_bytes,
        )
