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
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .api.utils import is_mllm_model
from .cli_arg_types import parse_positive_finite_float
from .engine.base import BaseEngine, suspend_cancellation
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
    specprefill_backbone_pct: float
    specprefill_draft_model: str | None
    prefix_trie_cache: bool
    prefix_trie_cache_size: int
    prefix_trie_cache_memory_mb: int | None
    stream_interval: int
    gpu_memory_utilization: float
    scheduler_config: SchedulerConfig | None
    max_tokens: int
    download_config: DownloadConfig
    auto_unload_idle_seconds: float = 0.0


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
    idle_unload_seconds: float = 0.0


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
    specprefill_backbone_pct: float | None = None
    specprefill_draft_model: str | None = None
    prefix_trie_cache: bool | None = None
    prefix_trie_cache_size: int | None = None
    prefix_trie_cache_memory_mb: int | None = None
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
    specprefill_backbone_pct: float
    specprefill_draft_model: str | None
    prefix_trie_cache: bool
    prefix_trie_cache_size: int
    prefix_trie_cache_memory_mb: int | None
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
    multiplier = 1024**3
    amount: str | int | float
    if isinstance(value, (int, float)):
        amount = value
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw.endswith("gb"):
            amount = raw[:-2]
        elif raw.endswith("mb"):
            amount = raw[:-2]
            multiplier = 1024**2
        elif raw.endswith("b"):
            amount = raw[:-1]
            multiplier = 1
        else:
            amount = raw
    else:
        raise TypeError(f"Unsupported memory budget value: {value!r}")

    value_name = "models-config manager memory budget"
    parsed = parse_positive_finite_float(amount, value_name)
    bytes_value = parse_positive_finite_float(parsed * multiplier, value_name)
    memory_budget_bytes = int(bytes_value)
    if memory_budget_bytes == 0:
        raise ValueError(f"{value_name} must be at least 1 byte")
    return memory_budget_bytes


def _safe_available_memory_bytes() -> int:
    """Best-effort available system memory."""
    if psutil is None:  # pragma: no cover - fallback only
        return 0
    return int(psutil.virtual_memory().available)


def _device_working_set_bytes() -> int | None:
    """Best-effort Metal recommended working-set size, or None when unavailable."""
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return None
        info = mx.device_info()
        raw = info.get(
            "max_recommended_working_set_size",
            info.get("memory_size", 0),
        )
        working_set = int(raw or 0)
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.debug("Could not query MLX device memory: %s", exc)
        return None
    return working_set or None


@dataclass(frozen=True)
class MemoryBudgetReport:
    """Reconciliation of the manager weight budget with the Metal ceiling.

    The manager budget counts model *weights* only, and the Metal allocation
    ceiling (``gpu_memory_utilization`` x device working set) is process-wide.
    Those two are directly comparable, so a budget above the ceiling is a
    deterministic conflict: the manager will keep models resident that MLX
    cannot allocate, and the load fails instead of evicting.

    The prefix-cache limit is deliberately *not* folded into that comparison.
    ``cache_memory_mb`` is a per-engine maximum — it is cloned into each
    resident continuous-batching engine and allocated lazily, and simple-mode
    entries never receive it at all — so it is neither a single process-wide
    reservation nor a bound that can be subtracted once. It is reported
    alongside the ceiling instead, with its own conflict check.
    """

    budget_bytes: int
    device_working_set_bytes: int | None
    gpu_memory_utilization: float | None
    gpu_memory_utilization_source: str | None
    per_engine_cache_limit_bytes: int | None
    per_engine_cache_percent: float | None
    continuous_batching_entries: int
    total_entries: int

    @property
    def allocation_ceiling_bytes(self) -> int | None:
        """Metal soft allocation limit that will be installed at engine start.

        ``None`` when no ceiling can be attributed: either MLX cannot report a
        device working set, or no entry will install one (only ``BatchedEngine``
        calls ``mx.set_memory_limit``).
        """
        if self.device_working_set_bytes is None or self.gpu_memory_utilization is None:
            return None
        return int(self.device_working_set_bytes * self.gpu_memory_utilization)

    @property
    def exceeds_ceiling(self) -> bool:
        """True when the weights budget alone cannot fit under the ceiling.

        Both sides are process-wide totals, so this is the deterministic check.
        """
        ceiling = self.allocation_ceiling_bytes
        return ceiling is not None and self.budget_bytes > ceiling

    @property
    def cache_limit_exceeds_ceiling(self) -> bool:
        """True when one engine's prefix cache could alone fill the ceiling."""
        ceiling = self.allocation_ceiling_bytes
        if ceiling is None or self.per_engine_cache_limit_bytes is None:
            return False
        return self.per_engine_cache_limit_bytes >= ceiling


def build_memory_budget_report(
    manager_config: RegistryManagerConfig,
    registry: dict[str, RegisteredModel],
    defaults: RegistryServeDefaults,
    *,
    device_working_set_bytes: int | None = None,
) -> MemoryBudgetReport:
    """Reconcile the manager weight budget against the Metal allocation ceiling.

    The Metal limit is process-wide but is re-installed by every
    ``BatchedEngine`` start, so the ceiling the manager has to live under is the
    *lowest* utilization among the entries that actually install one. Only
    continuous-batching entries qualify: ``SimpleEngine`` never calls
    ``mx.set_memory_limit`` and is not even given a ``gpu_memory_utilization``.
    A registry with no continuous-batching entries therefore gets no attributed
    ceiling rather than one derived from a value nothing installs.
    """
    if device_working_set_bytes is None:
        device_working_set_bytes = _device_working_set_bytes()

    # Only BatchedEngine installs a Metal allocation limit — SimpleEngine is not
    # even constructed with a gpu_memory_utilization — so a simple-mode entry's
    # override is inert and must not be treated as a ceiling candidate.
    candidates: list[tuple[float, int, str]] = []
    for name in sorted(registry):
        entry = registry[name]
        entry_continuous_batching = (
            entry.continuous_batching
            if entry.continuous_batching is not None
            else defaults.continuous_batching
        )
        if not entry_continuous_batching:
            continue
        if entry.gpu_memory_utilization is not None:
            # Rank 1: an override is only attributable to the entry declaring it.
            candidates.append(
                (entry.gpu_memory_utilization, 1, f"models-config entry '{name}'")
            )
        else:
            # Rank 0: prefer the serve default as the named source on ties.
            candidates.append((defaults.gpu_memory_utilization, 0, "serve default"))

    continuous_batching_entries = len(candidates)

    utilization: float | None = None
    utilization_source: str | None = None
    if candidates:
        utilization, _, utilization_source = min(candidates)

    # cache_memory_mb only binds for continuous-batching engines, and only when
    # the memory-aware prefix cache is the one actually in use.
    scheduler_config = defaults.scheduler_config
    per_engine_cache_limit_bytes: int | None = None
    per_engine_cache_percent: float | None = None
    cache_applies = (
        scheduler_config is not None
        and continuous_batching_entries > 0
        and getattr(scheduler_config, "enable_prefix_cache", False)
        and not getattr(scheduler_config, "use_paged_cache", False)
        and getattr(scheduler_config, "use_memory_aware_cache", False)
    )
    if cache_applies:
        cache_memory_mb = getattr(scheduler_config, "cache_memory_mb", None)
        if cache_memory_mb:
            per_engine_cache_limit_bytes = int(cache_memory_mb) * (1024**2)
        else:
            percent = getattr(scheduler_config, "cache_memory_percent", None)
            if percent:
                per_engine_cache_percent = float(percent)

    return MemoryBudgetReport(
        budget_bytes=manager_config.memory_budget_bytes,
        device_working_set_bytes=device_working_set_bytes,
        gpu_memory_utilization=utilization,
        gpu_memory_utilization_source=utilization_source,
        per_engine_cache_limit_bytes=per_engine_cache_limit_bytes,
        per_engine_cache_percent=per_engine_cache_percent,
        continuous_batching_entries=continuous_batching_entries,
        total_entries=len(registry),
    )


def log_memory_budget_report(report: MemoryBudgetReport) -> None:
    """Log the budget/ceiling reconciliation, warning when they conflict."""
    gb = 1024**3
    ceiling = report.allocation_ceiling_bytes

    if ceiling is None:
        if report.gpu_memory_utilization is None:
            reason = (
                "no continuous-batching entries, and --gpu-memory-utilization "
                "installs a Metal limit only for those"
            )
        else:
            reason = "MLX cannot report a device working set on this host"
        logger.info(
            "Registry memory budget: %.1f GB of model weights; no Metal "
            "allocation ceiling to reconcile it with (%s)",
            report.budget_bytes / gb,
            reason,
        )
        return

    engines = f"{report.continuous_batching_entries} of {report.total_entries} entries"
    if report.per_engine_cache_limit_bytes is not None:
        cache_desc = (
            f"{report.per_engine_cache_limit_bytes / gb:.1f} GB per "
            f"continuous-batching engine (--cache-memory-mb, {engines})"
        )
    elif report.per_engine_cache_percent is not None:
        cache_desc = (
            f"~{report.per_engine_cache_percent * 100:.0f}% of available RAM per "
            f"continuous-batching engine (--cache-memory-percent, {engines}); "
            "scales at runtime"
        )
    else:
        cache_desc = "none configured"

    logger.info(
        "Registry memory budget: %.1f GB of model weights; "
        "Metal allocation ceiling %.1f GB (%.0f%% of %.1f GB, from %s); "
        "prefix-cache maximum %s",
        report.budget_bytes / gb,
        ceiling / gb,
        report.gpu_memory_utilization * 100,
        (report.device_working_set_bytes or 0) / gb,
        report.gpu_memory_utilization_source,
        cache_desc,
    )

    if report.exceeds_ceiling:
        logger.warning(
            "models-config manager.memory_budget_gb (%.1f GB) exceeds the Metal "
            "allocation ceiling (%.1f GB). The budget counts model weights only, "
            "so the manager will keep models resident that MLX cannot allocate, "
            "and a load can fail with an out-of-memory error instead of evicting. "
            "Lower the budget below %.1f GB — further still, since the KV cache "
            "and activations also come out of the ceiling — or raise "
            "--gpu-memory-utilization.",
            report.budget_bytes / gb,
            ceiling / gb,
            ceiling / gb,
        )

    if report.cache_limit_exceeds_ceiling:
        logger.warning(
            "--cache-memory-mb (%.1f GB per continuous-batching engine) is at or "
            "above the Metal allocation ceiling (%.1f GB) on its own, leaving no "
            "room for model weights. Note this is a per-engine maximum: it is "
            "cloned into every resident continuous-batching engine, so the "
            "aggregate grows with the number of resident models.",
            (report.per_engine_cache_limit_bytes or 0) / gb,
            ceiling / gb,
        )

    if not report.exceeds_ceiling and not report.cache_limit_exceeds_ceiling:
        logger.info(
            "The registry budget covers model weights only; the KV cache, the "
            "prefix cache and activations are additional and are not reserved "
            "by it."
        )


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
    *,
    memory_budget_gb: float | None = None,
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

    idle_unload_seconds = manager_raw.get("idle_unload_seconds")
    manager = RegistryManagerConfig(
        memory_budget_bytes=_parse_memory_budget_bytes(
            memory_budget_gb
            if memory_budget_gb is not None
            else manager_raw.get("memory_budget_gb", manager_raw.get("memory_budget"))
        ),
        policy=policy,
        idle_unload_seconds=(
            float(idle_unload_seconds)
            if idle_unload_seconds is not None
            else defaults.auto_unload_idle_seconds
        ),
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

        raw_gpu_memory_utilization = item.get("gpu_memory_utilization")
        gpu_memory_utilization = None
        if raw_gpu_memory_utilization is not None:
            try:
                gpu_memory_utilization = float(raw_gpu_memory_utilization)
            except (TypeError, ValueError):
                raise ValueError(
                    f"models-config entry '{name}' gpu_memory_utilization must "
                    "be finite and within (0, 1]"
                ) from None
            if not math.isfinite(gpu_memory_utilization) or not (
                0.0 < gpu_memory_utilization <= 1.0
            ):
                raise ValueError(
                    f"models-config entry '{name}' gpu_memory_utilization must "
                    "be finite and within (0, 1]"
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
            specprefill_backbone_pct=item.get("specprefill_backbone_pct"),
            specprefill_draft_model=item.get("specprefill_draft_model"),
            prefix_trie_cache=item.get("prefix_trie_cache"),
            prefix_trie_cache_size=item.get("prefix_trie_cache_size"),
            prefix_trie_cache_memory_mb=item.get("prefix_trie_cache_memory_mb"),
            stream_interval=item.get("stream_interval"),
            gpu_memory_utilization=gpu_memory_utilization,
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
    def idle_unload_seconds(self) -> float:
        return self._config.idle_unload_seconds

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
                    "last_used_at": loaded.last_used_at if loaded is not None else None,
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

    async def unload_idle(self) -> list[str]:
        """Unload every loaded model idle past ``idle_unload_seconds``.

        No-op (returns an empty list) if idle-unload is disabled
        (``idle_unload_seconds <= 0``). Unlike memory-budget eviction, this
        proactively frees models even when no other model is being requested.
        Returns the names of models that were unloaded.
        """
        idle_seconds = self._config.idle_unload_seconds
        if idle_seconds <= 0:
            return []

        now = time.time()
        async with self._condition:
            stale = [
                loaded
                for loaded in self._idle_candidates_locked()
                if now - loaded.last_used_at >= idle_seconds
            ]
            unloads = [
                self._begin_unload_locked(loaded.config.entry.name) for loaded in stale
            ]
            self._condition.notify_all()

        if unloads:
            await self._run_unloads(unloads)

        return [loaded.config.entry.name for loaded in unloads]

    async def run_idle_reaper(self) -> None:
        """Background loop that proactively unloads idle models.

        Mirrors the single-model residency lifecycle loop: sleeps at half the
        configured idle timeout (bounded to 5s) so short timeouts stay
        responsive, and a failed pass is logged rather than killing the loop.
        """
        idle_seconds = self._config.idle_unload_seconds
        if idle_seconds <= 0:
            return

        while True:
            await asyncio.sleep(min(idle_seconds / 2, 5.0))
            try:
                await self.unload_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Idle unload pass failed")

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
        async def _stop_engines() -> None:
            for loaded in unloads:
                try:
                    await loaded.engine.stop()
                finally:
                    async with self._condition:
                        self._unloading.pop(loaded.config.entry.name, None)
                        self._condition.notify_all()

        unload_task = asyncio.create_task(_stop_engines())
        try:
            await asyncio.shield(unload_task)
        except asyncio.CancelledError:
            with suspend_cancellation():
                await unload_task
            raise

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

    def _idle_candidates_locked(
        self, *, exclude: str | None = None
    ) -> list[LoadedModel]:
        """Loaded, non-busy models eligible for eviction, oldest-used first."""
        return sorted(
            (
                loaded
                for name, loaded in self._loaded.items()
                if name != exclude and loaded.active_requests == 0
            ),
            key=lambda item: item.last_used_at,
        )

    def _collect_idle_unloads_locked(
        self, requested_model: str, required_bytes: int
    ) -> list[LoadedModel]:
        selected: list[LoadedModel] = []
        projected_bytes = self._committed_bytes_locked()
        candidates = self._idle_candidates_locked(exclude=requested_model)

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
                specprefill_backbone_pct=config.specprefill_backbone_pct,
                specprefill_draft_model=config.specprefill_draft_model,
                prefix_trie_cache=config.prefix_trie_cache,
                prefix_trie_cache_size=config.prefix_trie_cache_size,
                prefix_trie_cache_memory_mb=config.prefix_trie_cache_memory_mb,
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
        specprefill_backbone_pct = (
            entry.specprefill_backbone_pct
            if entry.specprefill_backbone_pct is not None
            else self._defaults.specprefill_backbone_pct
        )
        specprefill_draft_model = (
            entry.specprefill_draft_model
            if entry.specprefill_draft_model is not None
            else self._defaults.specprefill_draft_model
        )
        prefix_trie_cache = (
            entry.prefix_trie_cache
            if entry.prefix_trie_cache is not None
            else self._defaults.prefix_trie_cache
        )
        prefix_trie_cache_size = (
            entry.prefix_trie_cache_size
            if entry.prefix_trie_cache_size is not None
            else self._defaults.prefix_trie_cache_size
        )
        prefix_trie_cache_memory_mb = (
            entry.prefix_trie_cache_memory_mb
            if entry.prefix_trie_cache_memory_mb is not None
            else self._defaults.prefix_trie_cache_memory_mb
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
            specprefill_backbone_pct=specprefill_backbone_pct,
            specprefill_draft_model=specprefill_draft_model,
            prefix_trie_cache=prefix_trie_cache,
            prefix_trie_cache_size=prefix_trie_cache_size,
            prefix_trie_cache_memory_mb=prefix_trie_cache_memory_mb,
            stream_interval=stream_interval,
            gpu_memory_utilization=gpu_memory_utilization,
            scheduler_config=scheduler_config,
            estimated_memory_bytes=estimated_memory_bytes,
        )
