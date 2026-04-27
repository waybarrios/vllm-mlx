# SPDX-License-Identifier: Apache-2.0
"""Model lifecycle / residency management for vllm-mlx."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .engine.base import BaseEngine, suspend_cancellation


class ResidentState(str, Enum):
    """Runtime residency state for a configured model."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable engine construction inputs for a resident model."""

    model_key: str
    model_name: str
    use_batching: bool = False
    scheduler_config: Any | None = None
    stream_interval: int = 1
    max_tokens: int = 32768
    force_mllm: bool = False
    mtp: bool = False
    prefill_step_size: int = 2048
    specprefill_enabled: bool = False
    specprefill_threshold: int = 8192
    specprefill_keep_pct: float = 0.3
    specprefill_draft_model: str | None = None


@dataclass
class ResidentModel:
    """Runtime state for a single resident model."""

    spec: ModelSpec
    state: ResidentState = ResidentState.UNLOADED
    engine: BaseEngine | None = None
    active_requests: int = 0
    last_used_at: float | None = None
    loaded_at: float | None = None
    last_error: str | None = None
    estimated_memory_bytes: int | None = None
    _load_waiters: int = field(default=0, repr=False)
    _load_waiter_task: asyncio.Task[BaseEngine] | None = field(default=None, repr=False)
    _prepare_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _abandoned_loading_task: asyncio.Task[BaseEngine] | None = field(
        default=None, repr=False
    )
    _loading_task: asyncio.Task[BaseEngine] | None = field(default=None, repr=False)
    _unloading_task: asyncio.Task[bool] | None = field(default=None, repr=False)


class ResidencyManager:
    """Single-flight lifecycle manager for resident models."""

    def __init__(
        self,
        engine_factory: Callable[[ModelSpec], Awaitable[BaseEngine]],
        *,
        on_engine_loaded: (
            Callable[[ModelSpec, BaseEngine], Awaitable[None] | None] | None
        ) = None,
        on_engine_unloading: (
            Callable[[ModelSpec, BaseEngine], Awaitable[None] | None] | None
        ) = None,
        time_fn: Callable[[], float] | None = None,
        auto_unload_idle_seconds: float = 0,
    ) -> None:
        self._engine_factory = engine_factory
        self._on_engine_loaded = on_engine_loaded
        self._on_engine_unloading = on_engine_unloading
        self._time_fn = time_fn or __import__("time").time
        self.auto_unload_idle_seconds = auto_unload_idle_seconds
        self._residents: dict[str, ResidentModel] = {}
        self._lock = asyncio.Lock()

    def register_model(self, spec: ModelSpec) -> str:
        """Register a model spec, or replace a dormant resident entry."""
        existing = self._residents.get(spec.model_key)
        if existing is not None:
            is_dormant = (
                existing.engine is None
                and existing.active_requests == 0
                and existing._load_waiters == 0
                and existing._loading_task is None
                and existing._unloading_task is None
                and existing.state in {ResidentState.UNLOADED, ResidentState.FAILED}
            )
            if not is_dormant:
                raise RuntimeError(
                    f"Cannot replace resident model '{spec.model_key}' while it is live"
                )

        self._residents[spec.model_key] = ResidentModel(spec=spec)
        return spec.model_key

    def get_engine(self, model_key: str) -> BaseEngine | None:
        """Get the currently loaded engine, if any."""
        return self._resident(model_key).engine

    def get_status(self, model_key: str) -> dict[str, Any]:
        """Return a serializable snapshot of resident state."""
        resident = self._resident(model_key)
        return {
            "model_key": resident.spec.model_key,
            "model_name": resident.spec.model_name,
            "state": resident.state.value,
            "active_requests": resident.active_requests,
            "last_used_at": resident.last_used_at,
            "loaded_at": resident.loaded_at,
            "last_error": resident.last_error,
            "estimated_memory_bytes": resident.estimated_memory_bytes,
            "auto_unload_idle_seconds": self.auto_unload_idle_seconds,
        }

    async def ensure_loaded(self, model_key: str) -> BaseEngine:
        """Load and start a resident engine if needed."""
        while True:
            task: asyncio.Task[BaseEngine] | None = None
            unloading_task: asyncio.Task[bool] | None = None

            async with self._lock:
                resident = self._resident(model_key)
                if (
                    resident.state == ResidentState.LOADED
                    and resident.engine is not None
                ):
                    return resident.engine

                if resident._unloading_task is not None:
                    unloading_task = resident._unloading_task
                else:
                    if resident._loading_task is None:
                        resident.state = ResidentState.LOADING
                        resident.last_error = None
                        resident._loading_task = asyncio.create_task(
                            self._load_engine(resident)
                        )
                        resident._load_waiters = 0
                        resident._load_waiter_task = resident._loading_task
                        resident._abandoned_loading_task = None
                    task = resident._loading_task
                    resident._load_waiters += 1
                    resident._load_waiter_task = task

            if unloading_task is not None:
                await asyncio.shield(unloading_task)
                continue

            if task is None:
                raise RuntimeError(f"No load task available for resident {model_key}")
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                cancelling = getattr(current_task, "cancelling", None)
                if (
                    task.done()
                    and task.cancelled()
                    and (cancelling is None or cancelling() == 0)
                ):
                    async with self._lock:
                        resident = self._resident(model_key)
                        if resident._abandoned_loading_task is task:
                            continue
                raise
            finally:
                await self._release_load_waiter(model_key, task)

    async def acquire(
        self,
        model_key: str,
        *,
        count_activity: bool = True,
    ) -> BaseEngine:
        """Acquire a resident engine for request processing."""
        while True:
            engine = await self.ensure_loaded(model_key)
            async with self._lock:
                resident = self._resident(model_key)
                if (
                    resident.engine is not engine
                    or resident.state != ResidentState.LOADED
                    or resident._unloading_task is not None
                ):
                    continue
                resident.active_requests += 1
                if count_activity:
                    resident.last_used_at = self._time_fn()
                return engine

    async def release(self, model_key: str, *, count_activity: bool = True) -> None:
        """Release a previously acquired resident engine."""
        async with self._lock:
            resident = self._resident(model_key)
            if resident.active_requests > 0:
                resident.active_requests -= 1
            if count_activity:
                resident.last_used_at = self._time_fn()

    async def unload_if_idle(self, model_key: str) -> bool:
        """Unload a resident engine if it has been idle past the threshold."""
        if self.auto_unload_idle_seconds <= 0:
            return False

        while True:
            unloading_task: asyncio.Task[bool] | None = None
            async with self._lock:
                resident = self._resident(model_key)

                if resident._loading_task is not None:
                    return False

                if resident._unloading_task is not None:
                    unloading_task = resident._unloading_task
                else:
                    if (
                        resident.state != ResidentState.LOADED
                        or resident.engine is None
                        or resident.active_requests > 0
                        or resident.last_used_at is None
                    ):
                        return False

                    idle_for = self._time_fn() - resident.last_used_at
                    if idle_for < self.auto_unload_idle_seconds:
                        return False

                    resident.state = ResidentState.UNLOADING
                    resident._unloading_task = asyncio.create_task(
                        self._unload_engine(resident)
                    )
                    unloading_task = resident._unloading_task

            if unloading_task is None:
                return False
            return await asyncio.shield(unloading_task)

    async def shutdown(self) -> None:
        """Stop all loaded residents."""
        keys = list(self._residents.keys())
        failures: list[str] = []
        for model_key in keys:
            while True:
                loading_task: asyncio.Task[BaseEngine] | None = None
                unloading_task: asyncio.Task[bool] | None = None

                async with self._lock:
                    resident = self._resident(model_key)

                    if resident._loading_task is not None:
                        resident._loading_task.cancel()
                        loading_task = resident._loading_task
                    elif (
                        resident.engine is None
                        or resident.state == ResidentState.UNLOADED
                    ):
                        break
                    else:
                        if resident._unloading_task is None:
                            resident.state = ResidentState.UNLOADING
                            resident._unloading_task = asyncio.create_task(
                                self._unload_engine(resident)
                            )
                        unloading_task = resident._unloading_task

                if loading_task is not None:
                    with suppress(asyncio.CancelledError):
                        await loading_task
                    continue

                if unloading_task is not None:
                    # Shield the unload so that cancelling shutdown() does not
                    # orphan a half-stopped engine in UNLOADING state.
                    try:
                        unloaded = await asyncio.shield(unloading_task)
                    except asyncio.CancelledError:
                        # Shutdown itself was cancelled — finish the in-flight
                        # unload deterministically before propagating.
                        with suspend_cancellation():
                            unloaded = await unloading_task
                        raise
                    if not unloaded:
                        async with self._lock:
                            resident = self._resident(model_key)
                            error = resident.last_error or "resident remained loaded"
                        failures.append(
                            f"Failed to unload resident model '{model_key}' during shutdown: {error}"
                        )
                        break
                    break

        if failures:
            if len(failures) == 1:
                raise RuntimeError(failures[0])
            raise RuntimeError("; ".join(failures))

    async def _load_engine(self, resident: ResidentModel) -> BaseEngine:
        """Create and start a resident engine."""
        engine: BaseEngine | None = None
        try:
            engine = await self._engine_factory(resident.spec)
            await self._prepare_engine_start(resident, engine)
            await engine.start()
            await self._run_hook(self._on_engine_loaded, resident.spec, engine)
        except asyncio.CancelledError:
            await self._cleanup_cancelled_load(resident, engine)
            raise
        except Exception as exc:
            async with self._lock:
                abandoned = resident._abandoned_loading_task is asyncio.current_task()
            if abandoned:
                await self._cleanup_cancelled_load(resident, engine)
                raise asyncio.CancelledError() from exc
            if engine is not None:
                with suppress(Exception):
                    await engine.stop()
            async with self._lock:
                resident.state = ResidentState.FAILED
                resident.last_error = str(exc)
                resident._abandoned_loading_task = None
                resident._loading_task = None
            raise

        try:
            async with self._lock:
                resident.engine = engine
                resident.state = ResidentState.LOADED
                resident.loaded_at = self._time_fn()
                resident.last_used_at = resident.loaded_at
                resident.last_error = None
                resident._abandoned_loading_task = None
                resident._loading_task = None
        except asyncio.CancelledError:
            await self._cleanup_cancelled_load(resident, engine)
            raise

        return engine

    async def _unload_engine(self, resident: ResidentModel) -> bool:
        """Stop and drop a resident engine."""
        engine = resident.engine
        if engine is None:
            async with self._lock:
                resident.state = ResidentState.UNLOADED
                resident._unloading_task = None
            return False

        try:
            await self._run_hook(self._on_engine_unloading, resident.spec, engine)
            await engine.stop()
        except asyncio.CancelledError:
            async with self._lock:
                resident.state = ResidentState.LOADED
                resident._unloading_task = None
            raise
        except Exception as exc:
            async with self._lock:
                resident.engine = engine
                resident.state = ResidentState.LOADED
                resident.last_error = str(exc)
                resident._unloading_task = None
            return False

        async with self._lock:
            resident.engine = None
            resident.state = ResidentState.UNLOADED
            resident.loaded_at = None
            resident.last_error = None
            resident._unloading_task = None

        return True

    def _resident(self, model_key: str) -> ResidentModel:
        try:
            return self._residents[model_key]
        except KeyError as exc:
            raise KeyError(f"Resident model '{model_key}' is not registered") from exc

    async def _run_hook(
        self,
        hook: Callable[[ModelSpec, BaseEngine], Awaitable[None] | None] | None,
        spec: ModelSpec,
        engine: BaseEngine,
    ) -> None:
        if hook is None:
            return

        result = hook(spec, engine)
        if inspect.isawaitable(result):
            await result

    async def _prepare_engine_start(
        self,
        resident: ResidentModel,
        engine: BaseEngine,
    ) -> None:
        """Run blocking startup work away from the serving event loop."""
        prepare_for_start = getattr(engine, "prepare_for_start", None)
        if prepare_for_start is None:
            return

        uses_default_prepare = getattr(engine, "_uses_default_prepare_for_start", None)
        if callable(uses_default_prepare) and uses_default_prepare():
            # Keep default engine prepare on the event-loop thread so MLX
            # thread-local stream ownership matches subsequent streaming calls.
            prepare_for_start()
            return

        prepare_task = asyncio.create_task(asyncio.to_thread(prepare_for_start))
        async with self._lock:
            resident._prepare_task = prepare_task

        try:
            await asyncio.shield(prepare_task)
        except asyncio.CancelledError:
            with suspend_cancellation():
                while not prepare_task.done():
                    try:
                        await asyncio.shield(prepare_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            raise
        finally:
            async with self._lock:
                if resident._prepare_task is prepare_task:
                    resident._prepare_task = None

    async def _cleanup_cancelled_load(
        self,
        resident: ResidentModel,
        engine: BaseEngine | None,
    ) -> None:
        """Stop a partially loaded engine and unwind resident state."""
        with suspend_cancellation():
            if engine is not None:
                with suppress(Exception):
                    await engine.stop()
            async with self._lock:
                resident.engine = None
                resident.state = ResidentState.UNLOADED
                resident.loaded_at = None
                resident.last_error = None
                # Keep the abandoned-load marker until a new load task replaces it
                # so late waiters on the old task can still recognize a retryable
                # cancellation instead of inheriting CancelledError.
                resident._loading_task = None

    async def _release_load_waiter(
        self,
        model_key: str,
        task: asyncio.Task[BaseEngine],
    ) -> None:
        """Drop one waiter from a shared load, canceling abandoned solo loads."""
        task_to_cancel: asyncio.Task[BaseEngine] | None = None

        async with self._lock:
            resident = self._resident(model_key)
            if resident._load_waiter_task is not task or resident._load_waiters <= 0:
                return

            resident._load_waiters -= 1
            if resident._load_waiters == 0:
                resident._load_waiter_task = None
                if resident._loading_task is task and not task.done():
                    resident._abandoned_loading_task = task
                    task_to_cancel = task

        if task_to_cancel is None:
            return

        with suspend_cancellation():
            task_to_cancel.cancel()
            with suppress(asyncio.CancelledError):
                await task_to_cancel
