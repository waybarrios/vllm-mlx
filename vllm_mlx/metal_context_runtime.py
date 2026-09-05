# SPDX-License-Identifier: Apache-2.0
"""Phase-two page ownership runtime for the optional Metal Context Engine.

This module is deliberately below the scheduler boundary.  It owns physical
KV pages, request page tables, prefix references, copy-on-write, and the
low-level decode call, but it does not know about HTTP requests, sampling, or
continuous batching.

The runtime has two explicit execution modes:

``native``
    Require the optional :mod:`vllm_mlx._metal_context` extension and dispatch
    its compiled paged-decode kernel.  A missing capability or a native
    dispatch error is surfaced to the caller; there is no implicit fallback.

``numpy``
    Use the checked-in NumPy oracle for lifecycle and ownership tests on
    machines without Metal.  This mode is intentionally explicit and is not a
    serving fallback.

In native mode this class is an adapter around the compiled
``_metal_context.PageRuntime``.  It does not allocate a second Python KV pool
or maintain a parallel native page table.

The page storage uses the native foundation's logical layout, one physical
page shared by all layers:

``[page, layer, kv_head, block_offset, head_dim]`` (BF16 bits as ``uint16``)

``append_kv`` accepts ``[tokens, kv_heads, head_dim]`` values.  Prefixes are
immutable views of page chains.  If a request writes into a page referenced by
another request or prefix, the complete page is copied before the write.
Persistence/snapshot support is intentionally not implemented in this
package; that is the next PR-stack package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import importlib
import math
import platform
import threading
from typing import (
    Any,
    Callable,
    Concatenate,
    Literal,
    Mapping,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from .attention_backend import (
    AttentionGeometry,
    BackendCapabilities,
    ContextBackend,
    METAL_CONTEXT_ABI_VERSION,
    NATIVE_EXTENSION_MODULE,
    PageHandle,
    PrefixHandle,
    RequestHandle,
    SnapshotMetadata,
    numpy_paged_decode_attention,
)

ExecutionMode = Literal["native", "numpy"]
_INT32_MAX = int(np.iinfo(np.int32).max)
_P = ParamSpec("_P")
_Self = TypeVar("_Self", bound="_LockOwner")
_Return = TypeVar("_Return")


class _LockOwner(Protocol):
    _lock: threading.RLock


def _locked(
    method: Callable[Concatenate[_Self, _P], _Return],
) -> Callable[Concatenate[_Self, _P], _Return]:
    """Serialize all mutable page-runtime operations with the instance lock."""

    def wrapper(self: _Self, *args: _P.args, **kwargs: _P.kwargs) -> _Return:
        with self._lock:
            return method(self, *args, **kwargs)

    # ``functools.wraps`` preserves the public method metadata, while the
    # typeshed wrapper protocol cannot express the ParamSpec-preserving
    # decorator result precisely on every supported Python version.
    wrapped = wraps(method)(wrapper)
    return cast(Callable[Concatenate[_Self, _P], _Return], wrapped)


class MetalContextRuntimeError(RuntimeError):
    """Base error for page-runtime lifecycle failures."""


class MetalContextCapabilityError(MetalContextRuntimeError):
    """Raised when explicit native execution cannot be used."""


@dataclass(slots=True)
class _PageState:
    refcount: int = 0
    last_used: int = 0


@dataclass(slots=True)
class _RequestState:
    request_id: str
    max_tokens: int
    pages: list[int] = field(default_factory=list)
    layer_lengths: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return max(self.layer_lengths, default=0)


@dataclass(slots=True, frozen=True)
class _PrefixState:
    pages: tuple[int, ...]
    token_count: int


def _native_capabilities(
    native_module: Any,
    *,
    execution: ExecutionMode,
) -> BackendCapabilities:
    """Validate and normalize the small native capability ABI."""

    probe = getattr(native_module, "capabilities", None)
    if not callable(probe):
        raise MetalContextCapabilityError(
            f"{NATIVE_EXTENSION_MODULE} does not expose capabilities()"
        )
    try:
        raw = probe()
    except Exception as exc:  # pragma: no cover - native-only failures
        detail = str(exc).strip() or exc.__class__.__name__
        raise MetalContextCapabilityError(
            f"the native capability probe failed: {detail}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise MetalContextCapabilityError(
            "the native capability probe returned a non-mapping result"
        )

    abi = raw.get("abi_version")
    abi_version = abi if isinstance(abi, int) and not isinstance(abi, bool) else None
    advertised = bool(raw.get("available", False))
    metal_device = bool(raw.get("metal_device", advertised))
    compiled = bool(raw.get("compiled", True))
    reasons: list[str] = []
    if not compiled:
        reasons.append("the optional native extension is not compiled")
    if not metal_device:
        reasons.append("no usable Metal device was reported")
    if abi_version != METAL_CONTEXT_ABI_VERSION:
        reasons.append(
            f"native ABI {abi_version!r} does not match required ABI "
            f"{METAL_CONTEXT_ABI_VERSION}"
        )
    if not advertised:
        reason = raw.get("reason")
        if isinstance(reason, str) and reason.strip():
            reasons.append(reason.strip())
        else:
            reasons.append("the native extension did not advertise availability")

    if execution == "native" and (
        not compiled
        or not advertised
        or not metal_device
        or abi_version != METAL_CONTEXT_ABI_VERSION
    ):
        detail = "; ".join(dict.fromkeys(reasons))
        raise MetalContextCapabilityError(
            "the Metal Context page runtime was explicitly configured for native "
            f"execution but is unavailable: {detail}"
        )

    reason_value = raw.get("reason")
    reason = (
        reason_value.strip()
        if isinstance(reason_value, str) and reason_value.strip()
        else None
    )
    if reasons and not advertised:
        reason = "; ".join(dict.fromkeys(reasons))
    return BackendCapabilities(
        platform="darwin",
        native_extension=compiled,
        metal_device=metal_device,
        abi_version=abi_version,
        available=(
            advertised
            and compiled
            and metal_device
            and abi_version == METAL_CONTEXT_ABI_VERSION
        ),
        # The foundation intentionally reports serving_ready=False until the
        # scheduler executor is qualified.  This page runtime is a component,
        # not a claim that production serving is ready.
        serving_ready=bool(raw.get("serving_ready", False)),
        reason=reason,
        architecture=platform.machine().lower(),
        probed=True,
    )


def _oracle_capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        platform=platform.system().lower(),
        native_extension=False,
        metal_device=False,
        abi_version=None,
        available=False,
        serving_ready=False,
        reason="explicit NumPy oracle execution; native dispatch is disabled",
        architecture=platform.machine().lower(),
        probed=False,
    )


def _bf16_bits(value: Any, *, name: str) -> NDArray[np.uint16]:
    """Convert finite values to the native BF16 upper-bit representation."""

    array = np.asarray(value)
    if array.dtype.kind == "u" and array.dtype.itemsize == 2:
        if array.dtype.byteorder not in ("=", "<", "|"):
            raise ValueError(f"{name} must use native-endian uint16 BF16 storage")
        result: NDArray[np.uint16] = np.ascontiguousarray(array)
        exponent_all_ones = (result & np.uint16(0x7F80)) == np.uint16(0x7F80)
        if bool(np.any(exponent_all_ones)):
            raise ValueError(f"{name} must contain only finite BF16 values")
        return result

    try:
        float_values = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric BF16-compatible data") from exc
    if not bool(np.isfinite(float_values).all()):
        raise ValueError(f"{name} must contain only finite values")
    # Truncation matches the native foundation's documented conversion.  It is
    # deterministic and avoids a hidden float16/float32 storage mode.
    result = np.ascontiguousarray(
        (float_values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    )
    return result


def _query_bits(
    value: Any, *, expected_heads: int, head_dim: int
) -> tuple[np.ndarray, bool]:
    array = _bf16_bits(value, name="query")
    if array.ndim == 2:
        if array.shape != (expected_heads, head_dim):
            raise ValueError(
                "query must have shape [query_heads, 128] or " "[1, query_heads, 128]"
            )
        return array[None, ...], True
    if array.ndim == 3 and array.shape == (1, expected_heads, head_dim):
        return array, False
    raise ValueError(
        "query must have shape [query_heads, 128] or [1, query_heads, 128]"
    )


class MetalContextPageRuntime(ContextBackend):
    """Own phase-two physical KV pages and dispatch phase-one attention.

    Args:
        geometry: Supported Qwen dense/MoE attention geometry.
        max_pages: Number of preallocated physical pages.
        execution: ``"native"`` requires the compiled extension.  ``"numpy"``
            is an explicit oracle mode for portable lifecycle tests.
        native_module: Optional injected module implementing the foundation
            ``capabilities()``, ``paged_decode()``, and ``shutdown()`` ABI.

    The class is intentionally not connected to a scheduler.  Requests are
    opaque integer handles and are never serialized to disk.
    """

    # These fields are deliberately only materialized by the NumPy oracle.
    # Class annotations keep the oracle-only methods type-checkable without
    # creating a second lifecycle authority on native instances.
    _keys: np.ndarray
    _values: np.ndarray
    _pages: list[_PageState]
    _free_pages: set[int]
    _requests: dict[int, _RequestState]
    _prefixes: dict[int, _PrefixState]
    _released_requests: set[int]
    _released_prefixes: set[int]
    _next_request: int
    _next_prefix: int
    _clock: int
    _request_ids: set[str]
    _counters: dict[str, int | float]

    def __init__(
        self,
        geometry: AttentionGeometry,
        *,
        max_pages: int,
        execution: ExecutionMode = "native",
        native_module: Any | None = None,
        max_blocks_per_request: int | None = None,
        max_requests: int = 64,
    ) -> None:
        geometry.validate()
        if execution not in ("native", "numpy"):
            raise ValueError("execution must be 'native' or 'numpy'")
        if (
            not isinstance(max_pages, int)
            or isinstance(max_pages, bool)
            or max_pages <= 0
        ):
            raise ValueError("max_pages must be a positive integer")
        if max_pages > _INT32_MAX:
            raise ValueError("max_pages must not exceed INT32_MAX")
        if max_blocks_per_request is None:
            max_blocks_per_request = max_pages
        if (
            not isinstance(max_blocks_per_request, int)
            or isinstance(max_blocks_per_request, bool)
            or max_blocks_per_request <= 0
        ):
            raise ValueError("max_blocks_per_request must be a positive integer")
        if max_blocks_per_request > _INT32_MAX // geometry.block_size:
            raise ValueError(
                "max_blocks_per_request * block_size exceeds the int32 sequence "
                "limit"
            )
        if (
            not isinstance(max_requests, int)
            or isinstance(max_requests, bool)
            or max_requests <= 0
        ):
            raise ValueError("max_requests must be a positive integer")

        self.geometry = geometry
        self.max_pages = max_pages
        self.max_blocks_per_request = max_blocks_per_request
        self.max_requests = max_requests
        self.execution: ExecutionMode = execution
        self._lock = threading.RLock()
        self._native_module = native_module
        self._native_runtime: Any | None = None
        if execution == "native":
            if native_module is None:
                try:
                    native_module = importlib.import_module(NATIVE_EXTENSION_MODULE)
                except (ImportError, OSError) as exc:
                    detail = str(exc).strip() or exc.__class__.__name__
                    raise MetalContextCapabilityError(
                        "the Metal Context page runtime was explicitly configured "
                        f"for native execution but {NATIVE_EXTENSION_MODULE} "
                        f"could not be loaded: {detail}"
                    ) from exc
            self._native_module = native_module
            self._capabilities = _native_capabilities(
                native_module, execution=execution
            )
            page_runtime_type = getattr(native_module, "PageRuntime", None)
            if not callable(page_runtime_type):
                raise MetalContextCapabilityError(
                    f"{NATIVE_EXTENSION_MODULE} does not expose PageRuntime"
                )
            try:
                self._native_runtime = page_runtime_type(
                    num_layers=geometry.num_layers,
                    num_attention_heads=geometry.num_attention_heads,
                    num_key_value_heads=geometry.num_key_value_heads,
                    head_dim=geometry.head_dim,
                    block_size=geometry.block_size,
                    max_pages=max_pages,
                    max_blocks_per_request=max_blocks_per_request,
                    max_requests=max_requests,
                )
            except (MemoryError, ValueError, RuntimeError):
                raise
            except Exception as exc:  # pragma: no cover - native-only failures
                detail = str(exc).strip() or exc.__class__.__name__
                raise MetalContextRuntimeError(
                    f"could not construct the native page runtime: {detail}"
                ) from exc
        else:
            self._capabilities = _oracle_capabilities()

        self._shutdown = False
        if execution == "numpy":
            # The arrays are intentionally page-major and layer-major to make
            # the page table passed to the oracle a direct physical index.
            # Native mode never creates these parallel Python-owned buffers.
            storage_shape = (
                max_pages,
                geometry.num_layers,
                geometry.num_key_value_heads,
                geometry.block_size,
                geometry.head_dim,
            )
            try:
                self._keys = np.zeros(storage_shape, dtype=np.uint16)
                self._values = np.zeros(storage_shape, dtype=np.uint16)
            except (MemoryError, ValueError) as exc:
                raise MemoryError(
                    "Metal Context page runtime could not allocate its preallocated "
                    f"KV storage for {max_pages} pages"
                ) from exc

            self._pages = [_PageState() for _ in range(max_pages)]
            self._free_pages: set[int] = set(range(max_pages))
            self._requests: dict[int, _RequestState] = {}
            self._prefixes: dict[int, _PrefixState] = {}
            self._released_requests: set[int] = set()
            self._released_prefixes: set[int] = set()
            self._next_request = 1
            self._next_prefix = 1
            self._clock = 0
            self._request_ids: set[str] = set()
            self._counters: dict[str, int | float] = {
                "pages_allocated": 0,
                "pages_freed": 0,
                "append_tokens": 0,
                # ``dispatches``/``dispatch_failures`` are the stable totals
                # for this backend.  The mode-specific fields make the
                # source of the total explicit and are kept in parity with
                # native metrics.
                "dispatches": 0,
                "dispatch_failures": 0,
                "native_dispatches": 0,
                "native_failures": 0,
                "oracle_dispatches": 0,
                "oracle_failures": 0,
                "evictions": 0,
                "cow_events": 0,
                "oom_events": 0,
                "release_calls": 0,
                "released_requests": 0,
                "cancellations": 0,
                "prefix_attaches": 0,
                "prefix_forks": 0,
                "prefix_tokens_attached": 0,
                "attention_validation_bytes": 0,
                "metadata_bytes": 0,
                "decode_page_resolution_checks": 0,
                "snapshot_failures": 0,
                "restore_failures": 0,
            }

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @_locked
    def validate(self, geometry: AttentionGeometry) -> None:
        geometry.validate()
        if geometry != self.geometry:
            raise ValueError(
                "page runtime geometry does not match its preallocated storage"
            )

    def _ensure_open(self) -> None:
        if self._shutdown:
            raise MetalContextRuntimeError("Metal Context page runtime is shut down")

    @staticmethod
    def _handle(value: Any, *, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise KeyError(f"unknown {name} handle {value!r}")
        return int(value)

    def _request(self, request: RequestHandle) -> tuple[int, _RequestState]:
        request_id = self._handle(request, name="request")
        state = self._requests.get(request_id)
        if state is None:
            if request_id in self._released_requests:
                raise KeyError(f"request handle {request_id} was released")
            raise KeyError(f"unknown request handle {request_id}")
        return request_id, state

    def _prefix(self, prefix: PrefixHandle) -> tuple[int, _PrefixState]:
        prefix_id = self._handle(prefix, name="prefix")
        state = self._prefixes.get(prefix_id)
        if state is None:
            if prefix_id in self._released_prefixes:
                raise KeyError(f"prefix handle {prefix_id} was released")
            raise KeyError(f"unknown prefix handle {prefix_id}")
        return prefix_id, state

    def _native_request_id(self, request: RequestHandle) -> int:
        """Validate only the immutable Python representation of a handle.

        Generation/slot liveness is owned by the compiled runtime.  Keeping a
        second Python request map here would create a stale ownership authority.
        """

        return self._handle(request, name="request")

    def _native_prefix_id(self, prefix: PrefixHandle) -> int:
        """Validate only the immutable Python representation of a handle."""

        return self._handle(prefix, name="prefix")

    def _touch(self, page_id: int) -> None:
        self._clock += 1
        self._pages[page_id].last_used = self._clock

    def _plan_page_ids(self, count: int, *, record_oom: bool = True) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("page count cannot be negative")
        free = sorted(self._free_pages)
        if count <= len(free):
            return tuple(free[:count])
        evictable = sorted(
            (
                state.last_used,
                page_id,
            )
            for page_id, state in enumerate(self._pages)
            if state.refcount == 0 and page_id not in self._free_pages
        )
        needed = count - len(free)
        if needed > len(evictable):
            if record_oom:
                self._counters["oom_events"] += 1
            raise MemoryError(
                "Metal Context page runtime is out of pages: "
                f"requested {count}, available {len(free) + len(evictable)}"
            )
        return tuple(free + [page_id for _, page_id in evictable[:needed]])

    def _reserve_pages(
        self,
        count: int,
        *,
        page_ids: tuple[int, ...] | None = None,
        record_oom: bool = True,
    ) -> tuple[int, ...]:
        selected = (
            self._plan_page_ids(count, record_oom=record_oom)
            if page_ids is None
            else page_ids
        )
        if len(selected) != count:
            raise MetalContextRuntimeError("page reservation plan has wrong size")
        # Deterministic allocation makes page-table and COW tests reproducible.
        for page_id in selected:
            if page_id not in self._free_pages:
                if self._pages[page_id].refcount != 0:
                    raise MetalContextRuntimeError(
                        "page reservation selected a referenced page"
                    )
                self._counters["evictions"] += 1
                self._counters["pages_freed"] += 1
            else:
                self._free_pages.remove(page_id)
            self._pages[page_id] = _PageState(refcount=1, last_used=0)
            self._keys[page_id].fill(0)
            self._values[page_id].fill(0)
            self._touch(page_id)
        self._counters["pages_allocated"] += count
        return selected

    @_locked
    def allocate_request(self, request_id: str, *, max_tokens: int) -> RequestHandle:
        self._ensure_open()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if self.execution == "numpy" and request_id in self._request_ids:
            raise ValueError(f"request_id {request_id!r} is already active")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens > _INT32_MAX:
            raise ValueError("max_tokens must not exceed INT32_MAX")
        capacity = self.max_blocks_per_request * self.geometry.block_size
        if max_tokens > capacity:
            raise ValueError(
                "max_tokens must be no greater than the configured page-table "
                f"capacity ({capacity})"
            )
        if self.execution == "native":
            assert self._native_runtime is not None
            handle = self._native_runtime.allocate_request(
                request_id=request_id, max_tokens=max_tokens
            )
            # The compiled runtime publishes a valid generation/slot handle
            # atomically.  No Python lifecycle entry is created after this
            # call, so there is no adapter state that can diverge on OOM.
            handle_id = self._handle(handle, name="request")
            return RequestHandle(handle_id)
        if len(self._requests) >= self.max_requests:
            raise MemoryError("request capacity exhausted")
        handle = self._next_request
        self._next_request += 1
        self._requests[handle] = _RequestState(
            request_id=request_id,
            max_tokens=max_tokens,
            layer_lengths=[0] * self.geometry.num_layers,
        )
        self._request_ids.add(request_id)
        return RequestHandle(handle)

    @_locked
    def allocate_pages(
        self, request: RequestHandle, count: int
    ) -> tuple[PageHandle, ...]:
        self._ensure_open()
        if self.execution == "native":
            assert self._native_runtime is not None
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("page count must be a non-negative integer")
            request_id = self._native_request_id(request)
            if count == 0:
                return ()
            pages = self._native_runtime.allocate_pages(request=request_id, count=count)
            if len(pages) != count:
                raise MetalContextRuntimeError(
                    "native page runtime returned an unexpected page count"
                )
            if not isinstance(pages, tuple) or not all(
                isinstance(page, int) and not isinstance(page, bool) and page >= 0
                for page in pages
            ):
                raise MetalContextRuntimeError(
                    "native page runtime returned malformed page handles"
                )
            # The compiled bridge returns an immutable tuple.  Reuse it
            # directly so no Python lifecycle bookkeeping follows mutation.
            typed_pages: tuple[PageHandle, ...] = pages
            return typed_pages
        _, state = self._request(request)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("page count must be a non-negative integer")
        if count == 0:
            return ()
        capacity = math.ceil(state.max_tokens / self.geometry.block_size)
        if len(state.pages) + count > capacity:
            raise ValueError(
                "requested pages exceed the request max_tokens capacity "
                f"({capacity} pages)"
            )
        page_ids = self._reserve_pages(count)
        state.pages.extend(page_ids)
        return tuple(PageHandle(page_id) for page_id in page_ids)

    def _normalize_kv_arrays(
        self, layer: int, keys: Any, values: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise ValueError("layer must be an integer")
        if layer < 0 or layer >= self.geometry.num_layers:
            raise ValueError(f"layer must be in [0, {self.geometry.num_layers})")
        key_bits = _bf16_bits(keys, name="keys")
        value_bits = _bf16_bits(values, name="values")
        expected_tail = (
            self.geometry.num_key_value_heads,
            self.geometry.head_dim,
        )
        if key_bits.ndim != 3 or key_bits.shape[1:] != expected_tail:
            raise ValueError("keys must have shape [tokens, num_key_value_heads, 128]")
        if value_bits.shape != key_bits.shape:
            raise ValueError("keys and values must have identical shapes")
        return key_bits, value_bits

    def _normalize_append(
        self, request: _RequestState, layer: int, keys: Any, values: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        key_bits, value_bits = self._normalize_kv_arrays(layer, keys, values)
        if request.layer_lengths[layer] + key_bits.shape[0] > request.max_tokens:
            raise ValueError("append would exceed request max_tokens")
        return key_bits, value_bits

    def _cow_for_range(
        self, request: _RequestState, start: int, token_count: int
    ) -> None:
        if token_count == 0:
            return
        first_page = start // self.geometry.block_size
        last_page = (start + token_count - 1) // self.geometry.block_size
        if last_page >= len(request.pages):
            missing = last_page + 1 - len(request.pages)
            capacity = math.ceil(request.max_tokens / self.geometry.block_size)
            if len(request.pages) + missing > capacity:
                raise MemoryError(
                    "Metal Context page runtime has no page capacity for append"
                )
        else:
            missing = 0
        shared_positions = [
            position
            for position in range(first_page, min(last_page + 1, len(request.pages)))
            if self._pages[request.pages[position]].refcount > 1
        ]
        needed = missing + len(shared_positions)
        if needed == 0:
            return
        # Plan every missing and COW page before mutating free lists,
        # references, page tables, data, or metrics.  This is important for a
        # request that needs one new page and one COW page when only one free
        # page remains.
        planned = self._plan_page_ids(needed, record_oom=False)
        missing_ids = planned[:missing]
        cow_ids = planned[missing:]
        staged_copies = [
            (
                np.array(self._keys[request.pages[position]], copy=True),
                np.array(self._values[request.pages[position]], copy=True),
            )
            for position in shared_positions
        ]
        self._reserve_pages(needed, page_ids=planned, record_oom=False)
        new_pages = list(request.pages) + list(missing_ids)
        for position, new_page_id, (key_copy, value_copy) in zip(
            shared_positions, cow_ids, staged_copies
        ):
            old_page_id = request.pages[position]
            self._keys[new_page_id] = key_copy
            self._values[new_page_id] = value_copy
            new_pages[position] = new_page_id
            self._pages[old_page_id].refcount -= 1
            self._touch(new_page_id)
        request.pages[:] = new_pages
        self._counters["cow_events"] += len(shared_positions)

    @_locked
    def append_kv(
        self,
        request: RequestHandle,
        layer: int,
        keys: Any,
        values: Any,
    ) -> None:
        self._ensure_open()
        if self.execution == "native":
            key_bits, value_bits = self._normalize_kv_arrays(layer, keys, values)
            assert self._native_runtime is not None
            request_id = self._native_request_id(request)
            self._native_runtime.append_kv(
                request=request_id,
                layer=layer,
                keys=key_bits,
                values=value_bits,
            )
            return
        _, state = self._request(request)
        key_bits, value_bits = self._normalize_append(state, layer, keys, values)
        start = state.layer_lengths[layer]
        token_count = int(key_bits.shape[0])
        first_page = start // self.geometry.block_size
        last_page = (
            (start + token_count - 1) // self.geometry.block_size
            if token_count
            else first_page
        )
        existing_positions = range(first_page, min(last_page + 1, len(state.pages)))
        missing = max(0, last_page + 1 - len(state.pages)) if token_count else 0
        shared = [
            position
            for position in existing_positions
            if self._pages[state.pages[position]].refcount > 1
        ]
        needed = missing + len(shared)
        capacity = math.ceil(state.max_tokens / self.geometry.block_size)
        if len(state.pages) + missing > capacity:
            raise MemoryError(
                "Metal Context page runtime has no page capacity for append"
            )
        # Preflight all page capacity before changing any ownership, data, or
        # metrics.  In particular, a shared one-token prefix followed by a
        # 17-token append needs both a new tail page and a COW page.
        planned = self._plan_page_ids(needed, record_oom=False)
        affected = {state.pages[position] for position in existing_positions}
        affected.update(planned)
        page_state_snapshot = {
            page_id: _PageState(
                refcount=self._pages[page_id].refcount,
                last_used=self._pages[page_id].last_used,
            )
            for page_id in affected
        }
        data_snapshot = {
            page_id: (
                np.array(self._keys[page_id], copy=True),
                np.array(self._values[page_id], copy=True),
            )
            for page_id in affected
        }
        free_snapshot = set(self._free_pages)
        counters_snapshot = dict(self._counters)
        clock_snapshot = self._clock
        pages_snapshot = list(state.pages)
        lengths_snapshot = list(state.layer_lengths)
        try:
            self._cow_for_range(state, start, token_count)

            remaining = token_count
            cursor = 0
            while remaining:
                token_position = start + cursor
                page_position = token_position // self.geometry.block_size
                page_offset = token_position % self.geometry.block_size
                tokens = min(remaining, self.geometry.block_size - page_offset)
                page_id = state.pages[page_position]
                self._keys[page_id, layer, :, page_offset : page_offset + tokens, :] = (
                    np.transpose(key_bits[cursor : cursor + tokens], (1, 0, 2))
                )
                self._values[
                    page_id, layer, :, page_offset : page_offset + tokens, :
                ] = np.transpose(value_bits[cursor : cursor + tokens], (1, 0, 2))
                self._touch(page_id)
                cursor += tokens
                remaining -= tokens
            state.layer_lengths[layer] += token_count
            self._counters["append_tokens"] += token_count
        except BaseException:
            # Restore every observable part of the failed append, including
            # evictions, refcounts, page-table ownership, storage bytes, and
            # counters.  The preflight makes this path rare; it is retained so
            # an unexpected buffer/assignment failure cannot expose a partial
            # request state.
            state.pages[:] = pages_snapshot
            state.layer_lengths[:] = lengths_snapshot
            self._free_pages = free_snapshot
            self._clock = clock_snapshot
            self._counters.clear()
            self._counters.update(counters_snapshot)
            for page_id, page_state in page_state_snapshot.items():
                self._pages[page_id] = page_state
                self._keys[page_id] = data_snapshot[page_id][0]
                self._values[page_id] = data_snapshot[page_id][1]
            raise

    def _page_table(self, state: _RequestState, token_count: int) -> NDArray[np.int32]:
        blocks = math.ceil(token_count / self.geometry.block_size)
        if blocks > len(state.pages):
            raise ValueError(
                "request page table is shorter than its KV sequence; allocate_pages "
                "before decoding"
            )
        table: NDArray[np.int32] = np.full((1, blocks), -1, dtype=np.int32)
        if blocks:
            table[0, :blocks] = np.asarray(state.pages[:blocks], dtype=np.int32)
        return table

    def paged_decode_attention(
        self, request: RequestHandle, layer: int, query: Any
    ) -> Any:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise ValueError("layer must be an integer")
        if layer < 0 or layer >= self.geometry.num_layers:
            raise ValueError(f"layer must be in [0, {self.geometry.num_layers})")
        query_bits, squeeze = _query_bits(
            query,
            expected_heads=self.geometry.num_attention_heads,
            head_dim=self.geometry.head_dim,
        )
        if self.execution == "native":
            return self._paged_decode_native(
                request, layer, query_bits, squeeze=squeeze
            )
        with self._lock:
            self._ensure_open()
            _, state = self._request(request)
            token_count = state.layer_lengths[layer]
            if token_count == 0:
                output: NDArray[np.float32] = np.zeros(
                    (1, self.geometry.num_attention_heads, self.geometry.head_dim),
                    dtype=np.float32,
                )
                return output[0] if squeeze else output

            table = self._page_table(state, token_count)
            lengths = np.asarray([token_count], dtype=np.int32)
            key_pages = np.ascontiguousarray(self._keys[:, layer])
            value_pages = np.ascontiguousarray(self._values[:, layer])
            scale = 1.0 / math.sqrt(self.geometry.head_dim)

            try:
                output = numpy_paged_decode_attention(
                    query_bits,
                    key_pages,
                    value_pages,
                    table,
                    lengths,
                    block_size=self.geometry.block_size,
                    num_kv_heads=self.geometry.num_key_value_heads,
                    scale=scale,
                )
            except Exception:
                self._counters["oracle_failures"] += 1
                self._counters["dispatch_failures"] += 1
                raise
            self._counters["oracle_dispatches"] += 1
            self._counters["dispatches"] += 1
            # Decode is a read reference and therefore makes the page recently
            # used, but it never changes ownership or page contents.
            for page_id in state.pages[
                : math.ceil(token_count / self.geometry.block_size)
            ]:
                self._touch(page_id)
            return output[0] if squeeze else output

    def _paged_decode_native(
        self,
        request: RequestHandle,
        layer: int,
        query_bits: np.ndarray,
        *,
        squeeze: bool,
    ) -> Any:
        """Dispatch native decode outside the Python lifecycle lock.

        The compiled runtime owns its own request/page/dispatch locks and
        tracks in-flight dispatches through shutdown.  Python only protects
        the short handle validation/capture section; holding ``self._lock``
        across the native call would serialize otherwise independent requests.
        """

        with self._lock:
            self._ensure_open()
            assert self._native_runtime is not None
            request_id = self._native_request_id(request)
            native_runtime = self._native_runtime

        raw_output = native_runtime.paged_decode(
            request=request_id,
            layer=layer,
            query=query_bits[0],
            scale=1.0 / math.sqrt(self.geometry.head_dim),
        )
        expected = (
            self.geometry.num_attention_heads
            * self.geometry.head_dim
            * np.dtype(np.float32).itemsize
        )
        if not isinstance(raw_output, (bytes, bytearray, memoryview)):
            raise MetalContextRuntimeError(
                "native page runtime paged_decode returned a non-bytes result"
            )
        if len(raw_output) != expected:
            raise MetalContextRuntimeError(
                "native page runtime paged_decode returned an unexpected "
                f"output length: expected {expected}, got {len(raw_output)}"
            )
        output = np.frombuffer(raw_output, dtype=np.float32).reshape(
            (1, self.geometry.num_attention_heads, self.geometry.head_dim)
        )
        return output[0] if squeeze else output

    @_locked
    def create_prefix(
        self, request: RequestHandle, *, token_count: int | None = None
    ) -> PrefixHandle:
        """Create an immutable shared prefix view from a live request."""

        self._ensure_open()
        if self.execution == "native":
            assert self._native_runtime is not None
            request_id = self._native_request_id(request)
            if token_count is not None and (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count <= 0
            ):
                raise ValueError(
                    "token_count must equal the request's fully populated "
                    "positive sequence length"
                )
            native_length = int(self._native_runtime.sequence_length(request_id))
            if token_count is not None and token_count != native_length:
                raise ValueError(
                    "token_count must equal the request's fully populated "
                    "sequence length"
                )
            prefix = self._native_runtime.create_prefix(request_id)
            prefix_id = self._handle(prefix, name="prefix")
            return PrefixHandle(prefix_id)
        _, state = self._request(request)
        if (
            not state.layer_lengths
            or state.layer_lengths[0] == 0
            or len(set(state.layer_lengths)) > 1
        ):
            raise ValueError(
                "prefix creation requires all layers to have equal, non-zero "
                "populated lengths"
            )
        if token_count is None:
            token_count = state.length
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise ValueError(
                "token_count must equal the request's fully populated "
                "positive sequence length"
            )
        if token_count != state.length:
            raise ValueError(
                "token_count must equal the request's fully populated "
                "sequence length"
            )
        blocks = math.ceil(token_count / self.geometry.block_size)
        if blocks > len(state.pages):
            raise ValueError("request has no pages for the requested prefix")
        pages = tuple(state.pages[:blocks])
        for page_id in pages:
            self._pages[page_id].refcount += 1
            self._touch(page_id)
        prefix_id = self._next_prefix
        self._next_prefix += 1
        self._prefixes[prefix_id] = _PrefixState(pages=pages, token_count=token_count)
        return PrefixHandle(prefix_id)

    @_locked
    def attach_prefix(self, request: RequestHandle, prefix: PrefixHandle) -> None:
        self._ensure_open()
        if self.execution == "native":
            assert self._native_runtime is not None
            request_id = self._native_request_id(request)
            prefix_id = self._native_prefix_id(prefix)
            self._native_runtime.attach_prefix(request_id, prefix_id)
            return
        _, request_state = self._request(request)
        _, prefix_state = self._prefix(prefix)
        if request_state.pages or any(request_state.layer_lengths):
            raise ValueError("prefixes can only be attached to an empty request")
        required_blocks = len(prefix_state.pages)
        capacity = math.ceil(request_state.max_tokens / self.geometry.block_size)
        if required_blocks > capacity:
            raise ValueError("prefix exceeds request max_tokens capacity")
        for page_id in prefix_state.pages:
            self._pages[page_id].refcount += 1
            self._touch(page_id)
        request_state.pages.extend(prefix_state.pages)
        request_state.layer_lengths[:] = [
            prefix_state.token_count
        ] * self.geometry.num_layers
        self._counters["prefix_attaches"] += 1
        self._counters["prefix_tokens_attached"] += prefix_state.token_count

    @_locked
    def fork_prefix(self, prefix: PrefixHandle) -> PrefixHandle:
        self._ensure_open()
        if self.execution == "native":
            assert self._native_runtime is not None
            prefix_id = self._native_prefix_id(prefix)
            forked = self._native_runtime.fork_prefix(prefix_id)
            forked_id = self._handle(forked, name="prefix")
            return PrefixHandle(forked_id)
        _, prefix_state = self._prefix(prefix)
        for page_id in prefix_state.pages:
            self._pages[page_id].refcount += 1
            self._touch(page_id)
        prefix_id = self._next_prefix
        self._next_prefix += 1
        self._prefixes[prefix_id] = _PrefixState(
            pages=prefix_state.pages,
            token_count=prefix_state.token_count,
        )
        self._counters["prefix_forks"] += 1
        return PrefixHandle(prefix_id)

    @_locked
    def release_prefix(self, prefix: PrefixHandle) -> None:
        """Release a prefix handle; repeated release is an idempotent no-op."""

        if self.execution == "native":
            self._ensure_open()
            prefix_id = self._handle(prefix, name="prefix")
            assert self._native_runtime is not None
            self._native_runtime.release_prefix(prefix_id)
            return
        prefix_id = self._handle(prefix, name="prefix")
        if prefix_id in self._released_prefixes:
            return
        state = self._prefixes.pop(prefix_id, None)
        if state is None:
            raise KeyError(f"unknown prefix handle {prefix_id}")
        for page_id in state.pages:
            self._pages[page_id].refcount -= 1
            if self._pages[page_id].refcount < 0:  # pragma: no cover - invariant guard
                raise MetalContextRuntimeError("page reference count became negative")
        self._released_prefixes.add(prefix_id)

    @_locked
    def release(self, request: RequestHandle) -> None:
        """Release a request and its page references exactly once."""

        request_id = self._handle(request, name="request")
        if self.execution == "native":
            self._ensure_open()
            assert self._native_runtime is not None
            self._native_runtime.release(request=request_id)
            return
        self._counters["release_calls"] += 1
        if request_id in self._released_requests:
            return
        state = self._requests.pop(request_id, None)
        if state is None:
            raise KeyError(f"unknown request handle {request_id}")
        for page_id in state.pages:
            self._pages[page_id].refcount -= 1
            if self._pages[page_id].refcount < 0:  # pragma: no cover - invariant guard
                raise MetalContextRuntimeError("page reference count became negative")
        self._released_requests.add(request_id)
        self._request_ids.discard(state.request_id)
        self._counters["released_requests"] += 1

    @_locked
    def cancel(self, request: RequestHandle) -> None:
        """Cancel and release a request, preserving idempotent ownership."""

        request_id = self._handle(request, name="request")
        if self.execution == "native":
            self._ensure_open()
            assert self._native_runtime is not None
            cancel = getattr(self._native_runtime, "cancel", None)
            if not callable(cancel):
                raise MetalContextRuntimeError(
                    "native page runtime does not expose cancel"
                )
            cancel(request=request_id)
            return
        if request_id in self._released_requests:
            return
        self._counters["cancellations"] += 1
        self.release(RequestHandle(request_id))

    @_locked
    def evict(self, *, target_pages: int | None = None) -> int:
        """Evict unreferenced pages in LRU order and return the count."""

        self._ensure_open()
        if target_pages is not None and (
            not isinstance(target_pages, int)
            or isinstance(target_pages, bool)
            or target_pages < 0
        ):
            raise ValueError("target_pages must be a non-negative integer or None")
        if self.execution == "native":
            assert self._native_runtime is not None
            if target_pages is None:
                evicted = int(self._native_runtime.evict())
            else:
                evicted = int(self._native_runtime.evict(target_pages=target_pages))
            return evicted
        candidates = sorted(
            (
                state.last_used,
                page_id,
            )
            for page_id, state in enumerate(self._pages)
            if state.refcount == 0 and page_id not in self._free_pages
        )
        limit = len(candidates) if target_pages is None else target_pages
        evicted = 0
        for _, page_id in candidates[:limit]:
            self._keys[page_id].fill(0)
            self._values[page_id].fill(0)
            self._pages[page_id] = _PageState()
            self._free_pages.add(page_id)
            evicted += 1
        self._counters["evictions"] += evicted
        self._counters["pages_freed"] += evicted
        return evicted

    @_locked
    def snapshot(self, prefix: PrefixHandle, *, destination: str) -> SnapshotMetadata:
        if self.execution == "native":
            prefix_id = self._native_prefix_id(prefix)
            assert self._native_runtime is not None
            # The native bridge owns the authoritative failure counter and
            # fail-closed persistence error for the native execution mode.
            native_snapshot = getattr(self._native_runtime, "snapshot", None)
            if callable(native_snapshot):
                native_snapshot(prefix=prefix_id, destination=destination)
            else:
                raise NotImplementedError(
                    "Metal Context page-runtime snapshots are deferred to the "
                    "persistence package; no page bytes are written by this "
                    "backend"
                )
            raise AssertionError("native snapshot unexpectedly returned")
        del prefix, destination
        self._counters["snapshot_failures"] += 1
        raise NotImplementedError(
            "Metal Context page-runtime snapshots are deferred to the persistence "
            "package; no page bytes are written by this backend"
        )

    @_locked
    def restore(self, source: str) -> PrefixHandle:
        if self.execution == "native":
            assert self._native_runtime is not None
            native_restore = getattr(self._native_runtime, "restore", None)
            if callable(native_restore):
                native_restore(source=source)
            else:
                raise NotImplementedError(
                    "Metal Context page-runtime restore is deferred to the "
                    "persistence package; arbitrary Python objects are never "
                    "deserialized"
                )
            raise AssertionError("native restore unexpectedly returned")
        del source
        self._counters["restore_failures"] += 1
        raise NotImplementedError(
            "Metal Context page-runtime restore is deferred to the persistence "
            "package; arbitrary Python objects are never deserialized"
        )

    @_locked
    def metrics(self) -> Mapping[str, int | float | str | None]:
        """Return stable ownership/dispatch metrics without exposing objects."""

        if self.execution == "native":
            assert self._native_runtime is not None
            # Native metrics remain the sole ownership/counter authority.  The
            # adapter only adds stable aliases required by ContextBackend; it
            # never merges a second Python counter set into these values.
            raw = self._native_runtime.metrics()
            native_result: dict[str, int | float | str | None] = dict(raw)

            def native_count(name: str, fallback: int = 0) -> int:
                value = native_result.get(name, fallback)
                if not isinstance(value, int) or isinstance(value, bool):
                    return fallback
                return value

            dispatches = max(
                native_count("dispatches"), native_count("native_dispatches")
            )
            failures = max(
                native_count("dispatch_failures"), native_count("native_failures")
            )
            evictions = native_count("evictions")
            releases = native_count("releases")
            cancellations = native_count("cancellations")
            native_result.update(
                {
                    "backend": "metal-context",
                    "execution": "native",
                    "abi_version": self.capabilities.abi_version,
                    "capability_reason": self.capabilities.reason,
                    "block_size": native_count("block_size", self.geometry.block_size),
                    "head_dim": self.geometry.head_dim,
                    "native_available": self.capabilities.available,
                    "dispatches": dispatches,
                    "dispatch_failures": failures,
                    "native_dispatches": dispatches,
                    "native_failures": failures,
                    "oracle_dispatches": 0,
                    "oracle_failures": 0,
                    "pages_allocated": native_count(
                        "pages_allocated", native_count("page_allocations")
                    ),
                    "pages_freed": native_count("pages_freed", evictions),
                    "append_tokens": native_count("append_tokens"),
                    "evictions": evictions,
                    "cow_events": native_count("cow_events"),
                    "oom_events": native_count("oom_events"),
                    "release_calls": native_count(
                        "release_calls", releases + cancellations
                    ),
                    "released_requests": native_count(
                        "released_requests", releases + cancellations
                    ),
                    "cancellations": cancellations,
                    "prefix_attaches": native_count("prefix_attaches"),
                    "prefix_forks": native_count("prefix_forks"),
                    "prefix_tokens_attached": native_count("prefix_tokens_attached"),
                    "snapshot_failures": native_count("snapshot_failures"),
                    "restore_failures": native_count("restore_failures"),
                    "max_pages": native_count("max_pages", self.max_pages),
                    "max_blocks_per_request": native_count(
                        "max_blocks_per_request", self.max_blocks_per_request
                    ),
                }
            )
            return native_result

        resident = self.max_pages - len(self._free_pages)
        referenced = sum(state.refcount > 0 for state in self._pages)
        shared = sum(state.refcount > 1 for state in self._pages)
        oracle_result: dict[str, int | float | str | None] = {
            "backend": "metal-context",
            "execution": self.execution,
            "abi_version": self.capabilities.abi_version,
            "capability_reason": self.capabilities.reason,
            "resident_pages": resident,
            "referenced_pages": referenced,
            "shared_pages": shared,
            "free_pages": len(self._free_pages),
            "requests": len(self._requests),
            "prefixes": len(self._prefixes),
            "max_pages": self.max_pages,
            "block_size": self.geometry.block_size,
            "kv_dtype": self.geometry.kv_dtype,
            "head_dim": self.geometry.head_dim,
            "native_available": self.capabilities.available,
            "shutdown": self._shutdown,
        }
        oracle_result.update(self._counters)
        return oracle_result

    @_locked
    def shutdown(self) -> None:
        """Release all ownership and native resources; safe to repeat."""

        if self._shutdown:
            return
        if self.execution == "native":
            assert self._native_runtime is not None
            self._native_runtime.shutdown()
            self._shutdown = True
            return
        # Mark handles released through the normal reference path before
        # dropping tables, so a caller cannot accidentally reuse ownership.
        for request_id in tuple(self._requests):
            self.release(RequestHandle(request_id))
        for prefix_id in tuple(self._prefixes):
            self.release_prefix(PrefixHandle(prefix_id))
        self._keys.fill(0)
        self._values.fill(0)
        self._free_pages = set(range(self.max_pages))
        self._shutdown = True

    @_locked
    def page_table(
        self, request: RequestHandle, *, layer: int = 0
    ) -> NDArray[np.int32]:
        """Return a copy of a request's logical page table for diagnostics/tests."""

        self._ensure_open()
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise ValueError("layer must be an integer")
        if layer < 0 or layer >= self.geometry.num_layers:
            raise ValueError(f"layer must be in [0, {self.geometry.num_layers})")
        if self.execution == "native":
            request_id = self._native_request_id(request)
            assert self._native_runtime is not None
            # The compiled runtime owns per-layer sequence lengths and returns
            # the already-truncated logical page chain for this layer.  Do not
            # reconstruct a second page-table authority in Python.
            raw = cast(
                NDArray[np.int32],
                np.asarray(
                    self._native_runtime.page_table(request_id, layer),
                    dtype=np.int32,
                ),
            )
            if raw.ndim != 1:
                raw = cast(NDArray[np.int32], raw.reshape(-1))
            native_result = cast(NDArray[np.int32], raw.reshape(1, raw.size).copy())
            return native_result
        _, state = self._request(request)
        oracle_result = cast(
            NDArray[np.int32],
            self._page_table(state, state.layer_lengths[layer]).copy(),
        )
        return oracle_result

    @_locked
    def page_refcounts(self) -> tuple[int, ...]:
        """Return physical-page reference counts without exposing page storage."""

        if self.execution == "native":
            raise NotImplementedError(
                "native page references are opaque; use metrics() for ownership"
            )
        return tuple(state.refcount for state in self._pages)

    def __enter__(self) -> "MetalContextPageRuntime":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown()


__all__ = [
    "ExecutionMode",
    "MetalContextCapabilityError",
    "MetalContextPageRuntime",
    "MetalContextRuntimeError",
]
