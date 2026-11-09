# SPDX-License-Identifier: Apache-2.0
"""Attention backend selection and the phase-one context backend contract.

The Metal Context Engine is intentionally opt-in while its native implementation
is being qualified.  This module is kept importable on every platform and does
not import MLX or the optional native extension until a caller asks for the
corresponding capability probe/oracle.

There are two deliberately separate surfaces here:

* :func:`resolve_attention_backend` is the process-startup dispatch decision.
  ``mlx`` is the safe default, ``auto`` is conservative MLX routing until a
  qualified matrix is recorded, and an explicit ``metal-context`` request is a
  hard capability error rather than a silent fallback.
* :class:`ContextBackend` is the narrow runtime contract that the native page
  runtime will implement in a later package.  It is not wired into a scheduler
  in this phase.

The NumPy and MLX helpers use one explicit paged tensor layout so that native
kernel tests can compare identical inputs to a correctness oracle:

* query: ``[batch, query_heads, head_dim]`` (one decode query per request)
* key/value pages: ``[page, kv_heads, block_offset, head_dim]``
* page table: ``[batch, logical_block]`` containing physical page indices
* sequence lengths: ``[batch]``; only the first ``length`` tokens are read

The native ABI may use a different physical layout internally, but its adapter
must preserve these logical semantics at the Python boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import importlib
import math
import platform as _platform
import sys
from typing import Any, NewType, Protocol, runtime_checkable

ATTENTION_BACKEND_CHOICES = ("mlx", "metal-context", "auto")
"""CLI values accepted by the serving interfaces."""

NATIVE_EXTENSION_MODULE = "vllm_mlx._metal_context"
"""Import path reserved for the optional compiled extension."""

METAL_CONTEXT_ABI_VERSION = 1
APPLE_SILICON_ARCHITECTURES = frozenset({"arm64", "aarch64"})


class AttentionBackendName(str, Enum):
    """Names understood by the serving configuration."""

    MLX = "mlx"
    METAL_CONTEXT = "metal-context"
    AUTO = "auto"


def _coerce_backend_name(value: str | AttentionBackendName) -> AttentionBackendName:
    if isinstance(value, AttentionBackendName):
        return value
    try:
        return AttentionBackendName(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(ATTENTION_BACKEND_CHOICES)
        raise ValueError(
            f"Unknown attention backend {value!r}; expected one of: {choices}"
        ) from exc


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """A serializable snapshot of native backend availability.

    ``available`` means that the extension has passed its own capability probe
    and advertises the phase-one ABI.  It is deliberately stricter than merely
    being importable: a wheel can contain a native module that is not usable on
    the current OS/device or was built against an incompatible ABI.

    ``serving_ready`` is a separate gate.  Phase one ships the executable
    kernel foundation before a scheduler-owned executor exists, so a native
    module can be available for oracle/kernel tests while explicit serving
    selection still fails closed.
    """

    platform: str
    native_extension: bool
    metal_device: bool
    abi_version: int | None
    available: bool
    serving_ready: bool = False
    reason: str | None = None
    architecture: str = ""
    # ``mlx`` never needs to import or probe the optional native module.  Keep
    # that distinction visible to status consumers instead of representing an
    # unprobed default as a failed capability probe.
    probed: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return status data safe to expose in health/status responses."""

        return {
            "platform": self.platform,
            "native_extension": self.native_extension,
            "metal_device": self.metal_device,
            "abi_version": self.abi_version,
            "available": self.available,
            "serving_ready": self.serving_ready,
            "reason": self.reason,
            "architecture": self.architecture,
            "probed": self.probed,
        }


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """Result of resolving the requested backend at process startup."""

    requested: AttentionBackendName
    selected: AttentionBackendName
    capabilities: BackendCapabilities
    fallback_reason: str | None = None

    @property
    def is_fallback(self) -> bool:
        """Whether the requested value did not become the selected backend."""

        return self.requested is not self.selected

    def as_dict(self) -> dict[str, object]:
        """Return a stable status representation."""

        return {
            "requested": self.requested.value,
            "selected": self.selected.value,
            "fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "capabilities": self.capabilities.as_dict(),
        }


class AttentionBackendCapabilityError(RuntimeError):
    """Raised when an explicitly requested backend cannot be used."""

    def __init__(
        self,
        requested: AttentionBackendName,
        capabilities: BackendCapabilities,
    ) -> None:
        self.requested = requested
        self.capabilities = capabilities
        reason = capabilities.reason
        if not reason and capabilities.platform != "darwin":
            reason = "the Metal Context backend requires macOS"
        if (
            not reason
            and capabilities.architecture
            and capabilities.architecture not in APPLE_SILICON_ARCHITECTURES
        ):
            reason = (
                "the Metal Context backend requires Apple Silicon "
                f"(found {capabilities.architecture})"
            )
        if not reason and capabilities.available and not capabilities.serving_ready:
            reason = (
                "the native kernel is available, but no serving executor is "
                "registered yet"
            )
        reason = reason or "the required capability probe failed"
        super().__init__(
            "Attention backend 'metal-context' was explicitly requested but is "
            f"unavailable: {reason}. Install a vllm-mlx build with the optional "
            f"{NATIVE_EXTENSION_MODULE} extension on supported Apple Silicon, "
            "or select --attention-backend mlx."
        )


def _unprobed_capabilities() -> BackendCapabilities:
    """Return the neutral capability snapshot used by the MLX default.

    Selecting the established MLX backend does not require importing the
    optional native extension.  In particular, this function intentionally
    does not call :func:`discover_capabilities` so a normal installation stays
    independent of the Metal build and its device probe.
    """

    return BackendCapabilities(
        platform=sys.platform,
        native_extension=False,
        metal_device=False,
        abi_version=None,
        available=False,
        reason=None,
        architecture="",
        probed=False,
    )


def _unsupported_capabilities(reason: str) -> BackendCapabilities:
    return BackendCapabilities(
        platform=sys.platform,
        native_extension=False,
        metal_device=False,
        abi_version=None,
        available=False,
        reason=reason,
        architecture=_platform.machine().lower(),
        probed=True,
    )


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def discover_capabilities() -> BackendCapabilities:
    """Probe the optional native extension without importing MLX eagerly.

    The extension exposes a zero-argument ``capabilities()`` function returning
    a mapping.  Requiring that small ABI probe makes an importable but stale
    extension fail closed instead of being treated as a usable backend.
    """

    if sys.platform != "darwin":
        return _unsupported_capabilities(
            f"the Metal Context backend requires macOS (current platform: {sys.platform})"
        )

    machine = _platform.machine().lower()
    if machine not in APPLE_SILICON_ARCHITECTURES:
        return _unsupported_capabilities(
            "the Metal Context backend requires Apple Silicon "
            f"(arm64/aarch64; current architecture: {machine or 'unknown'})"
        )

    try:
        native = importlib.import_module(NATIVE_EXTENSION_MODULE)
    except (ImportError, OSError) as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        return _unsupported_capabilities(
            f"the optional native extension could not be loaded: {detail}"
        )

    probe = getattr(native, "capabilities", None)
    if not callable(probe):
        return BackendCapabilities(
            platform=sys.platform,
            native_extension=True,
            metal_device=False,
            abi_version=None,
            available=False,
            reason=(
                f"{NATIVE_EXTENSION_MODULE} does not expose the required "
                "capabilities() ABI probe"
            ),
            architecture=machine,
            probed=True,
        )

    try:
        raw = probe()
    except Exception as exc:  # native errors must become a precise status
        detail = str(exc).strip() or exc.__class__.__name__
        return BackendCapabilities(
            platform=sys.platform,
            native_extension=True,
            metal_device=False,
            abi_version=None,
            available=False,
            reason=f"the native capability probe failed: {detail}",
            architecture=machine,
            probed=True,
        )

    if not isinstance(raw, Mapping):
        return BackendCapabilities(
            platform=sys.platform,
            native_extension=True,
            metal_device=False,
            abi_version=None,
            available=False,
            reason="the native capability probe returned a non-mapping result",
            architecture=machine,
            probed=True,
        )

    native_extension = bool(raw.get("compiled", True))
    abi_version = _int_or_none(raw.get("abi_version"))
    advertised = bool(raw.get("available", raw.get("supported", False)))
    # Older phase-one native bridges use ``available`` as the result of a
    # successful Metal device/pipeline probe but do not expose a separate
    # metal_device field.  Preserve that ABI while still requiring a positive
    # probe result; newer bridges can report the field explicitly.
    metal_device = bool(raw.get("metal_device", advertised))
    reasons: list[str] = []
    if not native_extension:
        reasons.append("the optional native extension is not compiled")
    if not metal_device:
        reasons.append("no usable Metal device was reported")
    if abi_version != METAL_CONTEXT_ABI_VERSION:
        reasons.append(
            f"native ABI {abi_version!r} does not match required ABI "
            f"{METAL_CONTEXT_ABI_VERSION}"
        )
    if not advertised:
        native_reason = raw.get("reason")
        if isinstance(native_reason, str) and native_reason.strip():
            reasons.append(native_reason.strip())
        else:
            reasons.append("the native extension did not advertise availability")
    serving_ready = bool(raw.get("serving_ready", raw.get("executor_available", False)))
    if not serving_ready:
        reasons.append(
            "the native kernel is not serving-ready; the Metal Context executor "
            "has not been integrated and qualified"
        )

    available = metal_device and abi_version == METAL_CONTEXT_ABI_VERSION and advertised
    return BackendCapabilities(
        platform=sys.platform,
        native_extension=native_extension,
        metal_device=metal_device,
        abi_version=abi_version,
        available=available,
        serving_ready=serving_ready,
        reason=(
            None if available and serving_ready else "; ".join(dict.fromkeys(reasons))
        ),
        architecture=machine,
        probed=True,
    )


def resolve_attention_backend(
    requested: str | AttentionBackendName = AttentionBackendName.MLX,
    *,
    capabilities: BackendCapabilities | None = None,
) -> BackendSelection:
    """Resolve a serving request without silently enabling an unqualified path.

    ``auto`` intentionally selects MLX even when the extension is present.  A
    later qualification patch can add an explicit matrix gate while preserving
    this function's startup/error contract.
    """

    name = _coerce_backend_name(requested)

    # MLX is the established default and must remain independent of the
    # optional native extension.  A caller that already has a capability
    # snapshot may attach it for status/debugging, but the default path never
    # performs a probe or imports the native module.
    if name is AttentionBackendName.MLX:
        caps = capabilities if capabilities is not None else _unprobed_capabilities()
        return BackendSelection(
            requested=name,
            selected=AttentionBackendName.MLX,
            capabilities=caps,
        )

    # ``auto`` may probe conservatively so it can report why Metal was not
    # selected.  Explicit ``metal-context`` also probes here, then fails with
    # a precise capability error if the serving executor is unavailable.
    caps = capabilities if capabilities is not None else discover_capabilities()

    if name is AttentionBackendName.METAL_CONTEXT:
        if (
            caps.platform != "darwin"
            or (
                caps.architecture
                and caps.architecture not in APPLE_SILICON_ARCHITECTURES
            )
            or not caps.available
            or not caps.serving_ready
        ):
            raise AttentionBackendCapabilityError(name, caps)
        return BackendSelection(
            requested=name,
            selected=name,
            capabilities=caps,
        )

    if name is AttentionBackendName.AUTO:
        return BackendSelection(
            requested=name,
            selected=AttentionBackendName.MLX,
            capabilities=caps,
            fallback_reason=(
                "automatic Metal Context routing is disabled until the explicit "
                "qualification matrix passes; using the MLX correctness path"
            ),
        )

    raise AssertionError(f"unhandled attention backend {name!r}")


def validate_attention_backend(
    requested: str | AttentionBackendName,
    *,
    capabilities: BackendCapabilities | None = None,
) -> BackendSelection:
    """Compatibility alias emphasizing startup validation at call sites."""

    return resolve_attention_backend(requested, capabilities=capabilities)


@dataclass(frozen=True, slots=True)
class AttentionGeometry:
    """Phase-one geometry accepted by a concrete context backend."""

    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 128
    block_size: int = 16
    kv_dtype: str = "bfloat16"

    def validate(self) -> None:
        """Reject shapes outside the initial Qwen dense/MoE qualification scope."""

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for GQA"
            )
        if self.head_dim != 128:
            raise ValueError("Metal Context v1 supports head_dim=128 only")
        if self.block_size not in (16, 32):
            raise ValueError("Metal Context v1 supports block_size 16 or 32 only")
        if self.kv_dtype.lower() not in {"bfloat16", "bf16"}:
            raise ValueError("Metal Context v1 supports BF16 K/V pages only")


RequestHandle = NewType("RequestHandle", int)
PageHandle = NewType("PageHandle", int)
PrefixHandle = NewType("PrefixHandle", int)


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    """Opaque metadata returned by a future snapshot implementation."""

    identity: str
    page_count: int
    token_count: int
    content_hash: str


@runtime_checkable
class ContextBackend(Protocol):
    """Narrow lifecycle contract for the native context runtime.

    Implementations own page-table and command-queue state.  The protocol is
    intentionally free of scheduler/request-output types so it can be tested
    independently before a production executor is introduced.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        pass

    def validate(self, geometry: AttentionGeometry) -> None:
        pass

    def allocate_request(self, request_id: str, *, max_tokens: int) -> RequestHandle:
        pass

    def allocate_pages(
        self, request: RequestHandle, count: int
    ) -> tuple[PageHandle, ...]:
        pass

    def append_kv(
        self,
        request: RequestHandle,
        layer: int,
        keys: Any,
        values: Any,
    ) -> None:
        pass

    def paged_decode_attention(
        self, request: RequestHandle, layer: int, query: Any
    ) -> Any:
        pass

    def attach_prefix(self, request: RequestHandle, prefix: PrefixHandle) -> None:
        pass

    def fork_prefix(self, prefix: PrefixHandle) -> PrefixHandle:
        pass

    def release(self, request: RequestHandle) -> None:
        pass

    def evict(self, *, target_pages: int | None = None) -> int:
        pass

    def snapshot(self, prefix: PrefixHandle, *, destination: str) -> SnapshotMetadata:
        pass

    def restore(self, source: str) -> PrefixHandle:
        pass

    def metrics(self) -> Mapping[str, int | float | str | None]:
        pass

    def shutdown(self) -> None:
        pass


def _as_numpy(value: Any) -> Any:
    """Import NumPy lazily and provide a helpful dependency error."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency
        raise RuntimeError("NumPy is required for the attention oracle") from exc
    return np.asarray(value)


def _float32_values(value: Any) -> Any:
    """Normalize numeric oracle tensors, including native BF16 bit storage."""

    import numpy as np

    array = _as_numpy(value)
    if array.dtype == np.uint16:
        # The optional native ABI represents BF16 values as the high 16 bits
        # of an IEEE float32.  Accepting that representation here lets native
        # tests compare exactly the same buffers against the reference.
        return (array.astype(np.uint32) << 16).view(np.float32)
    return array.astype(np.float32, copy=False)


def _mlx_uint16_array(value: Any, mx: Any) -> bool:
    """Identify MLX arrays carrying the native BF16-bit representation."""

    dtype = getattr(value, "dtype", None)
    uint16_dtype = getattr(mx, "uint16", None)
    if uint16_dtype is not None and dtype == uint16_dtype:
        return True
    normalized = str(dtype).lower().replace(" ", "")
    return normalized in {"uint16", "uint16_t"}


def _normalize_mlx_input(value: Any, normalized_numpy: Any, mx: Any) -> Any:
    """Decode native BF16 bits before passing an MLX array to SDPA."""

    if isinstance(value, mx.array):
        if _mlx_uint16_array(value, mx):
            # _float32_values interprets uint16 as the upper half of float32.
            return mx.array(_float32_values(value))
        return value
    return mx.array(normalized_numpy)


def _validate_paged_inputs(
    query: Any,
    key_pages: Any,
    value_pages: Any,
    page_table: Any,
    sequence_lengths: Any,
    *,
    block_size: int,
    num_kv_heads: int | None,
) -> tuple[Any, Any, Any, Any, Any, int]:
    """Validate and normalize oracle input metadata."""

    import numpy as np

    q = _float32_values(query)
    k = _float32_values(key_pages)
    v = _float32_values(value_pages)
    table = _as_numpy(page_table)
    lengths = _as_numpy(sequence_lengths)
    if q.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "key_pages and value_pages must have shape "
            "[page, kv_heads, block, head_dim]"
        )
    if k.shape != v.shape:
        raise ValueError("key_pages and value_pages must have identical shapes")
    if table.ndim != 2 or lengths.ndim != 1 or table.shape[0] != q.shape[0]:
        raise ValueError(
            "page_table/sequence_lengths batch dimensions do not match query"
        )
    if lengths.shape[0] != q.shape[0]:
        raise ValueError("sequence_lengths batch dimension does not match query")
    if k.shape[2] != block_size:
        raise ValueError("page tensors do not match block_size")
    if k.shape[1] <= 0 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("attention head counts and head_dim must be positive")
    if q.shape[2] != k.shape[3]:
        raise ValueError("query and page head_dim values do not match")
    if num_kv_heads is not None and k.shape[1] != num_kv_heads:
        raise ValueError("page tensor kv head count does not match num_kv_heads")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("page_table must contain integer page indices")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("sequence_lengths must contain integer lengths")
    if np.any(lengths < 0):
        raise ValueError("sequence_lengths cannot be negative")
    max_tokens = table.shape[1] * block_size
    if np.any(lengths > max_tokens):
        raise ValueError("sequence_lengths exceed the page table capacity")
    if not (np.isfinite(q).all() and np.isfinite(k).all() and np.isfinite(v).all()):
        raise ValueError(
            "query, key_pages, and value_pages must contain only finite values"
        )
    return q, k, v, table, lengths, k.shape[1]


def numpy_paged_decode_attention(
    query: Any,
    key_pages: Any,
    value_pages: Any,
    page_table: Any,
    sequence_lengths: Any,
    *,
    block_size: int,
    num_kv_heads: int | None = None,
    scale: float | None = None,
) -> Any:
    """Compute one-token paged decode attention with a NumPy reference.

    The implementation gathers only the valid tail of each logical sequence,
    checks every physical page index, and computes a stable softmax in float32.
    It is a correctness oracle, not a serving implementation.
    """

    import numpy as np

    q, k, v, table, lengths, kv_heads = _validate_paged_inputs(
        query,
        key_pages,
        value_pages,
        page_table,
        sequence_lengths,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
    )
    batch, query_heads, head_dim = q.shape
    scale_value = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    output = np.zeros((batch, query_heads, head_dim), dtype=np.float32)
    group_size = query_heads // kv_heads
    page_count = k.shape[0]

    for batch_index in range(batch):
        sequence_length = int(lengths[batch_index])
        if sequence_length == 0:
            continue

        logical_blocks = (sequence_length + block_size - 1) // block_size
        page_ids = table[batch_index, :logical_blocks]
        if np.any(page_ids < 0) or np.any(page_ids >= page_count):
            raise ValueError("page_table contains a physical page outside page tensors")

        keys = np.empty((sequence_length, kv_heads, head_dim), dtype=np.float32)
        values = np.empty_like(keys)
        cursor = 0
        for logical_block, page_id_value in enumerate(page_ids):
            page_id = int(page_id_value)
            tokens = min(block_size, sequence_length - cursor)
            keys[cursor : cursor + tokens] = np.transpose(
                k[page_id, :, :tokens], (1, 0, 2)
            )
            values[cursor : cursor + tokens] = np.transpose(
                v[page_id, :, :tokens], (1, 0, 2)
            )
            cursor += tokens

        query_row = q[batch_index].astype(np.float32, copy=False)
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            with np.errstate(over="ignore", invalid="ignore"):
                logits = (keys[:, kv_head] @ query_row[query_head]) * scale_value
            if not np.isfinite(logits).all():
                raise ValueError("attention logits overflow for finite float32 inputs")
            max_logit = np.max(logits)
            weights = np.exp(logits - max_logit)
            denominator = np.sum(weights, dtype=np.float64)
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("attention softmax denominator is not finite")
            weights /= denominator
            output[batch_index, query_head] = weights @ values[:, kv_head]

    return output


def mlx_sdpa_paged_decode_attention(
    query: Any,
    key_pages: Any,
    value_pages: Any,
    page_table: Any,
    sequence_lengths: Any,
    *,
    block_size: int,
    num_kv_heads: int | None = None,
    scale: float | None = None,
) -> Any:
    """Run the same paged inputs through MLX's SDPA implementation.

    MLX SDPA uses ``[batch, heads, sequence, head_dim]``.  This adapter gathers
    each non-contiguous logical sequence and invokes SDPA one request at a time,
    explicitly expanding grouped KV heads to the query-head count.  That keeps
    GQA semantics identical to the NumPy oracle and avoids relying on an
    MLX-version-specific implicit GQA broadcast.
    """

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - exercised on non-Apple CI
        raise RuntimeError("MLX is required for the MLX SDPA attention oracle") from exc

    q_np, _k_np, _v_np, table_np, lengths_np, _ = _validate_paged_inputs(
        query,
        key_pages,
        value_pages,
        page_table,
        sequence_lengths,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
    )
    q = _normalize_mlx_input(query, q_np, mx)
    k_pages = _normalize_mlx_input(key_pages, _k_np, mx)
    v_pages = _normalize_mlx_input(value_pages, _v_np, mx)
    batch, query_heads, head_dim = q_np.shape
    scale_value = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    outputs = []

    for batch_index in range(batch):
        sequence_length = int(lengths_np[batch_index])
        if sequence_length == 0:
            outputs.append(mx.zeros((query_heads, head_dim), dtype=q.dtype))
            continue
        logical_blocks = (sequence_length + block_size - 1) // block_size
        page_ids = table_np[batch_index, :logical_blocks]
        if (page_ids < 0).any() or (page_ids >= int(k_pages.shape[0])).any():
            raise ValueError("page_table contains a physical page outside page tensors")
        key_parts = []
        value_parts = []
        cursor = 0
        for page_id_value in page_ids:
            page_id = int(page_id_value)
            tokens = min(block_size, sequence_length - cursor)
            key_parts.append(mx.transpose(k_pages[page_id, :, :tokens], (1, 0, 2)))
            value_parts.append(mx.transpose(v_pages[page_id, :, :tokens], (1, 0, 2)))
            cursor += tokens
        keys = mx.concatenate(key_parts, axis=0)
        values = mx.concatenate(value_parts, axis=0)

        # MLX SDPA's explicit layout is [batch, heads, sequence, head_dim].
        q_arg = q[batch_index][None, :, None, :]
        k_heads = mx.transpose(keys, (1, 0, 2))
        v_heads = mx.transpose(values, (1, 0, 2))
        if query_heads != int(k_heads.shape[0]):
            group_size = query_heads // int(k_heads.shape[0])
            # Repeat each KV head contiguously: q0/q1 use kv0, q2/q3 use
            # kv1 for the common 4-query-head/2-KV-head GQA shape.
            k_heads = mx.concatenate(
                [
                    k_heads[index : index + 1]
                    for index in range(int(k_heads.shape[0]))
                    for _ in range(group_size)
                ],
                axis=0,
            )
            v_heads = mx.concatenate(
                [
                    v_heads[index : index + 1]
                    for index in range(int(v_heads.shape[0]))
                    for _ in range(group_size)
                ],
                axis=0,
            )
        k_arg = k_heads[None, :, :, :]
        v_arg = v_heads[None, :, :, :]
        attended = mx.fast.scaled_dot_product_attention(
            q_arg,
            k_arg,
            v_arg,
            scale=scale_value,
        )
        outputs.append(attended[0, :, 0, :])

    return mx.stack(outputs) if outputs else mx.zeros((0, query_heads, head_dim))


__all__ = [
    "APPLE_SILICON_ARCHITECTURES",
    "ATTENTION_BACKEND_CHOICES",
    "AttentionBackendCapabilityError",
    "AttentionBackendName",
    "AttentionGeometry",
    "BackendCapabilities",
    "BackendSelection",
    "ContextBackend",
    "METAL_CONTEXT_ABI_VERSION",
    "NATIVE_EXTENSION_MODULE",
    "PageHandle",
    "PrefixHandle",
    "RequestHandle",
    "SnapshotMetadata",
    "discover_capabilities",
    "mlx_sdpa_paged_decode_attention",
    "numpy_paged_decode_attention",
    "resolve_attention_backend",
    "validate_attention_backend",
]
