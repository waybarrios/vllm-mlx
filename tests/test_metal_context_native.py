# SPDX-License-Identifier: Apache-2.0
"""Portable checks for the optional native Metal Context Engine surface."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vllm_mlx.attention_backend import (
    mlx_sdpa_paged_decode_attention,
    numpy_paged_decode_attention,
)


def test_capability_probe_never_requires_native_build():
    from vllm_mlx import _metal_context

    capabilities = _metal_context.capabilities()

    assert capabilities["backend"] == "metal-context"
    assert capabilities["abi_version"] == 1
    assert capabilities["block_sizes"] == (16, 32)
    assert capabilities["head_dims"] == (128,)
    assert capabilities["gqa"] is True
    assert capabilities["partial_blocks"] is True
    assert capabilities["online_softmax"] is True
    assert capabilities["kv_dtype"] == "bfloat16"
    assert isinstance(capabilities["apple_silicon"], bool)
    assert capabilities["serving_ready"] is False
    assert isinstance(capabilities["available"], bool)
    if not capabilities["available"]:
        assert capabilities["reason"]


def test_page_runtime_decode_has_no_hidden_pool_snapshot_or_dispatch_allocations():
    """Keep the allocation-free steady-state contract mechanically visible."""

    source = (
        Path(__file__).parents[1]
        / "native"
        / "metal_context"
        / "src"
        / "page_runtime.mm"
    ).read_text(encoding="utf-8")
    dispatch_start = source.index("bool dispatch_kernel(")
    dispatch_end = source.index("}  // namespace", dispatch_start)
    dispatch_source = source[dispatch_start:dispatch_end]
    assert "newBufferWithLength" not in dispatch_source
    assert "copy_decode_snapshot" not in source
    # Exactly one allocation site remains in the PageRuntime TU: the central
    # wrapper used by constructor-time preallocation.
    assert source.count("newBufferWithLength") == 1
    assert "std::vector<uint16_t> snapshot" not in source
    envelope_start = source.index("bool validate_attention_envelope(")
    envelope_end = source.index("std::string metallib_path(", envelope_start)
    envelope_source = source[envelope_start:envelope_end]
    assert "for (uint32_t token" not in envelope_source
    assert "max_value_accumulation" in envelope_source
    decode_start = source.index("bool PageRuntime::paged_decode(")
    decode_end = source.index("PageRuntimeMetrics PageRuntime::metrics(", decode_start)
    decode_source = source[decode_start:decode_end]
    assert "resolve_page" not in decode_source
    assert "for (" not in decode_source
    assert "decode_page_resolution_checks" in source
    assert "finish_dispatch(*result)" in decode_source
    assert "sync_table_range" in source
    bridge = (
        Path(__file__).parents[1]
        / "native"
        / "metal_context"
        / "src"
        / "page_runtime_python.mm"
    ).read_text(encoding="utf-8")
    assert "RuntimeCallLease" in bridge
    assert "Py_BEGIN_ALLOW_THREADS" in bridge
    assert "lifecycle_cv.wait" in bridge
    assert bridge.count("try {") >= 3
    assert bridge.count("current_exception") >= 3
    legacy_bridge = (
        Path(__file__).parents[1]
        / "native"
        / "metal_context"
        / "src"
        / "python_module.mm"
    ).read_text(encoding="utf-8")
    assert "LegacyDispatchLease" in legacy_bridge
    assert legacy_bridge.count("Py_BEGIN_ALLOW_THREADS") >= 1
    assert legacy_bridge.count("try {") >= 2
    assert legacy_bridge.count("current_exception") >= 2


def test_unavailable_native_dispatch_fails_precisely():
    from vllm_mlx import _metal_context

    capabilities = _metal_context.capabilities()
    if capabilities["available"]:
        pytest.skip("native extension is available; dispatch is covered on Apple CI")
    if capabilities.get("compiled"):
        # A compiled extension without a Metal device is still covered by the
        # host-storage PageRuntime lifecycle tests below.  This test targets
        # the pure-Python fail-closed dispatch surface only.
        pytest.skip("compiled native dispatch is unavailable without a Metal device")

    with pytest.raises(RuntimeError, match="metal-context backend unavailable"):
        _metal_context.paged_decode(None)


def test_native_paged_decode_matches_numpy_oracle():
    """Exercise the compiled Metal dispatch when the strict build is present."""

    from vllm_mlx import _metal_context

    capabilities = _metal_context.capabilities()
    if not capabilities["available"]:
        pytest.skip("strict native Metal build is unavailable")

    rng = np.random.default_rng(31)
    block_size = 16
    head_dim = 128
    query_heads = 4
    kv_heads = 2
    query_values = rng.normal(0.0, 0.25, size=(2, query_heads, head_dim)).astype(
        np.float32
    )
    key_values = rng.normal(0.0, 0.25, size=(4, kv_heads, block_size, head_dim)).astype(
        np.float32
    )
    value_values = rng.normal(0.0, 0.25, size=key_values.shape).astype(np.float32)

    def to_bf16_bits(values):
        return (values.view(np.uint32) >> 16).astype(np.uint16)

    def from_bf16_bits(values):
        return (values.astype(np.uint32) << 16).view(np.float32)

    query = to_bf16_bits(query_values)
    key_pages = to_bf16_bits(key_values)
    value_pages = to_bf16_bits(value_values)
    page_table = np.asarray([[2, 0], [1, 3]], dtype=np.int32)
    sequence_lengths = np.asarray([17, 30], dtype=np.int32)
    scale = 1.0 / math.sqrt(head_dim)

    native_bytes = _metal_context.paged_decode(
        query,
        key_pages,
        value_pages,
        page_table,
        sequence_lengths,
        num_kv_heads=kv_heads,
        block_size=block_size,
        scale=scale,
    )
    native_output = np.frombuffer(native_bytes, dtype=np.float32).reshape(
        query_values.shape
    )

    oracle_output = numpy_paged_decode_attention(
        from_bf16_bits(query),
        from_bf16_bits(key_pages),
        from_bf16_bits(value_pages),
        page_table,
        sequence_lengths,
        block_size=block_size,
        num_kv_heads=kv_heads,
        scale=scale,
    )

    np.testing.assert_allclose(native_output, oracle_output, rtol=2e-2, atol=2e-2)


def test_native_paged_decode_matches_mlx_oracle():
    """Compare the real Metal dispatch with MLX SDPA when both are present."""

    from vllm_mlx import _metal_context

    capabilities = _metal_context.capabilities()
    if not capabilities["available"]:
        pytest.skip("strict native Metal build is unavailable")
    pytest.importorskip("mlx.core")

    rng = np.random.default_rng(37)
    block_size = 32
    head_dim = 128
    query_heads = 4
    kv_heads = 2
    query = rng.normal(0.0, 0.2, size=(2, query_heads, head_dim)).astype(np.float32)
    key_pages = rng.normal(0.0, 0.2, size=(4, kv_heads, block_size, head_dim)).astype(
        np.float32
    )
    value_pages = rng.normal(0.0, 0.2, size=key_pages.shape).astype(np.float32)
    page_table = np.asarray([[3, 0], [2, 1]], dtype=np.int32)
    sequence_lengths = np.asarray([32, 33], dtype=np.int32)
    scale = 1.0 / math.sqrt(head_dim)

    def to_bf16_bits(values):
        return (values.view(np.uint32) >> 16).astype(np.uint16)

    query_bits = to_bf16_bits(query)
    key_bits = to_bf16_bits(key_pages)
    value_bits = to_bf16_bits(value_pages)
    native_output = np.frombuffer(
        _metal_context.paged_decode(
            query_bits,
            key_bits,
            value_bits,
            page_table,
            sequence_lengths,
            num_kv_heads=kv_heads,
            block_size=block_size,
            scale=scale,
        ),
        dtype=np.float32,
    ).reshape(query.shape)
    mlx_output = mlx_sdpa_paged_decode_attention(
        query_bits,
        key_bits,
        value_bits,
        page_table,
        sequence_lengths,
        block_size=block_size,
        num_kv_heads=kv_heads,
        scale=scale,
    )

    np.testing.assert_allclose(
        native_output, np.asarray(mlx_output), rtol=3e-2, atol=3e-2
    )


def test_native_rejects_non_native_formats_and_nonfinite_bf16():
    """The host ABI must reject ambiguous byte order and unsafe BF16 values."""

    from vllm_mlx import _metal_context

    capabilities = _metal_context.capabilities()
    if not capabilities["available"]:
        pytest.skip("strict native Metal build is unavailable")

    rng = np.random.default_rng(41)
    query = (
        rng.normal(size=(1, 2, 128)).astype(np.float32).view(np.uint32) >> 16
    ).astype(np.uint16)
    key = np.zeros((1, 1, 16, 128), dtype=np.uint16)
    value = np.zeros_like(key)
    table = np.asarray([[0]], dtype=np.int32)
    lengths = np.asarray([1], dtype=np.int32)
    kwargs = dict(
        num_kv_heads=1,
        block_size=16,
        scale=1.0 / math.sqrt(128),
    )

    with pytest.raises(ValueError, match="expected query"):
        _metal_context.paged_decode(
            query.reshape(2, 128), key, value, table, lengths, **kwargs
        )
    with pytest.raises(ValueError, match="uint16"):
        _metal_context.paged_decode(
            query.astype(">u2"), key, value, table, lengths, **kwargs
        )
    with pytest.raises(ValueError, match="uint16"):
        _metal_context.paged_decode(
            query.astype(np.float16), key, value, table, lengths, **kwargs
        )

    nonfinite_query = query.copy()
    nonfinite_query[0, 0, 0] = np.uint16(0x7F80)
    with pytest.raises(ValueError, match="finite BF16"):
        _metal_context.paged_decode(
            nonfinite_query, key, value, table, lengths, **kwargs
        )


def test_native_rejects_finite_bf16_dot_and_score_overflow():
    """Finite BF16 extremes must not reach an Inf/Inf softmax reduction."""

    from vllm_mlx import _metal_context

    if not _metal_context.capabilities()["available"]:
        pytest.skip("strict native Metal build is unavailable")

    key = np.zeros((1, 1, 16, 128), dtype=np.uint16)
    value = np.zeros_like(key)
    table = np.asarray([[0]], dtype=np.int32)
    lengths = np.asarray([1], dtype=np.int32)

    # 0x7f7f is the largest finite BF16 value.  The conservative host bound
    # rejects max_q * max_k * head_dim before any Metal allocation/dispatch.
    query = np.full((1, 1, 128), np.uint16(0x7F7F), dtype=np.uint16)
    key.fill(np.uint16(0x7F7F))
    with pytest.raises(ValueError, match="dot product"):
        _metal_context.paged_decode(
            query,
            key,
            value,
            table,
            lengths,
            num_kv_heads=1,
            block_size=16,
            scale=1.0,
        )

    # 0x5d31 is approximately 8e17.  Its 128-term dot bound is finite, but
    # multiplying that bound by scale=4 exceeds the guarded score envelope.
    query.fill(np.uint16(0x5D31))
    key.fill(np.uint16(0x5D31))
    with pytest.raises(ValueError, match="attention score"):
        _metal_context.paged_decode(
            query,
            key,
            value,
            table,
            lengths,
            num_kv_heads=1,
            block_size=16,
            scale=4.0,
        )

    # Value accumulation is a separate envelope from QK score safety.  Two
    # finite maximum-BF16 values must not be allowed to overflow the running
    # numerator even when the attention scores themselves are all zero.
    query.fill(np.uint16(0))
    key.fill(np.uint16(0))
    value.fill(np.uint16(0x7F7F))
    lengths[0] = 2
    with pytest.raises(ValueError, match="value accumulation"):
        _metal_context.paged_decode(
            query,
            key,
            value,
            table,
            lengths,
            num_kv_heads=1,
            block_size=16,
            scale=1.0,
        )


def test_legacy_foundation_rejects_value_accumulation_before_dispatch():
    """The raw foundation bridge shares the native V safety envelope."""

    from vllm_mlx import _metal_context

    if not _metal_context.capabilities().get("compiled"):
        pytest.skip("compiled native foundation bridge is unavailable")
    query = np.zeros((1, 1, 128), dtype=np.uint16)
    key = np.zeros((1, 1, 16, 128), dtype=np.uint16)
    value = np.full_like(key, np.uint16(0x7F7F))
    table = np.asarray([[0]], dtype=np.int32)
    lengths = np.asarray([2], dtype=np.int32)
    with pytest.raises(ValueError, match="value accumulation"):
        _metal_context.paged_decode(
            query,
            key,
            value,
            table,
            lengths,
            num_kv_heads=1,
            block_size=16,
            scale=1.0,
        )


def test_native_shutdown_can_repeat_without_stale_runtime_state():
    """Repeated capability probe/shutdown cycles must recreate the pipeline."""

    from vllm_mlx import _metal_context

    if not _metal_context.capabilities()["available"]:
        pytest.skip("strict native Metal build is unavailable")

    for _ in range(3):
        _metal_context.shutdown()
        capabilities = _metal_context.capabilities()
        assert capabilities["available"] is True
        assert capabilities["serving_ready"] is False


def _compiled_page_runtime():
    """Return the compiled ownership type without requiring a Metal device."""

    from vllm_mlx import _metal_context

    page_runtime = getattr(_metal_context, "PageRuntime", None)
    if page_runtime is None:
        pytest.skip("compiled native PageRuntime extension is unavailable")
    return _metal_context, page_runtime


def _native_zero_kv(tokens: int, kv_heads: int = 1) -> np.ndarray:
    return np.zeros((tokens, kv_heads, 128), dtype=np.uint16)


def test_compiled_page_runtime_host_lifecycle_runs_without_gpu():
    """Exercise ownership/prefix teardown; no dispatch or GPU is required."""

    native, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        max_pages=4,
        max_blocks_per_request=2,
        max_requests=4,
    )
    assert runtime.geometry()["num_key_value_heads"] == 2
    request = runtime.allocate_request("host-lifecycle", max_tokens=32)
    assert len(runtime.allocate_pages(request, 2)) == 2
    keys = _native_zero_kv(17, 2)
    runtime.append_kv(request, 0, keys, keys)
    runtime.append_kv(request, 1, keys, keys)
    prefix = runtime.create_prefix(request)
    branch = runtime.allocate_request("host-branch", max_tokens=32)
    runtime.attach_prefix(branch, prefix)

    metrics = runtime.metrics()
    assert metrics["append_tokens"] == 34
    assert metrics["max_pages"] == 4
    assert metrics["kv_dtype"] == "bfloat16"
    assert metrics["shared_pages"] == 2
    assert metrics["dispatches"] == 0
    assert metrics["dispatch_failures"] == 0

    # Release is idempotent for the same generation, and post-shutdown
    # diagnostics retain truthful capacity/counter values.
    runtime.release(branch)
    runtime.release(branch)
    runtime.release(request)
    runtime.release_prefix(prefix)
    runtime.shutdown()
    runtime.shutdown()
    after = runtime.metrics()
    assert after["shutdown"] is True
    assert after["resident_pages"] == 0
    assert after["free_pages"] == 4
    assert after["requests"] == 0
    assert after["prefixes"] == 0
    assert after["kv_bytes"] == 0
    assert after["append_tokens"] == 34
    native.shutdown()


def test_compiled_page_runtime_result_allocation_rolls_back_request(
    monkeypatch,
):
    """A failed PyLong result must not publish a request slot or generation."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 2, 2, 2)
    before = runtime.metrics()
    monkeypatch.setenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT", "allocate_request")
    with pytest.raises(MemoryError):
        runtime.allocate_request("result-fault-request", max_tokens=16)
    monkeypatch.delenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT")
    assert runtime.metrics() == before

    # The rolled-back slot and generation are immediately reusable.
    request = runtime.allocate_request("result-fault-request", max_tokens=16)
    runtime.release(request)
    runtime.shutdown()


@pytest.mark.parametrize("fault_value", ["allocate_pages", "tuple_item"])
def test_compiled_page_runtime_result_allocation_rolls_back_pages(
    monkeypatch,
    fault_value,
):
    """Tuple construction, including a partial-item failure, is transactional."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 4, 2, 1)
    request = runtime.allocate_request("result-fault-pages", max_tokens=32)
    before = (
        runtime.metrics(),
        runtime.request_pages(request),
        runtime.page_table(request),
    )
    monkeypatch.setenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT", fault_value)
    with pytest.raises(MemoryError):
        runtime.allocate_pages(request, 2)
    monkeypatch.delenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT")
    assert (
        runtime.metrics(),
        runtime.request_pages(request),
        runtime.page_table(request),
    ) == before

    pages = runtime.allocate_pages(request, 2)
    assert len(pages) == 2
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_result_allocation_rolls_back_prefixes(
    monkeypatch,
):
    """Prefix creation/forking restore refs, sequence length, and counters."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(2, 1, 1, 128, 16, 4, 2, 3)
    request = runtime.allocate_request("result-fault-prefix", max_tokens=32)
    runtime.allocate_pages(request, 2)
    one = _native_zero_kv(1)
    runtime.append_kv(request, 0, one, one)
    runtime.append_kv(request, 1, one, one)

    before_create = (
        runtime.metrics(),
        runtime.page_table(request, layer=0),
        runtime.sequence_length(request),
    )
    monkeypatch.setenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT", "create_prefix")
    with pytest.raises(MemoryError):
        runtime.create_prefix(request)
    monkeypatch.delenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT")
    assert (
        runtime.metrics(),
        runtime.page_table(request, layer=0),
        runtime.sequence_length(request),
    ) == before_create

    prefix = runtime.create_prefix(request)
    before_fork = runtime.metrics()
    monkeypatch.setenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT", "fork_prefix")
    with pytest.raises(MemoryError):
        runtime.fork_prefix(prefix)
    monkeypatch.delenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT")
    assert runtime.metrics() == before_fork
    forked = runtime.fork_prefix(prefix)
    runtime.release_prefix(forked)
    runtime.release_prefix(prefix)
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_result_allocation_rolls_back_eviction(
    monkeypatch,
):
    """A failed eviction PyLong restores page metadata and LRU counters."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 2, 1, 1)
    request = runtime.allocate_request("result-fault-evict", max_tokens=16)
    runtime.allocate_pages(request, 1)
    runtime.release(request)
    before = runtime.metrics()
    monkeypatch.setenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT", "evict")
    with pytest.raises(MemoryError):
        runtime.evict(1)
    monkeypatch.delenv("VLLM_MLX_METAL_CONTEXT_TEST_FAIL_RESULT")
    assert runtime.metrics() == before
    assert runtime.evict(1) == 1
    runtime.shutdown()


def test_compiled_page_runtime_snapshot_restore_fail_closed_with_native_metrics():
    """Persistence is deferred, but native failure counters remain truthful."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 2, 1, 1)
    request = runtime.allocate_request("snapshot-metrics", max_tokens=16)
    runtime.allocate_pages(request, 1)
    one = _native_zero_kv(1)
    runtime.append_kv(request, 0, one, one)
    prefix = runtime.create_prefix(request)

    with pytest.raises(NotImplementedError, match="snapshots are deferred"):
        runtime.snapshot(prefix=prefix, destination="/tmp/not-written")
    with pytest.raises(NotImplementedError, match="restore is deferred"):
        runtime.restore(source="/tmp/not-read")
    metrics = runtime.metrics()
    assert metrics["snapshot_failures"] == 1
    assert metrics["restore_failures"] == 1
    runtime.release_prefix(prefix)
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_rejects_int32_capacity_overflow():
    """Page IDs and sequence lengths are int32 at the native/Metal ABI."""

    _, page_runtime = _compiled_page_runtime()
    with pytest.raises(ValueError, match="INT32_MAX"):
        page_runtime(1, 1, 1, 128, 16, 2**31, 1, 1)
    with pytest.raises(ValueError, match="int32"):
        page_runtime(1, 1, 1, 128, 16, 1, 2**27, 1)
    with pytest.raises(ValueError, match="256 MiB"):
        page_runtime(1, 1, 1, 128, 16, 1, 1024, 70_000)

    runtime = page_runtime(1, 1, 1, 128, 16, 1, 1, 1)
    with pytest.raises(ValueError, match="INT32_MAX"):
        runtime.allocate_request("too-long", 2**31)
    # A failed replacement constructor must not discard the previously valid
    # runtime owned by this Python object.
    with pytest.raises(ValueError, match="INT32_MAX"):
        runtime.__init__(1, 1, 1, 128, 16, 2**31, 1, 1)
    handle = runtime.allocate_request("still-live", max_tokens=16)
    runtime.release(handle)
    runtime.shutdown()


def test_compiled_page_runtime_page_allocation_respects_request_max_tokens():
    """Native page reservation must match the NumPy request capacity rule."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 4, 4, 1)
    request = runtime.allocate_request("one-page", max_tokens=17)
    runtime.allocate_pages(request, 1)
    with pytest.raises(ValueError, match=r"max_tokens capacity \(2 pages\)"):
        runtime.allocate_pages(request, 2)
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_prefix_and_append_oom_are_transactional():
    """A multi-block COW append must not partially mutate ownership state."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(2, 1, 1, 128, 16, 2, 2, 3)
    base = runtime.allocate_request("oom-base", max_tokens=32)
    runtime.allocate_pages(base, 1)
    one = _native_zero_kv(1)
    runtime.append_kv(base, 0, one, one)
    runtime.append_kv(base, 1, one, one)
    prefix = runtime.create_prefix(base)
    branch = runtime.allocate_request("oom-branch", max_tokens=32)
    runtime.attach_prefix(branch, prefix)
    before_table = runtime.page_table(branch)
    before_metrics = runtime.metrics()

    many = _native_zero_kv(17)
    with pytest.raises(MemoryError, match="capacity exhausted"):
        runtime.append_kv(branch, 0, many, many)

    assert runtime.page_table(branch) == before_table
    assert runtime.metrics() == before_metrics
    assert runtime.metrics()["dispatch_failures"] == 0
    runtime.release(branch)
    runtime.release(base)
    runtime.release_prefix(prefix)
    runtime.shutdown()


def test_compiled_page_runtime_prefix_fork_growth_is_safe_without_gpu():
    """Forking live prefixes must survive repeated PrefixMeta reallocations."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 2, 1, 1)
    request = runtime.allocate_request("fork-growth", max_tokens=16)
    runtime.allocate_pages(request, 1)
    one = _native_zero_kv(1)
    runtime.append_kv(request, 0, one, one)
    prefix = runtime.create_prefix(request)

    forks = [runtime.fork_prefix(prefix) for _ in range(512)]
    assert runtime.metrics()["prefixes"] == len(forks) + 1
    for forked in forks:
        runtime.release_prefix(forked)
    runtime.release_prefix(prefix)
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_value_accumulation_guard_rejects_extreme_v():
    """Finite BF16 V must not overflow a multi-token shader accumulator."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 2, 2, 1)
    request = runtime.allocate_request("value-guard", max_tokens=32)
    runtime.allocate_pages(request, 2)
    keys = _native_zero_kv(2)
    values = np.full((2, 1, 128), np.uint16(0x7F7F), dtype=np.uint16)
    runtime.append_kv(request, 0, keys, values)
    query = np.zeros((1, 128), dtype=np.uint16)
    with pytest.raises(ValueError, match="value accumulation"):
        runtime.paged_decode(request, 0, query)
    assert runtime.metrics()["dispatch_failures"] == 1
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_append_rejects_nonfinite_kv_transactionally():
    """Nonfinite ingress must not change lengths, pages, or range metadata."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 1, 1, 1)
    request = runtime.allocate_request("append-finite", max_tokens=16)
    runtime.allocate_pages(request, 1)
    one = _native_zero_kv(1)
    runtime.append_kv(request, 0, one, one)
    before = runtime.metrics()
    bad = np.full((1, 1, 128), np.uint16(0x7F80), dtype=np.uint16)
    with pytest.raises(ValueError, match="finite BF16"):
        runtime.append_kv(request, 0, bad, one)
    assert runtime.sequence_length(request) == 1
    assert runtime.metrics() == before
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_attention_validation_bytes_ignore_context_length():
    """Decode validation accounts for Q bytes, not every historical KV token."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 3, 2, 2)
    short = runtime.allocate_request("validation-short", max_tokens=16)
    long = runtime.allocate_request("validation-long", max_tokens=32)
    runtime.allocate_pages(short, 1)
    runtime.allocate_pages(long, 2)
    one = _native_zero_kv(1)
    many = _native_zero_kv(32)
    runtime.append_kv(short, 0, one, one)
    runtime.append_kv(long, 0, many, many)
    query = np.zeros((1, 128), dtype=np.uint16)
    before = runtime.metrics()["attention_validation_bytes"]
    before_page_checks = runtime.metrics()["decode_page_resolution_checks"]
    for request in (short, long):
        try:
            runtime.paged_decode(request, 0, query)
        except RuntimeError:
            # A host-only build validates metadata, then fails closed before
            # pipeline dispatch because no Metal device is present.
            pass
    after = runtime.metrics()["attention_validation_bytes"]
    assert after - before == 2 * query.nbytes
    assert runtime.metrics()["decode_page_resolution_checks"] - before_page_checks == 0
    runtime.release(short)
    runtime.release(long)
    runtime.shutdown()


def test_compiled_page_runtime_numerical_guard_counts_failed_dispatch_without_gpu():
    """Finite BF16 extremes fail before Metal pipeline lookup/dispatch."""

    _, page_runtime = _compiled_page_runtime()
    runtime = page_runtime(1, 1, 1, 128, 16, 1, 1, 1)
    request = runtime.allocate_request("numerical-guard", max_tokens=16)
    runtime.allocate_pages(request, 1)
    extreme = np.full((1, 1, 128), np.uint16(0x7F7F), dtype=np.uint16)
    runtime.append_kv(request, 0, extreme, extreme)
    query = np.full((1, 128), np.uint16(0x7F7F), dtype=np.uint16)
    with pytest.raises(ValueError, match="dot product"):
        runtime.paged_decode(request, 0, query, scale=1.0)

    moderate = np.full((1, 1, 128), np.uint16(0x5D31), dtype=np.uint16)
    runtime.release(request)
    request = runtime.allocate_request("score-guard", max_tokens=16)
    runtime.allocate_pages(request, 1)
    runtime.append_kv(request, 0, moderate, moderate)
    query.fill(np.uint16(0x5D31))
    with pytest.raises(ValueError, match="attention score"):
        runtime.paged_decode(request, 0, query, scale=4.0)
    metrics = runtime.metrics()
    assert metrics["dispatches"] == 0
    assert metrics["dispatch_failures"] == 2
    assert metrics["native_dispatches"] == 0
    assert metrics["native_failures"] == metrics["dispatch_failures"]
    runtime.shutdown()


def test_compiled_page_runtime_reuses_decode_buffers_without_kv_pool_copies():
    """Warm decode must use preallocated scratch and live metadata only."""

    native, page_runtime = _compiled_page_runtime()
    if not native.capabilities()["available"]:
        pytest.skip("strict native Metal build is unavailable")

    runtime = page_runtime(1, 1, 1, 128, 16, 4, 2, 2)
    request = runtime.allocate_request("steady-state", max_tokens=16)
    runtime.allocate_pages(request, 1)
    zeros = np.zeros((1, 1, 128), dtype=np.uint16)
    runtime.append_kv(request, 0, zeros, zeros)
    query = np.zeros((1, 128), dtype=np.uint16)

    runtime.paged_decode(request, 0, query)
    warm = runtime.metrics()
    assert warm["buffer_allocations"] > 0
    assert warm["decode_buffer_allocations"] == 0
    assert warm["kv_pool_copies"] == 0
    assert warm["kv_copy_bytes"] > 0

    for _ in range(3):
        runtime.paged_decode(request, 0, query)
    after = runtime.metrics()
    assert after["buffer_allocations"] == warm["buffer_allocations"]
    assert after["decode_buffer_allocations"] == 0
    assert after["kv_pool_copies"] == 0
    assert after["kv_copy_bytes"] == warm["kv_copy_bytes"]
    assert after["metadata_copies"] == warm["metadata_copies"]
    assert after["query_copies"] - warm["query_copies"] == 3
    assert after["output_copies"] - warm["output_copies"] == 3
    assert after["dispatches"] - warm["dispatches"] == 3
    assert after["native_dispatches"] == after["dispatches"]
    runtime.release(request)
    runtime.shutdown()


def test_compiled_page_runtime_multi_slot_layer_lengths_use_distinct_offsets():
    """GPU decode must address sequence metadata as [slot, layer]."""

    native, page_runtime = _compiled_page_runtime()
    if not native.capabilities()["available"]:
        pytest.skip("strict native Metal build is unavailable")

    runtime = page_runtime(2, 1, 1, 128, 16, 4, 1, 2)
    first = runtime.allocate_request("offset-first", max_tokens=16)
    second = runtime.allocate_request("offset-second", max_tokens=16)
    runtime.allocate_pages(first, 1)
    runtime.allocate_pages(second, 1)

    def bf16_bits(value: float) -> np.uint16:
        word = np.asarray([value], dtype=np.float32).view(np.uint32)[0]
        return np.uint16(word >> 16)

    def kv(token_count: int) -> tuple[np.ndarray, np.ndarray]:
        keys = np.zeros((token_count, 1, 128), dtype=np.uint16)
        values = np.empty_like(keys)
        for token in range(token_count):
            values[token, 0, :] = bf16_bits(float(token + 1))
        return keys, values

    expected_lengths = (
        (first, 0, 1),
        (first, 1, 2),
        (second, 0, 3),
        (second, 1, 4),
    )
    for request, layer, token_count in expected_lengths:
        keys, values = kv(token_count)
        runtime.append_kv(request, layer, keys, values)

    query = np.zeros((1, 128), dtype=np.uint16)
    for request, layer, token_count in expected_lengths:
        output = np.frombuffer(
            runtime.paged_decode(request, layer, query), dtype=np.float32
        ).reshape(1, 128)
        expected = np.mean(np.arange(1, token_count + 1, dtype=np.float32))
        np.testing.assert_allclose(output, expected, rtol=2e-2, atol=2e-2)

    runtime.release(first)
    runtime.release(second)
    runtime.shutdown()


def test_compiled_page_runtime_shared_teardown_preserves_other_instances():
    """Module shutdown must not reset resources held by another runtime."""

    native, page_runtime = _compiled_page_runtime()
    first = page_runtime(1, 1, 1, 128, 16, 1, 1, 1)
    second = page_runtime(1, 1, 1, 128, 16, 1, 1, 1)
    first.shutdown()
    native.shutdown()
    request = second.allocate_request("survives-module-shutdown", max_tokens=16)
    second.allocate_pages(request, 1)
    second.release(request)
    second.shutdown()
