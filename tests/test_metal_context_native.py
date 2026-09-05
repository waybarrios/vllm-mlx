# SPDX-License-Identifier: Apache-2.0
"""Portable checks for the optional native Metal Context Engine surface."""

from __future__ import annotations

import math

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


def test_unavailable_native_dispatch_fails_precisely():
    from vllm_mlx import _metal_context

    capabilities = _metal_context.capabilities()
    if capabilities["available"]:
        pytest.skip("native extension is available; dispatch is covered on Apple CI")

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
