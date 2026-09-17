# SPDX-License-Identifier: Apache-2.0
"""Focused tests for phase-one attention backend dispatch and oracle semantics."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_mlx.attention_backend import (
    AttentionBackendCapabilityError,
    AttentionBackendName,
    AttentionGeometry,
    BackendCapabilities,
    ContextBackend,
    discover_capabilities,
    mlx_sdpa_paged_decode_attention,
    numpy_paged_decode_attention,
    resolve_attention_backend,
)


def unavailable_capabilities(reason: str = "test extension unavailable"):
    return BackendCapabilities(
        platform="darwin",
        native_extension=False,
        metal_device=False,
        abi_version=None,
        available=False,
        reason=reason,
    )


class TestBackendDispatch:
    def test_mlx_is_default_and_never_requires_native_extension(self):
        selection = resolve_attention_backend(capabilities=unavailable_capabilities())

        assert selection.requested is AttentionBackendName.MLX
        assert selection.selected is AttentionBackendName.MLX
        assert selection.is_fallback is False

    def test_default_mlx_does_not_probe_or_import_native(self, monkeypatch):
        def fail_probe():
            raise AssertionError("the MLX default must not probe native")

        monkeypatch.setattr(
            "vllm_mlx.attention_backend.discover_capabilities", fail_probe
        )
        monkeypatch.setattr(
            "vllm_mlx.attention_backend.importlib.import_module", fail_probe
        )

        selection = resolve_attention_backend()

        assert selection.selected is AttentionBackendName.MLX
        assert selection.capabilities.probed is False
        assert selection.capabilities.native_extension is False
        assert selection.capabilities.available is False
        assert selection.capabilities.architecture == ""

    def test_auto_is_conservative_even_when_native_is_available(self):
        capabilities = BackendCapabilities(
            platform="darwin",
            native_extension=True,
            metal_device=True,
            abi_version=1,
            available=True,
            serving_ready=True,
        )

        selection = resolve_attention_backend("auto", capabilities=capabilities)

        assert selection.selected is AttentionBackendName.MLX
        assert selection.is_fallback is True
        assert "qualification matrix" in (selection.fallback_reason or "")

    def test_explicit_metal_context_fails_without_capability(self):
        with pytest.raises(
            AttentionBackendCapabilityError,
            match="explicitly requested but is unavailable",
        ) as exc_info:
            resolve_attention_backend(
                "metal-context", capabilities=unavailable_capabilities("no device")
            )

        assert exc_info.value.capabilities.reason == "no device"
        assert "--attention-backend mlx" in str(exc_info.value)

    def test_explicit_metal_context_selects_only_after_probe(self):
        capabilities = BackendCapabilities(
            platform="darwin",
            native_extension=True,
            metal_device=True,
            abi_version=1,
            available=True,
            serving_ready=True,
        )

        selection = resolve_attention_backend(
            "metal-context", capabilities=capabilities
        )

        assert selection.selected is AttentionBackendName.METAL_CONTEXT
        assert selection.is_fallback is False

    def test_kernel_capability_without_executor_still_fails_closed(self):
        capabilities = BackendCapabilities(
            platform="darwin",
            native_extension=True,
            metal_device=True,
            abi_version=1,
            available=True,
        )

        with pytest.raises(
            AttentionBackendCapabilityError,
            match="no serving executor is registered",
        ):
            resolve_attention_backend("metal-context", capabilities=capabilities)

    def test_explicit_metal_context_rejects_non_macos_capability(self):
        capabilities = BackendCapabilities(
            platform="linux",
            native_extension=True,
            metal_device=True,
            abi_version=1,
            available=True,
            serving_ready=True,
        )

        with pytest.raises(AttentionBackendCapabilityError, match="requires macOS"):
            resolve_attention_backend("metal-context", capabilities=capabilities)

    @pytest.mark.parametrize("value", ["bogus", "", "METAL_CONTEXT"])
    def test_invalid_names_are_rejected(self, value):
        with pytest.raises(ValueError, match="Unknown attention backend"):
            resolve_attention_backend(value, capabilities=unavailable_capabilities())

    def test_selection_status_is_serializable(self):
        status = resolve_attention_backend(
            "auto", capabilities=unavailable_capabilities()
        ).as_dict()

        assert status["requested"] == "auto"
        assert status["selected"] == "mlx"
        assert status["capabilities"]["available"] is False
        assert status["capabilities"]["probed"] is False


class TestCapabilityProbe:
    def test_non_macos_fails_closed(self, monkeypatch):
        monkeypatch.setattr("vllm_mlx.attention_backend.sys.platform", "linux")

        capabilities = discover_capabilities()

        assert capabilities.available is False
        assert "requires macOS" in (capabilities.reason or "")

    def test_non_apple_silicon_fails_closed(self, monkeypatch):
        monkeypatch.setattr("vllm_mlx.attention_backend.sys.platform", "darwin")
        monkeypatch.setattr(
            "vllm_mlx.attention_backend._platform.machine", lambda: "x86_64"
        )

        capabilities = discover_capabilities()

        assert capabilities.available is False
        assert "requires Apple Silicon" in (capabilities.reason or "")


class TestGeometry:
    def test_qwen_phase_one_geometry_is_accepted(self):
        AttentionGeometry(
            num_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            block_size=32,
        ).validate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"head_dim": 64},
            {"block_size": 64},
            {"kv_dtype": "float16"},
            {"num_attention_heads": 30, "num_key_value_heads": 8},
        ],
    )
    def test_unsupported_geometry_is_rejected(self, kwargs):
        values = {
            "num_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        values.update(kwargs)
        with pytest.raises(ValueError):
            AttentionGeometry(**values).validate()


class TestNumpyOracle:
    def test_accepts_native_bf16_bit_storage(self):
        query_values = np.asarray([[[1.0], [-2.5]]], dtype=np.float32)
        query_bits = (query_values.view(np.uint32) >> 16).astype(np.uint16)
        key_values = np.full((1, 1, 16, 1), 0.125, dtype=np.float32)
        key_bits = (key_values.view(np.uint32) >> 16).astype(np.uint16)

        output = numpy_paged_decode_attention(
            query_bits,
            key_bits,
            key_bits,
            np.asarray([[0]], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            block_size=16,
            num_kv_heads=1,
            scale=1.0,
        )

        assert output.shape == query_values.shape
        assert np.isfinite(output).all()

    def test_noncontiguous_pages_gqa_and_partial_tail(self):
        rng = np.random.default_rng(11)
        block_size = 16
        query = rng.normal(size=(2, 4, 8)).astype(np.float32)
        key_pages = rng.normal(size=(4, 2, block_size, 8)).astype(np.float32)
        value_pages = rng.normal(size=key_pages.shape).astype(np.float32)
        # Request zero uses physical pages [2, 0] and has a one-token tail;
        # request one uses [1, 3] with a fourteen-token tail.
        page_table = np.asarray([[2, 0], [1, 3]], dtype=np.int32)
        sequence_lengths = np.asarray([17, 30], dtype=np.int32)

        output = numpy_paged_decode_attention(
            query,
            key_pages,
            value_pages,
            page_table,
            sequence_lengths,
            block_size=block_size,
            num_kv_heads=2,
        )

        assert output.shape == query.shape
        assert np.isfinite(output).all()
        assert output.dtype == np.float32

    @pytest.mark.parametrize(
        "mutator, message",
        [
            (lambda table, lengths: (table.astype(np.float32), lengths), "integer"),
            (lambda table, lengths: (table, lengths.astype(np.float32)), "integer"),
            (lambda table, lengths: (table, np.asarray([-1, 1])), "negative"),
            (
                lambda table, lengths: (np.asarray([[99, 0], [1, 3]]), lengths),
                "outside",
            ),
        ],
    )
    def test_invalid_page_metadata_is_rejected(self, mutator, message):
        rng = np.random.default_rng(3)
        query = rng.normal(size=(2, 4, 8)).astype(np.float32)
        key_pages = rng.normal(size=(4, 2, 16, 8)).astype(np.float32)
        value_pages = rng.normal(size=key_pages.shape).astype(np.float32)
        page_table = np.asarray([[2, 0], [1, 3]], dtype=np.int32)
        lengths = np.asarray([17, 30], dtype=np.int32)
        page_table, lengths = mutator(page_table, lengths)

        with pytest.raises(ValueError, match=message):
            numpy_paged_decode_attention(
                query,
                key_pages,
                value_pages,
                page_table,
                lengths,
                block_size=16,
                num_kv_heads=2,
            )

    def test_zero_length_request_returns_zero_without_page_access(self):
        query = np.ones((1, 4, 8), dtype=np.float32)
        key_pages = np.zeros((0, 2, 16, 8), dtype=np.float32)
        value_pages = np.zeros_like(key_pages)

        output = numpy_paged_decode_attention(
            query,
            key_pages,
            value_pages,
            np.zeros((1, 0), dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            block_size=16,
            num_kv_heads=2,
        )

        np.testing.assert_array_equal(output, np.zeros_like(query))

    def test_extreme_but_finite_logits_use_stable_softmax(self):
        query = np.asarray([[[1.0e20]]], dtype=np.float32)
        key_pages = np.zeros((1, 1, 16, 1), dtype=np.float32)
        key_pages[0, 0, :2, 0] = [1.0e-10, 2.0e-10]
        value_pages = np.zeros_like(key_pages)
        value_pages[0, 0, :2, 0] = [1.0, 3.0]

        output = numpy_paged_decode_attention(
            query,
            key_pages,
            value_pages,
            np.asarray([[0]], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
            block_size=16,
            num_kv_heads=1,
            scale=1.0,
        )

        assert np.isfinite(output).all()
        assert output[0, 0, 0] > 2.9

    def test_finite_inputs_with_overflowed_logits_fail_explicitly(self):
        query = np.asarray([[[1.0e20]]], dtype=np.float32)
        key_pages = np.zeros((1, 1, 16, 1), dtype=np.float32)
        key_pages[0, 0, 0, 0] = 1.0e20
        value_pages = np.ones_like(key_pages)

        with pytest.raises(ValueError, match="attention logits overflow"):
            numpy_paged_decode_attention(
                query,
                key_pages,
                value_pages,
                np.asarray([[0]], dtype=np.int32),
                np.asarray([1], dtype=np.int32),
                block_size=16,
                num_kv_heads=1,
                scale=1.0,
            )


class TestMlxOracle:
    @staticmethod
    def _real_mlx():
        mx = pytest.importorskip("mlx.core")
        if not hasattr(mx, "fast") or not hasattr(
            mx.fast, "scaled_dot_product_attention"
        ):
            pytest.skip("real MLX SDPA is unavailable")
        return mx

    def test_sdpa_matches_numpy_for_gqa_and_noncontiguous_pages(self):
        mx = self._real_mlx()
        rng = np.random.default_rng(23)
        block_size = 16
        query = rng.normal(size=(2, 4, 8)).astype(np.float32)
        key_pages = rng.normal(size=(4, 2, block_size, 8)).astype(np.float32)
        value_pages = rng.normal(size=key_pages.shape).astype(np.float32)
        page_table = np.asarray([[2, 0], [1, 3]], dtype=np.int32)
        sequence_lengths = np.asarray([17, 30], dtype=np.int32)

        numpy_output = numpy_paged_decode_attention(
            query,
            key_pages,
            value_pages,
            page_table,
            sequence_lengths,
            block_size=block_size,
            num_kv_heads=2,
        )
        mlx_output = mlx_sdpa_paged_decode_attention(
            mx.array(query),
            mx.array(key_pages),
            mx.array(value_pages),
            page_table,
            sequence_lengths,
            block_size=block_size,
            num_kv_heads=2,
        )
        mx.eval(mlx_output)

        np.testing.assert_allclose(
            np.asarray(mlx_output), numpy_output, rtol=2e-4, atol=2e-4
        )

    def test_sdpa_decodes_uint16_bf16_bits_before_attention(self):
        mx = self._real_mlx()
        rng = np.random.default_rng(29)
        block_size = 16
        query = rng.normal(size=(2, 4, 8)).astype(np.float32)
        key_pages = rng.normal(size=(4, 2, block_size, 8)).astype(np.float32)
        value_pages = rng.normal(size=key_pages.shape).astype(np.float32)

        def bf16_bits(values):
            return (values.view(np.uint32) >> 16).astype(np.uint16)

        query_bits = bf16_bits(query)
        key_bits = bf16_bits(key_pages)
        value_bits = bf16_bits(value_pages)
        page_table = np.asarray([[2, 0], [1, 3]], dtype=np.int32)
        sequence_lengths = np.asarray([17, 30], dtype=np.int32)

        numpy_output = numpy_paged_decode_attention(
            query_bits,
            key_bits,
            value_bits,
            page_table,
            sequence_lengths,
            block_size=block_size,
            num_kv_heads=2,
        )
        mlx_output = mlx_sdpa_paged_decode_attention(
            mx.array(query_bits),
            mx.array(key_bits),
            mx.array(value_bits),
            page_table,
            sequence_lengths,
            block_size=block_size,
            num_kv_heads=2,
        )
        mx.eval(mlx_output)

        np.testing.assert_allclose(
            np.asarray(mlx_output), numpy_output, rtol=2e-4, atol=2e-4
        )


class TestPythonContract:
    def test_protocol_is_runtime_checkable_for_future_adapters(self):
        class StubBackend:
            pass

        assert not isinstance(StubBackend(), ContextBackend)
