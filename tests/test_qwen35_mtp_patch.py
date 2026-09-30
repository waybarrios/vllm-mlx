# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3.5/3.6 MTP weight fixups."""

from types import SimpleNamespace

import pytest

import mlx.core as mx

from vllm_mlx.patches.qwen3_5_mtp import (
    _apply_qwen_mtp_rmsnorm_offset_fixups,
    _strip_mtp_key_prefix,
    _mtp_artifact_uses_mlx_norm_weights,
    _mtp_quantization_companion_key,
    _validate_qwen_moe_mtp_tensor_shapes,
)


def test_qwen_mtp_raw_offset_norm_weights_shift_once():
    weights = {
        "pre_fc_norm_hidden.weight": mx.array([-0.4, 0.0], dtype=mx.float32),
        "layers.0.input_layernorm.weight": mx.array([-0.2, 0.1], dtype=mx.float32),
    }

    shifted = _apply_qwen_mtp_rmsnorm_offset_fixups(weights)

    assert shifted == 2
    assert mx.allclose(
        weights["pre_fc_norm_hidden.weight"],
        mx.array([0.6, 1.0], dtype=mx.float32),
    )
    assert mx.allclose(
        weights["layers.0.input_layernorm.weight"],
        mx.array([0.8, 1.1], dtype=mx.float32),
    )


def test_qwen_mtp_actual_gamma_norm_weights_are_not_shifted_again():
    original = mx.array([0.56, 0.82], dtype=mx.float32)
    weights = {"pre_fc_norm_embedding.weight": original}

    shifted = _apply_qwen_mtp_rmsnorm_offset_fixups(weights)

    assert shifted == 0
    assert mx.allclose(weights["pre_fc_norm_embedding.weight"], original)


def test_qwen_mtp_non_norm_one_dimensional_weights_are_not_shifted():
    original = mx.array([-0.4, 0.0], dtype=mx.float32)
    weights = {"layers.0.mlp.shared_expert_gate.weight": original}

    shifted = _apply_qwen_mtp_rmsnorm_offset_fixups(weights)

    assert shifted == 0
    assert mx.allclose(weights["layers.0.mlp.shared_expert_gate.weight"], original)


def test_qwen_mtp_standalone_shards_accept_both_supported_prefixes():
    assert _strip_mtp_key_prefix("mtp.layers.0.fc.weight") == "layers.0.fc.weight"
    assert (
        _strip_mtp_key_prefix("language_model.mtp.layers.0.fc.weight")
        == "layers.0.fc.weight"
    )
    assert _strip_mtp_key_prefix("language_model.model.norm.weight") is None


def test_qwen_mtp_fused_expert_schema_matches_target_dimensions():
    args = SimpleNamespace(num_experts=2, hidden_size=4, moe_intermediate_size=3)
    weights = {
        "layers.0.mlp.experts.gate_up_proj": mx.zeros((2, 6, 4)),
        "layers.0.mlp.experts.down_proj": mx.zeros((2, 4, 3)),
    }

    _validate_qwen_moe_mtp_tensor_shapes(weights, args)


def test_qwen_mtp_rejects_fused_expert_bias_written_as_weight():
    args = SimpleNamespace(num_experts=2, hidden_size=4, moe_intermediate_size=3)
    weights = {
        "layers.0.mlp.experts.gate_up_proj": mx.zeros((2, 6, 1)),
        "layers.0.mlp.experts.down_proj": mx.zeros((2, 4, 1)),
    }

    with pytest.raises(ValueError, match="expert shape mismatch"):
        _validate_qwen_moe_mtp_tensor_shapes(weights, args)


def test_qwen_mtp_quantization_companions_preserve_fused_tensor_key():
    assert (
        _mtp_quantization_companion_key("layers.0.mlp.experts.gate_up_proj", "scales")
        == "layers.0.mlp.experts.gate_up_proj.scales"
    )
    assert (
        _mtp_quantization_companion_key("layers.0.self_attn.q_proj.weight", "biases")
        == "layers.0.self_attn.q_proj.biases"
    )


def test_qwen_mtp_manifest_marks_converter_norms_as_mlx_gamma(tmp_path):
    weights_path = tmp_path / "weights.safetensors"
    weights_path.touch()
    (tmp_path / "manifest.json").write_text(
        '{"artifact_schema_version": 1, "rmsnorm_weight_format": "mlx_gamma"}\n'
    )

    assert _mtp_artifact_uses_mlx_norm_weights(weights_path)


def test_qwen_mtp_missing_or_invalid_manifest_keeps_legacy_fixup(tmp_path):
    weights_path = tmp_path / "weights.safetensors"
    weights_path.touch()

    assert not _mtp_artifact_uses_mlx_norm_weights(weights_path)
    (tmp_path / "manifest.json").write_text("not json\n")
    assert not _mtp_artifact_uses_mlx_norm_weights(weights_path)

    (tmp_path / "manifest.json").write_text("[]\n")
    assert not _mtp_artifact_uses_mlx_norm_weights(weights_path)
