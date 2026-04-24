# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3.5/3.6 MTP weight fixups."""

import mlx.core as mx

from vllm_mlx.patches.qwen3_5_mtp import (
    _apply_qwen_mtp_rmsnorm_offset_fixups,
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
