# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5 MTP hidden-state checkpoint contract."""

import logging

import pytest

from vllm_mlx.patches.qwen3_5_mtp import (
    _resolve_qwen_mtp_hidden_state_mode,
    _select_qwen_mtp_hidden_state,
)


def test_qwen_mtp_hidden_state_mode_defaults_to_post_norm():
    assert _resolve_qwen_mtp_hidden_state_mode({}) == "post_norm"
    assert _resolve_qwen_mtp_hidden_state_mode({"text_config": {}}) == "post_norm"


@pytest.mark.parametrize(
    "config",
    [
        {"mtp_hidden_state_mode": "pre_norm"},
        {"text_config": {"mtp_hidden_state_mode": "pre_norm"}},
    ],
)
def test_qwen_mtp_hidden_state_mode_accepts_explicit_pre_norm(config):
    assert _resolve_qwen_mtp_hidden_state_mode(config) == "pre_norm"


def test_qwen_mtp_hidden_state_mode_uses_nested_setting_first():
    config = {
        "mtp_hidden_state_mode": "pre_norm",
        "text_config": {"mtp_hidden_state_mode": "post_norm"},
    }

    assert _resolve_qwen_mtp_hidden_state_mode(config) == "post_norm"


@pytest.mark.parametrize("value", ["unknown", None, 1])
def test_qwen_mtp_hidden_state_mode_rejects_unknown_value(value, caplog):
    caplog.set_level(logging.WARNING)

    mode = _resolve_qwen_mtp_hidden_state_mode(
        {"text_config": {"mtp_hidden_state_mode": value}}
    )

    assert mode == "post_norm"
    assert "Unsupported mtp_hidden_state_mode=" in caplog.text


def test_qwen_mtp_hidden_state_selection_preserves_checkpoint_contract():
    pre_norm = object()
    post_norm = object()

    assert _select_qwen_mtp_hidden_state("post_norm", pre_norm, post_norm) is post_norm
    assert _select_qwen_mtp_hidden_state("pre_norm", pre_norm, post_norm) is pre_norm
