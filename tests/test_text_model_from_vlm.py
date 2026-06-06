# SPDX-License-Identifier: Apache-2.0
"""Tests for building mlx_lm TextModel from mlx_vlm-loaded weights."""

import json
import sys
import types
from pathlib import Path

import pytest

import vllm_mlx.text_model_from_vlm as text_model_from_vlm
from vllm_mlx.text_model_from_vlm import build_text_model

# VLM+MTP model (created by merging mlx-community VLM + our MTP weights)
VLM_MTP_MODEL = Path.home() / "ai-models/mlx_models/Qwen3.5-35B-A3B-VLM-MTP-8bit"

# Text-only MTP model (no vision tower — can't test VLM loading)
TEXT_MTP_MODEL = Path.home() / "ai-models/mlx_models/Qwen3.5-35B-A3B-8bit"


def test_build_text_model_no_config():
    """Returns None when model path has no config.json."""
    result = build_text_model(None, "/nonexistent/path")
    assert result is None


def test_build_text_model_none_vlm():
    """Returns None when vlm_model is None."""
    result = build_text_model(None, TEXT_MTP_MODEL)
    assert result is None


def test_build_text_model_dispatches_gemma4_text_model(tmp_path, monkeypatch):
    """Gemma 4 text configs should use mlx_lm.models.gemma4_text classes."""

    model_path = tmp_path / "gemma4"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"text_config": {"model_type": "gemma4_text"}})
    )

    class FakeLanguageModel:
        def parameters(self):
            return {}

    class FakeVlmModel:
        language_model = FakeLanguageModel()

    class QwenTextModelArgs:
        @classmethod
        def from_dict(cls, _config):
            raise AssertionError("qwen3_5 dispatch should not be used for gemma4_text")

    class QwenTextModel:
        pass

    qwen_module = types.ModuleType("mlx_lm.models.qwen3_5")
    qwen_module.TextModel = QwenTextModel
    qwen_module.TextModelArgs = QwenTextModelArgs

    class GemmaModelArgs:
        seen_config = None

        @classmethod
        def from_dict(cls, config):
            cls.seen_config = config
            return "gemma4-args"

    class GemmaModel:
        def __init__(self, args):
            self.args = args
            self.loaded_weights = []

        def load_weights(self, weights, strict=False):
            self.loaded_weights.append((weights, strict))

    gemma_module = types.ModuleType("mlx_lm.models.gemma4_text")
    gemma_module.Model = GemmaModel
    gemma_module.ModelArgs = GemmaModelArgs

    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5", qwen_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.gemma4_text", gemma_module)
    monkeypatch.setattr(text_model_from_vlm.mlx.utils, "tree_flatten", lambda _p: [])

    text_model = build_text_model(FakeVlmModel(), model_path)

    assert isinstance(text_model, GemmaModel)
    assert text_model.args == "gemma4-args"
    assert GemmaModelArgs.seen_config == {"model_type": "gemma4_text"}


@pytest.mark.skipif(not VLM_MTP_MODEL.exists(), reason="VLM+MTP model not on disk")
def test_build_text_model_moe():
    """build_text_model creates a TextModel with shared weights and MTP (MoE)."""
    import runtime_patches

    runtime_patches.apply()

    from mlx_vlm import load as vlm_load

    vlm_model, processor = vlm_load(str(VLM_MTP_MODEL))
    text_model = build_text_model(vlm_model, VLM_MTP_MODEL)

    assert text_model is not None, "build_text_model returned None"

    # TextModel should have MTP (config has mtp_num_hidden_layers=1)
    assert hasattr(text_model, "mtp"), "TextModel missing .mtp attribute"
    assert text_model.mtp is not None, "TextModel.mtp is None"
    assert hasattr(text_model, "mtp_forward"), "TextModel missing mtp_forward method"
    assert hasattr(
        text_model, "make_mtp_cache"
    ), "TextModel missing make_mtp_cache method"

    # Verify MoE layer exists in MTP
    mtp_layer = text_model.mtp.layers[0]
    assert hasattr(mtp_layer, "mlp"), "MTP layer missing mlp"


@pytest.mark.skipif(not VLM_MTP_MODEL.exists(), reason="VLM+MTP model not on disk")
def test_text_model_mtp_forward():
    """TextModel.mtp_forward returns logits of correct vocab_size shape."""
    import mlx.core as mx
    import runtime_patches

    runtime_patches.apply()

    from mlx_vlm import load as vlm_load

    vlm_model, _ = vlm_load(str(VLM_MTP_MODEL))
    text_model = build_text_model(vlm_model, VLM_MTP_MODEL)

    config = json.loads((VLM_MTP_MODEL / "config.json").read_text())
    text_config = config.get("text_config", config)

    mtp_cache = text_model.make_mtp_cache()
    assert len(mtp_cache) > 0

    hidden = mx.zeros((1, 1, text_config["hidden_size"]))
    next_ids = mx.array([[0]])
    logits = text_model.mtp_forward(hidden, next_ids, mtp_cache)

    assert (
        logits.shape[-1] == text_config["vocab_size"]
    ), f"Expected vocab_size={text_config['vocab_size']}, got {logits.shape[-1]}"


@pytest.mark.skipif(not VLM_MTP_MODEL.exists(), reason="VLM+MTP model not on disk")
def test_text_model_return_hidden():
    """TextModel supports return_hidden=True (required by mtp_generate_step)."""
    import mlx.core as mx
    import runtime_patches

    runtime_patches.apply()

    from mlx_vlm import load as vlm_load

    vlm_model, _ = vlm_load(str(VLM_MTP_MODEL))
    text_model = build_text_model(vlm_model, VLM_MTP_MODEL)

    config = json.loads((VLM_MTP_MODEL / "config.json").read_text())
    text_config = config.get("text_config", config)

    cache = text_model.make_cache()
    tokens = mx.array([[1, 2, 3]])  # Dummy token IDs

    # return_hidden=True should return (logits, hidden_states)
    result = text_model(tokens, cache=cache, return_hidden=True)

    # Should be a tuple of (logits, hidden)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    logits, hidden = result
    assert logits.shape[-1] == text_config["vocab_size"]
    assert hidden.shape[-1] == text_config["hidden_size"]


@pytest.mark.skipif(not VLM_MTP_MODEL.exists(), reason="VLM+MTP model not on disk")
def test_weight_sharing():
    """Backbone weights are shared (zero-copy) between vlm and TextModel."""
    import mlx.core as mx
    import runtime_patches

    runtime_patches.apply()

    from mlx_vlm import load as vlm_load

    vlm_model, _ = vlm_load(str(VLM_MTP_MODEL))
    text_model = build_text_model(vlm_model, VLM_MTP_MODEL)

    # Compare a backbone weight reference.
    # Layer 0 may be linear_attn (GatedDeltaNet) on MoE models, so find a layer
    # with self_attn (full attention layers are at indices 11, 15, 19, 23, 27).
    for i in range(len(vlm_model.language_model.model.layers)):
        layer = vlm_model.language_model.model.layers[i]
        if hasattr(layer, "self_attn"):
            vlm_weight = layer.self_attn.q_proj.weight
            tm_weight = text_model.model.layers[i].self_attn.q_proj.weight
            assert mx.array_equal(
                vlm_weight, tm_weight
            ), f"Weights at layer {i} should be identical"
            break
    else:
        pytest.fail("No layer with self_attn found")
