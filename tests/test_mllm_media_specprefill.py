# SPDX-License-Identifier: Apache-2.0
"""Linux-runnable tests for the request-local media SpecPrefill adapter."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_mlx.mllm_specprefill import (
    SUPPORTED_QWEN_MEDIA_MODULES,
    ModelIdentity,
    SpecPrefillOutcome,
    SpecPrefillRequestConfig,
    can_compose_text_prefix_cache,
    capability_reason,
    model_identity,
    request_eligibility_reason,
    required_media_indices,
    run_media_specprefill,
)

SUPPORTED_IDENTITIES = (
    (
        "mlx_vlm.models.qwen3_vl.qwen3_vl",
        "mlx_vlm.models.qwen3_vl.language",
        "qwen3_vl",
    ),
    (
        "mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe",
        "mlx_vlm.models.qwen3_vl_moe.language",
        "qwen3_vl_moe",
    ),
    (
        "mlx_vlm.models.qwen3_5.qwen3_5",
        "mlx_vlm.models.qwen3_5.language",
        "qwen3_5",
    ),
    (
        "mlx_vlm.models.qwen3_5_moe.qwen3_5_moe",
        "mlx_vlm.models.qwen3_5_moe.language",
        "qwen3_5_moe",
    ),
)


def _model(
    model_module: str,
    language_module: str,
    model_type: str,
    *,
    image_token: int = 90,
    video_token: int = 91,
):
    language_cls = type("LanguageModel", (), {"__module__": language_module})
    model_cls = type(
        "Model",
        (),
        {
            "__module__": model_module,
            "get_input_embeddings": lambda self, **kwargs: None,
        },
    )
    instance = model_cls()
    instance.language_model = language_cls()
    if model_type.startswith("qwen3_5"):
        instance.config = SimpleNamespace(
            model_type=model_type,
            image_token_index=image_token,
            video_token_index=video_token,
            vision_start_token_id=92,
            vision_end_token_id=93,
        )
    else:
        instance.config = SimpleNamespace(
            model_type=model_type,
            image_token_id=image_token,
            video_token_id=video_token,
            vision_start_token_id=92,
            vision_end_token_id=93,
        )
    return instance


def test_public_dataclass_defaults_are_stable():
    assert SpecPrefillRequestConfig() == SpecPrefillRequestConfig(
        enabled=None,
        keep_pct=None,
        backbone_pct=None,
    )
    assert SpecPrefillOutcome() == SpecPrefillOutcome(
        requested=None,
        engaged=False,
        reason="not_evaluated",
        route="mllm_media",
        model_module=None,
        language_module=None,
        model_type=None,
        original_tokens=0,
        selected_tokens=0,
        cached_tokens=0,
    )


@pytest.mark.parametrize(
    ("model_module", "language_module", "model_type"), SUPPORTED_IDENTITIES
)
def test_exact_dense_and_moe_runtime_identities_are_supported(
    model_module, language_module, model_type
):
    model = _model(model_module, language_module, model_type)

    assert model_module in SUPPORTED_QWEN_MEDIA_MODULES
    assert model_identity(model) == ModelIdentity(
        model_module=model_module,
        language_module=language_module,
        model_type=model_type,
    )
    assert capability_reason(model) is None


def test_identity_checks_fail_closed_on_module_or_config_mismatch():
    unsupported = _model("custom.Model", "custom.LanguageModel", "qwen3_5")
    assert capability_reason(unsupported) == "unsupported_model_module"

    mismatched = _model(
        "mlx_vlm.models.qwen3_5.qwen3_5",
        "mlx_vlm.models.qwen3_5_moe.language",
        "qwen3_5_moe",
    )
    assert capability_reason(mismatched) == "model_module_mismatch"


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            image_token_index=90,
            video_token_index=91,
            vision_start_token_id=89,
            vision_end_token_id=92,
        ),
        {
            "image_token_id": 90,
            "video_token_id": 91,
            "vision_start_token_id": 89,
            "vision_end_token_id": 92,
        },
    ],
)
def test_required_media_indices_keep_visual_positions_and_final_token(config):
    input_ids = np.array([[89, 90, 90, 92, 91, 6]])

    assert required_media_indices(input_ids, config) == {0, 1, 2, 3, 4, 5}


def test_media_prefix_cache_only_composes_before_visual_placeholders():
    config = SimpleNamespace(
        image_token_index=90,
        video_token_index=91,
        vision_start_token_id=89,
    )
    input_ids = np.array([[1, 2, 89, 90, 90, 5, 6]])

    assert can_compose_text_prefix_cache(input_ids, config, cached_tokens=2)
    assert not can_compose_text_prefix_cache(input_ids, config, cached_tokens=3)
    assert not can_compose_text_prefix_cache(input_ids, config, cached_tokens=7)
    assert not can_compose_text_prefix_cache(
        np.array([[1, 2, 3]]), config, cached_tokens=1
    )


def test_request_eligibility_honors_controls_and_media_type():
    model = _model(*SUPPORTED_IDENTITIES[2])
    request = SimpleNamespace(
        input_ids=np.array([[1, 90, 2, 3]]),
        audio=None,
    )

    assert (
        request_eligibility_reason(
            model,
            object(),
            request,
            SpecPrefillRequestConfig(enabled=False),
        )
        == "disabled_by_request"
    )
    assert (
        request_eligibility_reason(
            model,
            object(),
            request,
            SpecPrefillRequestConfig(),
            threshold=4,
        )
        == "below_threshold"
    )
    assert (
        request_eligibility_reason(
            model,
            object(),
            request,
            SpecPrefillRequestConfig(enabled=True, keep_pct=0.3, backbone_pct=0.1),
            threshold=100,
        )
        is None
    )
    request.audio = ["clip.wav"]
    assert (
        request_eligibility_reason(
            model,
            object(),
            request,
            SpecPrefillRequestConfig(enabled=True),
        )
        == "unsupported_media_type"
    )


def test_request_eligibility_caps_draft_scoring_length():
    model = _model(*SUPPORTED_IDENTITIES[2])
    input_ids = np.zeros((1, 65537), dtype=np.int64)
    input_ids[0, 5] = 90
    request = SimpleNamespace(input_ids=input_ids, audio=None)

    assert (
        request_eligibility_reason(
            model,
            object(),
            request,
            SpecPrefillRequestConfig(enabled=True),
        )
        == "prompt_too_long"
    )


def _install_numpy_runtime(monkeypatch, selected):
    mx = types.ModuleType("mlx.core")
    mx.uint32 = np.uint32
    mx.array = lambda value, dtype=None: np.array(value, dtype=dtype)
    mx.eval = lambda *values: None
    mx.clear_cache = lambda: None

    mlx = types.ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    calls = {"score": [], "select": []}
    specprefill = types.ModuleType("vllm_mlx.specprefill")

    def score_tokens(model, tokens, **kwargs):
        calls["score"].append((model, list(tokens), kwargs))
        return np.arange(len(tokens), dtype=np.float32)

    def select_chunks(importance, **kwargs):
        calls["select"].append((importance.copy(), kwargs))
        return np.array(selected, dtype=np.uint32)

    specprefill.score_tokens = score_tokens
    specprefill.select_chunks = select_chunks
    specprefill._qwen35_extract_queries = object()
    monkeypatch.setitem(sys.modules, "vllm_mlx.specprefill", specprefill)
    return calls


class _CacheLayer:
    def __init__(self):
        self.state = [np.array([1], dtype=np.float32)]


def test_sparse_media_prefill_gathers_fused_state_and_adjusts_decode_delta(
    monkeypatch,
):
    runtime_calls = _install_numpy_runtime(monkeypatch, selected=[0, 1])
    model = _model(*SUPPORTED_IDENTITIES[0])
    sequence_length = 100
    input_ids = (np.arange(sequence_length, dtype=np.int64) % 50)[None, :]
    input_ids[0, 34] = 92
    input_ids[0, 35:37] = 90
    input_ids[0, 37] = 93
    visual_mask = np.zeros((1, sequence_length), dtype=bool)
    visual_mask[0, 35:37] = True
    fused = np.arange(sequence_length * 3, dtype=np.float32).reshape(1, 100, 3)
    positions = np.stack(
        [np.arange(sequence_length) + axis * 1000 for axis in range(3)], axis=0
    )[:, None, :]
    deepstack = [
        np.array([[10, 11], [12, 13]], dtype=np.float32),
        np.array([[20, 21], [22, 23]], dtype=np.float32),
    ]
    feature_calls = []

    def get_input_embeddings(**kwargs):
        feature_calls.append(kwargs)
        return SimpleNamespace(
            inputs_embeds=fused,
            position_ids=positions,
            rope_deltas=np.array([[-4]], dtype=np.int64),
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
        )

    model.get_input_embeddings = get_input_embeddings

    language_calls = []

    def language_model(tokens, **kwargs):
        language_calls.append((tokens.copy(), kwargs))
        return SimpleNamespace(
            logits=np.ones((1, tokens.shape[1], 7), dtype=np.float32)
        )

    model.language_model.__call__ = language_model
    # Special methods are resolved on the class, so install the callable there.
    type(model.language_model).__call__ = lambda self, tokens, **kwargs: language_model(
        tokens, **kwargs
    )

    request = SimpleNamespace(
        input_ids=input_ids,
        pixel_values=np.array([[5]], dtype=np.float32),
        attention_mask=np.ones_like(input_ids),
        image_grid_thw=np.array([[1, 2, 2]]),
        extra_kwargs={"video_grid_thw": np.array([[1, 2, 2]])},
    )
    cache = [_CacheLayer()]
    draft = object()
    cancellation_checks = []

    logits, decode_delta, selected_count = run_media_specprefill(
        model,
        model.language_model,
        draft,
        request,
        cache,
        keep_pct=0.2,
        backbone_pct=0.0,
        step_size=20,
        position_offset=7,
        cancel_check=lambda: cancellation_checks.append(True),
    )

    expected = [0, 1, *range(32, 64), *range(96, 100)]
    assert selected_count == len(expected) == 38
    assert len(language_calls) == 2
    np.testing.assert_array_equal(
        np.concatenate([call[0] for call in language_calls], axis=1),
        input_ids[:, expected],
    )
    np.testing.assert_array_equal(
        np.concatenate([call[1]["inputs_embeds"] for call in language_calls], axis=1),
        fused[:, expected, :],
    )
    np.testing.assert_array_equal(
        np.concatenate([call[1]["position_ids"] for call in language_calls], axis=-1),
        positions[..., expected] + 7,
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [call[1]["visual_pos_masks"] for call in language_calls], axis=-1
        ),
        visual_mask[..., expected],
    )
    assert "rope_deltas" not in language_calls[0][1]
    assert "mask" not in language_calls[0][1]
    np.testing.assert_array_equal(
        language_calls[0][1]["deepstack_visual_embeds"][0], deepstack[0]
    )
    assert language_calls[1][1]["deepstack_visual_embeds"][0].shape == (0, 2)
    np.testing.assert_array_equal(decode_delta, np.array([[58]]))
    assert logits.shape == (1, 18, 7)
    assert len(feature_calls) == 1
    assert feature_calls[0]["mask"].shape == (1, sequence_length)
    assert runtime_calls["score"][0][0] is draft
    assert runtime_calls["select"][0][1] == {
        "keep_pct": 0.2,
        "backbone_pct": 0.0,
    }
    assert cancellation_checks


def test_sparse_media_prefill_cancels_before_scoring(monkeypatch):
    runtime_calls = _install_numpy_runtime(monkeypatch, selected=[0])
    model = _model(*SUPPORTED_IDENTITIES[2])
    request = SimpleNamespace(input_ids=np.array([[1, 90, 2]]))

    def cancel():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        run_media_specprefill(
            model,
            model.language_model,
            object(),
            request,
            [],
            keep_pct=0.5,
            backbone_pct=0.0,
            step_size=32,
            cancel_check=cancel,
        )

    assert runtime_calls["score"] == []


def test_sparse_media_prefill_masks_only_out_of_vocab_placeholder_ids(monkeypatch):
    runtime_calls = _install_numpy_runtime(monkeypatch, selected=[0])
    model = _model(*SUPPORTED_IDENTITIES[2])
    draft = SimpleNamespace(args=SimpleNamespace(text_config={"vocab_size": 80}))
    input_ids = np.arange(64, dtype=np.int64)[None, :] % 20
    input_ids[0, 31:34] = [92, 90, 93]
    request = SimpleNamespace(
        input_ids=input_ids,
        pixel_values=np.array([[1]], dtype=np.float32),
        attention_mask=np.ones_like(input_ids),
        image_grid_thw=np.array([[1, 1, 1]]),
        extra_kwargs={},
    )
    fused = np.ones((1, 64, 4), dtype=np.float32)
    positions = np.stack([np.arange(64)] * 3, axis=0)[:, None, :]
    model.get_input_embeddings = lambda **kwargs: SimpleNamespace(
        inputs_embeds=fused,
        position_ids=positions,
        rope_deltas=np.zeros((1, 1), dtype=np.int64),
        visual_pos_masks=input_ids == 90,
        deepstack_visual_embeds=None,
    )
    type(model.language_model).__call__ = (
        lambda self, tokens, **kwargs: SimpleNamespace(
            logits=np.ones((1, tokens.shape[1], 7), dtype=np.float32)
        )
    )

    run_media_specprefill(
        model,
        model.language_model,
        draft,
        request,
        [_CacheLayer()],
        keep_pct=0.5,
        backbone_pct=0.0,
        step_size=32,
    )

    assert runtime_calls["score"][0][1][31:34] == [0, 0, 0]
    assert max(runtime_calls["score"][0][1]) < 80


@pytest.mark.parametrize(
    ("draft_model_type", "expects_explicit_extractor"),
    [("qwen3_5", True), ("qwen3_vl", False)],
)
def test_sparse_media_prefill_selects_extractor_from_draft_args(
    monkeypatch, draft_model_type, expects_explicit_extractor
):
    runtime_calls = _install_numpy_runtime(monkeypatch, selected=[0])
    model = _model(*SUPPORTED_IDENTITIES[2])
    draft = SimpleNamespace(
        args=SimpleNamespace(model_type=draft_model_type, vocab_size=100)
    )
    input_ids = np.arange(64, dtype=np.int64)[None, :] % 20
    input_ids[0, 31:34] = [92, 90, 93]
    request = SimpleNamespace(
        input_ids=input_ids,
        pixel_values=np.array([[1]], dtype=np.float32),
        attention_mask=np.ones_like(input_ids),
        image_grid_thw=np.array([[1, 1, 1]]),
        extra_kwargs={},
    )
    positions = np.stack([np.arange(64)] * 3, axis=0)[:, None, :]
    model.get_input_embeddings = lambda **kwargs: SimpleNamespace(
        inputs_embeds=np.ones((1, 64, 4), dtype=np.float32),
        position_ids=positions,
        rope_deltas=np.zeros((1, 1), dtype=np.int64),
        visual_pos_masks=input_ids == 90,
        deepstack_visual_embeds=None,
    )
    type(model.language_model).__call__ = (
        lambda self, tokens, **kwargs: SimpleNamespace(
            logits=np.ones((1, tokens.shape[1], 7), dtype=np.float32)
        )
    )

    run_media_specprefill(
        model,
        model.language_model,
        draft,
        request,
        [_CacheLayer()],
        keep_pct=0.5,
        backbone_pct=0.0,
        step_size=32,
    )

    assert (
        "query_extractor" in runtime_calls["score"][0][2]
    ) is expects_explicit_extractor


def test_sparse_media_prefill_rejects_nonmedia_token_outside_draft_vocab(
    monkeypatch,
):
    runtime_calls = _install_numpy_runtime(monkeypatch, selected=[0])
    model = _model(*SUPPORTED_IDENTITIES[2])
    draft = SimpleNamespace(args=SimpleNamespace(vocab_size=80))
    request = SimpleNamespace(input_ids=np.array([[81, 90, 2]]))

    with pytest.raises(ValueError, match="draft vocabulary"):
        run_media_specprefill(
            model,
            model.language_model,
            draft,
            request,
            [],
            keep_pct=0.5,
            backbone_pct=0.0,
            step_size=32,
        )

    assert runtime_calls["score"] == []
