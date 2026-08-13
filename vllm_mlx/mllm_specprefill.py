# SPDX-License-Identifier: Apache-2.0
"""Request-local SpecPrefill helpers for supported Qwen media models.

This module deliberately does not use :func:`specprefill.sparse_prefill`.
That helper installs RoPE wrappers on a shared model, which is unsuitable for
continuous batching.  Instead, this adapter asks mlx-vlm for the already fused
media/text embeddings and exact MRoPE positions, selects both at the same
indices, and passes them directly to the language model.

The public identity and eligibility helpers do not import MLX.  Runtime tensor
dependencies are imported lazily by :func:`run_media_specprefill`, allowing
Linux control-plane tests to import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_QWEN_CHUNK_SIZE = 32
_MAX_SCORED_TOKENS = 65536


@dataclass(frozen=True)
class ModelIdentity:
    """Runtime classes and model type used for fail-closed capability checks."""

    model_module: str | None = None
    language_module: str | None = None
    model_type: str | None = None


@dataclass(frozen=True)
class SpecPrefillRequestConfig:
    """Per-request overrides forwarded by the MLLM scheduler."""

    enabled: bool | None = None
    keep_pct: float | None = None
    backbone_pct: float | None = None


@dataclass
class SpecPrefillOutcome:
    """Stable request diagnostics for API and scheduler metadata."""

    requested: bool | None = None
    engaged: bool = False
    reason: str = "not_evaluated"
    route: str = "mllm_media"
    model_module: str | None = None
    language_module: str | None = None
    model_type: str | None = None
    original_tokens: int = 0
    selected_tokens: int = 0
    cached_tokens: int = 0


_SUPPORTED_IDENTITIES: dict[str, ModelIdentity] = {
    "mlx_vlm.models.qwen3_vl.qwen3_vl": ModelIdentity(
        model_module="mlx_vlm.models.qwen3_vl.qwen3_vl",
        language_module="mlx_vlm.models.qwen3_vl.language",
        model_type="qwen3_vl",
    ),
    "mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe": ModelIdentity(
        model_module="mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe",
        language_module="mlx_vlm.models.qwen3_vl_moe.language",
        model_type="qwen3_vl_moe",
    ),
    "mlx_vlm.models.qwen3_5.qwen3_5": ModelIdentity(
        model_module="mlx_vlm.models.qwen3_5.qwen3_5",
        language_module="mlx_vlm.models.qwen3_5.language",
        model_type="qwen3_5",
    ),
    "mlx_vlm.models.qwen3_5_moe.qwen3_5_moe": ModelIdentity(
        model_module="mlx_vlm.models.qwen3_5_moe.qwen3_5_moe",
        language_module="mlx_vlm.models.qwen3_5_moe.language",
        model_type="qwen3_5_moe",
    ),
}


SUPPORTED_QWEN_MEDIA_MODULES = frozenset(_SUPPORTED_IDENTITIES)


@dataclass(frozen=True)
class _PreparedMediaFeatures:
    input_ids: Any
    inputs_embeds: Any
    position_ids: Any
    rope_deltas: Any
    attention_mask: Any
    visual_pos_masks: Any
    deepstack_visual_embeds: Any


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def model_identity(model: Any) -> ModelIdentity:
    """Return the exact runtime identity used by the capability allowlist."""

    language_model = getattr(model, "language_model", None)
    config = getattr(model, "config", None)
    return ModelIdentity(
        model_module=getattr(type(model), "__module__", None),
        language_module=(
            getattr(type(language_model), "__module__", None)
            if language_model is not None
            else None
        ),
        model_type=_config_value(config, "model_type"),
    )


def capability_reason(model: Any) -> str | None:
    """Return ``None`` only for an exact supported Qwen runtime identity."""

    identity = model_identity(model)
    expected = _SUPPORTED_IDENTITIES.get(identity.model_module or "")
    if expected is None:
        return "unsupported_model_module"
    if (
        identity.language_module != expected.language_module
        or identity.model_type != expected.model_type
    ):
        return "model_module_mismatch"
    if not callable(getattr(model, "get_input_embeddings", None)):
        return "embedding_state_unavailable"
    return None


def _flatten_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_list(item))
        return flattened
    return [value]


def _token_list(input_ids: Any) -> list[int]:
    if input_ids is None:
        return []
    value = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    return [int(token) for token in _flatten_list(value)]


def _media_token_ids(model_config: Any) -> set[int]:
    token_ids = set()
    for name in (
        "image_token_index",
        "video_token_index",
        "image_token_id",
        "video_token_id",
    ):
        value = _config_value(model_config, name)
        if value is not None:
            token_ids.add(int(value))
    return token_ids


def _media_boundary_token_ids(model_config: Any) -> set[int]:
    token_ids = set()
    for name in (
        "vision_start_token_id",
        "vision_start_token_index",
        "vision_end_token_id",
        "vision_end_token_index",
    ):
        value = _config_value(model_config, name)
        if value is not None:
            token_ids.add(int(value))
    return token_ids


def _media_placeholder_indices(input_ids: Any, model_config: Any) -> set[int]:
    tokens = _token_list(input_ids)
    media_ids = _media_token_ids(model_config)
    return {index for index, token in enumerate(tokens) if token in media_ids}


def required_media_indices(input_ids: Any, model_config: Any) -> set[int]:
    """Return visual placeholder indices plus the required final prompt token."""

    tokens = _token_list(input_ids)
    if not tokens:
        return set()
    required_ids = _media_token_ids(model_config) | _media_boundary_token_ids(
        model_config
    )
    required = {i for i, token in enumerate(tokens) if token in required_ids}
    required.add(len(tokens) - 1)
    return required


def can_compose_text_prefix_cache(
    input_ids: Any, model_config: Any, cached_tokens: int
) -> bool:
    """Return whether a cached prefix preserves the full visual marker span."""
    tokens = _token_list(input_ids)
    if not tokens or cached_tokens < 0 or cached_tokens >= len(tokens):
        return False
    visual_indices = _media_placeholder_indices(input_ids, model_config)
    if not visual_indices:
        return False

    first_visual = min(visual_indices)
    vision_start_ids = {
        int(value)
        for name in ("vision_start_token_id", "vision_start_token_index")
        if (value := _config_value(model_config, name)) is not None
    }
    marker_indices = [
        index
        for index, token in enumerate(tokens[:first_visual])
        if token in vision_start_ids
    ]
    # get_rope_index uses the vision-start marker to distinguish image and
    # video spans. The uncached suffix must retain that marker. Unknown model
    # configs therefore fail closed unless the whole prompt is recomputed.
    media_span_start = marker_indices[-1] if marker_indices else 0
    return cached_tokens <= media_span_start


def request_eligibility_reason(
    model: Any,
    draft_model: Any,
    request: Any,
    config: SpecPrefillRequestConfig,
    *,
    threshold: int = 0,
) -> str | None:
    """Evaluate request-local controls and static compatibility fail closed."""

    if config.enabled is False:
        return "disabled_by_request"
    reason = capability_reason(model)
    if reason is not None:
        return reason
    if draft_model is None:
        return "draft_unavailable"
    if getattr(request, "audio", None):
        return "unsupported_media_type"
    input_ids = getattr(request, "input_ids", None)
    tokens = _token_list(input_ids)
    if not tokens:
        return "embedding_state_unavailable"
    if len(tokens) > _MAX_SCORED_TOKENS:
        return "prompt_too_long"
    media_indices = _media_placeholder_indices(
        input_ids, getattr(model, "config", None)
    )
    if not media_indices:
        return "unsupported_media_type"
    if config.enabled is not True and len(tokens) <= threshold:
        return "below_threshold"
    keep_pct = config.keep_pct
    backbone_pct = config.backbone_pct
    if keep_pct is not None and not 0.0 < keep_pct <= 1.0:
        return "invalid_request_controls"
    if backbone_pct is not None and not 0.0 <= backbone_pct <= 1.0:
        return "invalid_request_controls"
    if keep_pct is not None and backbone_pct is not None and backbone_pct > keep_pct:
        return "invalid_request_controls"
    return None


def _feature_value(features: Any, name: str) -> Any:
    if isinstance(features, dict):
        return features.get(name)
    return getattr(features, name, None)


def _ensure_batched(array: Any) -> Any:
    if getattr(array, "ndim", None) == 1:
        return array[None, :]
    return array


def _visual_mask_from_tokens(input_ids: Any, model_config: Any, mx: Any) -> Any:
    tokens = _token_list(input_ids)
    media_ids = _media_token_ids(model_config)
    return mx.array([[token in media_ids for token in tokens]])


def _prepare_media_features(
    model: Any, request: Any, mx: Any
) -> _PreparedMediaFeatures:
    input_ids = _ensure_batched(getattr(request, "input_ids", None))
    if input_ids is None or getattr(input_ids, "ndim", None) != 2:
        raise ValueError("media SpecPrefill requires one batched input_ids sequence")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("media SpecPrefill prepares one request at a time")

    kwargs = dict(getattr(request, "extra_kwargs", {}) or {})
    for duplicate in ("input_ids", "pixel_values", "attention_mask"):
        kwargs.pop(duplicate, None)

    attention_mask = getattr(request, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = _ensure_batched(attention_mask)
        kwargs["mask"] = attention_mask

    image_grid_thw = getattr(request, "image_grid_thw", None)
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw

    features = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=getattr(request, "pixel_values", None),
        **kwargs,
    )
    inputs_embeds = _feature_value(features, "inputs_embeds")
    position_ids = _feature_value(features, "position_ids")
    rope_deltas = _feature_value(features, "rope_deltas")
    if inputs_embeds is None or position_ids is None or rope_deltas is None:
        raise ValueError("mlx-vlm did not return complete media embedding state")

    sequence_length = int(input_ids.shape[-1])
    if int(inputs_embeds.shape[-2]) != sequence_length:
        raise ValueError("fused embedding length does not match input_ids")
    if int(position_ids.shape[-1]) != sequence_length:
        raise ValueError("position_ids length does not match input_ids")

    visual_pos_masks = _feature_value(features, "visual_pos_masks")
    if visual_pos_masks is None:
        visual_pos_masks = _visual_mask_from_tokens(
            input_ids, getattr(model, "config", None), mx
        )
    else:
        visual_pos_masks = _ensure_batched(visual_pos_masks)
    if int(visual_pos_masks.shape[-1]) != sequence_length:
        raise ValueError("visual position mask length does not match input_ids")

    return _PreparedMediaFeatures(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        attention_mask=attention_mask,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=_feature_value(features, "deepstack_visual_embeds"),
    )


def _as_indices(value: Any) -> list[int]:
    values = value.tolist() if hasattr(value, "tolist") else value
    return sorted({int(index) for index in _flatten_list(values)})


def _required_chunks(required: set[int], length: int) -> set[int]:
    indices: set[int] = set()
    for index in required:
        start = (index // _QWEN_CHUNK_SIZE) * _QWEN_CHUNK_SIZE
        indices.update(range(start, min(start + _QWEN_CHUNK_SIZE, length)))
    return indices


def _take_sequence(values: Any, indices: Any) -> Any:
    """Gather the sequence axis from a batched embedding tensor."""

    return values[:, indices, :]


def _take_last_axis(values: Any, indices: Any) -> Any:
    return values[..., indices]


def _visual_ordinals(mask: Any, selected: list[int]) -> list[int]:
    flat_mask = [bool(value) for value in _flatten_list(mask.tolist())]
    ordinal_by_position: dict[int, int] = {}
    ordinal = 0
    for position, is_visual in enumerate(flat_mask):
        if is_visual:
            ordinal_by_position[position] = ordinal
            ordinal += 1
    return [
        ordinal_by_position[index] for index in selected if index in ordinal_by_position
    ]


def _take_deepstack(deepstack: Any, ordinals: list[int], mx: Any) -> Any:
    if deepstack is None:
        return None
    indices = mx.array(ordinals, dtype=mx.uint32)
    return [embeds[indices] for embeds in deepstack]


def _cache_eval_values(cache: Any) -> list[Any]:
    values: list[Any] = []
    for layer in cache or ():
        state = getattr(layer, "state", None)
        if isinstance(state, (list, tuple)):
            values.extend(value for value in state if value is not None)
        elif state is not None:
            values.append(state)
        else:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if value is not None:
                    values.append(value)
    return values


def _slice_deepstack_window(
    deepstack: Any,
    selected_visual_mask: Any,
    start: int,
    end: int,
) -> Any:
    if deepstack is None:
        return None
    mask = [bool(value) for value in _flatten_list(selected_visual_mask.tolist())]
    before = sum(mask[:start])
    count = sum(mask[start:end])
    return [embeds[before : before + count] for embeds in deepstack]


def _draft_vocab_size(model: Any) -> int | None:
    args = getattr(model, "args", None)
    value = _config_value(args, "vocab_size")
    if value is None:
        value = _config_value(_config_value(args, "text_config"), "vocab_size")
    if value is None:
        value = _config_value(getattr(model, "config", None), "vocab_size")
    return int(value) if value is not None else None


def _draft_model_type(model: Any) -> str | None:
    for source in (
        getattr(model, "config", None),
        getattr(model, "args", None),
        model,
    ):
        value = _config_value(source, "model_type")
        if value:
            return str(value)
    return None


def run_media_specprefill(
    model: Any,
    language_model: Any,
    draft_model: Any,
    request: Any,
    cache: Any,
    keep_pct: float,
    backbone_pct: float,
    step_size: int,
    position_offset: int = 0,
    force_first_token: bool = False,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[Any, Any, int]:
    """Run media-aware sparse prefill without modifying shared RoPE modules.

    ``request.input_ids`` represents the uncached prompt portion when
    ``position_offset`` is non-zero.  The returned ``decode_rope_delta`` is
    adjusted for the difference between the original and sparse cache lengths
    and must be passed explicitly on every subsequent decode call.
    """

    reason = capability_reason(model)
    if reason is not None:
        raise ValueError(f"media SpecPrefill unavailable: {reason}")
    if language_model is not getattr(model, "language_model", None):
        raise ValueError("language_model does not belong to the target model")
    if draft_model is None:
        raise ValueError("media SpecPrefill requires a draft model")
    if not 0.0 < keep_pct <= 1.0:
        raise ValueError("keep_pct must be in (0, 1]")
    if not 0.0 <= backbone_pct <= keep_pct:
        raise ValueError("backbone_pct must be in [0, keep_pct]")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if position_offset < 0:
        raise ValueError("position_offset must be non-negative")

    import mlx.core as mx

    from .specprefill import _qwen35_extract_queries, score_tokens, select_chunks

    if cancel_check is not None:
        cancel_check()

    tokens = _token_list(getattr(request, "input_ids", None))
    if not tokens:
        raise ValueError("media SpecPrefill requires input_ids")
    config = getattr(model, "config", None)
    required = required_media_indices(getattr(request, "input_ids", None), config)
    media_positions = _media_placeholder_indices(
        getattr(request, "input_ids", None), config
    )
    if not media_positions:
        raise ValueError("media SpecPrefill requires visual placeholder tokens")

    draft_vocab_size = _draft_vocab_size(draft_model)
    if draft_vocab_size is not None:
        media_scoring_ids = _media_token_ids(config) | _media_boundary_token_ids(config)
        score_input_ids = [
            0 if token in media_scoring_ids and token >= draft_vocab_size else token
            for token in tokens
        ]
        if max(score_input_ids) >= draft_vocab_size:
            raise ValueError("draft vocabulary cannot score target prompt tokens")
    else:
        score_input_ids = tokens

    score_kwargs = {
        "prefill_step_size": step_size,
        "cancel_check": cancel_check,
    }
    if _draft_model_type(draft_model) in {"qwen3_5", "qwen3_5_moe"}:
        score_kwargs["query_extractor"] = _qwen35_extract_queries
    importance = score_tokens(draft_model, score_input_ids, **score_kwargs)
    if cancel_check is not None:
        cancel_check()
    selected = select_chunks(
        importance,
        keep_pct=keep_pct,
        backbone_pct=backbone_pct,
    )
    selected_indices = set(_as_indices(selected))
    selected_indices.update(_required_chunks(required, len(tokens)))
    if force_first_token:
        selected_indices.update(range(min(_QWEN_CHUNK_SIZE, len(tokens))))
    selected_list = sorted(selected_indices)
    if not selected_list or selected_list[-1] != len(tokens) - 1:
        raise RuntimeError("SpecPrefill selection omitted the final prompt token")

    if cancel_check is not None:
        cancel_check()
    features = _prepare_media_features(model, request, mx)
    index_array = mx.array(selected_list, dtype=mx.uint32)
    selected_tokens = features.input_ids[:, index_array]
    selected_embeds = _take_sequence(features.inputs_embeds, index_array)
    selected_positions = _take_last_axis(features.position_ids, index_array)
    if position_offset:
        selected_positions = selected_positions + position_offset
    selected_visual_mask = _take_last_axis(features.visual_pos_masks, index_array)
    visual_ordinals = _visual_ordinals(features.visual_pos_masks, selected_list)
    selected_deepstack = _take_deepstack(
        features.deepstack_visual_embeds, visual_ordinals, mx
    )

    selected_count = len(selected_list)
    output = None
    for start in range(0, selected_count, step_size):
        if cancel_check is not None:
            cancel_check()
        end = min(start + step_size, selected_count)
        call_kwargs = {
            "cache": cache,
            "inputs_embeds": selected_embeds[:, start:end, :],
            "position_ids": selected_positions[..., start:end],
            "visual_pos_masks": selected_visual_mask[..., start:end],
        }
        deepstack_window = _slice_deepstack_window(
            selected_deepstack, selected_visual_mask, start, end
        )
        if deepstack_window is not None:
            call_kwargs["deepstack_visual_embeds"] = deepstack_window
        output = language_model(selected_tokens[:, start:end], **call_kwargs)

        if end < selected_count:
            eval_values = _cache_eval_values(cache)
            if eval_values:
                mx.eval(*eval_values)
            mx.clear_cache()

    if output is None:
        raise RuntimeError("media SpecPrefill produced no target output")
    logits = getattr(output, "logits", output)
    mx.eval(logits)

    original_count = len(tokens)
    compression_adjustment = original_count - selected_count
    decode_rope_delta = features.rope_deltas + compression_adjustment
    mx.eval(decode_rope_delta)
    return logits, decode_rope_delta, selected_count


__all__ = [
    "SUPPORTED_QWEN_MEDIA_MODULES",
    "ModelIdentity",
    "SpecPrefillOutcome",
    "SpecPrefillRequestConfig",
    "can_compose_text_prefix_cache",
    "capability_reason",
    "model_identity",
    "request_eligibility_reason",
    "required_media_indices",
    "run_media_specprefill",
]
