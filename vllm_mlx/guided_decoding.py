# SPDX-License-Identifier: Apache-2.0
"""
Guided decoding helpers for structured output.

This module keeps the public ``response_format`` contract stable while using
Outlines-backed MLX logits processors underneath.
"""

from __future__ import annotations

import json
from typing import Any


def normalize_response_format(
    response_format: Any | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize ``ResponseFormat`` models into a plain dict."""
    if response_format is None:
        return None

    if hasattr(response_format, "type") and hasattr(response_format, "json_schema"):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
        return rf_dict

    return dict(response_format)


def response_format_to_schema(
    response_format: Any | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Convert a response-format request into the JSON schema used for guidance.

    ``json_object`` is treated as an unconstrained object schema, while
    ``json_schema`` uses the user-provided schema verbatim.
    """
    rf_dict = normalize_response_format(response_format)
    if rf_dict is None:
        return None

    format_type = rf_dict.get("type", "text")
    if format_type == "text":
        return None
    if format_type == "json_object":
        return {
            "type": "object",
            "additionalProperties": True,
        }
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema") or {}
        schema = json_schema_spec.get("schema")
        if not isinstance(schema, dict) or not schema:
            raise ValueError(
                "response_format.type='json_schema' requires a non-empty schema"
            )
        return schema

    raise ValueError(f"Unsupported response_format type: {format_type}")


def uses_guided_decoding(
    response_format: Any | dict[str, Any] | None,
) -> bool:
    """Return True when a response format should engage guided decoding."""
    return response_format_to_schema(response_format) is not None


class GuidedDecodingFactory:
    """Create fresh per-request logits processors for structured output."""

    def __init__(self, model: Any, tokenizer: Any):
        try:
            from outlines.models import from_mlxlm
        except ImportError as exc:  # pragma: no cover - dependency failure
            raise ImportError(
                "Structured output guided decoding requires the 'outlines' package"
            ) from exc

        # Outlines TransformerTokenizer reads eos_token_id, eos_token, and
        # all_special_tokens from tokenizer._tokenizer (the raw backend).
        # Modern transformers keeps these on the wrapper only. Patch them
        # onto the raw backend so Outlines can find them.
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None:
            for attr in (
                "eos_token_id",
                "eos_token",
                "pad_token_id",
                "pad_token",
                "all_special_tokens",
            ):
                if not hasattr(inner, attr):
                    val = getattr(tokenizer, attr, None)
                    if val is not None:
                        setattr(inner, attr, val)

        self._outlines_model = from_mlxlm(model, tokenizer)

    def build_processors(
        self,
        response_format: Any | dict[str, Any] | None,
    ) -> list[Any] | None:
        """Build fresh Outlines logits processors for a request."""
        schema = response_format_to_schema(response_format)
        if schema is None:
            return None

        from outlines.backends import get_json_schema_logits_processor

        processor = get_json_schema_logits_processor(
            None,
            self._outlines_model,
            json.dumps(schema, separators=(",", ":"), ensure_ascii=False),
        )
        # Wrap: BatchGenerator passes (tokens: list, logits: mx.array) but
        # Outlines expects (input_ids: mx.array, logits: mx.array).
        return [_wrap_outlines_processor(processor)]

    def build_raw_processors(
        self,
        response_format: Any | dict[str, Any] | None,
    ) -> list[Any] | None:
        """Build raw Outlines processors for serial stream_generate path.

        The serial path in mlx-lm already handles token tracking and passes
        (tokens: mx.array, logits: mx.array) — no wrapping needed.
        """
        schema = response_format_to_schema(response_format)
        if schema is None:
            return None

        from outlines.backends import get_json_schema_logits_processor

        processor = get_json_schema_logits_processor(
            None,
            self._outlines_model,
            json.dumps(schema, separators=(",", ":"), ensure_ascii=False),
        )
        return [processor]


def _wrap_outlines_processor(processor):
    """Adapt Outlines processor to BatchGenerator's (list, mx.array) interface.

    BatchGenerator passes the full token history (prompt + generated) but
    Outlines' FSM expects only generated tokens. Track the prompt length
    on the first call and slice subsequent calls to generated-only.
    """
    import mlx.core as mx

    prompt_len = None

    def wrapped(token_ids, logits):
        nonlocal prompt_len
        import logging

        _log = logging.getLogger("vllm_mlx.guided_decoding")
        if isinstance(token_ids, list):
            if prompt_len is None:
                prompt_len = len(token_ids)
                _log.info("Guided wrapper: prompt_len=%d", prompt_len)
            generated = token_ids[prompt_len:]
            _log.info(
                "Guided wrapper: generated=%d tokens, first_few=%s",
                len(generated),
                generated[:5],
            )
            token_ids = (
                mx.array(generated, dtype=mx.int32)
                if generated
                else mx.zeros((0,), dtype=mx.int32)
            )
        result = processor(token_ids, logits)
        # Debug: check if masking is effective
        if _log.isEnabledFor(logging.DEBUG):
            import numpy as np

            r = np.array(
                result.tolist()[0] if len(result.shape) == 2 else result.tolist()
            )
            finite = np.isfinite(r)
            _log.debug("Guided: %d/%d tokens allowed (non-inf)", finite.sum(), len(r))
        return result

    return wrapped
