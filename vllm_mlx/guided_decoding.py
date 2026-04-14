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
        return [processor]
