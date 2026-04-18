# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for JSON-schema constrained decoding.

Covers:

* ``vllm_mlx.constrained.cache`` — tokenizer-data construction and caching.
* ``vllm_mlx.constrained.json_schema_processor.JSONSchemaLogitsProcessor``
  — mask shape, token filtering, schema enforcement.
* ``vllm_mlx.api.tool_calling.build_json_logits_processor`` — builder glue.
* ``vllm_mlx.api.anthropic_adapter.anthropic_to_openai`` — propagation of
  the OpenAI-compatible ``response_format`` extension field from
  ``/v1/messages`` requests.

These tests are pure-logic: they use a minimal fake tokenizer (ASCII
vocabulary) to avoid pulling in a 200k-entry HF tokenizer at test time.
They skip gracefully when ``lm-format-enforcer`` is not installed, so CI
running with a minimal dependency set still passes.
"""

from __future__ import annotations

import pytest

from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicMessage, AnthropicRequest
from vllm_mlx.api.models import ResponseFormat, ResponseFormatJsonSchema
from vllm_mlx.api.tool_calling import build_json_logits_processor
from vllm_mlx.constrained import is_available
from vllm_mlx.constrained.cache import clear_cache, get_tokenizer_data

# Skip all processor-level tests if the optional dependency is missing.
pytestmark_lmfe = pytest.mark.skipif(
    not is_available(),
    reason="lm-format-enforcer not installed",
)


# ---------------------------------------------------------------------------
# Fake tokenizer — just enough API surface for the cache + processor.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """
    Deterministic, tiny tokenizer for unit tests.

    Vocabulary is a small ASCII set (digits, brackets, whitespace, quotes,
    a handful of letters).  IDs start at 0 and are stable across instances.
    Special tokens (EOS/BOS) occupy the end of the range.
    """

    def __init__(self) -> None:
        chars = list('0123456789{}[]:," \n\ttrueflasnul')
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        self._id_to_tok = unique + ["<eos>", "<bos>"]
        self._tok_to_id = {t: i for i, t in enumerate(self._id_to_tok)}
        self.vocab_size = len(self._id_to_tok)
        self.eos_token_id = self._tok_to_id["<eos>"]
        self.all_special_ids = [
            self._tok_to_id["<eos>"],
            self._tok_to_id["<bos>"],
        ]

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for ch in text:
            if ch in self._tok_to_id:
                out.append(self._tok_to_id[ch])
        return out

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for i in ids:
            if 0 <= i < len(self._id_to_tok):
                tok = self._id_to_tok[i]
                if not tok.startswith("<"):
                    parts.append(tok)
        return "".join(parts)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._tok_to_id)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Tokenizer-data cache.
# ---------------------------------------------------------------------------


@pytestmark_lmfe
class TestTokenizerDataCache:
    def test_builds_data_for_fake_tokenizer(self):
        tok = _FakeTokenizer()
        data = get_tokenizer_data(tok)
        assert data is not None
        assert data.vocab_size == tok.vocab_size

    def test_cache_reuse_same_tokenizer(self):
        tok = _FakeTokenizer()
        first = get_tokenizer_data(tok)
        second = get_tokenizer_data(tok)
        assert first is second  # cached object returned verbatim

    def test_separate_tokenizers_get_separate_entries(self):
        a = _FakeTokenizer()
        b = _FakeTokenizer()
        assert get_tokenizer_data(a) is not get_tokenizer_data(b)


# ---------------------------------------------------------------------------
# build_json_logits_processor() — builder glue.
# ---------------------------------------------------------------------------


class TestBuildJsonLogitsProcessor:
    """These tests don't require lm-format-enforcer — they just check glue."""

    def test_text_returns_none(self):
        tok = _FakeTokenizer()
        result = build_json_logits_processor({"type": "text"}, tok)
        assert result is None

    def test_none_returns_none(self):
        tok = _FakeTokenizer()
        result = build_json_logits_processor(None, tok)
        assert result is None

    def test_unsupported_type_returns_none(self):
        tok = _FakeTokenizer()
        result = build_json_logits_processor({"type": "xml"}, tok)
        assert result is None

    @pytestmark_lmfe
    def test_json_object_builds_processor(self):
        tok = _FakeTokenizer()
        result = build_json_logits_processor({"type": "json_object"}, tok)
        assert result is not None

    @pytestmark_lmfe
    def test_json_schema_builds_processor(self):
        tok = _FakeTokenizer()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }
        result = build_json_logits_processor(response_format, tok)
        assert result is not None

    @pytestmark_lmfe
    def test_json_schema_pydantic_model(self):
        tok = _FakeTokenizer()
        response_format = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="result",
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            ),
        )
        result = build_json_logits_processor(response_format, tok)
        assert result is not None


# ---------------------------------------------------------------------------
# JSONSchemaLogitsProcessor — mask semantics.
# ---------------------------------------------------------------------------


@pytestmark_lmfe
class TestProcessorMask:
    def _make_processor(self, schema: dict | None = None):
        from vllm_mlx.constrained import JSONSchemaLogitsProcessor

        tok = _FakeTokenizer()
        return JSONSchemaLogitsProcessor(schema, tok), tok

    def test_mask_shape_matches_logits(self):
        import mlx.core as mx

        processor, tok = self._make_processor()
        # Emulate: one prompt token already consumed, cursor at first
        # generation step (tokens contains prompt+1 generated).
        prompt = tok.encode(" ")  # a single harmless token as "prompt"
        tokens = mx.array(prompt)
        logits = mx.zeros((tok.vocab_size,))
        masked = processor(tokens, logits)
        assert masked.shape == logits.shape

    def test_mask_shape_matches_2d_logits(self):
        import mlx.core as mx

        processor, tok = self._make_processor()
        prompt = tok.encode(" ")
        tokens = mx.array(prompt)
        logits = mx.zeros((1, tok.vocab_size))
        masked = processor(tokens, logits)
        assert masked.shape == logits.shape

    def test_allows_at_least_one_token_at_start(self):
        """At the start of a JSON value, at least ``{`` or ``[`` must pass."""
        import mlx.core as mx

        processor, tok = self._make_processor()
        prompt = tok.encode(" ")
        tokens = mx.array(prompt)
        logits = mx.zeros((tok.vocab_size,))
        masked = processor(tokens, logits)
        # At least one finite entry must remain.
        finite = (masked != -float("inf")).sum().item()
        assert finite >= 1

    def test_processor_never_crashes_on_arbitrary_state(self):
        """Defensive: even with unexpected token history, returns logits."""
        import mlx.core as mx

        processor, _ = self._make_processor()
        # Pass nonsensical tokens — the processor's except handler should
        # catch any parser error and return the original logits unchanged.
        tokens = mx.array([999999])  # out-of-vocab id
        logits = mx.zeros((8,))
        result = processor(tokens, logits)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# Anthropic adapter — response_format propagation.
# ---------------------------------------------------------------------------


class TestAnthropicAdapterResponseFormat:
    def test_response_format_propagates_dict(self):
        req = AnthropicRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        openai_req = anthropic_to_openai(req)
        assert openai_req.response_format is not None
        # ChatCompletionRequest coerces the dict into ResponseFormat.
        rf_type = (
            openai_req.response_format.type
            if hasattr(openai_req.response_format, "type")
            else openai_req.response_format.get("type")
        )
        assert rf_type == "json_object"

    def test_response_format_json_schema_propagates(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        req = AnthropicRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Give a name")],
            max_tokens=50,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema},
            },
        )
        openai_req = anthropic_to_openai(req)
        assert openai_req.response_format is not None
        rf = openai_req.response_format
        rf_type = rf.type if hasattr(rf, "type") else rf.get("type")
        assert rf_type == "json_schema"

    def test_missing_response_format_is_none(self):
        req = AnthropicRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=50,
        )
        openai_req = anthropic_to_openai(req)
        assert openai_req.response_format is None


# ---------------------------------------------------------------------------
# _simplify_schema — metadata stripping and anyOf flattening.
# ---------------------------------------------------------------------------


class TestSimplifySchema:
    """Test ``_simplify_schema`` handles metadata and nested anyOf correctly."""

    def test_strips_default_and_metadata(self):
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "default": "unknown",
                    "title": "Name",
                    "description": "The name",
                    "examples": ["Alice"],
                },
            },
        }
        result = _simplify_schema(schema)
        name_prop = result["properties"]["name"]
        for kw in ("default", "title", "description", "examples"):
            assert kw not in name_prop, f"{kw!r} should have been stripped"
        assert name_prop["type"] == "string"

    def test_flattens_nested_anyof(self):
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "anyOf": [
                {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                {"type": "null"},
            ]
        }
        result = _simplify_schema(schema)
        # Nested anyOf should be flattened to 3 branches.
        assert "anyOf" in result
        assert len(result["anyOf"]) == 3

    def test_fact_batch_schema_simplifies_cleanly(self):
        """Regression test: the ``fact_batch`` schema that caused enforcer
        to get stuck in production (nested anyOf + default + $ref + not)."""
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        fact_batch_schema = {
            "type": "object",
            "properties": {
                "facts": {
                    "anyOf": [
                        {
                            "anyOf": [
                                {"not": {"$ref": "#/definitions/OpenAiAnyType"}},
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "kind": {
                                                "type": "string",
                                                "enum": [
                                                    "number",
                                                    "price",
                                                    "product_name",
                                                ],
                                            },
                                            "value": {
                                                "type": "string",
                                                "minLength": 1,
                                                "maxLength": 300,
                                            },
                                            "confidence": {
                                                "type": "string",
                                                "enum": [
                                                    "high",
                                                    "medium",
                                                    "low",
                                                ],
                                            },
                                        },
                                        "required": ["kind", "value", "confidence"],
                                        "additionalProperties": False,
                                    },
                                    "maxItems": 6,
                                },
                            ],
                            "default": [],
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
            "definitions": {
                "OpenAiAnyType": {
                    "type": ["string", "number", "integer", "boolean", "array", "null"],
                    "items": {"$ref": "#/definitions/OpenAiAnyType"},
                }
            },
            "$schema": "https://json-schema.org/draft/2019-09/schema#",
        }

        result = _simplify_schema(fact_batch_schema)

        # $schema must be stripped.
        assert "$schema" not in result
        # definitions must be consumed.
        assert "definitions" not in result

        # The ``facts`` property must exist and contain a flat anyOf.
        facts = result["properties"]["facts"]
        assert "anyOf" in facts
        # ``default`` must be stripped from all levels.
        assert "default" not in facts
        for branch in facts["anyOf"]:
            assert "default" not in branch
        # No nested anyOf wrappers should remain — each branch is either
        # the array schema or ``{type: null}``.
        for branch in facts["anyOf"]:
            if "anyOf" in branch:
                # Inner anyOf should have been flattened into the outer one.
                pytest.fail(f"Nested anyOf still present in branch: {branch!r}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
