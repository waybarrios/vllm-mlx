# SPDX-License-Identifier: Apache-2.0
"""
Targeted regression tests for the constrained JSON logits processor.

The existing ``test_constrained_decoding.py`` covers tokenizer caching,
mask shape, basic ``_simplify_schema`` flows, and incremental decode.
This module pins down the harder corners of the three functions that
issue #500 highlighted as the densest complexity in
``vllm_mlx/constrained/json_schema_processor.py``:

- ``_resolve`` — ``$ref`` cycles, ``$defs`` alias, type arrays, depth limit.
- ``_get_json_context`` — escaped quotes, braces inside strings,
  whitespace after structural openers, nested array-in-object.
- ``__call__`` — disabled-state EOS forcing, EOS guard at incomplete vs
  complete JSON, 2-D logits mask passthrough.

Tests skip cleanly when ``lm-format-enforcer`` is not installed.
"""

from __future__ import annotations

import pytest

from vllm_mlx.constrained import is_available
from vllm_mlx.constrained.cache import clear_cache

pytestmark_lmfe = pytest.mark.skipif(
    not is_available(),
    reason="lm-format-enforcer not installed",
)


# ---------------------------------------------------------------------------
# Minimal ASCII tokenizer with EOS — same shape as the one in
# test_constrained_decoding.py, kept self-contained to avoid cross-file
# imports.
# ---------------------------------------------------------------------------


class _AsciiTokenizer:
    def __init__(self) -> None:
        chars = list('0123456789{}[]:," \n\t\\trueflasnul')
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
        return [self._tok_to_id[c] for c in text if c in self._tok_to_id]

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


def _make_processor(schema: dict | None = None):
    from vllm_mlx.constrained import JSONSchemaLogitsProcessor

    tok = _AsciiTokenizer()
    return JSONSchemaLogitsProcessor(schema, tok), tok


# ---------------------------------------------------------------------------
# _resolve — $ref, $defs, type-as-array, cycles, depth limit
# ---------------------------------------------------------------------------


class TestResolveRefAndDefs:
    def test_resolve_ref_with_extra_keys_merged(self):
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/definitions/Item", "default": 0}},
            "definitions": {"Item": {"type": "integer"}},
        }
        result = _simplify_schema(schema)
        # default is stripped at the top level of the resolved node, but the
        # $ref path itself merges the referenced definition. The point is:
        # the property must end up as the resolved {"type": "integer"} body,
        # not the raw $ref placeholder.
        x = result["properties"]["x"]
        assert "$ref" not in x
        assert x.get("type") == "integer"

    def test_resolve_recursive_ref_breaks_cycle_returns_empty(self):
        """A $ref that resolves to a definition referencing itself must not
        recurse forever; the cycle guard should swap the inner instance for
        ``{}`` (= any), keeping the outer schema usable.
        """
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "$ref": "#/definitions/Node",
            "definitions": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                        "next": {"$ref": "#/definitions/Node"},
                    },
                }
            },
        }
        result = _simplify_schema(schema)
        # Outer has been resolved to the Node body.
        assert result["type"] == "object"
        # The recursive `next` must be flattened to {} (= any) by the cycle
        # guard, never an unresolved $ref.
        next_node = result["properties"]["next"]
        assert "$ref" not in next_node

    def test_resolve_unknown_ref_returns_empty(self):
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/definitions/Missing"}},
        }
        result = _simplify_schema(schema)
        # Unknown $ref must collapse to {} so the enforcer treats it as
        # "any value" rather than crashing.
        assert result["properties"]["x"] == {}

    def test_resolve_dollar_defs_alias_works(self):
        """``$defs`` is the modern alias for ``definitions`` (JSON Schema
        2019-09+). The resolver must accept both."""
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Item"}},
            "$defs": {"Item": {"type": "string"}},
        }
        result = _simplify_schema(schema)
        assert result["properties"]["x"].get("type") == "string"

    def test_resolve_type_as_array_becomes_anyof_with_items(self):
        """``type: [string, array]`` with shared ``items`` must produce a
        per-branch ``items`` only on the array branch."""
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema = {
            "type": ["string", "array"],
            "items": {"type": "integer"},
        }
        result = _simplify_schema(schema)
        assert "type" not in result
        assert "anyOf" in result
        types = sorted(branch["type"] for branch in result["anyOf"])
        assert types == ["array", "string"]
        for branch in result["anyOf"]:
            if branch["type"] == "array":
                assert branch["items"]["type"] == "integer"
            else:
                assert "items" not in branch

    def test_resolve_depth_limit_does_not_crash(self):
        """The resolver hard-caps recursion at depth 12. A pathologically
        deep schema must simplify without raising."""
        from vllm_mlx.constrained.json_schema_processor import _simplify_schema

        schema: dict = {"type": "object"}
        cur = schema
        for _ in range(20):
            cur["properties"] = {"n": {"type": "object"}}
            cur = cur["properties"]["n"]
        # Must not raise.
        result = _simplify_schema(schema)
        assert result["type"] == "object"


# ---------------------------------------------------------------------------
# _get_json_context — string escaping, structural depth, whitespace
# ---------------------------------------------------------------------------


@pytestmark_lmfe
class TestJsonContextStructuralEdges:
    def test_escaped_quote_in_string_does_not_close(self):
        """Inside a string, ``\\"`` must not be treated as a closing quote.
        After feeding ``{"a": "x\\"y`` we are still inside the string."""
        proc, tok = _make_processor()
        text = '{"a": "x\\"y'
        proc._prompt_len = 0
        ctx = proc._get_json_context(tok.encode(text))
        # Still inside the string value, not in_key, not key_start.
        assert ctx == "other"
        assert proc._json_ctx_in_string is True

    def test_braces_inside_string_do_not_change_depth(self):
        """A ``{`` or ``}`` literal that appears inside a JSON string value
        must not move the brace-depth counter."""
        proc, tok = _make_processor()
        text = '{"a": "literal {brace} text"'
        proc._prompt_len = 0
        proc._get_json_context(tok.encode(text))
        # One real ``{`` was opened; the braces inside the string don't count.
        assert proc._brace_depth == 1
        assert proc._json_ctx_in_string is False

    def test_whitespace_after_open_brace_still_key_start(self):
        """``{`` followed by whitespace must still report ``key_start``."""
        proc, tok = _make_processor()
        text = "{ \n\t"
        proc._prompt_len = 0
        ctx = proc._get_json_context(tok.encode(text))
        assert ctx == "key_start"

    def test_nested_array_inside_object_value_is_other(self):
        """``{"a": [`` — we are inside an array (value position), so context
        must be ``other`` even though we're nested in an object."""
        proc, tok = _make_processor()
        text = '{"a": ['
        proc._prompt_len = 0
        ctx = proc._get_json_context(tok.encode(text))
        assert ctx == "other"
        # Container stack: outer object, inner array.
        assert proc._container_stack == ["o", "a"]

    def test_multiple_escapes_in_string_handled(self):
        """Several backslashes in sequence must not desync the parser."""
        proc, tok = _make_processor()
        text = '{"a": "x\\\\\\"y"'
        proc._prompt_len = 0
        ctx = proc._get_json_context(tok.encode(text))
        # Closing quote of the string was processed; we exited the string.
        assert proc._json_ctx_in_string is False
        # Both braces still open (outer object still unclosed).
        assert proc._brace_depth == 1
        assert ctx == "other"

    def test_array_then_brace_not_misidentified_as_key_start(self):
        """Sanity check beyond ``test_brace_after_comma_in_array_is_allowed``:
        ``[{`` straight away (array element start with no comma) must still
        report ``key_start`` once we are inside the new object."""
        proc, tok = _make_processor()
        text = "[{"
        proc._prompt_len = 0
        ctx = proc._get_json_context(tok.encode(text))
        assert ctx == "key_start"
        assert proc._container_stack == ["a", "o"]


# ---------------------------------------------------------------------------
# __call__ — disabled state, EOS guard, 2-D logits
# ---------------------------------------------------------------------------


@pytestmark_lmfe
class TestCallSemantics:
    def test_disabled_processor_forces_eos_via_mask(self):
        """When the processor is disabled, ``__call__`` must mask everything
        except EOS so the request unblocks instead of generating garbage up
        to ``max_tokens``."""
        import mlx.core as mx
        import numpy as np

        proc, tok = _make_processor()
        proc._disabled = True
        prompt = tok.encode(" ")
        tokens = mx.array(prompt)
        logits = mx.zeros((tok.vocab_size,))
        masked = proc(tokens, logits)
        arr = np.array(masked)
        eos = tok.eos_token_id
        assert arr[eos] == 0.0, "EOS must remain finite after masking"
        # Every non-EOS slot must be -inf.
        for i in range(tok.vocab_size):
            if i != eos:
                assert arr[i] == -np.inf, f"slot {i} should be -inf"

    def test_eos_rejected_when_json_incomplete(self):
        """The EOS guard must zero EOS out of the allowed set when the
        decoded suffix does not parse as a complete JSON value yet."""
        import mlx.core as mx
        import numpy as np

        proc, tok = _make_processor()
        # Prime prompt_len with one token, then set state for an open object.
        prompt = tok.encode(" ")
        proc(mx.array(prompt, dtype=mx.int32), mx.zeros((tok.vocab_size,)))
        # Now feed an incomplete JSON suffix.
        partial = "{"
        suffix_ids = tok.encode(partial)
        full = prompt + suffix_ids
        masked = proc(mx.array(full, dtype=mx.int32), mx.zeros((tok.vocab_size,)))
        arr = np.array(masked)
        # Output should not be complete JSON yet, so EOS must be -inf.
        assert arr[tok.eos_token_id] == -np.inf

    def test_eos_allowed_when_json_complete(self):
        """The completion predicate must report ``True`` once the suffix
        decodes to a valid JSON value. The ``__call__`` EOS guard branches on
        this predicate, so its correctness here is the contract that lets
        EOS pass through after the model has finished a JSON object."""
        proc, tok = _make_processor()
        proc._prompt_len = 0
        suffix_ids = tok.encode("{}")
        # Updating the JSON context populates brace/bracket depth so the
        # fast pre-check inside _suffix_is_complete_json can pass.
        proc._get_json_context(suffix_ids)
        assert proc._brace_depth == 0
        assert proc._suffix_is_complete_json(suffix_ids) is True

    def test_2d_logits_returns_2d_mask(self):
        """1-row 2-D logits in must return a 2-D output of the same shape."""
        import mlx.core as mx

        proc, tok = _make_processor()
        prompt = tok.encode(" ")
        tokens = mx.array(prompt)
        logits = mx.zeros((1, tok.vocab_size))
        masked = proc(tokens, logits)
        assert masked.shape == (1, tok.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
