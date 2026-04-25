# SPDX-License-Identifier: Apache-2.0
"""
``JSONSchemaLogitsProcessor`` — a ``mlx_lm``-compatible logits processor that
masks the vocabulary so the model can only emit tokens forming a valid JSON
value (optionally matching a JSON schema).

The processor implements the signature expected by ``mlx_lm.generate.generate_step``
and ``vllm_mlx``'s batched engine alike:

    processor(tokens: mx.array, logits: mx.array) -> mx.array

``tokens`` contains the full sequence generated for this request so far
(prompt + previously emitted tokens), and ``logits`` is the last-step logits
row.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import mlx.core as mx
import numpy as np

from .cache import get_tokenizer_data

logger = logging.getLogger(__name__)


class LMFormatEnforcerNotAvailableError(RuntimeError):
    """Raised when ``lm-format-enforcer`` is required but not installed."""


def is_available() -> bool:
    """Return ``True`` iff ``lm-format-enforcer`` is importable."""
    try:
        import lmformatenforcer  # noqa: F401
    except ImportError:
        return False
    return True


# A permissive "any JSON value" schema for ``response_format.type = json_object``.
# JSON spec allows any of {object, array, string, number, boolean, null} at the
# top level, but realistic OpenAI ``json_object`` mode expects an object or
# array at the root.
_GENERIC_JSON_SCHEMA: dict = {
    "anyOf": [
        {"type": "object"},
        {"type": "array"},
    ]
}


def _simplify_schema(schema: dict) -> dict:
    """Pre-process a JSON Schema for ``lm-format-enforcer`` compatibility.

    ``lm-format-enforcer`` does not support ``$ref``, ``not``, ``type`` as an
    array, or recursive definitions.  This function:

    1. Resolves ``$ref`` by inlining referenced definitions (with cycle
       detection so recursive definitions are truncated to ``{}``).
    2. Removes ``not`` sub-schemas (makes the schema more permissive).
    3. Strips metadata / serialisation-hint keywords that the enforcer does
       not understand: ``default``, ``examples``, ``title``, ``description``,
       ``$schema``, ``$id``.
    4. Converts ``type: [t1, t2, ...]`` to ``anyOf: [{type: t1}, ...]``.
    5. Cleans up empty ``anyOf`` / ``oneOf`` branches.
    6. Flattens nested ``anyOf``/``oneOf`` (e.g.
       ``anyOf: [{anyOf: [A, B]}, C]`` → ``anyOf: [A, B, C]``).
    """
    schema = copy.deepcopy(schema)
    definitions: dict = {}
    definitions.update(schema.pop("definitions", {}))
    definitions.update(schema.pop("$defs", {}))

    resolving: set[str] = set()  # cycle guard

    def _resolve(node: Any, depth: int = 0) -> Any:
        if depth > 12 or not isinstance(node, dict):
            return node

        # --- resolve $ref --------------------------------------------------
        if "$ref" in node:
            ref: str = node["$ref"]
            parts = ref.split("/")
            if (
                len(parts) == 3
                and parts[0] == "#"
                and parts[1] in ("definitions", "$defs")
            ):
                name = parts[2]
                if name in definitions and ref not in resolving:
                    resolving.add(ref)
                    resolved = copy.deepcopy(definitions[name])
                    # Merge extra keys (e.g. ``default``) from the $ref node.
                    for k, v in node.items():
                        if k != "$ref" and k not in resolved:
                            resolved[k] = v
                    result = _resolve(resolved, depth + 1)
                    resolving.discard(ref)
                    return result
            # Circular or unresolvable — return empty (= any).
            return {}

        # --- remove unsupported keywords -----------------------------------
        node.pop("not", None)
        node.pop("$schema", None)
        node.pop("$id", None)
        # Metadata / serialisation hints that lm-format-enforcer doesn't
        # understand — keeping them causes the parser to mis-navigate.
        node.pop("default", None)
        node.pop("examples", None)
        node.pop("title", None)
        node.pop("description", None)

        # --- type array → anyOf --------------------------------------------
        if isinstance(node.get("type"), list):
            types = node.pop("type")
            items_schema = node.pop("items", None)
            branches: list[dict] = []
            for t in types:
                branch: dict[str, Any] = {"type": t}
                if t == "array" and items_schema is not None:
                    branch["items"] = _resolve(copy.deepcopy(items_schema), depth + 1)
                branches.append(branch)
            existing = node.pop("anyOf", [])
            node["anyOf"] = existing + branches

        # --- recurse into sub-schemas --------------------------------------
        if "properties" in node and isinstance(node["properties"], dict):
            for k in list(node["properties"]):
                node["properties"][k] = _resolve(node["properties"][k], depth + 1)

        for key in ("items", "additionalProperties"):
            if key in node and isinstance(node[key], dict):
                node[key] = _resolve(node[key], depth + 1)

        for key in ("allOf", "anyOf", "oneOf"):
            if key in node and isinstance(node[key], list):
                # Resolve each branch; drop empty dicts (= "any", redundant
                # inside anyOf since they make the whole constraint trivially
                # true — but keeping one "any" branch confuses the enforcer).
                resolved_items = [_resolve(item, depth + 1) for item in node[key]]
                node[key] = [it for it in resolved_items if it != {}]
                if not node[key]:
                    del node[key]

        # --- flatten nested anyOf/oneOf ------------------------------------
        # ``anyOf: [{anyOf: [A, B]}, C]`` → ``anyOf: [A, B, C]`` when the
        # wrapper dict has no extra keys.  This removes one level of nesting
        # that confuses lm-format-enforcer's UnionParser.
        for key in ("anyOf", "oneOf"):
            if key in node and isinstance(node[key], list):
                flattened: list[Any] = []
                for item in node[key]:
                    if isinstance(item, dict) and key in item and len(item) == 1:
                        flattened.extend(item[key])
                    else:
                        flattened.append(item)
                node[key] = flattened

        return node

    return _resolve(schema)


def _force_no_additional_properties(schema: dict) -> dict:
    """Return a deep copy of *schema* with ``additionalProperties: false``
    injected into every object-type sub-schema that declares ``properties``.

    ``lm-format-enforcer`` has a bug where multi-character tokens spanning
    JSON structural boundaries (e.g., a single token that decodes to ``""``)
    can produce empty or whitespace-only keys, causing ``KeyError`` crashes in
    ``jsonschemaparser.py``.  Setting ``additionalProperties: false`` tells the
    enforcer's trie traversal that only the declared property names are valid
    keys, which significantly narrows the allowed tokens and prevents most of
    these boundary-spanning issues.
    """
    schema = copy.deepcopy(schema)
    _inject_no_additional_props(schema)
    return schema


def _inject_no_additional_props(node: Any) -> None:
    """Recursively inject ``additionalProperties: false`` into *node*."""
    if not isinstance(node, dict):
        return
    if "properties" in node and "additionalProperties" not in node:
        node["additionalProperties"] = False
    for value in node.values():
        if isinstance(value, dict):
            _inject_no_additional_props(value)
        elif isinstance(value, list):
            for item in value:
                _inject_no_additional_props(item)


def _collect_property_names(schema: dict | None) -> set[str]:
    """Collect all property names declared anywhere in *schema*."""
    names: set[str] = set()
    if schema is None:
        return names
    _walk_properties(schema, names)
    return names


def _walk_properties(node: Any, names: set[str]) -> None:
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if isinstance(props, dict):
        names.update(props.keys())
        for v in props.values():
            _walk_properties(v, names)
    for key in ("items", "additionalProperties", "not"):
        if key in node and isinstance(node[key], dict):
            _walk_properties(node[key], names)
    for key in ("allOf", "anyOf", "oneOf"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _walk_properties(item, names)


class JSONSchemaLogitsProcessor:
    """
    Logits processor that constrains generation to valid JSON.

    Parameters
    ----------
    schema:
        The JSON Schema the output must match.  When ``None``, any valid JSON
        object/array is accepted (``json_object`` mode).
    tokenizer:
        The tokenizer used for generation.  Its vocabulary is iterated once
        (via :mod:`vllm_mlx.constrained.cache`) and cached for subsequent
        requests.
    """

    def __init__(
        self,
        schema: dict | None,
        tokenizer: Any,
    ) -> None:
        if not is_available():
            raise LMFormatEnforcerNotAvailableError(
                "lm-format-enforcer is not installed. "
                'Install it with `pip install "lm-format-enforcer>=0.10.9"`.'
            )

        from lmformatenforcer import JsonSchemaParser, TokenEnforcer

        self._tokenizer = tokenizer
        self._schema = schema
        self._tok_data = get_tokenizer_data(tokenizer)
        if self._tok_data is None:
            raise LMFormatEnforcerNotAvailableError(
                "Could not build TokenEnforcerTokenizerData for this tokenizer."
            )

        # Pre-process schema: resolve $ref, remove 'not', convert type arrays.
        # Then harden: force ``additionalProperties: false`` on all object
        # sub-schemas to prevent the enforcer from allowing arbitrary keys.
        self._disabled = False
        if schema is not None:
            parser_schema = _simplify_schema(schema)
            parser_schema = _force_no_additional_properties(parser_schema)
        else:
            parser_schema = _GENERIC_JSON_SCHEMA

        try:
            self._parser = JsonSchemaParser(parser_schema)
            self._enforcer = TokenEnforcer(self._tok_data, self._parser)
        except Exception as exc:
            logger.warning(
                "JSONSchemaLogitsProcessor: enforcer init failed (%s); "
                "falling back to unconstrained generation",
                exc,
            )
            self._disabled = True
            self._parser = None  # type: ignore[assignment]
            self._enforcer = None  # type: ignore[assignment]

        # Bootstrap the enforcer's ``prefix_states`` with the empty tuple so
        # that subsequent ``get_allowed_tokens([t1, t2, ...])`` calls can find
        # their ``prev_step_tuple`` and apply characters incrementally rather
        # than treating the whole sequence as a prompt and resetting to the
        # root parser.
        if not self._disabled:
            try:
                self._enforcer.get_allowed_tokens([])
            except Exception as exc:
                logger.warning(
                    "TokenEnforcer bootstrap failed (%s); "
                    "falling back to unconstrained generation",
                    exc,
                )
                self._disabled = True

        self._prompt_len: int | None = None
        self._vocab_size: int = self._tok_data.vocab_size

        # EOS/stop tokens cache.
        eos_id = getattr(self._tok_data, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple, set)):
            self._eos_set: set[int] = {int(e) for e in eos_id}
        elif eos_id is not None:
            self._eos_set = {int(eos_id)}
        else:
            self._eos_set = set()

        # Pre-compute valid property name prefixes for key-start filtering.
        all_names = _collect_property_names(schema)
        self._valid_key_first_chars: set[str] = {n[0] for n in all_names if n}
        self._valid_key_names: set[str] = all_names

        # Lazy decode cache — populated on demand.
        self._token_decode_cache: dict[int, str | None] = {}

        # Suffix decode cache keyed by length.  Full tokenizer.decode()
        # is always used (incremental per-token decode is incorrect for
        # BPE/SentencePiece tokenizers where whitespace is a token prefix).
        self._cached_suffix_text: str = ""
        self._cached_suffix_len: int = 0

        # Incremental JSON context state — avoids re-scanning the full
        # decoded text on every step.
        self._json_ctx_in_string: bool = False
        self._json_ctx_last_quote_pos: int = -1
        self._json_ctx_scanned_len: int = 0

        # Bracket/brace depth counters for fast _suffix_is_complete_json
        # pre-check.  Updated by _get_json_context incrementally.  JSON
        # is only potentially complete when both are zero.
        self._brace_depth: int = 0
        self._bracket_depth: int = 0

    # ------------------------------------------------------------------

    def _suffix(self, tokens_list: list[int]) -> list[int]:
        """Return the slice of ``tokens`` that corresponds to generated output."""
        if self._prompt_len is None:
            self._prompt_len = max(0, len(tokens_list) - 1)
        return tokens_list[self._prompt_len :]

    def _decode_token_cached(self, tok_id: int) -> str | None:
        """Return the decoded text for a single token (cached)."""
        cached = self._token_decode_cache.get(tok_id)
        if cached is not None:
            return cached
        if tok_id in self._token_decode_cache:
            return None  # previously cached as None
        try:
            decoded = self._tokenizer.decode([tok_id])
        except Exception:
            self._token_decode_cache[tok_id] = None
            return None
        result = decoded if isinstance(decoded, str) else None
        self._token_decode_cache[tok_id] = result
        return result

    def _decode_suffix(self, suffix: list[int]) -> str | None:
        """Decode suffix tokens to text.

        Always uses full ``tokenizer.decode(suffix)`` which is correct for
        all tokenizer families (BPE, SentencePiece, etc.).  Per-token
        concatenation is NOT safe because whitespace may be encoded as a
        token prefix (e.g. ``decode([1526]) = "world"`` but in context
        ``decode([22557, 1526]) = "Hello world"``).

        Results are cached by suffix length to avoid redundant decodes
        within the same generation step (``_get_json_context`` and
        ``_suffix_is_complete_json`` both call this method).
        """
        if not suffix:
            self._cached_suffix_text = ""
            self._cached_suffix_len = 0
            return ""

        suffix_len = len(suffix)

        # Fast path: already decoded this exact suffix length.
        if suffix_len == self._cached_suffix_len:
            return self._cached_suffix_text

        # Full decode (correct for all tokenizer families).
        try:
            decoded = self._tokenizer.decode(list(suffix))
        except Exception:
            return None
        result = decoded if isinstance(decoded, str) else ""

        # Validate prefix stability for incremental JSON context scanning.
        # decode(tokens[:n]) must be a prefix of decode(tokens[:n+1]) for
        # the incremental scanner in _get_json_context to be correct.
        if (
            suffix_len > self._cached_suffix_len
            and self._cached_suffix_len > 0
            and not result.startswith(self._cached_suffix_text)
        ):
            # Prefix changed — reset incremental context state.
            self._json_ctx_scanned_len = 0
            self._json_ctx_in_string = False
            self._json_ctx_last_quote_pos = -1
            self._brace_depth = 0
            self._bracket_depth = 0

        self._cached_suffix_text = result
        self._cached_suffix_len = suffix_len
        return result

    def _suffix_is_complete_json(self, suffix: list[int]) -> bool:
        """Return True if the decoded ``suffix`` parses as a complete JSON value.

        Uses cached bracket/brace depth from ``_get_json_context`` as a
        fast pre-check: JSON cannot be complete when brackets are
        unbalanced or we are inside a string.  This avoids the expensive
        ``json.loads`` call on ~99% of steps.
        """
        if not suffix:
            return False
        # Fast pre-check using cached structural state.
        if self._brace_depth != 0 or self._bracket_depth != 0:
            return False
        if self._json_ctx_in_string:
            return False
        text = self._decode_suffix(suffix)
        if not text:
            return False
        text = text.strip()
        if not text:
            return False
        try:
            json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return False
        return True

    def _get_json_context(self, suffix: list[int]) -> str:
        """Determine the JSON structural context of the current suffix.

        Processes only newly appended characters instead of re-scanning
        the full decoded text on every call (O(1) amortised per step
        instead of O(n)).

        Returns one of:
        - ``"key_start"``: expecting a new key (after ``{`` or ``,``)
        - ``"in_key"``: inside an open key string
        - ``"other"``: any other position
        """
        text = self._decode_suffix(suffix)
        if text is None or not text:
            return "other"

        text_len = len(text)

        if text_len > self._json_ctx_scanned_len and self._json_ctx_scanned_len > 0:
            # Incremental scan: process only new characters.
            in_string = self._json_ctx_in_string
            last_quote_pos = self._json_ctx_last_quote_pos
            brace_depth = self._brace_depth
            bracket_depth = self._bracket_depth
            i = self._json_ctx_scanned_len
            while i < text_len:
                ch = text[i]
                if in_string:
                    if ch == "\\" and i + 1 < text_len:
                        i += 2
                        continue
                    if ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                        last_quote_pos = i
                    elif ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                    elif ch == "[":
                        bracket_depth += 1
                    elif ch == "]":
                        bracket_depth -= 1
                i += 1
            self._json_ctx_in_string = in_string
            self._json_ctx_last_quote_pos = last_quote_pos
            self._json_ctx_scanned_len = text_len
            self._brace_depth = brace_depth
            self._bracket_depth = bracket_depth
        else:
            # Full scan (first call or text shrank/reset).
            in_string = False
            last_quote_pos = -1
            brace_depth = 0
            bracket_depth = 0
            i = 0
            while i < text_len:
                ch = text[i]
                if in_string:
                    if ch == "\\" and i + 1 < text_len:
                        i += 2
                        continue
                    if ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                        last_quote_pos = i
                    elif ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                    elif ch == "[":
                        bracket_depth += 1
                    elif ch == "]":
                        bracket_depth -= 1
                i += 1
            self._json_ctx_in_string = in_string
            self._json_ctx_last_quote_pos = last_quote_pos
            self._json_ctx_scanned_len = text_len
            self._brace_depth = brace_depth
            self._bracket_depth = bracket_depth

        if self._json_ctx_in_string:
            before = text[: self._json_ctx_last_quote_pos].rstrip()
            if not before or before[-1] in ("{", ","):
                return "in_key"
            return "other"

        stripped = text.rstrip()
        if not stripped:
            return "other"
        if stripped[-1] in ("{", ","):
            return "key_start"
        return "other"

    def _filter_at_key_context(
        self, context: str, suffix: list[int], allowed: list[int]
    ) -> list[int]:
        """Apply schema-aware filtering when in key-related context.

        At ``key_start``: only allow tokens that begin a valid key, whitespace,
        ``}``, or just ``"``.
        At ``in_key``: only allow tokens compatible with continuing a valid
        property name (no leading whitespace; content must be a valid prefix).
        """
        if not self._valid_key_names:
            return allowed  # no schema info → skip filtering

        if context == "key_start":
            return self._filter_key_start_tokens(suffix, allowed)
        elif context == "in_key":
            return self._filter_in_key_tokens(suffix, allowed)
        return allowed

    def _filter_key_start_tokens(
        self, suffix: list[int], allowed: list[int]
    ) -> list[int]:
        """Filter tokens at key-start position.

        Only permit tokens that:
        - Are whitespace-only (before the key ``"``)
        - Decode to ``}`` (close object)
        - Start a valid key: ``"`` followed by a valid first char
        """
        result = []
        for tok_id in allowed:
            if tok_id in self._eos_set:
                continue  # EOS at key-start is handled separately
            tok_text = self._decode_token_cached(tok_id)
            if tok_text is None:
                result.append(tok_id)
                continue
            stripped = tok_text.lstrip()
            if not stripped:
                # Pure whitespace — allowed before key
                result.append(tok_id)
                continue
            if stripped[0] == "}":
                # Closing brace — end of object
                result.append(tok_id)
                continue
            if stripped[0] == '"':
                # Opening a key — validate content
                rest = stripped[1:]
                if not rest:
                    # Just ``"`` — will be validated on next step
                    result.append(tok_id)
                    continue
                # Check if rest starts with a valid key character
                if rest[0] in self._valid_key_first_chars:
                    # Further check: does the key content (up to closing ``"``)
                    # match a prefix of a known property name?
                    close_idx = rest.find('"')
                    if close_idx < 0:
                        # Key not yet closed — check prefix
                        if self._is_valid_key_prefix(rest):
                            result.append(tok_id)
                    else:
                        # Key fully contained in this token
                        key_name = rest[:close_idx]
                        if key_name in self._valid_key_names:
                            result.append(tok_id)
                    continue
                # First char not in valid set → skip
                continue
            # Other chars (digits, letters without quote) → skip at key-start
            continue
        return result if result else allowed  # safety: never return empty

    def _filter_in_key_tokens(self, suffix: list[int], allowed: list[int]) -> list[int]:
        """Filter tokens when we're inside an open key string.

        Only allow tokens whose content continues a valid property name.
        Reject whitespace-only/leading-whitespace tokens.
        """
        # Figure out what key content we've accumulated so far.
        text = self._decode_suffix(suffix)
        if text is None:
            return allowed

        # Find the last unmatched ``"`` — everything after it is key content
        # accumulated so far.
        last_open = text.rfind('"')
        if last_open < 0:
            return allowed
        key_so_far = text[last_open + 1 :]

        result = []
        for tok_id in allowed:
            tok_text = self._decode_token_cached(tok_id)
            if tok_text is None:
                result.append(tok_id)
                continue
            # Token must not start with whitespace (no ws inside keys)
            if tok_text and tok_text[0] in (" ", "\t", "\n", "\r"):
                continue
            # Check if key_so_far + tok_text is a valid key prefix
            candidate = key_so_far + tok_text
            # If the closing ``"`` is in tok_text, extract the full key
            close_idx = tok_text.find('"')
            if close_idx >= 0:
                full_key = key_so_far + tok_text[:close_idx]
                if full_key in self._valid_key_names:
                    result.append(tok_id)
            else:
                # Key still open — check if it's a valid prefix
                if self._is_valid_key_prefix(candidate):
                    result.append(tok_id)
        return result if result else allowed

    def _is_valid_key_prefix(self, prefix: str) -> bool:
        """Return True if *prefix* is a prefix of at least one valid key name."""
        return any(name.startswith(prefix) for name in self._valid_key_names)

    def _build_allow_mask(self, allowed: list[int], vocab_size: int) -> mx.array:
        """
        Build a 1-D mask of length ``vocab_size`` where allowed positions are
        ``0`` and disallowed positions are ``-inf``.

        Uses numpy for mask construction (C-level speed) instead of a
        Python loop over ``vocab_size`` elements.
        """
        if not allowed:
            return mx.full((vocab_size,), -float("inf"))
        allowed_clamped = [i for i in allowed if 0 <= i < vocab_size]
        if not allowed_clamped:
            return mx.full((vocab_size,), -float("inf"))
        buf = np.full(vocab_size, -np.inf, dtype=np.float32)
        buf[allowed_clamped] = 0.0
        return mx.array(buf)

    # ------------------------------------------------------------------

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply the allowed-tokens mask to ``logits``."""
        if self._disabled:
            return logits

        try:
            tokens_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            if isinstance(tokens_list, int):
                tokens_list = [tokens_list]
            elif tokens_list and isinstance(tokens_list[0], list):
                tokens_list = tokens_list[0]

            suffix = self._suffix(tokens_list)
            # Use prompt_len directly instead of O(n) list comparison.
            pass_to_enforcer = suffix if self._prompt_len else tokens_list
            allowed_result = self._enforcer.get_allowed_tokens(pass_to_enforcer)
            allowed = getattr(allowed_result, "allowed_tokens", allowed_result)
            if allowed is None:
                return logits

            allowed_list = list(allowed)

            # --- Schema-aware key filter (before EOS guard so that the
            # incremental JSON context state and bracket depth counters
            # are up-to-date for the _suffix_is_complete_json pre-check).
            context = self._get_json_context(suffix)
            if context in ("key_start", "in_key"):
                allowed_list = self._filter_at_key_context(
                    context, suffix, allowed_list
                )

            # --- EOS guard: only permit EOS when output is valid JSON ---
            if (
                self._eos_set
                and any(t in self._eos_set for t in allowed_list)
                and not self._suffix_is_complete_json(suffix)
            ):
                allowed_list = [t for t in allowed_list if t not in self._eos_set]

            # --- Recovery: if enforcer returns empty set AND output is not
            # complete JSON, the schema is likely unsupported — disable the
            # processor and let the model generate freely (system prompt +
            # post-validation still apply).  Only force EOS if the output
            # already parses as valid JSON (generation is done).
            if not allowed_list:
                if self._suffix_is_complete_json(suffix) and self._eos_set:
                    allowed_list = sorted(self._eos_set)
                else:
                    logger.warning(
                        "JSONLP: enforcer stuck (empty allowed-set at "
                        "suffix_len=%d); disabling constrained decoding "
                        "for this request",
                        len(suffix),
                    )
                    self._disabled = True
                    return logits

            actual_vocab = logits.shape[-1]
            mask = self._build_allow_mask(allowed_list, actual_vocab)
            if logits.ndim == 2 and logits.shape[0] == 1:
                mask = mask[None, :]
            return logits + mask
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "JSONSchemaLogitsProcessor crashed; disabling for this request: %s",
                exc,
            )
            self._disabled = True
            return logits

    # Diagnostic helpers -------------------------------------------------

    @property
    def schema(self) -> dict | None:
        return self._schema

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
