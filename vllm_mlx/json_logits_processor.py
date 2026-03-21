"""JSON schema constrained decoding via outlines-core.

Provides a logits processor that masks tokens to guarantee valid JSON output
conforming to a given JSON schema. Uses outlines-core's regex-based guide
built from the JSON schema.

Requires: pip install outlines-core
"""

import json
import logging

logger = logging.getLogger(__name__)

try:
    from outlines_core import Guide, Index, Vocabulary
    from outlines_core.json_schema import build_regex_from_schema

    HAS_OUTLINES = True
except ImportError:
    HAS_OUTLINES = False


def _build_vocabulary(tokenizer) -> "Vocabulary":
    """Build outlines-core Vocabulary from a HuggingFace tokenizer."""
    eos_id = tokenizer.eos_token_id
    vocab_map = {}
    for token_str, token_id in tokenizer.get_vocab().items():
        if token_id == eos_id:
            continue
        vocab_map.setdefault(token_str, []).append(token_id)
    return Vocabulary(eos_id, vocab_map)


def build_json_logits_processor(schema, tokenizer):
    """Build a logits processor that constrains output to valid JSON.

    Args:
        schema: JSON schema as dict or string.
        tokenizer: HuggingFace tokenizer with get_vocab() and eos_token_id.

    Returns:
        A callable (tokens, logits) -> logits suitable for mlx-lm's
        logits_processors parameter, or None if outlines-core is not installed.
    """
    if not HAS_OUTLINES:
        logger.warning(
            "outlines-core not installed, JSON schema constraint disabled. "
            "Install with: pip install outlines-core"
        )
        return None

    if isinstance(schema, dict):
        schema = json.dumps(schema)

    try:
        regex = build_regex_from_schema(schema)
    except Exception as e:
        logger.warning("Failed to build regex from JSON schema: %s", e)
        return None

    try:
        vocab = _build_vocabulary(tokenizer)
        index = Index(regex, vocab)
        guide = Guide(index)
    except Exception as e:
        logger.warning("Failed to build constrained decoding guide: %s", e)
        return None

    logger.info("JSON schema constraint active (regex length: %d)", len(regex))

    import mlx.core as mx

    class JsonSchemaProcessor:
        """Stateful logits processor that tracks the guide state."""

        def __init__(self, guide):
            self._guide = guide
            self._started = False

        def __call__(self, tokens, logits):
            # Advance guide with the last generated token
            if self._started and len(tokens) > 0:
                last_token = tokens[-1].item()
                try:
                    self._guide.advance(last_token)
                except Exception:
                    # Token not in guide's allowed set — shouldn't happen
                    # if processor was applied, but fail gracefully
                    logger.warning("Guide advance failed for token %d", last_token)
                    return logits
            self._started = True

            allowed = self._guide.get_tokens()
            if not allowed:
                return logits

            mask = mx.full(logits.shape, float("-inf"))
            allowed_list = list(allowed)
            # Set allowed positions to 0 (logits + 0 = unchanged, logits + -inf = masked)
            for idx in allowed_list:
                if idx < logits.shape[-1]:
                    mask[idx] = 0.0

            return logits + mask

    return JsonSchemaProcessor(guide)
