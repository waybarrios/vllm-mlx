# SPDX-License-Identifier: Apache-2.0
"""
Per-token log probabilities for OpenAI-compatible ``logprobs``.

The engines already compute ``log_softmax`` over the vocabulary at every
decode step. That happens after logits processors (JSON-schema masks, logit
bias, repetition/presence penalties) and before temperature, top-p, top-k and
min-p, so the values reported here are the processed distribution at T=1.

``extract_token_logprob`` turns one row of that array into plain Python
values. Call it on the engine thread: MLX arrays must be evaluated on the
thread that built them, and no MLX arrays should reach the API layer.
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import mlx.core as mx

if TYPE_CHECKING:
    from .api.models import ChoiceLogprobs, CompletionLogprobs

logger = logging.getLogger(__name__)

# OpenAI reports -9999.0 for very unlikely tokens. The same floor replaces the
# -inf values that masked logits (e.g. JSON-schema constraints) produce, which
# are not valid JSON.
LOGPROB_FLOOR = -9999.0
# OpenAI's limit; ``ChatCompletionRequest.top_logprobs`` enforces the same bound.
MAX_TOP_LOGPROBS = 20


class TopToken(NamedTuple):
    """One of the most likely tokens at a position."""

    token_id: int
    token: str
    logprob: float
    # Raw UTF-8 bytes of the token; ``None`` falls back to the text's bytes.
    bytes: Optional[List[int]] = None


@dataclass
class TokenLogprob:
    """Log probability of one generated token and its most likely alternatives."""

    token_id: int
    token: str
    logprob: float
    # Most likely first
    top_logprobs: List[TopToken] = field(default_factory=list)
    # Raw UTF-8 bytes of the token; ``None`` falls back to the text's bytes.
    bytes: Optional[List[int]] = None


def _clamp_to_floor(value: float) -> float:
    if math.isnan(value) or value < LOGPROB_FLOOR:
        return LOGPROB_FLOOR
    return value


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        logger.debug(
            "Could not decode token id %d for logprobs", token_id, exc_info=True
        )
        return ""


def _gpt2_byte_decoder() -> Dict[str, int]:
    """Invert GPT-2's byte-to-unicode map, which byte-level BPE vocabularies use."""
    direct = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    remapped = [b for b in range(256) if b not in direct]
    decoder = {chr(b): b for b in direct}
    decoder.update({chr(256 + n): b for n, b in enumerate(remapped)})
    return decoder


_GPT2_BYTE_DECODER = _gpt2_byte_decoder()
_BYTE_FALLBACK_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _byte_level_bytes(piece: str) -> List[int]:
    return [_GPT2_BYTE_DECODER[char] for char in piece]


def _byte_fallback_bytes(piece: str) -> Optional[List[int]]:
    match = _BYTE_FALLBACK_PIECE.fullmatch(piece)
    return [int(match.group(1), 16)] if match else None


def _piece_decoder(tokenizer: Any) -> Optional[Callable[[str], Optional[List[int]]]]:
    """
    Return a function from vocabulary piece to raw bytes, or ``None``.

    Decoding one token returns U+FFFD when the token holds part of a
    multi-byte character, so raw bytes come from the vocabulary piece. Only
    Hugging Face fast tokenizers whose decoder fixes the piece-to-byte mapping
    qualify: byte-level BPE (GPT-2, Llama 3, Qwen) and SentencePiece byte
    fallback (``<0xF0>`` pieces).
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    try:
        spec = json.loads(backend.decoder.__getstate__())
    except Exception:
        return None
    if spec.get("type") == "ByteLevel":
        return _byte_level_bytes
    if spec.get("type") == "Sequence" and any(
        step.get("type") == "ByteFallback" for step in spec.get("decoders", [])
    ):
        return _byte_fallback_bytes
    return None


def _text_bytes(text: str) -> Optional[List[int]]:
    # U+FFFD stands for bytes the text cannot carry, so encoding it would lie.
    if "\ufffd" in text:
        return None
    return list(text.encode("utf-8"))


def _token_bytes(
    tokenizer: Any,
    piece_decoder: Optional[Callable[[str], Optional[List[int]]]],
    token_id: int,
    text: str,
) -> Optional[List[int]]:
    if piece_decoder is not None:
        try:
            raw = piece_decoder(tokenizer.convert_ids_to_tokens(token_id))
        except Exception:
            # Added tokens stored as plain text, or ids past the tokenizer
            # vocabulary (``convert_ids_to_tokens`` returns ``None``).
            raw = None
        if raw is not None:
            return raw
    return _text_bytes(text)


def extract_token_logprob(
    vocab_logprobs: mx.array,
    token_id: int,
    num_top: int,
    tokenizer: Any,
) -> Optional[TokenLogprob]:
    """
    Build a ``TokenLogprob`` for ``token_id`` from a vocabulary logprob row.

    Returns ``None`` when ``vocab_logprobs`` does not cover ``token_id``, as
    with the placeholder arrays that error paths emit.
    """
    row = vocab_logprobs.reshape(-1)
    vocab_size = row.shape[0]
    if token_id < 0 or token_id >= vocab_size:
        return None
    num_top = max(0, min(int(num_top), MAX_TOP_LOGPROBS, vocab_size))
    chosen = row[token_id]
    pairs: List[Tuple[int, float]] = []
    if num_top:
        top_ids = mx.argpartition(-row, kth=num_top - 1)[:num_top]
        top_values = row[top_ids]
        mx.eval(chosen, top_ids, top_values)
        pairs = sorted(
            zip(top_ids.tolist(), top_values.tolist()), key=lambda pair: -pair[1]
        )
    else:
        mx.eval(chosen)

    piece_decoder = _piece_decoder(tokenizer)

    def describe(tid: int) -> Tuple[str, Optional[List[int]]]:
        text = _decode_token(tokenizer, tid)
        return text, _token_bytes(tokenizer, piece_decoder, tid, text)

    token, token_bytes = describe(token_id)
    top_logprobs = []
    for tid, value in pairs:
        top_token, top_bytes = describe(tid)
        top_logprobs.append(TopToken(tid, top_token, _clamp_to_floor(value), top_bytes))
    return TokenLogprob(
        token_id=token_id,
        token=token,
        logprob=_clamp_to_floor(chosen.item()),
        top_logprobs=top_logprobs,
        bytes=token_bytes,
    )


def record_step_logprobs(
    request: Any,
    response: Any,
    tokenizer: Any,
) -> Optional[List[TokenLogprob]]:
    """
    Append the logprob of ``response.token`` to ``request.output_logprobs``.

    Returns ``None`` without touching the request when it did not ask for
    logprobs (``request.sampling_params.logprobs is None``). Otherwise it
    creates ``request.output_logprobs`` on first use and returns this step's
    entries, which are empty for stop tokens because those are not content.
    """
    num_top = request.sampling_params.logprobs
    if num_top is None:
        return None
    if request.output_logprobs is None:
        request.output_logprobs = []
    if response.finish_reason == "stop":
        return []
    entry = extract_token_logprob(response.logprobs, response.token, num_top, tokenizer)
    if entry is None:
        return []
    request.output_logprobs.append(entry)
    return [entry]


def _entry_bytes(entry: Any) -> Optional[List[int]]:
    if entry.bytes is not None:
        return entry.bytes
    return _text_bytes(entry.token)


def to_chat_logprobs(entries: List[TokenLogprob]) -> "ChoiceLogprobs":
    """Format entries as an OpenAI chat ``ChoiceLogprobs``."""
    # Imported here so engine modules can use this file without the API layer.
    from .api.models import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob

    return ChoiceLogprobs(
        content=[
            ChatCompletionTokenLogprob(
                token=entry.token,
                logprob=entry.logprob,
                bytes=_entry_bytes(entry),
                top_logprobs=[
                    TopLogprob(
                        token=top.token, logprob=top.logprob, bytes=_entry_bytes(top)
                    )
                    for top in entry.top_logprobs
                ],
            )
            for entry in entries
        ]
    )


def to_completion_logprobs(
    entries: List[TokenLogprob], text_offset: int = 0
) -> "CompletionLogprobs":
    """Format entries in the legacy ``/v1/completions`` logprobs shape."""
    from .api.models import CompletionLogprobs

    tokens: List[str] = []
    token_logprobs: List[float] = []
    top_logprobs: List[dict] = []
    offsets: List[int] = []
    offset = text_offset
    for entry in entries:
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        # The legacy API always includes the sampled token among the top ones.
        top = {top.token: top.logprob for top in entry.top_logprobs}
        top.setdefault(entry.token, entry.logprob)
        top_logprobs.append(top)
        offsets.append(offset)
        offset += len(entry.token)
    return CompletionLogprobs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=offsets,
    )
