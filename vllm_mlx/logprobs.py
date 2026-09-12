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

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

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


@dataclass
class TokenLogprob:
    """Log probability of one generated token and its most likely alternatives."""

    token_id: int
    token: str
    logprob: float
    # (token_id, token, logprob), most likely first
    top_logprobs: List[Tuple[int, str, float]] = field(default_factory=list)


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
    return TokenLogprob(
        token_id=token_id,
        token=_decode_token(tokenizer, token_id),
        logprob=_clamp_to_floor(chosen.item()),
        top_logprobs=[
            (tid, _decode_token(tokenizer, tid), _clamp_to_floor(value))
            for tid, value in pairs
        ],
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


def _utf8(token: str) -> List[int]:
    return list(token.encode("utf-8", errors="replace"))


def to_chat_logprobs(entries: List[TokenLogprob]) -> "ChoiceLogprobs":
    """Format entries as an OpenAI chat ``ChoiceLogprobs``."""
    # Imported here so engine modules can use this file without the API layer.
    from .api.models import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob

    return ChoiceLogprobs(
        content=[
            ChatCompletionTokenLogprob(
                token=entry.token,
                logprob=entry.logprob,
                bytes=_utf8(entry.token),
                top_logprobs=[
                    TopLogprob(token=token, logprob=logprob, bytes=_utf8(token))
                    for _, token, logprob in entry.top_logprobs
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
        top = {token: logprob for _, token, logprob in entry.top_logprobs}
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
