"""Thinking-aware logits processor for reasoning models.

Manages the full thinking lifecycle: budget enforcement, phase transitions,
and content-phase constrained decoding delegation.
"""

from __future__ import annotations

import enum
from collections import deque
from typing import Callable

import mlx.core as mx


class BoundedSuffixMatcher:
    """Detect a target token sequence in a stream using a rolling suffix buffer.

    Unlike a naive sequential matcher that resets to position 0 on mismatch,
    this uses a bounded buffer that catches overlapping prefixes.
    """

    __slots__ = ("target", "_buf", "_max_len")

    def __init__(self, target_ids: list[int]) -> None:
        if not target_ids:
            raise ValueError("target_ids must be non-empty")
        self.target = tuple(target_ids)
        self._max_len = len(target_ids)
        self._buf: deque[int] = deque(maxlen=self._max_len)

    def feed(self, token_id: int) -> bool:
        """Feed one token. Returns True when the buffer suffix equals the target."""
        self._buf.append(token_id)
        return len(self._buf) == self._max_len and tuple(self._buf) == self.target

    def reset(self) -> None:
        """Clear the buffer."""
        self._buf.clear()


class Phase(enum.Enum):
    """Thinking lifecycle phases."""

    IDLE = "idle"
    THINKING = "thinking"
    TRANSITIONING = "transitioning"
    CONTENT = "content"


class ThinkingAwareLogitsProcessor:
    """Unified logits processor for thinking-model lifecycle management.

    Manages a four-phase state machine:
      IDLE -> THINKING -> TRANSITIONING -> CONTENT

    - IDLE: before reasoning start tokens. Pass through.
    - THINKING: inside reasoning span. Count tokens, pass through.
    - TRANSITIONING: forcing reasoning end sequence via logits masking.
    - CONTENT: after reasoning closed. Delegate to inner processor.

    No re-entry into THINKING after CONTENT is reached.
    """

    __slots__ = (
        "_start_matcher",
        "_end_matcher",
        "_end_token_ids",
        "_thinking_token_budget",
        "_inner",
        "_vocab_size",
        "_state",
        "_thinking_tokens",
        "_transition_index",
    )

    def __init__(
        self,
        start_token_ids: list[int],
        end_token_ids: list[int],
        thinking_token_budget: int,
        inner: Callable[[mx.array, mx.array], mx.array] | None = None,
        vocab_size: int = 152064,
        prompt_has_think_tag: bool = False,
    ) -> None:
        self._start_matcher = BoundedSuffixMatcher(start_token_ids)
        self._end_matcher = BoundedSuffixMatcher(end_token_ids)
        self._end_token_ids = list(end_token_ids)
        self._thinking_token_budget = thinking_token_budget
        self._inner = inner
        self._vocab_size = vocab_size
        self._thinking_tokens = 0
        self._transition_index = 0
        # When the chat template already injected <think> into the prompt,
        # the first generated token is already inside the thinking span.
        # Start in THINKING (or TRANSITIONING if budget=0) instead of IDLE.
        if prompt_has_think_tag:
            if thinking_token_budget == 0:
                self._state = Phase.TRANSITIONING
            else:
                self._state = Phase.THINKING
        else:
            self._state = Phase.IDLE

    @property
    def state(self) -> Phase:
        return self._state

    @property
    def thinking_tokens(self) -> int:
        return self._thinking_tokens

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # Extract the last token ID from the sequence.
        last_token = tokens[-1].item()

        if self._state == Phase.IDLE:
            if self._start_matcher.feed(last_token):
                self._state = Phase.THINKING
                # Check budget=0: transition immediately
                if self._thinking_token_budget == 0:
                    self._state = Phase.TRANSITIONING
                    self._transition_index = 0
            return logits

        if self._state == Phase.THINKING:
            # Check for natural end of thinking
            if self._end_matcher.feed(last_token):
                self._state = Phase.CONTENT
                return self._call_inner(tokens, logits)

            # Count this as a thinking token (start sequence tokens are not
            # counted because state was IDLE when they were emitted).
            self._thinking_tokens += 1

            # Check if budget is now exhausted: the *next* token would exceed.
            if self._thinking_tokens >= self._thinking_token_budget:
                self._state = Phase.TRANSITIONING
                self._transition_index = 0

            return logits

        if self._state == Phase.TRANSITIONING:
            return self._force_transition(logits)

        # Phase.CONTENT
        return self._call_inner(tokens, logits)

    def _force_transition(self, logits: mx.array) -> mx.array:
        """Force the next token in the reasoning end sequence."""
        target_id = self._end_token_ids[self._transition_index]
        # Mask all logits to -inf, then set the target token to 0.
        # Handle both 1-D (vocab,) and 2-D (1, vocab) logits shapes.
        masked = mx.full(logits.shape, float("-inf"))
        if masked.ndim == 1:
            masked[target_id] = 0.0
        else:
            masked[..., target_id] = 0.0
        self._transition_index += 1
        if self._transition_index >= len(self._end_token_ids):
            self._state = Phase.CONTENT
            self._end_matcher.reset()
        return masked

    def _call_inner(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Delegate to inner processor if present."""
        if self._inner is not None:
            return self._inner(tokens, logits)
        return logits
