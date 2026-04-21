# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Gemma 4 models.

Gemma 4 uses a channel-based protocol for reasoning:

    <|channel>thought
    ...thinking content...
    <channel|>
    ...response content...

Where:
    <|channel> = token 100 (channel switch marker)
    <channel|> = token 101 (end-of-channel marker)

The channel names "thought" and "response" appear as text after the
special tokens and should be stripped from the output.

Some model variants may use <|channel>response instead of <channel|>
to transition from thinking to response mode. This parser handles both.

When thinking is disabled or not triggered, output contains no tags.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

# Channel names that follow <|channel> — stripped from output
_THOUGHT_PREFIX = "thought"
_RESPONSE_MARKER = "<|channel>response"


def _strip_channel_name(text: str, prefix: str) -> str:
    """Strip channel name and leading whitespace/newline from text start."""
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.lstrip("\n")


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Gemma 4 models.

    Handles two transition formats:
    1. <|channel>thought...<channel|>response  (standard: token 100 + 101)
    2. <|channel>thought...<|channel>response   (alternative: token 100 + 100)

    Channel names ("thought", "response") are stripped from output.

    Example:
        Input:  "<|channel>thought\\nLet me think...<channel|>The answer is 42."
        Output: reasoning="Let me think...", content="The answer is 42."

    When no tags are present, the entire output is treated as content.

    Streaming buffering:
        Partial markers at a delta boundary (e.g. "<|channel>" without a
        following "response" yet) are buffered internally so they don't
        leak into reasoning/content. The buffer is either consumed when
        the marker completes in a later delta, or flushed as reasoning
        via finalize_stream() when the stream ends.
    """

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Trailing text withheld because it could complete into a marker
        # (e.g. "<|channel>" that might become "<|channel>response" next delta).
        self._pending: str = ""
        # Tracks whether we have emitted the first content delta past the
        # <|channel>response transition — used to strip the leading newline.
        self._content_seen: bool = False
        # Gemma 4's template injects `<|channel>thought\n<channel|>` as
        # preamble, so the model starts emitting content, not reasoning.
        # Start in "content" phase; mid-stream thought blocks re-enter via
        # the base class's content-phase path.
        self._phase = "content"
        # Counters for eating the "thought\n" channel-name prefix at the
        # start of each reasoning block, tolerant to chunk-boundary splits.
        self._thought_prefix_consumed: int = 0
        self._thought_newline_consumed: bool = False

    def reset_state(self):
        super().reset_state()
        self._pending = ""
        self._content_seen = False
        self._phase = "content"
        self._thought_prefix_consumed = 0
        self._thought_newline_consumed = False

    def _arm_thought_strip(self) -> None:
        """Re-arm the thought-prefix stripper when entering a new thinking block."""
        self._thought_prefix_consumed = 0
        self._thought_newline_consumed = False

    def _trailing_partial_marker_len(self, text: str) -> int:
        """
        Return length of trailing substring of `text` that is a proper prefix
        of any transition marker (<channel|>, <|channel>response, <|channel>).

        Only counts PROPER prefixes — if the marker is already complete in
        `text`, no buffering is needed. Returns 0 if no partial match.

        We must never buffer legitimate content. For <|channel>, only buffer
        when it appears AT THE END and is not followed by more text (i.e.,
        `response` or `thought` hasn't arrived yet).
        """
        markers = (_RESPONSE_MARKER, self.end_token, self.start_token)
        max_len = 0
        for marker in markers:
            # Scan from longest proper prefix downwards
            for i in range(min(len(marker) - 1, len(text)), 0, -1):
                if text.endswith(marker[:i]):
                    # Proper prefix match; ensure we're not inside a completed
                    # marker (which would already be handled by other logic).
                    if not text.endswith(marker):
                        if i > max_len:
                            max_len = i
                        break
        return max_len

    def finalize_stream(self) -> DeltaMessage | None:
        """
        Flush any buffered partial marker at the end of stream.

        If the stream ends while we have a partial marker buffered (e.g.
        model emitted "<|channel>" as its last token and got truncated by
        max_tokens), emit it as reasoning so the client doesn't lose the
        text. Content phase flushes as content.
        """
        if not self._pending:
            return None
        pending = self._pending
        self._pending = ""
        if self._phase == "content":
            return DeltaMessage(content=pending)
        return DeltaMessage(reasoning=pending)

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles both <channel|> and <|channel>response as transition markers.
        Strips channel names ("thought", "response") from output.
        """
        text = model_output

        # Try standard format first: <|channel>thought...<channel|>response
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.partition(self.end_token)
            reasoning = _strip_channel_name(reasoning.strip(), _THOUGHT_PREFIX)
            content = content.strip()
            return reasoning or None, content or None

        # Try alternative format: <|channel>thought...<|channel>response...
        if text.count(self.start_token) >= 2 and _RESPONSE_MARKER in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.partition(_RESPONSE_MARKER)
            reasoning = _strip_channel_name(reasoning.strip(), _THOUGHT_PREFIX)
            content = content.lstrip("\n").strip()
            return reasoning or None, content or None

        # Only closing tag (think injected in prompt)
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            reasoning = _strip_channel_name(reasoning.strip(), _THOUGHT_PREFIX)
            content = content.strip()
            return reasoning or None, content or None

        # Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            reasoning = _strip_channel_name(reasoning.strip(), _THOUGHT_PREFIX)
            return reasoning or None, None

        # No tags at all — pure content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Handles:
        - No tags: treat as content (Gemma 4 doesn't inject tags in prompt)
        - <|channel>thought: enter reasoning mode, strip channel name
        - <channel|> or <|channel>response: transition to content mode

        Partial markers at delta boundaries are buffered internally to
        prevent leaking them as reasoning/content.
        """
        # Buffer trailing partial marker so it doesn't leak into output.
        # Process only the "safe" portion (without the partial trailing bytes).
        trailing = self._trailing_partial_marker_len(current_text)
        safe_current = current_text[:-trailing] if trailing else current_text
        prev_trailing = self._trailing_partial_marker_len(previous_text)
        safe_previous = (
            previous_text[:-prev_trailing] if prev_trailing else previous_text
        )

        # Update buffered pending text for external flush (finalize_stream).
        self._pending = current_text[len(safe_current) :]

        # If no new safe text this delta, suppress emission — everything
        # new is buffered as a potential marker prefix.
        if len(safe_current) <= len(safe_previous):
            return None

        safe_delta = safe_current[len(safe_previous) :]

        return self._extract_from_safe_text(safe_previous, safe_current, safe_delta)

    def _extract_from_safe_text(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Parse safe (non-buffered) text using the original logic."""
        # No channel tokens at all — plain content
        if self.start_token not in current_text and self.end_token not in current_text:
            return DeltaMessage(content=delta_text)

        # Check for alternative transition: <|channel>response
        if _RESPONSE_MARKER in current_text:
            if _RESPONSE_MARKER not in previous_text:
                # Transition happening in this delta
                self._phase = "content"
                marker_pos = current_text.find(_RESPONSE_MARKER)
                after_marker = current_text[marker_pos + len(_RESPONSE_MARKER) :]
                after_marker = after_marker.lstrip("\n")
                if after_marker:
                    self._content_seen = True
                    return DeltaMessage(content=after_marker)
                return None  # Suppress the marker itself
            else:
                # Already past transition — pure content.
                # Strip leading newline(s) from the FIRST content delta after
                # the marker (tokenizer often emits "\n" as its own token).
                if not self._content_seen:
                    stripped = delta_text.lstrip("\n")
                    self._content_seen = bool(stripped)
                    if not stripped:
                        return None
                    return DeltaMessage(content=stripped)
                return DeltaMessage(content=delta_text)

        # Re-arm the prefix stripper each time a new thought block opens.
        if self.start_token in delta_text:
            self._arm_thought_strip()

        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

        # Eat the "thought\n" label from the start of each reasoning block.
        if result is not None and result.reasoning is not None:
            r = result.reasoning
            while (
                self._thought_prefix_consumed < len(_THOUGHT_PREFIX)
                and r
                and r[0] == _THOUGHT_PREFIX[self._thought_prefix_consumed]
            ):
                r = r[1:]
                self._thought_prefix_consumed += 1
            if (
                self._thought_prefix_consumed == len(_THOUGHT_PREFIX)
                and not self._thought_newline_consumed
                and r.startswith("\n")
            ):
                r = r.lstrip("\n")
                self._thought_newline_consumed = True

            if not r and not result.content:
                return None
            return DeltaMessage(reasoning=r or None, content=result.content)

        return result
