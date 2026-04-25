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

Degenerate cycling:
    On long prompts with tools, Gemma 4 may oscillate between thought and
    response channels many times, producing garbage reasoning before finally
    emitting valid content/tool_calls. The parser handles this by splitting
    at the LAST <channel|> so all cycles go into reasoning_content and only
    the final response goes into content. Channel tokens are stripped from
    both sides.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

# Channel names that follow <|channel> — stripped from output
_THOUGHT_PREFIX = "thought"
_RESPONSE_MARKER = "<|channel>response"
# Full "open-thinking" marker. Kept as a buffering target so that partial
# prefixes arriving mid-stream (e.g. "<|channel>th", "<|channel>thoug") are
# held back until the label completes, instead of leaking 'th', 'tho', etc.
# into the reasoning output.
_THOUGHT_MARKER = "<|channel>thought"


def _strip_channel_name(text: str, prefix: str) -> str:
    """Strip channel name and leading whitespace/newline from text start."""
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.lstrip("\n")


def _strip_channel_tokens(text: str) -> str:
    """Remove all channel special tokens and bare channel names from text.

    Handles degenerate model output with multiple thought/response cycles
    by stripping all protocol tokens, leaving only the actual text content.
    """
    # Remove special tokens
    text = text.replace("<channel|>", "")
    text = text.replace("<|channel>", "")
    # Remove channel names on standalone lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if s in ("thought", "response"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    # Strip leading channel name
    text = text.strip()
    for name in ("thought", "response"):
        if text.startswith(name + "\n"):
            text = text[len(name) + 1 :]
            break
        if text.startswith(name) and (
            len(text) == len(name) or not text[len(name)].isalpha()
        ):
            text = text[len(name) :]
            break
    return text.strip()


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

    Degenerate cycling (long prompts + tools):
        Uses rpartition to split at the LAST <channel|>, so all intermediate
        thought/response cycles go into reasoning and only the final response
        goes into content.

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

    def reset_state(self):
        super().reset_state()
        self._pending = ""
        self._content_seen = False

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
        markers = (_RESPONSE_MARKER, _THOUGHT_MARKER, self.end_token, self.start_token)
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

        Uses rpartition (LAST <channel|>) to handle degenerate cycling:
        all intermediate thought/response cycles go into reasoning,
        only the final response goes into content. Channel tokens
        are stripped from both sides.
        """
        text = model_output

        # Standard format: <|channel>thought...<channel|>content
        # Use rpartition on LAST <channel|> to handle multiple cycles
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.rpartition(self.end_token)
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        # Alternative format: <|channel>thought...<|channel>response...
        # Use rfind for the LAST <|channel>response marker
        if text.count(self.start_token) >= 2 and _RESPONSE_MARKER in text:
            _, _, after_start = text.partition(self.start_token)
            last_resp = after_start.rfind(_RESPONSE_MARKER)
            reasoning = after_start[:last_resp]
            content = after_start[last_resp + len(_RESPONSE_MARKER) :]
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        # Only closing tag (think injected in prompt)
        if self.end_token in text:
            reasoning, _, content = text.rpartition(self.end_token)
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        # Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            reasoning = _strip_channel_tokens(reasoning)
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
        - Re-entry into thought from content (degenerate cycling): back to reasoning

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

    @staticmethod
    def _strip_channel_tokens_from_delta(
        msg: DeltaMessage | None,
    ) -> DeltaMessage | None:
        """Strip channel special tokens from content and reasoning in a delta."""
        if msg is None:
            return None
        c = msg.content
        r = msg.reasoning
        if c is not None:
            c = c.replace("<channel|>", "").replace("<|channel>", "")
        if r is not None:
            r = r.replace("<channel|>", "").replace("<|channel>", "")
        if not c and not r:
            return None
        if c == msg.content and r == msg.reasoning:
            return msg
        return DeltaMessage(reasoning=r or None, content=c or None)

    def _extract_from_safe_text(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Parse safe (non-buffered) text.

        Uses count-based detection for channel tokens so that multiple
        thought/response cycles (degenerate model behaviour) are handled
        correctly — each NEW <|channel> re-enters reasoning, each NEW
        <channel|> transitions to content.
        """
        # No channel tokens at all — plain content
        if self.start_token not in current_text and self.end_token not in current_text:
            return DeltaMessage(content=delta_text)

        # ── Alternative transition: <|channel>response ──
        # Check BEFORE generic <|channel> count — <|channel>response contains
        # <|channel> but is a content transition, not a re-entry to reasoning.
        if _RESPONSE_MARKER in current_text and _RESPONSE_MARKER not in previous_text:
            self._phase = "content"
            self._content_seen = False
            marker_pos = current_text.find(_RESPONSE_MARKER)
            after_marker = current_text[marker_pos + len(_RESPONSE_MARKER) :]
            after_marker = after_marker.lstrip("\n")
            if after_marker:
                self._content_seen = True
                return self._strip_channel_tokens_from_delta(
                    DeltaMessage(content=after_marker)
                )
            return None

        cur_starts = current_text.count(self.start_token)
        prev_starts = previous_text.count(self.start_token)
        cur_ends = current_text.count(self.end_token)
        prev_ends = previous_text.count(self.end_token)

        # ── NEW <|channel> (enter / re-enter reasoning) ──
        if cur_starts > prev_starts:
            if self._phase != "thinking":
                self._phase = "thinking"
                self._content_seen = False
            return None  # suppress marker + channel name

        # ── NEW <channel|> (transition to content) ──
        if cur_ends > prev_ends:
            self._phase = "content"
            self._content_seen = False
            # Text after the last <channel|> in this delta is content
            last_end = delta_text.rfind(self.end_token)
            if last_end >= 0:
                after = delta_text[last_end + len(self.end_token) :]
                after = _strip_channel_name(after.lstrip("\n"), _THOUGHT_PREFIX)
                after = _strip_channel_name(after, "response")
                if after:
                    self._content_seen = True
                    return DeltaMessage(content=after)
            return None

        # ── Content phase ──
        if self._phase == "content":
            if not self._content_seen:
                stripped = delta_text.lstrip("\n")
                stripped = _strip_channel_name(stripped, _THOUGHT_PREFIX)
                stripped = _strip_channel_name(stripped, "response")
                self._content_seen = bool(stripped)
                if not stripped:
                    return None
                return self._strip_channel_tokens_from_delta(
                    DeltaMessage(content=stripped)
                )
            return self._strip_channel_tokens_from_delta(
                DeltaMessage(content=delta_text)
            )

        # ── Thinking phase: emit as reasoning ──
        if self._phase == "thinking":
            # Strip "thought" channel name from initial reasoning delta
            if cur_starts > 0:
                after_ch = current_text.split(self.start_token, 1)[1]
                if after_ch.startswith(_THOUGHT_PREFIX):
                    clean = after_ch[len(_THOUGHT_PREFIX) :].lstrip("\n")
                    prev_after = ""
                    if self.start_token in previous_text:
                        prev_after = previous_text.split(self.start_token, 1)[1]
                        if prev_after.startswith(_THOUGHT_PREFIX):
                            prev_after = prev_after[len(_THOUGHT_PREFIX) :].lstrip("\n")
                    r = clean[len(prev_after) :]
                    return DeltaMessage(reasoning=r) if r else None
            return DeltaMessage(reasoning=delta_text) if delta_text else None

        # ── pre_think: first delta has no markers yet — emit as reasoning ──
        return DeltaMessage(reasoning=delta_text) if delta_text else None
