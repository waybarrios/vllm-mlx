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
    """

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

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
        """
        # No channel tokens at all — plain content
        if self.start_token not in current_text and self.end_token not in current_text:
            return DeltaMessage(content=delta_text)

        # Check for alternative transition: <|channel>response
        if _RESPONSE_MARKER in current_text:
            if _RESPONSE_MARKER not in previous_text:
                # Transition happening in this delta
                # Find what (if any) content comes after the marker
                marker_pos = current_text.find(_RESPONSE_MARKER)
                after_marker = current_text[marker_pos + len(_RESPONSE_MARKER) :]
                after_marker = after_marker.lstrip("\n")
                if after_marker:
                    return DeltaMessage(content=after_marker)
                return None  # Suppress the marker itself
            else:
                # Already past transition — pure content
                # But we need to only emit the NEW text (delta)
                return DeltaMessage(content=delta_text)

        # Delegate to base class for standard <|channel>/<channel|> handling
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

        # Strip "thought" channel name from initial reasoning
        if result is not None and result.reasoning is not None:
            r = result.reasoning
            # First reasoning delta after <|channel> will be "thought" or "thought\n"
            if self.start_token in current_text:
                # Check if this is the very first reasoning content
                after_channel = current_text.split(self.start_token, 1)[1]
                if after_channel.startswith(_THOUGHT_PREFIX):
                    # Remove "thought" prefix from the accumulated reasoning so far
                    clean = after_channel[len(_THOUGHT_PREFIX) :].lstrip("\n")
                    # Compute what portion of clean text is in this delta
                    prev_after = ""
                    if self.start_token in previous_text:
                        prev_after = previous_text.split(self.start_token, 1)[1]
                        if prev_after.startswith(_THOUGHT_PREFIX):
                            prev_after = prev_after[len(_THOUGHT_PREFIX) :].lstrip("\n")
                    # The new reasoning text is clean minus what was already emitted
                    new_reasoning = clean[len(prev_after) :]
                    if new_reasoning:
                        return DeltaMessage(reasoning=new_reasoning)
                    return None  # Suppress channel name token

        return result
