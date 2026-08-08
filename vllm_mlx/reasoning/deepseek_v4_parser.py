# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for DeepSeek-V4 (Pro/Flash).

V4 shares ``<think>``/``</think>`` with R1, and the prompt encoder closes the
generation prompt on ``<think>``, so the opening tag is usually absent from the
output — the lenient R1 behaviour this builds on already covers that.

What V4 adds is the interaction with tool calls: the model must finish reasoning
before it may call a tool, so an opening ``<｜DSML｜tool_calls>`` marker
terminates the reasoning block even when ``</think>`` never arrives. Without
this, a tool-calling turn would have its entire DSML payload swallowed as
reasoning and the caller would see no tool call at all.
"""

from ..utils.deepseek_v4_encoding import (
    DSML_TOKEN,
    TOOL_CALLS_START,
    partial_marker_len,
)
from .base import DeltaMessage
from .deepseek_r1_parser import DeepSeekR1ReasoningParser

__all__ = ["DeepSeekV4ReasoningParser", "DSML_TOKEN", "TOOL_CALLS_START"]


class DeepSeekV4ReasoningParser(DeepSeekR1ReasoningParser):
    """Reasoning parser for DeepSeek-V4.

    Example::

        Input:  "weighing options<｜DSML｜tool_calls>\\n<｜DSML｜invoke ..."
        Output: reasoning="weighing options",
                content="<｜DSML｜tool_calls>\\n<｜DSML｜invoke ..."

    The tool markup is deliberately left in ``content`` — extracting it is the
    tool parser's job, and it needs the markup intact.
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Characters of the accumulated text already emitted on either channel,
        # and whether the stream has crossed into tool markup.
        self._emitted_len: int = 0
        self._in_tool_markup: bool = False

    def reset_state(self):
        """Reset state machine for a new streaming request."""
        super().reset_state()
        self._emitted_len = 0
        self._in_tool_markup = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, honouring the implicit close before a tool call."""
        tool_idx = model_output.find(TOOL_CALLS_START)
        if tool_idx != -1:
            end_idx = model_output.find(self.end_token)
            if end_idx == -1 or end_idx > tool_idx:
                # A tool call opened while still reasoning: everything before it
                # is reasoning, the markup and beyond is content.
                reasoning = model_output[:tool_idx]
                if reasoning.startswith(self.start_token):
                    reasoning = reasoning[len(self.start_token) :]
                content = model_output[tool_idx:]
                return reasoning.strip() or None, content or None

        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Stream reasoning, closing it when tool markup opens.

        Tracks how much of the accumulated text has been emitted rather than
        trusting the delta, because the marker straddles delta boundaries: a
        parser that streams each fragment as it arrives leaks markup into the
        reasoning channel and then repeats the whole marker as content once it
        recognises it.
        """
        if self._in_tool_markup:
            new = current_text[self._emitted_len :]
            self._emitted_len = len(current_text)
            return DeltaMessage(content=new) if new else None

        marker_idx = current_text.find(TOOL_CALLS_START)
        end_idx = current_text.find(self.end_token)
        reasoning_open = end_idx == -1

        if marker_idx != -1 and (reasoning_open or end_idx > marker_idx):
            # The marker completed while reasoning was still open. Withholding
            # its prefix above guarantees nothing past marker_idx went out yet.
            head = current_text[self._emitted_len : marker_idx]
            tail = current_text[marker_idx:]
            self._emitted_len = len(current_text)
            self._in_tool_markup = True
            self._phase = "content"
            self._content_started = True
            return DeltaMessage(reasoning=head or None, content=tail or None)

        # Hold back a tail that could still grow into one of the markers.
        # <｜DSML｜tool_calls> is assembled from several tokens, so it always
        # straddles deltas; the think tags have their own ids and normally
        # arrive whole, but are covered too because a detokenizer that splits
        # them would otherwise leak fragments into the reasoning channel and
        # emit the remainder as content.
        hold = (
            partial_marker_len(
                current_text, TOOL_CALLS_START, self.start_token, self.end_token
            )
            if reasoning_open
            else 0
        )
        limit = len(current_text) - hold
        if limit <= self._emitted_len:
            return None

        effective_previous = current_text[: self._emitted_len]
        effective_current = current_text[:limit]
        effective_delta = current_text[self._emitted_len : limit]
        self._emitted_len = limit

        return super().extract_reasoning_streaming(
            effective_previous, effective_current, effective_delta
        )
