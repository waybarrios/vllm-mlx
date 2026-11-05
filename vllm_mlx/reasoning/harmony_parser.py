# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using Harmony format.

Harmony uses channels for reasoning vs final content:

    <|channel|>analysis
    <|message|>Let me think about this...
    <|end|>
    <|channel|>final
    <|message|>The answer is 42.
    <|return|>

The analysis channel contains reasoning, and the final channel
contains the user-facing response.
"""

import re

from .base import DeltaMessage, ReasoningParser

_CHANNEL_HEADER_PATTERN = re.compile(
    r"(?:<\|start\|>assistant(?P<role_metadata>[^<]*))?"
    r"<\|channel\|>(?P<channel>analysis|final|commentary)\b"
    r"(?P<channel_metadata>[^<]*)"
    r"(?:<\|constrain\|>[^<]*)?<\|message\|>"
)
_RECIPIENT_PATTERN = re.compile(r"(?:^|\s)to=\S+")

_STRUCTURAL_TOKENS = (
    "<|channel|>",
    "<|message|>",
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|constrain|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
)
_STRUCTURAL_PATTERN = re.compile("|".join(map(re.escape, _STRUCTURAL_TOKENS)))


class HarmonyReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models using Harmony format.

    Extracts reasoning from the 'analysis' channel and content from
    the 'final' channel. Unaddressed commentary is public content; addressed
    messages are handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."
    """

    def __init__(
        self,
        tokenizer=None,
        *,
        include_plain: bool = False,
        include_reasoning: bool = True,
    ):
        super().__init__(tokenizer)
        self._current_channel: str | None = None
        self._in_message: bool = False
        self._stream_offset = 0
        self._stream_text = ""
        self._include_plain = include_plain
        self._include_reasoning = include_reasoning

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel blocks as reasoning and the
        final and unaddressed commentary channel blocks as content.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        # EOS may be consumed before detokenization, so a body does not need
        # a decoded terminator. Use the same boundaries as the stream path.
        projection = HarmonyReasoningParser(include_plain=self._include_plain)
        projection._stream_text = model_output
        reasoning = "\n".join(
            body.strip()
            for header, body in projection._channel_bodies(len(model_output))
            if header.group("channel") == "analysis" and body.strip()
        )
        delta = projection._extract_stream_delta(finished=True)
        if delta is None:
            return None, None
        return reasoning or None, (delta.content or "").strip() or None

    def _channel_bodies(self, safe_end: int):
        for header in _CHANNEL_HEADER_PATTERN.finditer(self._stream_text):
            metadata = (
                (header.group("role_metadata") or "")
                + " "
                + header.group("channel_metadata")
            )
            if _RECIPIENT_PATTERN.search(metadata):
                continue
            boundary = _STRUCTURAL_PATTERN.search(self._stream_text, header.end())
            end = min(boundary.start(), safe_end) if boundary else safe_end
            start = max(header.end(), self._stream_offset)
            yield header, self._stream_text[start:end] if start < end else ""

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming Harmony output.

        Tracks the current channel and emits reasoning deltas for
        analysis channel content and content deltas for final channel.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None.
        """
        self._stream_text = current_text
        return self._extract_stream_delta()

    def _extract_stream_delta(self, *, finished: bool = False) -> DeltaMessage | None:
        # Inspect raw context, not just this delta: a channel token and its name
        # can arrive separately, or a complete header and body can arrive together.
        # Only newly available body text is emitted; routing headers and tool
        # arguments never become reasoning or visible content.
        current_text = self._stream_text
        reasoning, content = [], []
        safe_end = len(current_text)
        if not finished:
            # Hold only real structural-token prefixes. A disproved prefix
            # (e.g. <div> or <|unknown|>) remains ordinary body text.
            for length in range(
                1, min(len(current_text), max(map(len, _STRUCTURAL_TOKENS))) + 1
            ):
                suffix = current_text[-length:]
                if any(
                    token.startswith(suffix) and token != suffix
                    for token in _STRUCTURAL_TOKENS
                ):
                    safe_end = len(current_text) - length
        if self._include_plain:
            boundary = _STRUCTURAL_PATTERN.search(current_text)
            plain_end = min(boundary.start(), safe_end) if boundary else safe_end
            content.append(current_text[self._stream_offset : plain_end])
        for header, body in self._channel_bodies(safe_end):
            if header.group("channel") == "analysis":
                if self._include_reasoning:
                    reasoning.append(body)
            else:
                # Unaddressed commentary is a public assistant preamble.
                content.append(body)
        self._stream_offset = safe_end
        if not any(reasoning) and not any(content):
            return None
        return DeltaMessage(
            reasoning="".join(reasoning) or None,
            content="".join(content) or None,
        )

    def finalize_stream(self) -> DeltaMessage | None:
        """Release a held literal prefix at EOS, without replaying body text."""
        return self._extract_stream_delta(finished=True)

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
        self._stream_offset = 0
        self._stream_text = ""
