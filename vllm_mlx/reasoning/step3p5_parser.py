# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Step3p5/Step 3.7 Flash models."""

from __future__ import annotations

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class Step3p5ReasoningParser(BaseThinkingReasoningParser):
    """Split Step3p5 ``<think>...</think>`` reasoning from final content.

    Step chat templates seed generation with ``<think>``. The model may emit
    only the closing ``</think>`` token before final content, so keep that path
    explicit and trim the newline commonly emitted next to the closing tag.
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output)
        if reasoning is not None:
            reasoning = reasoning.removesuffix("\n").strip() or None
        if content is not None:
            content = content.removeprefix("\n").strip() or None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        newline_delta = self._newline_after_end_delta(previous_text, delta_text)
        if newline_delta is not False:
            return DeltaMessage(content=newline_delta) if newline_delta else None

        message = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
        )
        return self._normalize_streaming_message(message, delta_text)

    def _newline_after_end_delta(
        self,
        previous_text: str,
        delta_text: str,
    ) -> str | bool:
        if not previous_text.endswith(self.end_token) or not delta_text:
            return False
        remaining = delta_text.removeprefix("\n")
        if remaining == delta_text:
            return False
        return remaining

    def _normalize_streaming_message(
        self,
        message: DeltaMessage | None,
        delta_text: str,
    ) -> DeltaMessage | None:
        if message is None:
            return None
        reasoning = message.reasoning
        content = message.content
        if reasoning is not None:
            reasoning = reasoning.removesuffix("\n") or None
        if content is not None and self.end_token in delta_text:
            content = content.removeprefix("\n") or None
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)
