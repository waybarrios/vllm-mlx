# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GLM-4 models (GLM-4.5-Air, GLM-4.6V, GLM-4.7, etc.).

GLM-4 uses <think>...</think> tags for reasoning content, same as Qwen3.
However, unlike Qwen3, GLM-4 does NOT inject <think> in the prompt —
the model decides autonomously whether to reason.

This means:
- Output without tags = normal response (no reasoning)
- Output with tags = reasoning + content

This is the opposite of Qwen3 where no tags = pure reasoning (because
<think> was injected in the prompt and the model hit max_tokens).

GLM-4.6V also wraps responses in <|begin_of_box|>...<|end_of_box|> container
tags which must be stripped before returning content.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_BOX_START = "<|begin_of_box|>"
_BOX_END = "<|end_of_box|>"


class Glm4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for GLM-4 models.

    GLM-4 uses <think>...</think> tokens to denote reasoning text.
    Unlike Qwen3, the template does NOT inject <think> in the prompt,
    so output without tags is a normal response (not truncated reasoning).

    Supports three scenarios:
    1. Both tags in output: <think>reasoning</think>content
    2. Only closing tag (think in prompt): reasoning</think>content
    3. No tags: pure content (NOT reasoning)

    Example (with thinking):
        Input: "<think>Let me analyze...</think>The answer is 42."
        Output: reasoning="Let me analyze...", content="The answer is 42."

    Example (no thinking):
        Input: "The answer is 42."
        Output: reasoning=None, content="The answer is 42."
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
        cleaned = model_output.replace(_BOX_START, "").replace(_BOX_END, "")
        return super().extract_reasoning(cleaned)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Overrides base class pre_think behavior: when no tags have been seen,
        emit delta as content (not reasoning). GLM-4 doesn't inject <think>
        in the prompt, so early tokens without tags are normal content.

        Once <think> is seen, delegates to base class state machine.
        """
        # Strip GLM-4.6V box container tags (special tokens, always whole)
        delta_text = delta_text.replace(_BOX_START, "").replace(_BOX_END, "")
        if not delta_text:
            return None

        start_tok = self.start_token
        end_tok = self.end_token

        # In pre_think phase: check if we should treat as content
        if self._phase == "pre_think":
            # If start tag appeared, transition to thinking via base class
            if start_tok in current_text:
                return super().extract_reasoning_streaming(
                    previous_text, current_text, delta_text
                )

            # If end tag appeared without start (implicit mode from agent)
            if end_tok in current_text:
                return super().extract_reasoning_streaming(
                    previous_text, current_text, delta_text
                )

            # No tags yet — GLM-4 doesn't inject <think>, so this is content
            return DeltaMessage(content=delta_text)

        # In thinking or content phase, delegate to base class
        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )
