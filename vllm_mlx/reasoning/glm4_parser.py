# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GLM-4 models (GLM-4.5-Air, GLM-4.6V, GLM-4.7, etc.).

GLM-4 uses <think>...</think> tags for reasoning content, same as Qwen3.
GLM-4.7's chat template injects <think> in the prompt when thinking is
enabled (and a pre-closed </think> when disabled), so model output with
thinking on carries only the closing tag — implicit reasoning mode, like
Qwen3. Thinking-disabled requests bypass reasoning parsing server-side
(``_thinking_disabled``), so no-tag output is not misclassified.

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

        GLM-4.7's chat template injects ``<think>`` at the end of the
        generation prompt when thinking is enabled (and a pre-closed
        ``</think>`` when disabled), so with thinking on the model output
        contains only the CLOSING tag — the base class's implicit-reasoning
        mode (default to reasoning until ``</think>``) is exactly right.
        Thinking-disabled requests never reach this parser: the server
        bypasses reasoning parsing via ``_thinking_disabled``, so plain
        content is not at risk of being swallowed into reasoning.

        (An earlier version assumed GLM-4 never injects ``<think>`` and
        emitted pre-tag deltas as content; on GLM-4.7 that streamed the
        entire thinking block into ``content``.)
        """
        # Strip GLM-4.6V box container tags (special tokens, always whole)
        delta_text = delta_text.replace(_BOX_START, "").replace(_BOX_END, "")
        if not delta_text:
            return None

        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )
