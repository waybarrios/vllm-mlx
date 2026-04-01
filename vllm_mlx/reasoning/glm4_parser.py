# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GLM4 models.

GLM4 uses <think>...</think> tags for reasoning content. The GLM-4.7 chat
template injects <think> directly into the prompt, so the model never
outputs it natively - it only outputs </think> to end thinking.

This desynchronizes standard parsers, so we need special handling.
"""

from .think_parser import BaseThinkingReasoningParser


class GLM4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for GLM4 models.

    Handles the GLM-specific case where:
    1. The chat template injects <think> into the prompt
    2. The model starts its output already "in reasoning"
    3. The model only outputs </think> to end thinking

    This is different from Qwen3 where both tags may appear in output.

    Example:
        Model output: "Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."
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
        """
        Extract reasoning from GLM4 output.

        GLM4 typically only outputs </think> (not <think>) because the start
        token was injected in the prompt by the chat template.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        text = model_output

        # Case 1: Both tags present (rare, but handle it)
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only closing tag (most common for GLM)
        # Model was already "in reasoning" due to prompt injection
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (reasoning in progress)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No tags - pure content (thinking disabled)
        return None, model_output
