# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Qwen3.5 models.

Qwen3.5 uses <think>...</think> tags for reasoning content.
Same as Qwen3, but this parser is kept separate for potential future differences.

Supports:
1. Both tags in output: <think>reasoning</think>content
2. Only end tag (think in prompt): reasoning</think>content
3. No tags: pure content
"""

from .think_parser import BaseThinkingReasoningParser


class Qwen35ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3.5 models."""

    @property
    def start_token(self) -> str:
        """Return the start token for Qwen3.5 reasoning."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """Return the end token for Qwen3.5 reasoning."""
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Qwen3.5 output.

        Qwen3.5 uses <think>...</think> tags to denote reasoning text.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # If no end token at all, treat as pure content
        if self.end_token not in model_output:
            return None, model_output

        # Use base class implementation (handles both explicit and implicit)
        return super().extract_reasoning(model_output)