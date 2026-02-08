# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for DeepSeek-R1 models.

DeepSeek-R1 uses <think>...</think> tags for reasoning content.
The model may sometimes start outputting reasoning without the explicit
<think> tag, so this parser is more lenient than Qwen3.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek-R1 model.

    DeepSeek-R1 uses <think>...</think> tokens to denote reasoning text.
    This parser is more lenient than Qwen3:
    - The <think> tag may not be explicitly generated (model assumes it)
    - If only </think> is found, everything before it is reasoning

    Example:
        Input: "<think>Step 1: analyze...\nStep 2: solve...</think>The answer is 42."
        Output: reasoning="Step 1: analyze...\nStep 2: solve...", content="The answer is 42."

        Input: "reasoning content</think>final answer"  # No opening tag
        Output: reasoning="reasoning content", content="final answer"
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
        Extract reasoning from DeepSeek-R1 output.

        More lenient than Qwen3 - handles cases where start tag is implicit.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # If we have end token but no start token, treat beginning as reasoning
        if self.end_token in model_output and self.start_token not in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            reasoning = reasoning.strip() or None
            content = content.strip() or None
            return reasoning, content

        # If neither token, return as pure content
        if self.end_token not in model_output and self.start_token not in model_output:
            return None, model_output

        # Use base class for standard case
        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Handles DeepSeek-R1's pattern where <think> may be implicit.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # First try base class logic
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

        # Handle DeepSeek-R1 special case: no start token seen but end token appears
        if result is not None:
            start_in_prev = self.start_token in previous_text
            start_in_delta = self.start_token in delta_text
            end_in_delta = self.end_token in delta_text

            # If end token in delta but we never saw start token
            if not start_in_prev and not start_in_delta and end_in_delta:
                # Everything before end token is reasoning
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )

            # Note: DeepSeek-R1 may omit <think> but still be in reasoning mode.
            # However, we can't reliably detect implicit reasoning without context,
            # so we default to treating unmarked content as regular content.

        return result
