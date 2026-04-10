# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Gemma 4 models.

Gemma 4 uses channel-based tokens for reasoning:
    <|channel>thought
    [reasoning content]
    <channel|>
    [final content]

The newline between the tag and content is optional — the model
sometimes omits it before <channel|>.

Unlike <think>/<​/think> which are single tokens, Gemma 4's tags span
multiple tokens (e.g. <|channel> + thought), so the streaming parser
uses stateful phase tracking instead of the base class text matching.
"""

from .base import DeltaMessage, ReasoningParser


class Gemma4ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Gemma 4 models.

    Gemma 4 uses <|channel>thought / <channel|> tokens to denote reasoning.

    Example:
        Input: "<|channel>thought\\nLet me think...<channel|>The answer is 42."
        Output: reasoning="Let me think...", content="The answer is 42."
    """

    START_TAG = "<|channel>thought"
    END_TAG = "<channel|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._phase = "init"  # init -> reasoning -> content

    def reset_state(self):
        self._phase = "init"

    def extract_reasoning(
        self,
        model_output: str,
        implicit_think: bool = False,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete Gemma 4 output."""
        if self.START_TAG in model_output and self.END_TAG in model_output:
            _, _, after_start = model_output.partition(self.START_TAG)
            reasoning, _, content = after_start.partition(self.END_TAG)
            return reasoning.strip() or None, content.strip() or None

        if self.START_TAG in model_output:
            _, _, reasoning = model_output.partition(self.START_TAG)
            return reasoning.strip() or None, None

        if self.END_TAG in model_output:
            reasoning, _, content = model_output.partition(self.END_TAG)
            return reasoning.strip() or None, content.strip() or None

        if implicit_think:
            return model_output.strip() or None, None

        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming deltas.

        Uses stateful phase tracking because the start/end tags span
        multiple MLX tokens (e.g. <|channel> and thought arrive separately).
        """
        # Phase: waiting for start tag to complete
        if self._phase == "init":
            if self.START_TAG in current_text:
                # Start tag is now complete — switch to reasoning
                self._phase = "reasoning"
                # Emit any reasoning content after the start tag
                after = current_text.partition(self.START_TAG)[2]
                # Check if end tag also arrived in this chunk
                if self.END_TAG in after:
                    self._phase = "content"
                    reasoning, _, content = after.partition(self.END_TAG)
                    return DeltaMessage(
                        reasoning=reasoning.lstrip("\n") or None,
                        content=content.lstrip("\n") or None,
                    )
                text = after.lstrip("\n")
                return DeltaMessage(reasoning=text) if text else None
            # Still accumulating start tag — suppress output
            # Check if current_text could be a prefix of the start tag
            if self.START_TAG.startswith(current_text) or current_text.startswith("<"):
                return None
            # Not a thinking response — treat as plain content
            return DeltaMessage(content=delta_text)

        # Phase: inside reasoning, waiting for end tag
        if self._phase == "reasoning":
            if self.END_TAG in delta_text:
                self._phase = "content"
                reasoning, _, content = delta_text.partition(self.END_TAG)
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content.lstrip("\n") or None,
                )
            if self.END_TAG in current_text and self.END_TAG not in previous_text:
                # End tag spans this delta boundary
                self._phase = "content"
                reasoning, _, content = current_text.partition(self.END_TAG)
                # Only emit what's new (after previous_text's reasoning)
                prev_reasoning = previous_text.partition(self.START_TAG)[2]
                new_reasoning = reasoning[len(prev_reasoning):]
                return DeltaMessage(
                    reasoning=new_reasoning or None,
                    content=content.lstrip("\n") or None,
                )
            return DeltaMessage(reasoning=delta_text)

        # Phase: after end tag — pure content
        return DeltaMessage(content=delta_text)
