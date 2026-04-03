# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 reasoning parser for vllm-mlx.

Handles Gemma 4's channel-based thinking format:
    <|channel>thought
    ...reasoning content...
    <channel|>
    Final answer content

The thinking channel is activated when <|think|> appears in the system prompt.
"""

from .base import DeltaMessage, ReasoningParser

START_TOKEN = "<|channel>thought\n"
END_TOKEN = "<channel|>"


class Gemma4ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Gemma 4 models.

    Extracts reasoning from <|channel>thought ... <channel|> blocks.
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._in_reasoning = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        text = model_output

        # Case 1: Both markers present
        if START_TOKEN in text and END_TOKEN in text:
            _, _, after_start = text.partition(START_TOKEN)
            reasoning, _, content = after_start.partition(END_TOKEN)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only start (reasoning still in progress or truncated)
        if START_TOKEN in text:
            _, _, reasoning = text.partition(START_TOKEN)
            return reasoning.strip() or None, None

        # Case 3: Only end (start was in prompt/previous turn)
        if END_TOKEN in text:
            reasoning, _, content = text.partition(END_TOKEN)
            return reasoning.strip() or None, content.strip() or None

        # Case 4: No markers — pure content
        return None, text.strip() or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        had_start = START_TOKEN in previous_text
        has_start = START_TOKEN in current_text
        had_end = END_TOKEN in previous_text
        has_end = END_TOKEN in current_text

        # Transition: start token just appeared
        if has_start and not had_start:
            # Everything after start is reasoning; suppress the marker
            _, _, after = current_text.partition(START_TOKEN)
            if after:
                return DeltaMessage(reasoning=after)
            return None  # suppress the marker itself

        # In reasoning phase (start seen, end not yet)
        if has_start and not has_end:
            # Check if delta contains partial end token — buffer it
            if "<channel" in delta_text or "channel|>" in delta_text:
                return None  # buffer partial end markers
            return DeltaMessage(reasoning=delta_text)

        # Transition: end token just appeared
        if has_end and not had_end:
            # Split: reasoning before end, content after
            before_end, _, after_end = current_text.partition(END_TOKEN)
            if START_TOKEN in before_end:
                _, _, reasoning_part = before_end.partition(START_TOKEN)
            else:
                reasoning_part = before_end

            # Only emit the new content after the end marker
            prev_after_end = ""
            if END_TOKEN in previous_text:
                _, _, prev_after_end = previous_text.partition(END_TOKEN)
            new_content = after_end[len(prev_after_end) :]
            if new_content:
                return DeltaMessage(content=new_content)
            return DeltaMessage(content="")  # signal transition

        # Post-reasoning: both markers seen
        if has_start and has_end:
            return DeltaMessage(content=delta_text)

        # No markers at all — pure content
        return DeltaMessage(content=delta_text)

    def reset_state(self):
        self._in_reasoning = False
