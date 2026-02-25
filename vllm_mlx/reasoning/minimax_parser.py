# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for MiniMax models.

MiniMax models (e.g., MiniMax-M2.5) generate inline reasoning text without
explicit <think> tags. This parser detects and strips common reasoning patterns
that leak into visible output, such as:
  - "The user asks..." / "The user wants..."
  - "I need to..." / "Let me think..."
  - Step-by-step analysis before the actual answer

Unlike tag-based parsers, this uses heuristic pattern matching on the
beginning of the output to separate reasoning preamble from content.
"""

from __future__ import annotations

import re

from .base import DeltaMessage, ReasoningParser


class MiniMaxReasoningParser(ReasoningParser):
    """
    Reasoning parser for MiniMax models that think inline without tags.

    Strategy:
    - Buffer the first N characters of output
    - Detect reasoning preamble patterns
    - Find the transition point where real content begins
    - Emit reasoning as reasoning_content, rest as content
    """

    # Max chars to buffer before deciding (covers typical reasoning preamble)
    BUFFER_SIZE = 512

    # Patterns that indicate the START of reasoning (at beginning of output)
    _REASONING_START_RE = re.compile(
        r"^(?:\s*)"  # optional whitespace
        r"(?:"
        r"(?:The\s+user\s+(?:asks|wants|is\s+asking|requests|said|query|question))"
        r"|(?:I\s+(?:need\s+to|should|will|can|want\s+to|have\s+to|must|am\s+going\s+to))"
        r"|(?:Let\s+me\s+(?:think|check|analyze|figure|consider|look|read|review|process))"
        r"|(?:This\s+(?:is\s+a|requires|seems|looks\s+like|appears))"
        r"|(?:First,?\s+(?:I|let|we))"
        r"|(?:(?:So|Now|OK|Okay|Alright|Well),?\s+(?:the\s+user|I\s+need|let\s+me|I\s+should))"
        r"|(?:what's\s+worth\s+storing)"
        r"|(?:(?:Analyzing|Thinking|Processing|Considering|Evaluating|Extracting)\s)"
        r")",
        re.IGNORECASE,
    )

    # Patterns that indicate transition FROM reasoning TO content
    # These mark where the actual answer/response begins
    _CONTENT_TRANSITION_RE = re.compile(
        r"(?:"
        # Common answer starters after reasoning
        r"(?:^|\n\n)(?:(?:The\s+)?(?:answer|result|output|response|solution)\s*(?:is|:))"
        # MiniMax-specific patterns: "Thus answer:", "Thus final"
        r"|(?:^|\n)(?:Thus\s+(?:answer|final|the\s+answer|response)\s*[:\.])"
        # Direct content markers
        r"|(?:^|\n\n)(?:```)"  # code block start
        r"|(?:^|\n\n)(?:Here\s+(?:is|are)\s)"
        r"|(?:^|\n\n)(?:(?:Sure|Of\s+course|Absolutely)[!,.]?\s)"
        r"|(?:^|\n\n)(?:I'(?:d|ll|m)\s+(?:happy|glad)\s+to\s)"
        # Tool call markers (should NOT be stripped)
        r"|(?:<minimax:tool_call>)"
        r"|(?:<tool_call>)"
        r"|(?:<invoke\s)"
        # Structured output after reasoning
        r"|(?:^|\n\n)(?:\d+\.\s+\*\*)"  # numbered bold list
        r"|(?:^|\n\n)(?:##\s)"  # markdown heading
        # MiniMax meta-reasoning followed by actual answer
        r"|(?:^|\n\n)\*\*"  # bold text start (often the answer)
        r")",
        re.IGNORECASE | re.MULTILINE,
    )

    # If the output starts with these, it's NOT reasoning (direct content)
    _DIRECT_CONTENT_RE = re.compile(
        r"^(?:\s*)"
        r"(?:"
        r"```"  # code block
        r"|(?:<minimax:tool_call>)"
        r"|(?:<tool_call>)"
        r"|(?:<invoke\s)"
        r"|(?:#+\s)"  # markdown heading
        r"|(?:\{)"  # JSON object
        r"|(?:\[)"  # JSON array
        r")",
    )

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._buffer = ""
        self._decided = False
        self._is_reasoning = False
        self._transition_pos = 0

    def reset_state(self):
        """Reset state for a new stream."""
        self._buffer = ""
        self._decided = False
        self._is_reasoning = False
        self._transition_pos = 0

    def extract_reasoning(
        self, model_output: str
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete MiniMax output.

        Returns:
            (reasoning, content) tuple.
        """
        # Handle explicit <think> tags first (MiniMax sometimes uses them)
        if "<think>" in model_output or "</think>" in model_output:
            if "</think>" in model_output:
                parts = model_output.split("</think>", 1)
                reasoning = parts[0].replace("<think>", "").strip()
                content = parts[1].strip() if len(parts) > 1 else None
                return reasoning or None, content or None

        # Check for direct content (no reasoning)
        if self._DIRECT_CONTENT_RE.match(model_output):
            return None, model_output

        # Check if output starts with reasoning patterns
        if not self._REASONING_START_RE.match(model_output):
            return None, model_output

        # Find transition point
        match = self._CONTENT_TRANSITION_RE.search(model_output)
        if match:
            reasoning = model_output[: match.start()].strip()
            content = model_output[match.start() :].strip()
            # Don't strip if the "reasoning" is very short (likely false positive)
            if len(reasoning) < 10:
                return None, model_output
            return reasoning or None, content or None

        # No clear transition found - if the whole thing looks like reasoning
        # followed by a short answer, try splitting on double newline
        parts = model_output.split("\n\n", 1)
        if len(parts) == 2:
            first, second = parts
            # Only split if first part matches reasoning and second is shorter
            if self._REASONING_START_RE.match(first) and len(second.strip()) > 0:
                return first.strip(), second.strip()

        # Can't separate - return as content (conservative)
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta for MiniMax models.
        """
        # Handle explicit </think> tag transition
        if "</think>" in delta_text:
            idx = delta_text.find("</think>")
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len("</think>"):]
            self._decided = True
            self._is_reasoning = False
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # Skip <think> tag itself
        if "<think>" in delta_text:
            cleaned = delta_text.replace("<think>", "")
            self._decided = True
            self._is_reasoning = True
            if cleaned:
                return DeltaMessage(reasoning=cleaned)
            return None  # Skip the tag

        if self._decided:
            if self._is_reasoning:
                # Still in reasoning phase - check for transition
                match = self._CONTENT_TRANSITION_RE.search(current_text[self._transition_pos:])
                if match:
                    # Found transition to content
                    abs_pos = self._transition_pos + match.start()
                    self._decided = True
                    self._is_reasoning = False

                    # The delta might contain the transition
                    prev_len = len(previous_text)
                    if abs_pos >= prev_len:
                        # Transition is in this delta
                        reasoning_part = delta_text[: abs_pos - prev_len]
                        content_part = delta_text[abs_pos - prev_len :]
                        # Strip any leading newlines from content
                        content_part = content_part.lstrip("\n")
                        return DeltaMessage(
                            reasoning=reasoning_part if reasoning_part else None,
                            content=content_part if content_part else None,
                        )
                    else:
                        # Transition was before this delta - emit as content
                        return DeltaMessage(content=delta_text)

                # No transition yet - emit as reasoning
                return DeltaMessage(reasoning=delta_text)
            else:
                # In content phase - pass through
                return DeltaMessage(content=delta_text)

        # Still buffering - accumulate and decide
        self._buffer = current_text

        if len(self._buffer) < min(self.BUFFER_SIZE, 80):
            # Not enough text yet to decide
            # But check for direct content markers early
            if self._DIRECT_CONTENT_RE.match(self._buffer):
                self._decided = True
                self._is_reasoning = False
                return DeltaMessage(content=delta_text)
            # Buffer without emitting
            return DeltaMessage(reasoning=delta_text)

        # Enough text to decide
        self._decided = True

        if not self._REASONING_START_RE.match(self._buffer):
            # Not reasoning - emit everything as content
            self._is_reasoning = False
            return DeltaMessage(content=delta_text)

        # It IS reasoning - check for transition already in buffer
        self._is_reasoning = True
        match = self._CONTENT_TRANSITION_RE.search(self._buffer)
        if match:
            self._is_reasoning = False
            abs_pos = match.start()
            prev_len = len(previous_text)
            if abs_pos >= prev_len:
                reasoning_part = delta_text[: abs_pos - prev_len]
                content_part = delta_text[abs_pos - prev_len :].lstrip("\n")
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            return DeltaMessage(content=delta_text)

        self._transition_pos = max(0, len(self._buffer) - 20)
        return DeltaMessage(reasoning=delta_text)

    def finalize_streaming(
        self, accumulated_text: str
    ) -> DeltaMessage | None:
        """
        Finalize streaming - if everything was classified as reasoning
        with no content ever emitted, try to extract the answer portion.
        """
        if not self._is_reasoning:
            return None

        # Try to extract answer from accumulated reasoning text
        reasoning, content = self.extract_reasoning(accumulated_text)
        if content and content != accumulated_text:
            return DeltaMessage(content=content)

        # Can't separate - reclassify as content to avoid empty response
        return DeltaMessage(content=accumulated_text)
