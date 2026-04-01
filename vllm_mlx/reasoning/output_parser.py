# SPDX-License-Identifier: Apache-2.0
"""
ReasoningOutputParser - Token-based streaming reasoning extraction.

This module provides ReasoningOutputParser, which extends the base
ReasoningParser to support token-aware streaming extraction. This is
particularly important for models like Qwen3/Qwen3.5 where thinking
markers may span multiple tokens.

Key features:
1. Token ID awareness for precise marker detection
2. previous_token_ids state tracking
3. Multi-token delta handling (prevent marker leakage)
4. Compatible with existing text-based parsers

Design reference: vLLM PR #34779
"""

from typing import List, Optional

from .base import DeltaMessage, ReasoningParser


class ReasoningOutputParser(ReasoningParser):
    """
    Enhanced reasoning parser with token-based streaming support.

    This parser extends ReasoningParser to handle streaming output
    with token ID awareness. It maintains state to track whether
    thinking phase has ended, enabling correct separation of
    reasoning and content even when markers span multiple tokens.

    Key state:
    - previous_token_ids: Tracks tokens seen so far
    - _in_content_phase: Whether we've passed the end marker

    Example:
        >>> parser = ReasoningOutputParser(
        ...     start_token="<｜begin▁of▁thinking▁｜>",
        ...     end_token="<｜end▁of▁thinking▁｜>",
        ...     start_token_id=151648,
        ...     end_token_id=151649
        ... )
        >>> parser.reset_state()
        >>> for delta_text, delta_token_ids in stream:
        ...     msg = parser.extract_streaming(delta_text, delta_token_ids)
        ...     if msg:
        ...         # msg.reasoning or msg.content populated
        ...         pass
    """

    def __init__(
        self,
        start_token: str,
        end_token: str,
        start_token_id: Optional[int] = None,
        end_token_id: Optional[int] = None,
        tokenizer: Optional[object] = None,
    ):
        """
        Initialize the reasoning output parser.

        Args:
            start_token: The text marker for thinking start (e.g., "<｜begin▁of▁thinking▁｜>")
            end_token: The text marker for thinking end (e.g., "<｜end▁of▁thinking▁｜>")
            start_token_id: Optional token ID for start marker (enables token-based detection)
            end_token_id: Optional token ID for end marker (enables token-based detection)
            tokenizer: Optional tokenizer for encoding markers to get IDs
        """
        super().__init__(tokenizer)
        self._start_token = start_token
        self._end_token = end_token
        self._start_token_id = start_token_id
        self._end_token_id = end_token_id

        # State for streaming
        self._previous_token_ids: List[int] = []
        self._in_content_phase: bool = False
        
        # Memory management: cap previous_token_ids to prevent unbounded growth
        # Default limit: 10000 tokens (sufficient for most reasoning chains)
        self._max_previous_tokens: int = 10000

        # If tokenizer provided and IDs not given, try to encode
        if tokenizer and start_token_id is None:
            try:
                self._start_token_id = self._encode_marker(start_token)
            except Exception:
                pass
        if tokenizer and end_token_id is None:
            try:
                self._end_token_id = self._encode_marker(end_token)
            except Exception:
                pass

    def _encode_marker(self, marker: str) -> Optional[int]:
        """Encode a marker string to token ID using tokenizer."""
        if self.tokenizer is None:
            return None
        try:
            # Handle different tokenizer types
            if hasattr(self.tokenizer, "encode"):
                ids = self.tokenizer.encode(marker, add_special_tokens=False)
                if ids:
                    return ids[0]
            elif hasattr(self.tokenizer, "tokenizer"):
                # Wrapped tokenizer (like MLX)
                ids = self.tokenizer.tokenizer.encode(marker)
                if ids:
                    return ids[0]
        except Exception:
            pass
        return None

    @property
    def start_token(self) -> str:
        """The text marker that starts reasoning content."""
        return self._start_token

    @property
    def end_token(self) -> str:
        """The text marker that ends reasoning content."""
        return self._end_token

    @property
    def start_token_id(self) -> Optional[int]:
        """The token ID for start marker (if available)."""
        return self._start_token_id

    @property
    def end_token_id(self) -> Optional[int]:
        """The token ID for end marker (if available)."""
        return self._end_token_id

    def reset_state(self) -> None:
        """Reset streaming state for a new request."""
        self._previous_token_ids = []
        self._in_content_phase = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete model output (non-streaming).

        Args:
            model_output: Complete text output from the model.

        Returns:
            Tuple of (reasoning_content, final_content).
        """
        text = model_output

        # Case 1: Both markers present (standard case)
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only end marker (implicit reasoning mode)
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            # Strip start marker if somehow present in reasoning part
            reasoning = reasoning.replace(self.start_token, "").strip()
            return reasoning or None, content.strip() or None

        # Case 3: Only start marker (incomplete reasoning)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No markers - pure content
        return None, text

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta (text-based).

        This method provides text-based streaming extraction for
        backward compatibility. For token-aware extraction, use
        extract_streaming_with_tokens().

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Skip pure marker tokens
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            return None
        if stripped_delta == self.end_token:
            return None

        # Check marker positions
        start_in_prev = self.start_token in previous_text
        start_in_current = self.start_token in current_text
        end_in_prev = self.end_token in previous_text
        end_in_delta = self.end_token in delta_text

        # Explicit <think> found in current
        if start_in_current and not start_in_prev:
            return self._handle_explicit_think_start(
                previous_text, delta_text, end_in_prev, end_in_delta
            )

        # Implicit mode: only </think> found
        if self.end_token in current_text and not start_in_prev:
            return self._handle_implicit_think(
                delta_text, end_in_prev, end_in_delta
            )

        # Already in content phase
        if end_in_prev or self._in_content_phase:
            return DeltaMessage(content=delta_text)

        # Still in reasoning phase (or before any markers)
        return DeltaMessage(reasoning=delta_text)

    def extract_streaming_with_tokens(
        self,
        delta_text: str,
        delta_token_ids: List[int],
    ) -> Optional[DeltaMessage]:
        """
        Extract reasoning from streaming delta with token ID awareness.

        This is the preferred method for streaming extraction when
        token IDs are available. It handles multi-token deltas and
        prevents marker leakage.

        Design reference: vLLM PR #34779

        Args:
            delta_text: The new text in this streaming chunk.
            delta_token_ids: Token IDs corresponding to delta_text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Step 1: Strip start token if present in delta
        # (Old template compatibility - model may output start marker)
        if self._start_token_id is not None:
            if self._start_token_id in delta_token_ids:
                idx = delta_text.find(self.start_token)
                if idx >= 0:
                    delta_text = delta_text[idx + len(self.start_token) :]
                    # Also strip corresponding token IDs
                    # (Simplified: we assume single marker token or handle via text)
                    delta_token_ids = [
                        t for t in delta_token_ids
                        if t != self._start_token_id
                    ]

        # Step 2: Detect end token and split
        if self._end_token_id is not None:
            if self._end_token_id in delta_token_ids:
                end_idx = delta_text.find(self.end_token)
                if end_idx >= 0:
                    reasoning_part = delta_text[:end_idx]
                    content_part = delta_text[end_idx + len(self.end_token) :]
                    
                    # Transition: mark as content phase
                    self._in_content_phase = True
                    
                    return DeltaMessage(
                        reasoning=reasoning_part if reasoning_part else None,
                        content=content_part if content_part else None,
                    )

        # Step 3: Check state - are we past thinking phase?
        if self._end_token_id is not None:
            if self._end_token_id in self._previous_token_ids or self._in_content_phase:
                # Already past thinking phase - this is content
                # Memory no longer needed after content phase
                self._previous_token_ids = []
                return DeltaMessage(content=delta_text)

        # Step 4: Still in reasoning phase
        # (Either we haven't seen end marker, or no token ID info)
        if not delta_text:
            return None
        
        # Update state with memory management
        self._previous_token_ids.extend(delta_token_ids)
        
        # Cap memory to prevent unbounded growth
        # Only need recent tokens to detect end marker crossing delta boundaries
        if len(self._previous_token_ids) > self._max_previous_tokens:
            # Keep the most recent tokens (end marker detection needs recent history)
            self._previous_token_ids = self._previous_token_ids[-self._max_previous_tokens:]
        
        # Default: treat as reasoning (safe for implicit mode)
        return DeltaMessage(reasoning=delta_text)

    def _handle_explicit_think_start(
        self,
        previous_text: str,
        delta_text: str,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where start marker appears in delta."""
        start_idx = delta_text.find(self.start_token)

        if end_in_delta:
            # Both markers in this delta
            end_idx = delta_text.find(self.end_token)
            reasoning_part = delta_text[start_idx + len(self.start_token) : end_idx]
            content_part = delta_text[end_idx + len(self.end_token) :]
            self._in_content_phase = True
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        else:
            # Only start marker - beginning of reasoning
            reasoning_part = delta_text[start_idx + len(self.start_token) :]
            return DeltaMessage(reasoning=reasoning_part if reasoning_part else None)

    def _handle_implicit_think(
        self,
        delta_text: str,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where start marker is implicit (only end marker visible)."""
        if end_in_delta:
            # Transition: end marker in this delta
            idx = delta_text.find(self.end_token)
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len(self.end_token) :]
            self._in_content_phase = True
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        elif end_in_prev:
            # Already past reasoning phase - pure content
            return DeltaMessage(content=delta_text)
        else:
            # Still in implicit reasoning phase
            return DeltaMessage(reasoning=delta_text)


# Model-specific marker configurations
# Reference: vllm_mlx/reasoning/qwen3_parser.py, qwen35_parser.py

QWEN3_MARKERS = {
    "start_token": "<｜begin▁of▁thinking▁｜>",
    "end_token": "<｜end▁of▁thinking▁｜>",
    # Token IDs from Qwen3 tokenizer
    # Note: These may vary by model version - verify with actual tokenizer
    "start_token_id": 151648,  # <|begin_of_thinking|>
    "end_token_id": 151649,    # <|end_of_thinking|>
}

QWEN35_MARKERS = {
    "start_token": "<｜begin▁of▁thinking▁｜>",
    "end_token": "<｜end▁of▁thinking▁｜>",
    # Token IDs (same as Qwen3 for thinking markers)
    "start_token_id": 151648,
    "end_token_id": 151649,
}

DEEPSEEK_R1_MARKERS = {
    "start_token": "<｜begin▁of▁thinking▁｜>",
    "end_token": "<｜end▁of▁thinking▁｜>",
    # DeepSeek-R1 uses similar markers
    "start_token_id": None,  # Verify with actual tokenizer
    "end_token_id": None,
}


def create_qwen3_parser(tokenizer: Optional[object] = None) -> ReasoningOutputParser:
    """Create a ReasoningOutputParser configured for Qwen3."""
    return ReasoningOutputParser(
        start_token=QWEN3_MARKERS["start_token"],
        end_token=QWEN3_MARKERS["end_token"],
        start_token_id=QWEN3_MARKERS["start_token_id"],
        end_token_id=QWEN3_MARKERS["end_token_id"],
        tokenizer=tokenizer,
    )


def create_qwen35_parser(tokenizer: Optional[object] = None) -> ReasoningOutputParser:
    """Create a ReasoningOutputParser configured for Qwen3.5."""
    return ReasoningOutputParser(
        start_token=QWEN35_MARKERS["start_token"],
        end_token=QWEN35_MARKERS["end_token"],
        start_token_id=QWEN35_MARKERS["start_token_id"],
        end_token_id=QWEN35_MARKERS["end_token_id"],
        tokenizer=tokenizer,
    )


def create_deepseek_r1_parser(tokenizer: Optional[object] = None) -> ReasoningOutputParser:
    """Create a ReasoningOutputParser configured for DeepSeek-R1."""
    return ReasoningOutputParser(
        start_token=DEEPSEEK_R1_MARKERS["start_token"],
        end_token=DEEPSEEK_R1_MARKERS["end_token"],
        start_token_id=DEEPSEEK_R1_MARKERS["start_token_id"],
        end_token_id=DEEPSEEK_R1_MARKERS["end_token_id"],
        tokenizer=tokenizer,
    )