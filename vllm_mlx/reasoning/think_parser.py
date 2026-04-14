# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

This module provides BaseThinkingReasoningParser, a concrete implementation
for extracting reasoning content from models that use thinking tags.

Supports three scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. No tags: pure content

Performance: The streaming parser uses a simple state machine to track the
current phase (pre-think / thinking / content). Tag completion is detected
against the accumulated text for correctness when `<think>` / `</think>` are
split across delta boundaries, but phase tracking still avoids the old
whole-output rescanning behavior.
"""

from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base parser for models using <think>...</think> style tags.

    This parser handles the common pattern where reasoning content is wrapped
    in special tags. Subclasses define the specific start and end tokens.

    Supports "implicit reasoning mode" where <think> is injected in the prompt
    and only </think> appears in the model output. This is common with AI agents
    like OpenCode that force models to reason by injecting thinking tags.

    The streaming parser uses a state machine with three phases:

        pre_think -> thinking -> content

    Transitions are tracked by parser state. Accumulated text is consulted only
    to detect when a start/end tag has completed across delta boundaries.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Streaming state — reset per request via reset_state()
        self._phase: str = "pre_think"  # "pre_think" | "thinking" | "content"

    def reset_state(self):
        """Reset state machine for a new streaming request."""
        self._phase = "pre_think"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles three cases:
        1. Both tags present: <think>reasoning</think>content
        2. Only closing tag: reasoning</think>content (think in prompt)
        3. No tags: pure content

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        text = model_output

        # Case 1: Both tags present (normal case)
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.partition(self.end_token)
            # Strip duplicate end tokens (some models generate <think></think></think>)
            while content.lstrip().startswith(self.end_token):
                content = content.lstrip()[len(self.end_token) :]
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only closing tag (think was injected in prompt)
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            while content.lstrip().startswith(self.end_token):
                content = content.lstrip()[len(self.end_token) :]
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No tags at all — pure content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from a streaming delta using state-machine tracking.

        Instead of rescanning the full accumulated text on every token, this
        method tracks the current phase (pre_think / thinking / content) and
        only consults accumulated text to detect completed start/end tags that
        were split across delta boundaries.

        Handles three scenarios:
        1. Explicit <think>...</think> in model output
        2. Implicit mode (<think> in prompt, only </think> in output)
        3. No tags at all (pure content after first token with no reasoning)

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text in this chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None to skip.
        """
        if not delta_text:
            return None

        start_tok = self.start_token
        end_tok = self.end_token

        # ── Phase: pre_think ──────────────────────────────────────
        # Haven't seen a completed tag yet. Could be:
        # - About to see <think> (explicit reasoning)
        # - Already inside implicit reasoning (think was in prompt)
        # - No reasoning at all (pure content model)
        if self._phase == "pre_think":
            if start_tok in current_text:
                self._phase = "thinking"
                idx = delta_text.find(start_tok)
                after = delta_text[idx + len(start_tok) :] if idx >= 0 else delta_text

                if end_tok in after:
                    self._phase = "content"
                    eidx = after.find(end_tok)
                    reasoning = after[:eidx]
                    content = after[eidx + len(end_tok) :]
                    if not reasoning and not content:
                        return None
                    return DeltaMessage(
                        reasoning=reasoning or None,
                        content=content or None,
                    )
                return DeltaMessage(reasoning=after) if after else None

            # Implicit mode: </think> completed without an explicit <think>.
            if end_tok in current_text:
                self._phase = "content"
                idx = delta_text.find(end_tok)
                if idx >= 0:
                    reasoning = delta_text[:idx]
                    content = delta_text[idx + len(end_tok) :]
                else:
                    reasoning = None
                    content = delta_text
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )

            # No tags — default to reasoning (implicit mode assumption).
            # If the model doesn't use thinking at all, the server's
            # non-parser path handles it. This path only activates when
            # a reasoning parser is explicitly configured.
            return DeltaMessage(reasoning=delta_text)

        # ── Phase: thinking ───────────────────────────────────────
        # Inside a reasoning block, waiting for end tag.
        if self._phase == "thinking":
            if end_tok in current_text and end_tok not in previous_text:
                self._phase = "content"
                idx = delta_text.find(end_tok)
                if idx >= 0:
                    reasoning = delta_text[:idx]
                    content = delta_text[idx + len(end_tok) :]
                else:
                    reasoning = delta_text
                    content = None
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            return DeltaMessage(reasoning=delta_text)

        # ── Phase: content ────────────────────────────────────────
        # Past the reasoning block — everything is content.
        return DeltaMessage(content=delta_text)
