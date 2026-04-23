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
        self._content_started = False
        self._content_buffer = ""

    def reset_state(self):
        """Reset state machine for a new streaming request."""
        self._phase = "pre_think"
        self._content_started = False
        self._content_buffer = ""

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

        # Cases 1 and 2: consume one or more leading reasoning spans. Some
        # thinking models emit an extra empty ``<think></think>`` block after
        # the forced transition; that block still belongs to reasoning, not
        # final content.
        if self.end_token in text:
            return self._extract_complete_reasoning(text)

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
                    return self._transition_to_content(reasoning, content)
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
                return self._transition_to_content(reasoning, content)

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
                return self._transition_to_content(reasoning, content)
            return DeltaMessage(reasoning=delta_text)

        # ── Phase: content ────────────────────────────────────────
        # Past the reasoning block — everything is content.
        return self._content_delta(delta_text)

    def _extract_complete_reasoning(self, text: str) -> tuple[str | None, str | None]:
        """Split complete output into leading reasoning spans and final content."""
        reasoning_parts: list[str] = []
        remainder = text

        while remainder:
            stripped = remainder.lstrip()

            if stripped.startswith(self.start_token):
                after_start = stripped[len(self.start_token) :]
                reasoning, found, after_end = after_start.partition(self.end_token)
                if not found:
                    reasoning_parts.append(reasoning)
                    remainder = ""
                    break
                if reasoning.strip():
                    reasoning_parts.append(reasoning.strip())
                remainder = after_end
                continue

            start_idx = stripped.find(self.start_token)
            end_idx = stripped.find(self.end_token)
            if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
                reasoning = stripped[:end_idx]
                if reasoning.strip():
                    reasoning_parts.append(reasoning.strip())
                remainder = stripped[end_idx + len(self.end_token) :]
                continue

            remainder = stripped
            break

        reasoning = "\n".join(reasoning_parts).strip() or None
        content = remainder.strip() or None
        return reasoning, content

    def _transition_to_content(
        self, reasoning: str | None, content: str | None
    ) -> DeltaMessage | None:
        """Return a delta while suppressing leading post-transition think blocks."""
        content_msg = self._content_delta(content or "")
        extra_reasoning = content_msg.reasoning if content_msg else None
        final_content = content_msg.content if content_msg else None
        reasoning_text = (reasoning or "") + (extra_reasoning or "")
        if not reasoning_text and not final_content:
            return None
        return DeltaMessage(
            reasoning=reasoning_text or None,
            content=final_content or None,
        )

    def _content_delta(self, delta_text: str) -> DeltaMessage | None:
        """Emit content after consuming repeated leading think blocks."""
        if not delta_text and not self._content_buffer:
            return None

        if self._content_started:
            return DeltaMessage(content=delta_text) if delta_text else None

        self._content_buffer += delta_text
        buffer = self._content_buffer.lstrip()
        reasoning_parts: list[str] = []

        while buffer:
            if buffer.startswith(self.end_token):
                buffer = buffer[len(self.end_token) :].lstrip()
                continue

            if buffer.startswith(self.start_token):
                after_start = buffer[len(self.start_token) :]
                end_idx = after_start.find(self.end_token)
                if end_idx == -1:
                    self._content_buffer = buffer
                    return None
                reasoning = after_start[:end_idx]
                if reasoning:
                    reasoning_parts.append(reasoning)
                buffer = after_start[end_idx + len(self.end_token) :].lstrip()
                continue

            if self.start_token.startswith(buffer):
                self._content_buffer = buffer
                return None

            if self.end_token.startswith(buffer):
                self._content_buffer = buffer
                return None

            self._content_started = True
            self._content_buffer = ""
            return DeltaMessage(
                reasoning="".join(reasoning_parts) or None,
                content=buffer,
            )

        self._content_buffer = ""
        if reasoning_parts:
            return DeltaMessage(reasoning="".join(reasoning_parts))
        return None
