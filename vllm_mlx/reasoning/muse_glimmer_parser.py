# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Muse Glimmer models (ATEM channel format).

Muse Glimmer addresses each assistant message to a recipient:
    <|start|>assistant to=self<|message|>[reasoning]<|eom|>
    <|start|>assistant to=user<|message|>[content]<|eot|>
or, for a tool call, to the tool itself:
    <|start|>assistant to=get_weather<|message|><atem:function_calls>...

The generation prompt ends with ``<|start|>assistant``, so output begins with
`` to=self``. ``<|start|>`` and ``<|message|>`` are special tokens: streamed text
still contains them, but the MLLM engine strips them from complete output via
``clean_output_text``. Both forms are normalized by dropping those tokens, which
leaves a plain start/end pair (``to=self`` ... ``<|eom|>``) followed by a
channel header (``assistant to=user``) that is removed from the content. A tool
channel header is removed too, leaving ``<atem:function_calls>`` for the
``muse_glimmer`` tool parser.

The chat template renders one ``to=self`` block per assistant message, so only
the first block is treated as reasoning. A recipient followed by
``<atem:function_calls>`` is a tool channel even when its name starts with
``user`` (``to=user_lookup``); otherwise ``user`` is the user channel and what
follows it is the reply. ``self`` and ``user`` are themselves reserved
recipients per the ATEM protocol and cannot be tool names; the server rejects
those names at request-validation time (``_validate_muse_glimmer_tool_names``
in ``server.py``) rather than relying on this lookahead to resolve every case.
"""

import re

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_FRAMING_RE = re.compile(r"<\|start\|>|<\|message\|>")
_TOOL_CALLS = "<atem:function_calls>"
_HEADER = "assistant to="
# A turn's first channel has no ``<|start|>assistant`` prefix (the generation
# prompt already ended in it), so a header opening straight into content or a
# tool call -- skipping ``to=self`` -- is bare, not ``assistant``-prefixed.
_BARE_HEADER = "to="
_RECIPIENT_RE = re.compile(r"[\w.\-]*")


def _header_pattern(prefix: str) -> re.Pattern[str]:
    # Tool recipient first, so ``to=user_lookup<atem:...>`` is not read as
    # the user channel followed by ``_lookup``.
    return re.compile(
        re.escape(prefix) + r"(?:[\w.\-]+(?=\s*<atem:function_calls>)|user)"
    )


_HEADER_PATTERNS = {
    _HEADER: _header_pattern(_HEADER),
    _BARE_HEADER: _header_pattern(_BARE_HEADER),
}


def _strip_header(content: str, prefix: str = _HEADER) -> str:
    return _HEADER_PATTERNS[prefix].sub("", content.lstrip(), count=1)


def _header_incomplete(text: str, prefix: str = _HEADER) -> bool:
    """True while ``text`` could still grow into a channel header."""
    if prefix.startswith(text):
        return True
    if not text.startswith(prefix):
        return False
    recipient = text[len(prefix) :]
    # Until the text after the name rules out <atem:function_calls>, a
    # ``user``-prefixed name may still be a tool (``user_lookup``).
    rest = recipient[_RECIPIENT_RE.match(recipient).end() :].lstrip()
    return _TOOL_CALLS.startswith(rest) and rest != _TOOL_CALLS


class MuseGlimmerReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Muse Glimmer's ATEM channels.

    Example (engine-cleaned output):
        Input: " to=selfThe user wants a greeting.<|eom|>assistant to=userHello"
        Output: reasoning="The user wants a greeting.", content="Hello"
    """

    @property
    def start_token(self) -> str:
        return "to=self"

    @property
    def end_token(self) -> str:
        return "<|eom|>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        text = _FRAMING_RE.sub("", model_output)
        if self.end_token not in text:
            # No reasoning close tag. Genuine truncated reasoning starts with
            # ``to=self`` and is left to the base class unchanged. Otherwise,
            # the model skipped the reasoning block and opened straight with
            # a bare channel header (``to=user`` or a tool) -- strip it rather
            # than let the base class's no-tags fallback leak it as content.
            stripped = text.lstrip()
            if stripped.startswith(_BARE_HEADER) and not stripped.startswith(
                self.start_token
            ):
                return None, _strip_header(stripped, _BARE_HEADER).strip() or None
            return super().extract_reasoning(text)
        # The reasoning message ends at <|eom|>; one reply or tool message follows.
        reasoning, _, content = text.partition(self.end_token)
        reasoning = reasoning.strip().removeprefix(self.start_token).strip()
        return reasoning or None, _strip_header(content).strip() or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        previous_text = _FRAMING_RE.sub("", previous_text)
        current_text = _FRAMING_RE.sub("", current_text)
        delta_text = _FRAMING_RE.sub("", delta_text)

        if self._phase == "pre_think":
            if self.start_token in current_text:
                # The reasoning block opened after all; hand the base class
                # anything withheld below plus this delta so it can find the
                # (possibly split) start token as one piece of text.
                if self._content_buffer:
                    delta_text = self._content_buffer + delta_text
                    self._content_buffer = ""
            else:
                return self._pre_think_bare_header(delta_text)

        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

    def _pre_think_bare_header(self, delta_text: str) -> DeltaMessage | None:
        """Handle pre-``to=self`` streaming text that may skip reasoning.

        Mirrors ``Glm4ReasoningParser``'s ``pre_think`` override: while no
        reasoning block has opened, untagged text is never genuine reasoning
        here either, so don't delegate to the base class's "assume reasoning"
        fallback. Instead withhold text (reusing ``_content_buffer``, which is
        otherwise idle until the post-``<|eom|>`` case -- the two never
        overlap in time) until it resolves into a bare ``to=<recipient>``
        header, strip that header, and switch to the content phase.
        """
        if not delta_text:
            return None
        buffer = (self._content_buffer + delta_text).lstrip()
        self._content_buffer = ""
        if _header_incomplete(buffer, _BARE_HEADER):
            self._content_buffer = buffer
            return None
        if buffer.startswith(_BARE_HEADER):
            self._phase = "content"
            return self._transition_to_content(
                None, _strip_header(buffer, _BARE_HEADER)
            )
        # Doesn't resolve into a header at all -- shouldn't happen given the
        # model's format, but fall back to the base class's default behavior.
        return super().extract_reasoning_streaming("", buffer, buffer)

    def _content_delta(self, delta_text: str) -> DeltaMessage | None:
        if self._content_started:
            return super()._content_delta(delta_text)
        # Withhold the channel header until it can be removed whole.
        buffer = (self._content_buffer + delta_text).lstrip()
        self._content_buffer = ""
        if _header_incomplete(buffer):
            self._content_buffer = buffer
            return None
        return super()._content_delta(_strip_header(buffer))

    def finalize_stream(self) -> DeltaMessage | None:
        if not self._content_started and self._content_buffer:
            leftover, self._content_buffer = self._content_buffer, ""
            # A held ``to=user<word>`` reply still loses its header here.
            # Whichever phase held it decides which header shape to strip:
            # a bare header if the stream ended before a ``to=self`` block
            # ever opened, an ``assistant``-prefixed one after it did.
            prefix = _BARE_HEADER if self._phase == "pre_think" else _HEADER
            return DeltaMessage(content=_strip_header(leftover, prefix) or None)
        return super().finalize_stream()
