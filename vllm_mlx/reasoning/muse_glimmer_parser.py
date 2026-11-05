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
_RECIPIENT_WORD_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


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

# Where an unfinished tool-call block starts: the whole tail from the block
# marker on is dropped (non-streaming) or withheld (streaming) whenever no
# ``</atem:function_calls>`` follows, so a max_tokens cut inside a block can
# never leak markup as visible content.
_CALLS_OPEN_RE = re.compile(
    r"<atem:function_calls>(?!.*</atem:function_calls>)", re.DOTALL
)

_CALLS_END = "</atem:function_calls>"
_INVOKE_END = "</atem:invoke>"

# A channel header directly in front of an unfinished block belongs to the
# tool channel, so it is held/dropped with the block instead of leaking as
# reply text.
_PRE_HEADER_RE = re.compile(
    r"(?:<\|message\|>\s*)?(?:assistant\s+)?to=[\w.\-]+\s*(?:<\|message\|>\s*)?$"
)


def _split_at_unfinished_call(text: str) -> tuple[str, str]:
    """Split ``text`` at the start of an unfinished tool-call block.

    Returns ``(visible, tail)`` where ``tail`` starts at an
    ``<atem:function_calls>`` with no later ``</atem:function_calls>``
    (empty when every opened block is closed). Any channel header directly
    in front of the unfinished block stays with the tail.
    """
    match = _CALLS_OPEN_RE.search(text)
    if not match:
        return text, ""
    start = match.start()
    header = _PRE_HEADER_RE.search(text, 0, start)
    if header and header.group().strip():
        start = header.start()
    return text[:start], text[start:]


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


def _self_channel_state(text: str) -> bool | None:
    """Classify whether ``text`` opens the reasoning channel.

    Returns True (genuine ``to=self``), False (a tool/user channel header), or
    None while the text can still grow into a tool channel -- withhold, don't
    commit.

    ``to=self`` is a prefix of the legal tool recipient ``self_check``, so a
    substring test cannot distinguish them; this routes through the same
    complete-marker machinery used for every other recipient. Once ``to=self``
    is complete it commits to the reasoning channel even if a call marker
    follows -- that marker is then an example inside reasoning. A markerless
    ``to=self_check`` resolves True too; per ATEM a tool channel's body is its
    marker, so such a channel is malformed either way and misreading it as
    reasoning is harmless.
    """
    stripped = text.lstrip()
    if not stripped.startswith(_BARE_HEADER):
        return False
    after = stripped[len(_BARE_HEADER) :]
    if after == "":
        return None
    m = _RECIPIENT_RE.match(after)
    name = after[: m.end()] if m else ""
    tail = after[m.end() :].lstrip() if m else after
    if name == "self":
        # ``self_check`` continues the name, so ``to=self`` is still ambiguous.
        if tail and tail[:1] in _RECIPIENT_WORD_CHARS:
            return None
        return True
    if _header_incomplete(stripped, _BARE_HEADER):
        return None
    if _HEADER_PATTERNS[_BARE_HEADER].match(stripped):
        return False
    # No marker followed: ``to=self`` with content run straight onto it
    # (``to=selfThe plan…``) is still the reasoning channel.
    return stripped.startswith("to=self")


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

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset_state()

    def reset_state(self, implicit_mode: bool = False):
        super().reset_state(implicit_mode)
        # Trailing text withheld while an <atem:function_calls> block is open;
        # dropped at stream end (a cut-off block never becomes content).
        self._calls_hold = ""

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
            state = _self_channel_state(stripped)
            if state is False:
                return None, _strip_header(stripped, _BARE_HEADER).strip() or None
            # state True (truncated reasoning) or None (ambiguous prefix such
            # as ``to=self`` that a complete output can no longer grow into a
            # tool channel): leave to the base class.
            return super().extract_reasoning(text)
        # The reasoning message ends at <|eom|>; one reply or tool message follows.
        state = _self_channel_state(text)
        if state is False:
            # Output skipped the reasoning channel (bare tool/user header):
            # nothing is reasoning; the tool parser strips call markup later.
            return None, _strip_header(text.lstrip(), _BARE_HEADER).strip() or None
        reasoning, _, content = text.partition(self.end_token)
        reasoning = reasoning.strip().removeprefix(self.start_token).strip()
        visible, tail = _split_at_unfinished_call(content)
        if not tail and _TOOL_CALLS in visible:
            # Only an *example* call markup inside a (possibly redirected)
            # reasoning channel: keep it verbatim for the tool parser.
            visible = content
        content = visible
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
            head = (self._content_buffer + delta_text).lstrip()
            state = _self_channel_state(head)
            if state is None:
                # Ambiguous: ``to=self`` might still resolve into
                # ``to=self_check``; hold until the header resolves.
                self._content_buffer = head
                return None
            if state:
                # Genuine reasoning channel; hand the base class everything
                # withheld plus this delta so it can find the (possibly
                # split) start token as one piece of text.
                delta_text = head
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
        if _header_incomplete(buffer, _BARE_HEADER) or (
            _self_channel_state(buffer) is None
        ):
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
        if self._content_started or self._calls_hold:
            return self._calls_guard(delta_text)
        # Withhold the channel header until it can be removed whole.
        buffer = (self._content_buffer + delta_text).lstrip()
        self._content_buffer = ""
        if _header_incomplete(buffer):
            self._content_buffer = buffer
            return None
        return self._calls_guard(_strip_header(buffer))

    def _calls_guard(self, delta_text: str) -> DeltaMessage | None:
        """Emit ``delta_text`` unless it ends inside an unfinished block.

        A tool-call block that has opened but not closed is withheld (streaming)
        or dropped (non-streaming, via ``extract_reasoning``) so a max_tokens
        cut inside a block can never leak markup as visible content. A block
        that closes is released verbatim for the tool parser downstream.
        """
        combined = self._calls_hold + delta_text
        self._calls_hold = ""
        if not combined:
            return None
        idx = combined.find(_TOOL_CALLS)
        if idx == -1:
            # Hold a trailing run that could still grow into the marker.
            for n in range(min(len(_TOOL_CALLS) - 1, len(combined)), 0, -1):
                if combined.endswith(_TOOL_CALLS[:n]):
                    visible, self._calls_hold = combined[:-n], combined[-n:]
                    break
            else:
                visible, self._calls_hold = combined, ""
            return super()._content_delta(visible) if visible else None
        visible, rest = combined[:idx], combined[idx:]
        if _CALLS_END in rest:
            return super()._content_delta(_strip_header(visible + rest))
        if _INVOKE_END in rest:
            # The invoke itself closed; the block wrapper may yet arrive, but
            # the call is already extractable -- release what is complete
            # (matching non-streaming, which accepts unwrapped blocks) instead
            # of holding markup that a max_tokens cut would drop entirely.
            return super()._content_delta(_strip_header(visible + rest))
        visible, self._calls_hold = _split_at_unfinished_call(combined)
        return super()._content_delta(visible) if visible else None

    def finalize_stream(self) -> DeltaMessage | None:
        if not self._content_started and self._content_buffer:
            leftover, self._content_buffer = self._content_buffer, ""
            # A held ``to=user<word>`` reply still loses its header here.
            # Whichever phase held it decides which header shape to strip:
            # a bare header if the stream ended before a ``to=self`` block
            # ever opened, an ``assistant``-prefixed one after it did.
            prefix = _BARE_HEADER if self._phase == "pre_think" else _HEADER
            return DeltaMessage(content=_strip_header(leftover, prefix) or None)
        # At stream end the held tail can no longer grow into a marker: emit
        # it as content unless it is a cut-off block's tail, which stays
        # suppressed exactly as in non-streaming mode.
        if self._calls_hold:
            hold, self._calls_hold = self._calls_hold, ""
            if _TOOL_CALLS in hold and _CALLS_END not in hold:
                return super().finalize_stream()
            # Emit via the base content path: _calls_guard would re-withhold
            # text that is final now.
            delta = super()._content_delta(hold)
            return delta if delta else super().finalize_stream()
        return super().finalize_stream()
