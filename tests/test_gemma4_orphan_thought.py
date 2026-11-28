# SPDX-License-Identifier: Apache-2.0
"""Gemma 4: a thought channel without its opening ``<|channel>`` must not leak.

With thinking disabled the chat template ends the generation prompt with an empty
``<|channel>thought\\n<channel|>``. Gemma 4 sometimes writes that empty channel
again **without** the opening special token: ``thought`` + ``\\n`` + ``<channel|>``
(token ids 45518, 107, 101), at the start of the answer or after a sentence of
content. With mlx-community/gemma-4-26B-A4B-it-qat-4bit and a tool-calling
conversation, 14 of 20 replays of the same request did this.

The server used to re-enable the reasoning parser only when ``<|channel>`` was
present, so these fragments reached ``content``, and a tool call after them
arrived after streamed text. vLLM's Gemma 4 parser always runs (also with
thinking off) and treats output that starts with ``thought\\n`` or meets a bare
``<channel|>`` as reasoning (vllm/parser/gemma4.py ``_preprocess_feed``). These
tests pin the same behaviour, plus the label echoed after a sentence of content.
"""

import pytest

import vllm_mlx.server as srv
from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser


def _stream(pieces):
    parser = Gemma4ReasoningParser()
    parser.reset_state(implicit_mode=False)
    accumulated = ""
    reasoning, content = [], []
    for piece in pieces:
        prev = accumulated
        accumulated += piece
        msg = parser.extract_reasoning_streaming(prev, accumulated, piece)
        if msg is not None:
            reasoning.append(msg.reasoning or "")
            content.append(msg.content or "")
    final = parser.finalize_stream()
    if final is not None:
        reasoning.append(final.reasoning or "")
        content.append(final.content or "")
    return "".join(reasoning), "".join(content)


# Token-by-token shapes observed from the model (answer text replaced).
LEAK_CASES = {
    "orphan_empty_thought_at_start": (
        ["thought", "\n", "<channel|>", "The", " knowledge", " base was searched."],
        "The knowledge base was searched.",
    ),
    "bare_end_marker_at_start": (
        ["<channel|>", "Hi", " there, ", "nothing found."],
        "Hi there, nothing found.",
    ),
    "newline_then_bare_end_marker": (
        ["\n", "<channel|>", "The", " know", "ledge"],
        "\nThe knowledge",
    ),
    "orphan_thought_after_content": (
        [
            "The knowledge base",
            " is checked.",
            "\n\n",
            "thought",
            "\n",
            "<channel|>",
            "Hi there, ",
            "the report …",
        ],
        "The knowledge base is checked.\n\nHi there, the report …",
    ),
    "label_glued_to_junk": (
        [" enough", "thought", "\n", "<channel|>", "Hi there, ", " was searched."],
        " enoughHi there,  was searched.",
    ),
    "label_after_url_like_junk": (
        ["://", "thought", "\n", "<channel|>", "Hi there, "],
        "://Hi there, ",
    ),
    "orphan_thought_with_body_at_start": (
        [
            "thought",
            "\n",
            "Use ",
            "the last ",
            "timestamp.",
            "<channel|>",
            "the report ",
            "says…",
        ],
        "the report says…",
    ),
}


@pytest.mark.parametrize("name", sorted(LEAK_CASES))
def test_streaming_orphan_thought_does_not_reach_content(name):
    pieces, expected_content = LEAK_CASES[name]
    _, content = _stream(pieces)
    assert content == expected_content
    assert "thought" not in content
    assert "<channel|>" not in content


def test_streaming_orphan_thought_body_goes_to_reasoning():
    reasoning, _ = _stream(LEAK_CASES["orphan_thought_with_body_at_start"][0])
    assert reasoning == "Use the last timestamp."


@pytest.mark.parametrize(
    "pieces, expected",
    [
        (["No ", "stock left."], "No stock left."),
        # The English word stays content once the next token shows it is not a channel label
        (["I ", "thought", " about", " it."], "I thought about it."),
        (["afterthought", "s matter"], "afterthoughts matter"),
        # Stream ends right after the word: flushed as content, not lost
        (["My ", "thought"], "My thought"),
    ],
)
def test_streaming_plain_text_unchanged(pieces, expected):
    reasoning, content = _stream(pieces)
    assert reasoning == ""
    assert content == expected


def test_streaming_orphan_thought_never_closed_is_flushed_as_content():
    # If the model opens an orphan thought and never closes it, the answer must
    # not disappear (thinking-off clients drop reasoning). Flush it as content.
    _, content = _stream(["thought", "\n", "No ", "stock left."])
    assert content == "No stock left."


def test_streaming_standard_thought_channel_unchanged():
    reasoning, content = _stream(
        ["<|channel>", "thought", "\n", "thinking", "<channel|>", "The answer."]
    )
    assert reasoning == "thinking"
    assert content == "The answer."


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "thought\n<channel|>The knowledge base was searched.",
            "The knowledge base was searched.",
        ),
        ("<channel|>Hi there, nothing found.", "Hi there, nothing found."),
        (
            " is checked.\n\nthought\n<channel|>Hi there, the report …",
            " is checked.\n\nHi there, the report …",
        ),
        (
            "thought\nUse the last timestamp.<channel|>the report says…",
            "the report says…",
        ),
        ("No stock left.", "No stock left."),
    ],
)
def test_non_streaming_orphan_thought(monkeypatch, text, expected):
    monkeypatch.setattr(srv, "_reasoning_parser", Gemma4ReasoningParser())
    _, cleaned, _ = srv._extract_reasoning_and_tool_calls(
        text, None, allow_reasoning=False
    )
    assert cleaned.strip() == expected.strip()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("thought", True),
        ("thought\n", True),
        ("<channel|>Hi there", True),
        ("\n<channel|>The", True),
        (" is checked.\n\nthought", True),
        (" is checked.\n\nthought\n<channel|>Hi there", True),
        # Always on for Gemma 4; plain text is protected by the parser itself
        ("No ", True),
        ("I thought about it.", True),
    ],
)
def test_thinking_off_latch_for_gemma4(text, expected):
    # The server re-enables the parser with thinking off via this check.
    parser = Gemma4ReasoningParser()
    assert srv._explicit_reasoning_markers_present(text, parser) is expected


def _server_stream(pieces):
    """Mimic the chat streaming loop with thinking off (server.py latch + parser).

    The stream ends with an empty finished chunk, as the engine sends it, so the
    parser's buffer is flushed through srv._extract_streaming_reasoning_delta.
    Reasoning is dropped, as the server does with thinking off.
    """
    parser = Gemma4ReasoningParser()
    parser.reset_state(implicit_mode=False)
    raw, accumulated, latched = "", "", False
    content = []
    chunks = [(piece, False) for piece in pieces] + [("", True)]
    for piece, finished in chunks:
        if piece:
            raw += piece
            if not latched and srv._explicit_reasoning_markers_present(raw, parser):
                latched = True
        if not latched:
            content.append(piece)
            continue
        prev = accumulated
        accumulated += piece
        msg = srv._extract_streaming_reasoning_delta(
            parser, prev, accumulated, piece, finished=finished
        )
        if msg is not None and msg.content:
            content.append(msg.content)
    return "".join(content)


@pytest.mark.parametrize("name", sorted(LEAK_CASES))
def test_server_loop_thinking_off_does_not_leak(name):
    pieces, expected = LEAK_CASES[name]
    assert _server_stream(pieces) == expected


@pytest.mark.parametrize(
    "pieces, expected",
    [
        (["No ", "stock left."], "No stock left."),
        (["I ", "thought", " about", " it."], "I thought about it."),
        (["Before ", "<channel|>", " after"], "Before <channel|> after"),
        (["My ", "thought"], "My thought"),
        (["afterthought", "s matter"], "afterthoughts matter"),
        # Held back as a possible marker prefix, then the stream ends: still content
        (["The comparison is 2 ", "<"], "The comparison is 2 <"),
        (["Use ", "<chan"], "Use <chan"),
        (["Use ", "<|chan"], "Use <|chan"),
        (["a ", "<", " b"], "a < b"),
    ],
)
def test_server_loop_thinking_off_plain_text_unchanged(pieces, expected):
    assert _server_stream(pieces) == expected
