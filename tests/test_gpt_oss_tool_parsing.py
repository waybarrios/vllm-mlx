# SPDX-License-Identifier: Apache-2.0
"""
Tests for GPT-OSS harmony-format tool call parsing at the server boundary.

Tests exercise _extract_reasoning_and_tool_calls directly: when the model emits
an analysis (reasoning) channel followed by a commentary channel, the analysis
block must be stripped before the text is handed to the tool parser so reasoning
prose is never run through the generic fallback (which can extract spurious tool
calls from JSON-shaped text) and never duplicates into response content.

Usage:
    pytest tests/test_gpt_oss_tool_parsing.py -v
"""

from types import SimpleNamespace

import json

import pytest


@pytest.mark.parametrize("tools", [None, [], [{"type": "function"}]])
def test_disabled_tools_preserve_empty_public_harmony_content(monkeypatch, tools):
    import vllm_mlx.server as server
    from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

    monkeypatch.setattr(server, "_reasoning_parser", HarmonyReasoningParser())
    raw = (
        "<|channel|>analysis<|message|>thinking..."
        "<|channel|>commentary to=functions.read_file"
        '<|message|>{"path":"/etc/hosts"}<|call|>'
    )
    request = SimpleNamespace(tools=tools, tool_choice="none")

    assert server._extract_reasoning_and_tool_calls(raw, request) == (
        "thinking...",
        "",
        None,
    )


def test_tools_branch_strips_analysis_block_before_parser(monkeypatch):
    import vllm_mlx.server as server

    class FakeReasoningParser:
        def extract_reasoning(self, text):
            return (
                'I will use {"name": "get_weather", "arguments": {"city": "SF"}}',
                None,
            )

    seen = []

    def fake_parse(text, request, **_):
        seen.append(text)
        return text, None

    raw = (
        '<|channel|>analysis<|message|>I will use {"name": "get_weather", '
        '"arguments": {"city": "SF"}}<|end|>'
        "<|channel|>commentary to=functions.get_weather"
        '<|message|>{"city": "S'
    )
    request = SimpleNamespace(tools=[{"type": "function"}])
    monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
    monkeypatch.setattr(server, "_parse_tool_calls_with_parser", fake_parse)

    reasoning, cleaned, tool_calls = server._extract_reasoning_and_tool_calls(
        raw, request
    )

    assert reasoning is not None and reasoning != ""
    assert tool_calls is None
    assert len(seen) == 1
    assert "<|channel|>analysis" not in seen[0]
    assert seen[0] == (
        "<|end|><|channel|>commentary to=functions.get_weather" '<|message|>{"city": "S'
    )


def test_tools_branch_analysis_then_commentary_call_passes_through(monkeypatch):
    """Happy path: analysis channel followed by a commentary channel with a
    <|call|> terminator. The call must still be passed through to the parser
    (commentary block preserved), but the analysis block must be stripped."""
    import vllm_mlx.server as server

    class FakeReasoningParser:
        def extract_reasoning(self, text):
            return "analysis content", None

    seen = []

    def fake_parse(text, request, **_):
        seen.append(text)
        return None, ["call_extracted"]

    raw = (
        "<|channel|>analysis<|message|>thinking..."
        "<|channel|>commentary to=functions.read_file"
        '<|message|>{"path":"/etc/hosts"}<|call|>'
    )
    request = SimpleNamespace(tools=[{"type": "function"}])
    monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
    monkeypatch.setattr(server, "_parse_tool_calls_with_parser", fake_parse)

    reasoning, cleaned, tool_calls = server._extract_reasoning_and_tool_calls(
        raw, request
    )

    assert reasoning == "analysis content"
    assert tool_calls == ["call_extracted"]
    assert len(seen) == 1
    assert "<|channel|>analysis" not in seen[0]
    assert seen[0] == (
        "<|channel|>commentary to=functions.read_file"
        '<|message|>{"path":"/etc/hosts"}<|call|>'
    )


# ============================================================================
# Streaming gate activation for gpt-oss harmony commentary
# ============================================================================


def test_streaming_marker_possible_for_commentary():
    """The streaming marker gate must activate for a gpt-oss commentary block.

    _STREAMING_TOOL_MARKERS now includes "<|channel|>commentary" and "<|call|>",
    otherwise _streaming_tool_markup_possible would never fire for harmony
    tool calls and the streaming parser path would stay dormant.
    """
    import vllm_mlx.server as server

    commentary = (
        "<|channel|>commentary to=functions.get_weather"
        '<|message|>{"city": "SF"}<|call|>'
    )
    assert server._streaming_tool_markup_possible(commentary) is True

    # A commentary block that starts before the current delta also activates
    # the gate (boundary-window scan).
    assert (
        server._streaming_tool_markup_possible_after_delta(
            "prefix text ", "<|channel|>commentary to=functions.get_weather"
        )
        is True
    )
    # The terminator alone is a marker too (e.g. split across chunks).
    assert (
        server._streaming_tool_markup_possible_after_delta(
            "<|channel|>commentary to=functions.get_weather"
            '<|message|>{"city": "SF"}',
            "<|call|>",
        )
        is True
    )


def test_streaming_marker_false_for_plain_text():
    """Plain content must not activate the harmony tool-markup gate."""
    import vllm_mlx.server as server

    plain = "The weather in San Francisco is 72 degrees and sunny."
    assert server._streaming_tool_markup_possible(plain) is False
    assert server._streaming_tool_markup_possible_after_delta("", "72 degrees") is False
    # Final-channel content (no commentary) is not tool markup either.
    final_only = "<|channel|>final<|message|>The answer is 42.<|return|>"
    assert server._streaming_tool_markup_possible(final_only) is False


_HARMONY_CHUNKS = [
    "<|channel|>",
    "analysis",
    "<|message|>",
    "Need weather.",
    "<|end|>",
    "<|start|>",
    "assistant",
    "<|channel|>",
    "commentary to=functions.get_weather ",
    "<|constrain|>",
    "json",
    "<|message|>",
    '{"city":"Paris"}',
]
_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.fixture
def harmony_api(monkeypatch):
    """Use real HTTP handlers, parsers and SimpleEngine.chat, without model I/O."""
    from fastapi.testclient import TestClient

    import vllm_mlx.server as server
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.engine.simple import SimpleEngine

    class StreamEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False
        _loaded = True
        chat = SimpleEngine.chat

        def __init__(self):
            self.chunks = list(_HARMONY_CHUNKS)

        async def stream_chat(self, messages, **kwargs):
            text = ""
            for i, delta in enumerate(self.chunks):
                text += delta
                yield GenerationOutput(
                    text=text,
                    new_text=delta,
                    finished=False,
                    finish_reason=None,
                    prompt_tokens=7,
                    completion_tokens=i + 1,
                )
            # EOS is deliberately a separate empty output, as in the scheduler.
            yield GenerationOutput(
                text=text,
                new_text="",
                finished=True,
                finish_reason="stop",
                prompt_tokens=7,
                completion_tokens=len(self.chunks),
            )

    engine = StreamEngine()
    for key, value in {
        "_engine": engine,
        "_model_manager": None,
        "_model_name": "test-model",
        "_api_key": None,
        "_default_timeout": 30.0,
        "_default_max_tokens": 128,
        "_default_chat_template_kwargs": None,
        "_default_thinking_token_budget": None,
        "_enable_auto_tool_choice": True,
        "_tool_call_parser": "harmony",
        "_tool_parser_instance": None,
        "_rate_limiter": server.RateLimiter(enabled=False),
    }.items():
        monkeypatch.setattr(server, key, value)

    def configure(reasoning):
        monkeypatch.setattr(server, "_reasoning_parser_name", reasoning)
        monkeypatch.setattr(
            server,
            "_reasoning_parser",
            server.get_reasoning_parser(reasoning)() if reasoning else None,
        )

    # Startup/shutdown owns model resources; only the HTTP application is under
    # test here, just as in test_server.py's endpoint fixtures.
    yield TestClient(server.app), engine, configure


def _harmony_request(client, api, stream, *, tool_choice=None, tools=True, **extra):
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Weather?"}],
        "max_tokens": 64,
        "stream": stream,
        **extra,
    }
    if tools:
        if api == "openai":
            body["tools"] = [{"type": "function", "function": _WEATHER_TOOL}]
        else:
            body["tools"] = [
                {
                    "name": _WEATHER_TOOL["name"],
                    "description": _WEATHER_TOOL["description"],
                    "input_schema": _WEATHER_TOOL["parameters"],
                }
            ]
    if tool_choice:
        body["tool_choice"] = tool_choice if api == "openai" else {"type": tool_choice}
    response = client.post(
        "/v1/chat/completions" if api == "openai" else "/v1/messages", json=body
    )
    assert response.status_code == 200, response.text
    return response


def _sse_data(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reasoning", ["gpt_oss", "harmony", None])
def test_harmony_nonstream_api_preserves_engine_framing(harmony_api, api, reasoning):
    client, engine, configure = harmony_api
    configure(reasoning)
    body = _harmony_request(client, api, False).json()
    if api == "openai":
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        calls = message["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
        assert not message.get("content")
        assert message.get("reasoning_content") == (
            "Need weather." if reasoning else None
        )
    else:
        assert body["stop_reason"] == "tool_use"
        calls = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["input"] == {"city": "Paris"}
        assert not [b for b in body["content"] if b["type"] == "text" and b["text"]]
        thoughts = [b["thinking"] for b in body["content"] if b["type"] == "thinking"]
        assert thoughts == (["Need weather."] if reasoning else [])


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reasoning", ["gpt_oss", "harmony", None])
@pytest.mark.parametrize("fragmented", [True, False])
@pytest.mark.parametrize("terminator", ["", "<|call|>"])
@pytest.mark.parametrize("assistant_prefix", [False, True])
def test_harmony_stream_api_preserves_raw_tool_channels(
    harmony_api, api, reasoning, fragmented, terminator, assistant_prefix
):
    client, engine, configure = harmony_api
    configure(reasoning)
    if not fragmented:
        engine.chunks = ["".join(_HARMONY_CHUNKS[:4]), "".join(_HARMONY_CHUNKS[4:])]
    if terminator:
        engine.chunks.append(terminator)
    if assistant_prefix:
        engine.chunks[:0] = ["<|start|>", "assistant"]
    events = _sse_data(_harmony_request(client, api, True))
    if api == "openai":
        choices = [e["choices"][0] for e in events if e.get("choices")]
        calls = [tc for c in choices for tc in c["delta"].get("tool_calls") or []]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
        assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
            "tool_calls"
        ]
        visible = "".join(c["delta"].get("content") or "" for c in choices)
        thoughts = "".join(c["delta"].get("reasoning_content") or "" for c in choices)
    else:
        calls = [
            e["content_block"]
            for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        args = "".join(
            e["delta"]["partial_json"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(args) == {"city": "Paris"}
        assert [
            e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"
        ] == ["tool_use"]
        visible = "".join(
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        )
        thoughts = "".join(
            e["delta"]["thinking"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "thinking_delta"
        )
    assert visible == ""
    assert thoughts == ("Need weather." if reasoning else "")


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("final_answer", [False, True])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss", None])
def test_harmony_api_tool_gates_do_not_invent_calls(
    harmony_api, api, stream, tools, final_answer, reasoning
):
    client, engine, configure = harmony_api
    configure(reasoning)
    engine.chunks.append("<|call|>")
    if final_answer:
        engine.chunks += [
            "<|channel|>",
            "final",
            "<|message|>",
            "Safe answer.",
            "<|return|>",
        ]
    response = _harmony_request(
        client, api, stream, tool_choice="none" if tools else None, tools=tools
    )
    assert "get_weather" not in response.text
    assert "Paris" not in response.text
    assert "commentary" not in response.text
    assert "<|" not in response.text
    if final_answer:
        assert "Safe answer." in response.text
    if stream:
        events = _sse_data(response)
        if api == "openai":
            choices = [e["choices"][0] for e in events if e.get("choices")]
            assert not [
                tc for c in choices for tc in c["delta"].get("tool_calls") or []
            ]
            assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
                "stop"
            ]
        else:
            assert [
                e["delta"]["stop_reason"]
                for e in events
                if e["type"] == "message_delta"
            ] == ["end_turn"]
    if not stream:
        body = response.json()
        if api == "openai":
            assert body["choices"][0]["finish_reason"] == "stop"
            assert not body["choices"][0]["message"].get("tool_calls")
        else:
            assert body["stop_reason"] == "end_turn"
            assert not [b for b in body["content"] if b["type"] == "tool_use"]


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_harmony_stream_final_content_is_clean_without_reasoning(harmony_api, api):
    client, engine, configure = harmony_api
    configure(None)
    engine.chunks = ["<|channel|>final<|message|>Safe answer.<|return|>"]
    response = _harmony_request(client, api, True)
    assert "Safe answer." in response.text
    assert "<|" not in response.text
    assert "final" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize("mllm", [False, True])
async def test_batched_aggregate_keeps_raw_parser_input(mllm):
    from unittest.mock import AsyncMock

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.api.utils import clean_output_text

    raw = "".join(_HARMONY_CHUNKS)
    result = SimpleNamespace(
        output_text=raw,
        output_token_ids=[1, 2],
        prompt_tokens=7,
        completion_tokens=13,
        finish_reason="stop",
        mtp_drafts=0,
        mtp_accepted=0,
        specprefill_outcome=None,
    )
    backend = SimpleNamespace(generate=AsyncMock(return_value=result))
    engine = SimpleNamespace(
        _loaded=True,
        _is_mllm=mllm,
        _mllm_scheduler=backend if mllm else None,
        _engine=backend,
        _default_mllm_draft=False,
    )
    output = await BatchedEngine.generate(engine, "prompt")
    assert output.text == clean_output_text(raw)
    assert getattr(output, "raw_text", None) == raw


def _stream_fields(response, api):
    events = _sse_data(response)
    if api == "openai":
        choices = [e["choices"][0] for e in events if e.get("choices")]
        return (
            "".join(c["delta"].get("content") or "" for c in choices),
            "".join(c["delta"].get("reasoning_content") or "" for c in choices),
        )
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    return (
        "".join(d["text"] for d in deltas if d["type"] == "text_delta"),
        "".join(d["thinking"] for d in deltas if d["type"] == "thinking_delta"),
    )


def _response_fields(response, api, stream):
    if stream:
        return _stream_fields(response, api)
    body = response.json()
    if api == "openai":
        message = body["choices"][0]["message"]
        return message.get("content") or "", message.get("reasoning_content") or ""
    return (
        "".join(b["text"] for b in body["content"] if b["type"] == "text"),
        "".join(b["thinking"] for b in body["content"] if b["type"] == "thinking"),
    )


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss", None])
def test_harmony_api_disabled_thinking_keeps_only_public_bodies(
    harmony_api, api, stream, reasoning
):
    client, engine, configure = harmony_api
    configure(reasoning)
    engine.chunks = list(
        "<|channel|>analysis<|message|>Private thoughts.<|end|>"
        "<|channel|>final<|message|>Safe answer."
    )
    response = _harmony_request(
        client,
        api,
        stream,
        tools=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert _response_fields(response, api, stream) == ("Safe answer.", "")


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss"])
def test_harmony_api_controls_never_leak_from_fragmented_bodies(
    harmony_api, api, stream, reasoning
):
    client, engine, configure = harmony_api
    configure(reasoning)
    engine.chunks = list(
        "<|channel|>analysis<|message|>Thinking.<|im_end|>"
        "<|channel|>final<|message|>Safe.<|im_end|><|return|>"
    )
    response = _harmony_request(client, api, stream)
    assert _response_fields(response, api, stream) == ("Safe.", "Thinking.")


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_harmony_api_preserves_two_identical_tool_blocks(harmony_api, api, stream):
    client, engine, configure = harmony_api
    configure("harmony")
    block = (
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|message|>{"city":"Paris"}<|call|>'
    )
    engine.chunks = [block, block]
    response = _harmony_request(client, api, stream)
    assert _response_fields(response, api, stream) == ("", "")
    if not stream:
        body = response.json()
        calls = (
            body["choices"][0]["message"]["tool_calls"]
            if api == "openai"
            else [b for b in body["content"] if b["type"] == "tool_use"]
        )
    elif api == "openai":
        calls = [
            tc
            for e in _sse_data(response)
            if e.get("choices")
            for tc in e["choices"][0]["delta"].get("tool_calls") or []
        ]
        assert [tc["index"] for tc in calls] == [0, 1]
    else:
        calls = [
            e["content_block"]
            for e in _sse_data(response)
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
    assert len(calls) == 2
    assert len({call["id"] for call in calls}) == 2


def _stream_tool_calls(response, api):
    events = _sse_data(response)
    if api == "openai":
        choices = [e["choices"][0] for e in events if e.get("choices")]
        fragments = [tc for c in choices for tc in c["delta"].get("tool_calls") or []]
        indices = sorted({tc["index"] for tc in fragments})
        assert indices == list(range(len(indices)))
        calls = []
        for index in indices:
            parts = [tc for tc in fragments if tc["index"] == index]
            identities = [tc["id"] for tc in parts if tc.get("id")]
            names = [
                tc["function"]["name"] for tc in parts if tc["function"].get("name")
            ]
            assert len(identities) == len(names) == 1
            calls.append(
                {
                    "id": identities[0],
                    "name": names[0],
                    "arguments": json.loads(
                        "".join(tc["function"].get("arguments", "") for tc in parts)
                    ),
                }
            )
        assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
            "tool_calls"
        ]
    else:
        starts = [
            e
            for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        calls = []
        for start in starts:
            index = start["index"]
            assert (
                len(
                    [
                        e
                        for e in events
                        if e["type"] == "content_block_stop" and e["index"] == index
                    ]
                )
                == 1
            )
            block = start["content_block"]
            calls.append(
                {
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": json.loads(
                        "".join(
                            e["delta"]["partial_json"]
                            for e in events
                            if e["type"] == "content_block_delta"
                            and e["index"] == index
                            and e["delta"]["type"] == "input_json_delta"
                        )
                    ),
                }
            )
        assert [
            e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"
        ] == ["tool_use"]
    assert len({call["id"] for call in calls}) == len(calls)
    return calls


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss", None])
@pytest.mark.parametrize("identical", [False, True])
@pytest.mark.parametrize("second_terminator", ["", "<|call|>"])
@pytest.mark.parametrize("buffered", [False, True])
def test_harmony_stream_finalizes_only_unemitted_tool_blocks(
    harmony_api, api, reasoning, identical, second_terminator, buffered
):
    client, engine, configure = harmony_api
    configure(reasoning)
    header = "<|channel|>commentary to=functions.get_weather<|message|>"
    first_args = {"city": "Paris<|im_end|>Ω\\n"}
    second_args = first_args if identical else {"city": "Tokyo<|endoftext|>\\n"}
    engine.chunks = [
        "<|channel|>analysis<|message|>Need weather.<|end|>",
        header + json.dumps(first_args) + "<|call|>",
        *list(header + json.dumps(second_args) + second_terminator),
    ]
    extra = (
        {"tool_argument_recovery": "buffered"} if buffered and api == "openai" else {}
    )
    response = _harmony_request(client, api, True, **extra)
    assert _stream_fields(response, api) == ("", "Need weather." if reasoning else "")
    calls = _stream_tool_calls(response, api)
    assert len(calls) == 2
    assert [call["name"] for call in calls] == ["get_weather", "get_weather"]
    assert [call["arguments"] for call in calls] == [first_args, second_args]


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss"])
def test_harmony_stream_does_not_finalize_incomplete_json_after_emitted_call(
    harmony_api, api, reasoning
):
    client, engine, configure = harmony_api
    configure(reasoning)
    header = "<|channel|>commentary to=functions.get_weather<|message|>"
    engine.chunks = [
        header + '{"city":"Paris"}<|call|>',
        *list(header + '{"city":"Tok'),
    ]
    response = _harmony_request(client, api, True)
    assert _stream_fields(response, api) == ("", "")
    calls = _stream_tool_calls(response, api)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "Paris"}


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss"])
@pytest.mark.parametrize("gate", ["no_tools", "choice_none", "thinking_disabled"])
def test_harmony_stream_eos_finalization_respects_request_gates(
    harmony_api, api, reasoning, gate
):
    client, engine, configure = harmony_api
    configure(reasoning)
    header = "<|channel|>commentary to=functions.get_weather<|message|>"
    engine.chunks = [
        "<|channel|>analysis<|message|>Need weather.<|end|>",
        header + '{"city":"Paris"}<|call|>',
        *list(header + '{"city":"Tokyo"}'),
    ]
    extra = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if gate == "thinking_disabled"
        else {}
    )
    response = _harmony_request(
        client,
        api,
        True,
        tools=gate != "no_tools",
        tool_choice="none" if gate == "choice_none" else None,
        **extra,
    )
    assert _stream_fields(response, api) == (
        "",
        "" if gate == "thinking_disabled" else "Need weather.",
    )
    if gate == "thinking_disabled":
        calls = _stream_tool_calls(response, api)
        assert [call["arguments"] for call in calls] == [
            {"city": "Paris"},
            {"city": "Tokyo"},
        ]
    else:
        assert "get_weather" not in response.text
        assert "Paris" not in response.text
        assert "Tokyo" not in response.text
        events = _sse_data(response)
        if api == "openai":
            choices = [e["choices"][0] for e in events if e.get("choices")]
            assert not [
                tc for c in choices for tc in c["delta"].get("tool_calls") or []
            ]
            assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
                "stop"
            ]
        else:
            assert not [
                e
                for e in events
                if e["type"] == "content_block_start"
                and e["content_block"]["type"] == "tool_use"
            ]
            assert [
                e["delta"]["stop_reason"]
                for e in events
                if e["type"] == "message_delta"
            ] == ["end_turn"]


def test_harmony_tool_finalization_is_idempotent_and_resettable():
    import vllm_mlx.server as server
    from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

    parser = HarmonyToolParser()
    header = "<|channel|>commentary to=functions.get_weather<|message|>"
    first = header + '{"city":"Paris"}<|call|>'
    second = header + '{"city":"Tokyo"}'
    text, emitted = server._extract_streaming_tool_delta(parser, "", first, {})
    assert len(emitted["tool_calls"]) == 1
    first_id = emitted["tool_calls"][0]["id"]
    for chunk in second:
        text, emitted = server._extract_streaming_tool_delta(parser, text, chunk, {})
        assert not (emitted or {}).get("tool_calls")
    pending = server._finalize_streaming_tool_result(parser, text)
    assert pending is not None
    assert len(pending["tool_calls"]) == 1
    assert pending["tool_calls"][0]["index"] == 1
    assert pending["tool_calls"][0]["id"] != first_id
    assert json.loads(pending["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Tokyo"
    }
    assert server._finalize_streaming_tool_result(parser, text) is None
    parser.reset()
    fresh = server._finalize_streaming_tool_result(parser, second)
    assert len(fresh["tool_calls"]) == 1
    assert fresh["tool_calls"][0]["index"] == 0
    assert fresh["tool_calls"][0]["id"] != pending["tool_calls"][0]["id"]


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning", ["harmony", "gpt_oss", None])
@pytest.mark.parametrize("fragmented", [False, True])
def test_harmony_api_projects_renderer_headers_and_eos(
    harmony_api, api, stream, reasoning, fragmented
):
    client, engine, configure = harmony_api
    configure(reasoning)
    raw = (
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|channel|>commentary<|message|>Checking now.<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary"
        '<|message|>{"city":"Paris"}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>Safe answer."
    )
    engine.chunks = list(raw) if fragmented else [raw]
    response = _harmony_request(client, api, stream)
    expected_thoughts = "Need weather." if reasoning else ""
    if stream:
        assert _stream_fields(response, api) == (
            "Checking now.Safe answer.",
            expected_thoughts,
        )
        events = _sse_data(response)
        if api == "openai":
            choices = [e["choices"][0] for e in events if e.get("choices")]
            calls = [tc for c in choices for tc in c["delta"].get("tool_calls") or []]
            assert len(calls) == 1
            assert calls[0]["function"]["name"] == "get_weather"
            assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
            assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
                "tool_calls"
            ]
        else:
            calls = [
                e["content_block"]
                for e in events
                if e["type"] == "content_block_start"
                and e["content_block"]["type"] == "tool_use"
            ]
            assert len(calls) == 1
            assert calls[0]["name"] == "get_weather"
            assert json.loads(
                "".join(
                    e["delta"]["partial_json"]
                    for e in events
                    if e["type"] == "content_block_delta"
                    and e["delta"]["type"] == "input_json_delta"
                )
            ) == {"city": "Paris"}
            assert [
                e["delta"]["stop_reason"]
                for e in events
                if e["type"] == "message_delta"
            ] == ["tool_use"]
    else:
        body = response.json()
        if api == "openai":
            choice = body["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            message = choice["message"]
            assert message["content"] == "Checking now.Safe answer."
            assert (message.get("reasoning_content") or "") == expected_thoughts
            assert len(message["tool_calls"]) == 1
            call = message["tool_calls"][0]["function"]
            assert call["name"] == "get_weather"
            assert json.loads(call["arguments"]) == {"city": "Paris"}
        else:
            assert body["stop_reason"] == "tool_use"
            assert "".join(
                b["text"] for b in body["content"] if b["type"] == "text"
            ) == ("Checking now.Safe answer.")
            assert (
                "".join(
                    b["thinking"] for b in body["content"] if b["type"] == "thinking"
                )
                == expected_thoughts
            )
            calls = [b for b in body["content"] if b["type"] == "tool_use"]
            assert len(calls) == 1
            assert calls[0]["name"] == "get_weather"
            assert calls[0]["input"] == {"city": "Paris"}


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("final_channel", [False, True])
def test_harmony_stream_preserves_public_whitespace(harmony_api, api, final_channel):
    client, engine, configure = harmony_api
    configure(None)
    engine.chunks = ["Hello", " world", "\n", "    indented", "\n\n"]
    if final_channel:
        engine.chunks[:0] = ["<|channel|>final", "<|message|>"]
        engine.chunks.append("<|return|>")
    content, reasoning = _stream_fields(_harmony_request(client, api, True), api)
    assert content == "Hello world\n    indented\n\n"
    assert reasoning == ""


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_non_harmony_nonstream_reasoning_still_removes_controls(harmony_api, api):
    client, engine, configure = harmony_api
    configure("deepseek_r1")
    engine.chunks = ["<think>Thinking.<|im_end|></think>", "Hello"]
    response = _harmony_request(client, api, False, tools=False)
    body = response.json()
    if api == "openai":
        assert body["choices"][0]["message"]["reasoning_content"] == "Thinking."
    else:
        assert body["content"][0]["thinking"] == "Thinking."
    assert "<|im_end|>" not in response.text


def test_non_harmony_anthropic_stream_removes_public_controls(harmony_api):
    client, engine, configure = harmony_api
    configure("deepseek_r1")
    engine.chunks = ["<think>Thinking.<|im_end|></think>", "Hello", "<|im_end|>"]
    response = _harmony_request(client, "anthropic", True, tools=False)
    assert _stream_fields(response, "anthropic") == ("Hello", "Thinking.")


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "chunks, expected",
    [
        (["<", "div>", "literal", "</div>"], "<div>literal</div>"),
        (["<"], "<"),
        (["<|ch", "aracter>"], "<|character>"),
    ],
)
def test_harmony_stream_releases_disproved_or_terminal_prefix(
    harmony_api, api, chunks, expected
):
    client, engine, configure = harmony_api
    configure(None)
    engine.chunks = chunks
    assert _stream_fields(_harmony_request(client, api, True), api) == (expected, "")


@pytest.mark.parametrize("channel", ["analysis", "final"])
@pytest.mark.parametrize(
    "chunks, expected",
    [
        (
            ["Example: <|unknown|> suffix", " after"],
            "Example: <|unknown|> suffix after",
        ),
        (["Example: <|", "unknown|> suffix"], "Example: <|unknown|> suffix"),
        (["Comparison: x <"], "Comparison: x <"),
    ],
)
def test_harmony_reasoning_preserves_literal_body_and_flushes(
    channel, chunks, expected
):
    from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

    parser = HarmonyReasoningParser()
    text = ""
    emitted = []
    for chunk in ["<|channel|>", channel, "<|message|>", *chunks]:
        previous, text = text, text + chunk
        delta = parser.extract_reasoning_streaming(previous, text, chunk)
        if delta:
            emitted.append(delta.reasoning if channel == "analysis" else delta.content)
    delta = parser.finalize_stream()
    if delta:
        emitted.append(delta.reasoning if channel == "analysis" else delta.content)
    assert "".join(value or "" for value in emitted) == expected
    assert parser.finalize_stream() is None


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "chunks, expected",
    [
        (
            ["Example: <|unknown|> suffix", " after"],
            "Example: <|unknown|> suffix after",
        ),
        (["Comparison: x <"], "Comparison: x <"),
    ],
)
def test_harmony_stream_preserves_literal_final_body(
    harmony_api, api, chunks, expected
):
    client, engine, configure = harmony_api
    configure("harmony")
    engine.chunks = ["<|channel|>final", "<|message|>", *chunks]
    assert _stream_fields(_harmony_request(client, api, True, tools=False), api) == (
        expected,
        "",
    )


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("buffered", [False, True])
def test_harmony_stream_keeps_final_content_alongside_calls(
    harmony_api, api, coalesced, buffered
):
    client, engine, configure = harmony_api
    configure("harmony")
    engine.chunks = list(_HARMONY_CHUNKS)
    final = [
        "<|call|>",
        "<|channel|>final",
        "<|message|>",
        "Safe answer.",
        "<|return|>",
    ]
    engine.chunks += ["".join(final)] if coalesced else final
    full = _harmony_request(client, api, False).json()
    extra = (
        {"tool_argument_recovery": "buffered"} if buffered and api == "openai" else {}
    )
    response = _harmony_request(client, api, True, **extra)
    assert _stream_fields(response, api) == ("Safe answer.", "Need weather.")
    events = _sse_data(response)
    if api == "openai":
        assert full["choices"][0]["message"]["content"] == "Safe answer."
        choices = [e["choices"][0] for e in events if e.get("choices")]
        calls = [tc for c in choices for tc in c["delta"].get("tool_calls") or []]
        # Buffered recovery may send identity and arguments in separate deltas;
        # they must still describe exactly one call, with neither duplicated.
        assert {tc["index"] for tc in calls} == {0}
        assert len([tc for tc in calls if tc.get("id")]) == 1
        assert [
            tc["function"]["name"] for tc in calls if tc["function"].get("name")
        ] == ["get_weather"]
        assert json.loads(
            "".join(tc["function"].get("arguments", "") for tc in calls)
        ) == {"city": "Paris"}
        assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
            "tool_calls"
        ]
    else:
        assert (
            next(b["text"] for b in full["content"] if b["type"] == "text")
            == "Safe answer."
        )
        calls = [
            e["content_block"]
            for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        arguments = "".join(
            e["delta"]["partial_json"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(arguments) == {"city": "Paris"}
