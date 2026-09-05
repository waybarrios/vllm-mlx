# SPDX-License-Identifier: Apache-2.0
"""End-to-end streaming coverage for text arriving alongside tool calls.

The unit tests next door only exercise ``_parse_streaming_tool_content``. That
helper backs the two Anthropic call sites; the Responses path and the OpenAI
path each carry their own copy of the same decision, so all three have to be
driven for real or two of them can regress while the suite stays green.

Each test drives the actual streaming generator with a parser that returns
content and tool calls in one delta — the shape a buffering parser produces
when a whole tool-call block arrives at once.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm_mlx.server as srv

# The streaming paths only consult the parser when the delta looks like it
# could contain tool markup, so the fake output has to trip that gate.
MARKER = "<tool_call>"
TEXT = "Checking the weather."
CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Prague"}'},
}

# What the model actually emitted by the time a buffering parser hands back
# `TEXT` alongside the completed call: the prose is in the accumulated deltas,
# ahead of the markup. Feeding only the marker would ask the streaming path to
# emit text that appears nowhere in the model output.
BUFFERED_OUTPUT = TEXT + MARKER


class _BothInOneDelta:
    """Emits the buffered text together with the completed calls, once."""

    def __init__(self):
        self.done = False

    def reset(self):
        self.done = False

    def extract_tool_calls_streaming(self, *args, **kwargs):
        if self.done:
            return None
        self.done = True
        return {"content": TEXT, "tool_calls": [CALL]}


def _stream_output(new_text, finish_reason=None):
    return SimpleNamespace(
        new_text=new_text,
        text=new_text,
        prompt_tokens=7,
        completion_tokens=1,
        finish_reason=finish_reason,
        finished=finish_reason is not None,
    )


def _engine(*outputs):
    engine = MagicMock()
    engine.model_name = "test-model"
    engine.preserve_native_tool_format = False

    async def _stream_chat(**kwargs):
        for output in outputs:
            yield output

    engine.stream_chat = _stream_chat
    return engine


def _tool_call_object():
    """Shape the terminal re-parse hands back, as a real parser would."""
    return SimpleNamespace(
        id=CALL["id"],
        type="function",
        function=SimpleNamespace(
            name=CALL["function"]["name"], arguments=CALL["function"]["arguments"]
        ),
    )


@pytest.fixture()
def both_in_one_delta(monkeypatch):
    parser = _BothInOneDelta()
    monkeypatch.setattr(
        srv, "_get_streaming_tool_parser", lambda *a, **k: parser, raising=False
    )
    monkeypatch.setattr(srv, "_reasoning_parser", None, raising=False)
    # The Anthropic and Responses paths do not emit tool calls from the
    # streaming deltas; they re-parse the accumulated text once at the end.
    # A real parser answers both, so the stub has to as well or those paths
    # look like they emit no calls at all.
    monkeypatch.setattr(
        srv,
        "_parse_tool_calls_with_parser",
        lambda text, *a, **k: (TEXT, [_tool_call_object()]),
        raising=False,
    )
    return parser


async def _collect(generator):
    return [chunk async for chunk in generator]


def _data_payloads(chunks):
    out = []
    for raw in chunks:
        for line in raw.splitlines():
            if line.startswith("data: "):
                body = line.removeprefix("data: ").strip()
                if body and body != "[DONE]":
                    out.append(json.loads(body))
    return out


class TestOpenAIStream:
    @pytest.mark.anyio
    async def test_text_precedes_the_call_and_each_appears_once(
        self, both_in_one_delta
    ):
        from vllm_mlx.api.models import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        )
        engine = _engine(_stream_output(BUFFERED_OUTPUT), _stream_output("", "stop"))

        chunks = await _collect(
            srv.stream_chat_completion(engine, request.messages, request)
        )
        payloads = _data_payloads(chunks)

        texts = [
            d["choices"][0]["delta"]["content"]
            for d in payloads
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content")
        ]
        calls = [
            tc
            for d in payloads
            if d.get("choices")
            for tc in (d["choices"][0].get("delta", {}).get("tool_calls") or [])
        ]

        assert texts == [TEXT], f"text emitted {len(texts)} times: {texts}"
        assert len(calls) == 1, f"tool call emitted {len(calls)} times"

        first_text = next(
            i
            for i, d in enumerate(payloads)
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content")
        )
        first_call = next(
            i
            for i, d in enumerate(payloads)
            if d.get("choices") and d["choices"][0].get("delta", {}).get("tool_calls")
        )
        assert first_text < first_call, "text must be emitted before the tool call"

        assert any(
            d.get("choices") and d["choices"][0].get("finish_reason") for d in payloads
        ), "stream ended without a finish_reason"
        assert any("[DONE]" in raw for raw in chunks), "stream ended without [DONE]"

    @pytest.mark.anyio
    async def test_reasoning_parser_active_does_not_lose_the_text(
        self, both_in_one_delta, monkeypatch
    ):
        """The reasoning path reaches the same tool branch by another route."""
        from vllm_mlx.api.models import ChatCompletionRequest

        class _Reasoning:
            def extract_reasoning_streaming(self, previous, current, delta):
                return SimpleNamespace(reasoning=None, content=delta)

            def reset(self):
                pass

            def reset_state(self, *args, **kwargs):
                pass

        monkeypatch.setattr(srv, "_reasoning_parser", _Reasoning(), raising=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
        )
        engine = _engine(_stream_output(BUFFERED_OUTPUT), _stream_output("", "stop"))

        payloads = _data_payloads(
            await _collect(
                srv.stream_chat_completion(engine, request.messages, request)
            )
        )
        texts = [
            d["choices"][0]["delta"]["content"]
            for d in payloads
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content")
        ]
        assert TEXT in "".join(texts)
        calls = [
            tc
            for d in payloads
            if d.get("choices")
            for tc in (d["choices"][0].get("delta", {}).get("tool_calls") or [])
        ]
        assert len(calls) == 1, f"tool call emitted {len(calls)} times"
        assert any(
            d.get("choices") and d["choices"][0].get("finish_reason") for d in payloads
        ), "stream ended without a finish_reason"


class TestAnthropicStream:
    @pytest.mark.anyio
    async def test_text_block_is_emitted_before_the_call(self, both_in_one_delta):
        from vllm_mlx.api.models import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
        )
        engine = _engine(_stream_output(BUFFERED_OUTPUT), _stream_output("", "stop"))
        prepared = SimpleNamespace(
            messages=request.messages, tools=None, chat_kwargs={}
        )
        anthropic_request = SimpleNamespace(
            model="test-model", max_tokens=64, stream=True
        )

        chunks = await _collect(
            srv._stream_anthropic_messages(engine, request, anthropic_request, prepared)
        )
        body = "".join(chunks)

        assert TEXT in body, "assistant text never reached the Anthropic stream"
        assert body.count(TEXT) == 1, "text emitted more than once"
        assert body.count(CALL["function"]["name"]) == 1, "tool call not emitted once"
        assert body.index(TEXT) < body.index(
            CALL["function"]["name"]
        ), "text must precede the tool call"
        assert "message_stop" in body, "stream ended without message_stop"


class TestResponsesStream:
    @pytest.mark.anyio
    async def test_text_survives_the_responses_path(
        self, both_in_one_delta, monkeypatch
    ):
        from vllm_mlx.api.models import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
        )
        engine = _engine(_stream_output(BUFFERED_OUTPUT), _stream_output("", "stop"))
        monkeypatch.setattr(
            srv,
            "_prepare_streaming_responses_request",
            lambda req: (engine, request, request.messages, {}),
            raising=False,
        )

        from vllm_mlx.api.responses_models import ResponsesRequest

        responses_request = ResponsesRequest(
            model="test-model", input="weather?", stream=True
        )
        chunks = await _collect(srv._stream_responses_request(responses_request))
        body = "".join(chunks)

        assert TEXT in body, "assistant text never reached the Responses stream"

        # The Responses protocol repeats the final text in its done/completed
        # events by design, so count the incremental deltas rather than the
        # whole body.
        deltas = [
            payload["delta"]
            for payload in _data_payloads(chunks)
            if payload.get("type") == "response.output_text.delta"
        ]
        assert deltas == [TEXT], f"text streamed as {deltas!r}"

        # Like the text, the call is repeated across added/done/completed by
        # design, so count the item-added events rather than substring hits.
        payloads = _data_payloads(chunks)
        added = [
            d
            for d in payloads
            if d.get("type") == "response.output_item.added"
            and (d.get("item") or {}).get("type") == "function_call"
        ]
        assert len(added) == 1, f"tool call announced {len(added)} times"
        assert body.index(TEXT) < body.index(
            CALL["function"]["name"]
        ), "text must precede the tool call"
        assert "response.completed" in body, "stream ended without response.completed"

    @pytest.mark.anyio
    async def test_reasoning_content_is_routed_through_the_tool_parser(
        self, both_in_one_delta, monkeypatch
    ):
        """The reasoning branch used to emit and `continue`, skipping the parser.

        Tool markup then left the reasoning parser as ordinary text and went
        out as visible output with no tool calls parsed at all — a worse
        failure than the one this PR started from, and invisible to every test
        that runs without a reasoning parser.
        """
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.api.responses_models import ResponsesRequest

        class _Reasoning:
            def extract_reasoning_streaming(self, previous, current, delta):
                return SimpleNamespace(reasoning=None, content=delta)

            def reset_state(self, *args, **kwargs):
                pass

        monkeypatch.setattr(
            srv, "_build_reasoning_parser", lambda *a, **k: _Reasoning(), raising=False
        )
        monkeypatch.setattr(srv, "_thinking_disabled", lambda *a, **k: False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
        )
        engine = _engine(_stream_output(BUFFERED_OUTPUT), _stream_output("", "stop"))
        monkeypatch.setattr(
            srv,
            "_prepare_streaming_responses_request",
            lambda req: (engine, request, request.messages, {}),
            raising=False,
        )

        chunks = await _collect(
            srv._stream_responses_request(
                ResponsesRequest(model="test-model", input="weather?", stream=True)
            )
        )
        payloads = _data_payloads(chunks)
        deltas = [
            d["delta"]
            for d in payloads
            if d.get("type") == "response.output_text.delta"
        ]

        assert MARKER not in "".join(
            deltas
        ), "raw tool markup leaked to the client as output text"
        assert deltas == [TEXT], f"text streamed as {deltas!r}"

        added = [
            d
            for d in payloads
            if d.get("type") == "response.output_item.added"
            and (d.get("item") or {}).get("type") == "function_call"
        ]
        assert len(added) == 1, f"tool call announced {len(added)} times"
        assert "response.completed" in "".join(chunks)


class TestRealPoolsideParserParity:
    """Streaming must answer the same assistant text as non-streaming.

    The canned parser above cannot settle this: it is asked for one delta and
    returns whatever the fixture says. The real PoolsideV1ToolParser buffers
    prose and, when the whole block lands at once, hands back the text from
    *both* sides of the call concatenated. Non-streaming keeps only the prefix
    (``cleaned_text[:cleaned_text.find("<tool_call>")]``), so streaming that
    forwarded the parser's content verbatim would answer "BeforeAfter" where
    non-streaming answers "Before".
    """

    OUTPUT = (
        "Before"
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value>"
        "</tool_call>"
        "After"
    )
    REQUEST_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]

    def _parser(self):
        from vllm_mlx.tool_parsers import ToolParserManager

        return ToolParserManager.get_tool_parser("poolside_v1")()

    def test_non_streaming_keeps_only_the_prefix(self):
        """Pin the contract streaming has to match, straight from the parser."""
        result = self._parser().extract_tool_calls(
            self.OUTPUT, {"tools": self.REQUEST_TOOLS}
        )
        assert result.tools_called is True
        assert result.content == "Before"
        assert result.tool_calls[0]["name"] == "write_file"

    @pytest.mark.parametrize(
        "deltas",
        [
            pytest.param(None, id="single-delta"),
            pytest.param(
                [
                    "Before"
                    "<tool_call>write_file"
                    "<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value>"
                    "</tool_call>",
                    "After",
                ],
                id="text-after-call-in-a-later-delta",
            ),
            pytest.param(
                [
                    "Before",
                    "<tool_call>write_file"
                    "<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value>"
                    "</tool_call>",
                    "After",
                ],
                id="prefix-call-and-suffix-each-in-their-own-delta",
            ),
        ],
    )
    @pytest.mark.anyio
    async def test_streamed_text_matches_non_streaming(self, monkeypatch, deltas):
        from vllm_mlx.api.models import ChatCompletionRequest

        parser = self._parser()
        monkeypatch.setattr(
            srv, "_get_streaming_tool_parser", lambda *a, **k: parser, raising=False
        )
        monkeypatch.setattr(srv, "_reasoning_parser", None, raising=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "write it"}],
            stream=True,
            tools=self.REQUEST_TOOLS,
        )
        chunks_in = deltas if deltas is not None else [self.OUTPUT]
        engine = _engine(
            *[_stream_output(c) for c in chunks_in], _stream_output("", "stop")
        )

        payloads = _data_payloads(
            await _collect(
                srv.stream_chat_completion(engine, request.messages, request)
            )
        )
        texts = [
            d["choices"][0]["delta"]["content"]
            for d in payloads
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content")
        ]

        expected = self._parser().extract_tool_calls(
            self.OUTPUT, {"tools": self.REQUEST_TOOLS}
        )
        assert (
            "".join(texts) == expected.content
        ), f"streamed {texts!r}, non-streaming returned {expected.content!r}"
        assert "After" not in "".join(texts), "text following the call must be dropped"
