# SPDX-License-Identifier: Apache-2.0
"""Server tests for OpenAI-compatible logprobs on chat and text completions."""

import json
import platform
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


def _entry(token_id, token, logprob, top=()):
    from vllm_mlx.logprobs import TokenLogprob

    return TokenLogprob(
        token_id=token_id, token=token, logprob=logprob, top_logprobs=list(top)
    )


def _sse_payloads(chunks):
    payloads = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payloads.append(json.loads(line[len("data: ") :]))
    return payloads


def _streamed_chat_tokens(chunks):
    return [
        token["token"]
        for payload in _sse_payloads(chunks)
        for choice in payload.get("choices", [])
        if choice.get("logprobs")
        for token in choice["logprobs"]["content"]
    ]


class FakeChatEngine:
    """Chat engine double that records its kwargs and returns canned outputs."""

    model_name = "fake-engine"
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self, result=None, stream=(), supports_logprobs=True):
        self.result = result
        self.stream = list(stream)
        self.supports_logprobs = supports_logprobs
        self.kwargs = None

    async def chat(self, messages, **kwargs):
        self.kwargs = kwargs
        return self.result

    async def stream_chat(self, messages, **kwargs):
        self.kwargs = kwargs
        for output in self.stream:
            yield output


class FakeCompletionEngine:
    """Completion engine double that records its kwargs and returns canned outputs."""

    def __init__(self, result=None, stream=(), supports_logprobs=True):
        self.result = result
        self.stream = list(stream)
        self.supports_logprobs = supports_logprobs
        self.kwargs = None

    async def generate(self, **kwargs):
        self.kwargs = kwargs
        return self.result

    async def stream_generate(self, **kwargs):
        self.kwargs = kwargs
        for output in self.stream:
            yield output


@pytest.fixture
def chat_server(monkeypatch):
    """Patch server globals so chat handlers run against ``state["engine"]``."""
    import vllm_mlx.server as server

    state = {"engine": None}

    async def fake_acquire(
        raw_request,
        *,
        total_timeout=None,
        deadline=None,
        count_activity=True,
        model=None,
    ):
        return state["engine"]

    async def fake_release(*, count_activity=True):
        return None

    monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", fake_acquire)
    monkeypatch.setattr(server, "_release_default_engine", fake_release)
    monkeypatch.setattr(server, "_model_name", "served-model")
    monkeypatch.setattr(server, "_default_max_tokens", 128)
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    monkeypatch.setattr(server, "_reasoning_parser_name", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    return state


@pytest.fixture
def hides_tag_parser(monkeypatch):
    """Install a reasoning parser that consumes ``<tag>`` without emitting a chunk."""
    import vllm_mlx.server as server
    from vllm_mlx.reasoning import DeltaMessage

    class HidesTagParser:
        def __init__(self, tokenizer=None):
            pass

        def reset_state(self, implicit_mode: bool = False):
            pass

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            if delta_text == "<tag>":
                return None
            return DeltaMessage(content=delta_text)

    monkeypatch.setattr(server, "_reasoning_parser_name", "hides-tag")
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "get_reasoning_parser", lambda name: HidesTagParser)


@pytest.fixture
def completion_server(monkeypatch):
    """Patch server globals so completion handlers use ``state["engine"]``."""
    import vllm_mlx.server as server

    state = {"engine": None}
    monkeypatch.setattr(server, "_model_name", "test-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)
    monkeypatch.setattr(server, "get_engine", lambda: state["engine"])
    return state


def _chat_request(**fields):
    from vllm_mlx.server import ChatCompletionRequest, Message

    return ChatCompletionRequest(
        model="served-model",
        messages=[Message(role="user", content="Hello")],
        **fields,
    )


async def _stream_chat(engine, request):
    from vllm_mlx.server import stream_chat_completion

    return [
        chunk
        async for chunk in stream_chat_completion(engine, request.messages, request)
    ]


class TestChatCompletionLogprobs:
    @pytest.mark.anyio
    async def test_nonstream_returns_logprobs_and_forwards_top(self, chat_server):
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import create_chat_completion

        engine = FakeChatEngine(
            result=GenerationOutput(
                text="Hi!",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
                logprobs=[
                    _entry(0, "Hi", -0.1, [(0, "Hi", -0.1), (1, "Hey", -2.0)]),
                    _entry(2, "!", -0.3),
                ],
            )
        )
        chat_server["engine"] = engine

        response = await create_chat_completion(
            _chat_request(logprobs=True, top_logprobs=2), raw_request=None
        )

        assert engine.kwargs["logprobs"] == 2
        content = response.choices[0].logprobs.content
        assert [token.token for token in content] == ["Hi", "!"]
        assert [top.token for top in content[0].top_logprobs] == ["Hi", "Hey"]
        assert content[1].bytes == [ord("!")]

    @pytest.mark.anyio
    async def test_nonstream_without_logprobs_sends_no_flag(self, chat_server):
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import create_chat_completion

        engine = FakeChatEngine(
            result=GenerationOutput(text="ok", finish_reason="stop"),
            supports_logprobs=False,
        )
        chat_server["engine"] = engine

        response = await create_chat_completion(_chat_request(), raw_request=None)

        assert "logprobs" not in engine.kwargs
        assert response.choices[0].logprobs is None

    @pytest.mark.anyio
    async def test_engine_without_support_is_rejected(self, chat_server):
        from fastapi import HTTPException

        from vllm_mlx.server import create_chat_completion

        engine = FakeChatEngine(supports_logprobs=False)
        chat_server["engine"] = engine

        with pytest.raises(HTTPException) as excinfo:
            await create_chat_completion(_chat_request(logprobs=True), raw_request=None)

        assert excinfo.value.status_code == 400
        assert "continuous-batching" in excinfo.value.detail
        assert engine.kwargs is None

    @pytest.mark.anyio
    async def test_stream_emits_each_token_logprob_once(self, chat_server):
        from vllm_mlx.engine.base import GenerationOutput

        engine = FakeChatEngine(
            stream=[
                GenerationOutput(
                    text="",
                    new_text="Hi",
                    finished=False,
                    new_logprobs=[_entry(0, "Hi", -0.1)],
                ),
                GenerationOutput(
                    text="",
                    new_text="!",
                    finished=True,
                    finish_reason="stop",
                    new_logprobs=[_entry(2, "!", -0.3)],
                ),
            ]
        )

        chunks = await _stream_chat(engine, _chat_request(stream=True, logprobs=True))

        assert _streamed_chat_tokens(chunks) == ["Hi", "!"]

    @pytest.mark.anyio
    async def test_stream_keeps_logprobs_of_a_consumed_final_token(
        self, chat_server, hides_tag_parser
    ):
        """A parser that swallows the finishing output forces the terminal chunk."""
        from vllm_mlx.engine.base import GenerationOutput

        engine = FakeChatEngine(
            stream=[
                GenerationOutput(
                    text="",
                    new_text="Hi",
                    finished=False,
                    new_logprobs=[_entry(0, "Hi", -0.1)],
                ),
                GenerationOutput(
                    text="",
                    new_text="<tag>",
                    finished=True,
                    finish_reason="stop",
                    new_logprobs=[_entry(9, "<tag>", -0.2)],
                ),
            ]
        )

        chunks = await _stream_chat(engine, _chat_request(stream=True, logprobs=True))

        assert _streamed_chat_tokens(chunks) == ["Hi", "<tag>"]
        finishing = [
            choice
            for payload in _sse_payloads(chunks)
            for choice in payload.get("choices", [])
            if choice.get("finish_reason")
        ]
        assert finishing and finishing[-1]["logprobs"]["content"][0]["token"] == "<tag>"

    @pytest.mark.anyio
    async def test_stream_flushes_consumed_tokens_when_stream_ends_early(
        self, chat_server, hides_tag_parser
    ):
        from vllm_mlx.engine.base import GenerationOutput

        engine = FakeChatEngine(
            stream=[
                GenerationOutput(
                    text="",
                    new_text="Hi",
                    finished=False,
                    new_logprobs=[_entry(0, "Hi", -0.1)],
                ),
                GenerationOutput(
                    text="",
                    new_text="<tag>",
                    finished=False,
                    new_logprobs=[_entry(9, "<tag>", -0.2)],
                ),
            ]
        )

        chunks = await _stream_chat(engine, _chat_request(stream=True, logprobs=True))

        assert _streamed_chat_tokens(chunks) == ["Hi", "<tag>"]


class TestAttachPendingLogprobs:
    def _chunk_with_choice(self):
        from vllm_mlx.api.models import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            ChatCompletionChunkDelta,
        )

        return ChatCompletionChunk(
            model="m",
            choices=[
                ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content="Hi"))
            ],
        )

    def test_chunk_without_choices_leaves_entries_pending(self):
        from vllm_mlx.api.models import ChatCompletionChunk
        from vllm_mlx.server import _attach_pending_logprobs

        pending = [_entry(0, "Hi", -0.1)]
        _attach_pending_logprobs(ChatCompletionChunk(model="m", choices=[]), pending)

        assert len(pending) == 1

    def test_entries_move_onto_the_next_chunk_with_choices(self):
        from vllm_mlx.server import _attach_pending_logprobs

        pending = [_entry(0, "Hi", -0.1)]
        chunk = self._chunk_with_choice()
        _attach_pending_logprobs(chunk, pending)

        assert [t.token for t in chunk.choices[0].logprobs.content] == ["Hi"]
        assert pending == []


class TestCompletionLogprobs:
    @pytest.mark.anyio
    async def test_nonstream_completion_returns_legacy_logprobs(
        self, completion_server
    ):
        from vllm_mlx.server import CompletionRequest, create_completion

        engine = FakeCompletionEngine(
            result=SimpleNamespace(
                text="ab",
                finish_reason="stop",
                completion_tokens=2,
                prompt_tokens=1,
                logprobs=[
                    _entry(0, "a", -0.2, [(0, "a", -0.2), (1, "b", -1.9)]),
                    _entry(1, "b", -0.4),
                ],
            )
        )
        completion_server["engine"] = engine

        response = await create_completion(
            CompletionRequest(model="test-model", prompt="hello", logprobs=1),
            raw_request=None,
        )

        assert engine.kwargs["logprobs"] == 1
        logprobs = response.choices[0].logprobs
        assert logprobs.tokens == ["a", "b"]
        assert logprobs.token_logprobs == [-0.2, -0.4]
        assert logprobs.text_offset == [0, 1]
        assert logprobs.top_logprobs[0] == {"a": -0.2, "b": -1.9}

    @pytest.mark.anyio
    async def test_completion_engine_without_support_is_rejected(
        self, completion_server
    ):
        from fastapi import HTTPException

        from vllm_mlx.server import CompletionRequest, create_completion

        engine = FakeCompletionEngine(supports_logprobs=False)
        completion_server["engine"] = engine

        with pytest.raises(HTTPException) as excinfo:
            await create_completion(
                CompletionRequest(model="test-model", prompt="hello", logprobs=0),
                raw_request=None,
            )

        assert excinfo.value.status_code == 400
        assert engine.kwargs is None

    @pytest.mark.anyio
    async def test_stream_completion_offsets_continue_across_chunks(self):
        from vllm_mlx.api.models import CompletionRequest
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import stream_completion

        engine = FakeCompletionEngine(
            stream=[
                GenerationOutput(
                    text="",
                    new_text="ab",
                    finished=False,
                    new_logprobs=[_entry(0, "ab", -0.5)],
                ),
                GenerationOutput(
                    text="",
                    new_text="c",
                    finished=True,
                    finish_reason="stop",
                    completion_tokens=2,
                    prompt_tokens=1,
                    new_logprobs=[_entry(2, "c", -0.7)],
                ),
            ]
        )
        request = CompletionRequest(model="test-model", prompt="hello", logprobs=0)

        chunks = [
            chunk
            async for chunk in stream_completion(engine, "hello", request, max_tokens=8)
        ]

        assert engine.kwargs["logprobs"] == 0
        choice_logprobs = [
            payload["choices"][0]["logprobs"] for payload in _sse_payloads(chunks)
        ]
        assert [lp["tokens"] for lp in choice_logprobs] == [["ab"], ["c"]]
        assert [lp["text_offset"] for lp in choice_logprobs] == [[0], [2]]
