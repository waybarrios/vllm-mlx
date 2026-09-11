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


@pytest.fixture
def chat_server(monkeypatch):
    """Patch server globals so chat handlers run against a fake engine."""
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


class TestChatCompletionLogprobs:
    @pytest.mark.anyio
    async def test_nonstream_returns_logprobs_and_forwards_top(self, chat_server):
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            create_chat_completion,
        )

        captured = {}

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False
            supports_logprobs = True

            async def chat(self, messages, **kwargs):
                captured.update(kwargs)
                return GenerationOutput(
                    text="Hi!",
                    prompt_tokens=3,
                    completion_tokens=2,
                    finish_reason="stop",
                    logprobs=[
                        _entry(0, "Hi", -0.1, [(0, "Hi", -0.1), (1, "Hey", -2.0)]),
                        _entry(2, "!", -0.3),
                    ],
                )

        chat_server["engine"] = FakeEngine()
        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="Hello")],
            logprobs=True,
            top_logprobs=2,
        )

        response = await create_chat_completion(request, raw_request=None)

        assert captured["logprobs"] == 2
        content = response.choices[0].logprobs.content
        assert [token.token for token in content] == ["Hi", "!"]
        assert [top.token for top in content[0].top_logprobs] == ["Hi", "Hey"]
        assert content[1].bytes == [33]

    @pytest.mark.anyio
    async def test_nonstream_without_logprobs_sends_no_flag(self, chat_server):
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            create_chat_completion,
        )

        captured = {}

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                captured.update(kwargs)
                return GenerationOutput(text="ok", finish_reason="stop")

        chat_server["engine"] = FakeEngine()
        request = ChatCompletionRequest(
            model="served-model", messages=[Message(role="user", content="Hello")]
        )

        response = await create_chat_completion(request, raw_request=None)

        assert "logprobs" not in captured
        assert response.choices[0].logprobs is None

    @pytest.mark.anyio
    async def test_engine_without_support_is_rejected(self, chat_server):
        from fastapi import HTTPException

        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            create_chat_completion,
        )

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                raise AssertionError("engine must not be called")

        chat_server["engine"] = FakeEngine()
        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="Hello")],
            logprobs=True,
        )

        with pytest.raises(HTTPException) as excinfo:
            await create_chat_completion(request, raw_request=None)
        assert excinfo.value.status_code == 400
        assert "continuous-batching" in excinfo.value.detail

    @pytest.mark.anyio
    async def test_stream_emits_each_token_logprob_once(self, chat_server):
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )

        class FakeEngine:
            model_name = "fake-engine"
            supports_logprobs = True

            async def stream_chat(self, messages, **kwargs):
                yield GenerationOutput(
                    text="",
                    new_text="Hi",
                    finished=False,
                    logprobs=[_entry(0, "Hi", -0.1)],
                )
                yield GenerationOutput(
                    text="",
                    new_text="!",
                    finished=True,
                    finish_reason="stop",
                    logprobs=[_entry(2, "!", -0.3)],
                )

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="Hello")],
            stream=True,
            logprobs=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]

        tokens = [
            token["token"]
            for payload in _sse_payloads(chunks)
            for choice in payload.get("choices", [])
            if choice.get("logprobs")
            for token in choice["logprobs"]["content"]
        ]
        assert tokens == ["Hi", "!"]

    def test_pending_logprobs_wait_for_a_chunk_with_choices(self):
        from vllm_mlx.api.models import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            ChatCompletionChunkDelta,
        )
        from vllm_mlx.server import _attach_pending_logprobs

        pending = [_entry(0, "Hi", -0.1)]

        usage_only = _attach_pending_logprobs(
            ChatCompletionChunk(model="m", choices=[]), pending
        )
        assert usage_only.choices == []
        assert len(pending) == 1

        chunk = _attach_pending_logprobs(
            ChatCompletionChunk(
                model="m",
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content="Hi")
                    )
                ],
            ),
            pending,
        )
        assert [t.token for t in chunk.choices[0].logprobs.content] == ["Hi"]
        assert pending == []


class TestCompletionLogprobs:
    @pytest.mark.anyio
    async def test_nonstream_completion_returns_legacy_logprobs(self, monkeypatch):
        import vllm_mlx.server as server
        from vllm_mlx.server import CompletionRequest, create_completion

        captured = {}

        class DummyEngine:
            supports_logprobs = True

            async def generate(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    text="ab",
                    finish_reason="stop",
                    completion_tokens=2,
                    prompt_tokens=1,
                    logprobs=[
                        _entry(0, "a", -0.2, [(0, "a", -0.2), (1, "b", -1.9)]),
                        _entry(1, "b", -0.4),
                    ],
                )

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_model_manager", None)
        monkeypatch.setattr(server, "_residency_manager", None)
        monkeypatch.setattr(server, "_default_model_key", None)
        monkeypatch.setattr(server, "get_engine", lambda: DummyEngine())

        request = CompletionRequest(model="test-model", prompt="hello", logprobs=1)
        response = await create_completion(request, raw_request=None)

        assert captured["logprobs"] == 1
        logprobs = response.choices[0].logprobs
        assert logprobs.tokens == ["a", "b"]
        assert logprobs.token_logprobs == [-0.2, -0.4]
        assert logprobs.text_offset == [0, 1]
        assert logprobs.top_logprobs[0] == {"a": -0.2, "b": -1.9}

    @pytest.mark.anyio
    async def test_completion_engine_without_support_is_rejected(self, monkeypatch):
        from fastapi import HTTPException

        import vllm_mlx.server as server
        from vllm_mlx.server import CompletionRequest, create_completion

        class DummyEngine:
            async def generate(self, **kwargs):
                raise AssertionError("engine must not be called")

        monkeypatch.setattr(server, "_model_name", "test-model")
        monkeypatch.setattr(server, "_model_manager", None)
        monkeypatch.setattr(server, "_residency_manager", None)
        monkeypatch.setattr(server, "_default_model_key", None)
        monkeypatch.setattr(server, "get_engine", lambda: DummyEngine())

        request = CompletionRequest(model="test-model", prompt="hello", logprobs=0)
        with pytest.raises(HTTPException) as excinfo:
            await create_completion(request, raw_request=None)
        assert excinfo.value.status_code == 400

    @pytest.mark.anyio
    async def test_stream_completion_offsets_continue_across_chunks(self):
        from vllm_mlx.api.models import CompletionRequest
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.server import stream_completion

        captured = {}

        class DummyEngine:
            async def stream_generate(self, **kwargs):
                captured.update(kwargs)
                yield GenerationOutput(
                    text="",
                    new_text="ab",
                    finished=False,
                    logprobs=[_entry(0, "ab", -0.5)],
                )
                yield GenerationOutput(
                    text="",
                    new_text="c",
                    finished=True,
                    finish_reason="stop",
                    completion_tokens=2,
                    prompt_tokens=1,
                    logprobs=[_entry(2, "c", -0.7)],
                )

        request = CompletionRequest(model="test-model", prompt="hello", logprobs=0)
        chunks = [
            chunk
            async for chunk in stream_completion(
                DummyEngine(), "hello", request, max_tokens=8
            )
        ]

        assert captured["logprobs"] == 0
        choice_logprobs = [
            payload["choices"][0]["logprobs"] for payload in _sse_payloads(chunks)
        ]
        assert [lp["tokens"] for lp in choice_logprobs] == [["ab"], ["c"]]
        assert [lp["text_offset"] for lp in choice_logprobs] == [[0], [2]]
