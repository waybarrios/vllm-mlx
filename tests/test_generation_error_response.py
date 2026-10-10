# SPDX-License-Identifier: Apache-2.0
"""Engine-reported generation failures must reach the client as errors.

The schedulers report a request that failed inside the engine (for example a
Metal out-of-memory error during prefill) as a finished output with
``finish_reason="error"`` and no tokens. Before this fix the server passed that
through: non-streaming requests got HTTP 200 with empty content, and streams
ended normally with ``finish_reason: "error"`` (not an OpenAI value), which
clients read as a successful, empty completion.
"""

import json

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.engine.base import GenerationOutput


def _error_output() -> GenerationOutput:
    return GenerationOutput(
        text="",
        new_text="",
        finished=True,
        finish_reason="error",
        prompt_tokens=0,
        completion_tokens=0,
    )


class FailingEngine:
    """Engine whose every request fails the way BatchedEngine reports it."""

    model_name = "test-model"
    is_mllm = False
    preserve_native_tool_format = False
    tokenizer = None

    async def chat(self, messages, **kwargs):
        return _error_output()

    async def generate(self, **kwargs):
        return _error_output()

    async def stream_chat(self, messages, **kwargs):
        yield _error_output()

    async def stream_generate(self, **kwargs):
        yield _error_output()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(srv, "_engine", FailingEngine())
    monkeypatch.setattr(srv, "_model_name", "test-model")
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(srv, "_tool_call_parser", None)
    monkeypatch.setattr(srv, "_tool_parser_instance", None)
    return TestClient(srv.app)


def _sse_data(body: str) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _assert_openai_stream_error(body: str) -> None:
    data = _sse_data(body)
    assert data[-1] == "[DONE]"
    payloads = [json.loads(d) for d in data if d != "[DONE]"]
    errors = [p["error"] for p in payloads if "error" in p]
    assert len(errors) == 1
    assert errors[0]["code"] == 500
    assert errors[0]["type"] == "InternalServerError"
    assert errors[0]["message"] == "Internal server error during generation"
    # The internal value must not leak out as a finish reason.
    finish_reasons = [
        choice.get("finish_reason") for p in payloads for choice in p.get("choices", [])
    ]
    assert "error" not in finish_reasons


def test_chat_non_stream_returns_500(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500


def test_chat_stream_emits_error_event_then_done(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    # Headers were already sent when the engine failed, so the status stays 200;
    # the failure is carried by the error event.
    assert response.status_code == 200
    _assert_openai_stream_error(response.text)


def test_completion_non_stream_returns_500(client):
    response = client.post(
        "/v1/completions", json={"model": "test-model", "prompt": "hi"}
    )
    assert response.status_code == 500


def test_completion_stream_emits_error_event_then_done(client):
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "hi", "stream": True},
    )
    assert response.status_code == 200
    _assert_openai_stream_error(response.text)


def test_anthropic_non_stream_returns_500(client):
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 500


def test_anthropic_stream_emits_error_event_then_message_stop(client):
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert "error" in events
    assert events[-1] == "message_stop"
    error = next(
        json.loads(d) for d in _sse_data(response.text) if '"type": "error"' in d
    )
    assert error["error"]["type"] == "api_error"


def test_responses_non_stream_returns_500(client):
    response = client.post("/v1/responses", json={"model": "test-model", "input": "hi"})
    assert response.status_code == 500


@pytest.mark.anyio
async def test_ensure_sse_terminal_reports_unexpected_exceptions_generically():
    """Any mid-stream exception becomes an error event; internals stay in the log."""

    async def exploding():
        yield "data: {}\n\n"
        raise RuntimeError("[METAL] Command buffer execution failed: secret detail")

    chunks = [
        chunk
        async for chunk in srv._ensure_sse_terminal(exploding(), "data: [DONE]\n\n")
    ]
    assert chunks[0] == "data: {}\n\n"
    assert chunks[-1] == "data: [DONE]\n\n"
    error = json.loads(chunks[1].removeprefix("data: "))["error"]
    assert error == {
        "message": "Internal server error",
        "type": "InternalServerError",
        "param": None,
        "code": 500,
    }


@pytest.mark.anyio
async def test_ensure_sse_terminal_no_error_event_on_success():
    async def happy():
        yield "data: {}\n\n"
        yield "data: [DONE]\n\n"

    chunks = [
        chunk async for chunk in srv._ensure_sse_terminal(happy(), "data: [DONE]\n\n")
    ]
    assert chunks == ["data: {}\n\n", "data: [DONE]\n\n"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
