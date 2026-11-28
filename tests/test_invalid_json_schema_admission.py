# SPDX-License-Identifier: Apache-2.0
"""Reject broken strict schemas before they can enter a generation batch."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

import vllm_mlx.server as server
from vllm_mlx.api.tool_calling import build_json_logits_processor
from vllm_mlx.constrained.json_schema_processor import ConstrainedDecodingError
from vllm_mlx.engine.base import GenerationOutput

_VALID_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ['{"ok":true}'],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<unk>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>"
    )


@pytest.fixture(
    params=[
        pytest.param(
            ({"type": "object", "properties": {"value": {"type": "strng"}}}, "strng"),
            id="unknown-nested-type",
        ),
        pytest.param(
            ({"$ref": "#/$defs/missing"}, "missing"), id="unresolved-reference"
        ),
        pytest.param(
            ({"type": "string", "pattern": "(unclosed"}, "unclosed"),
            id="invalid-pattern",
        ),
    ]
)
def invalid_schema(request):
    return request.param


def _response_format(schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": "result", "schema": schema, "strict": True},
    }


def test_invalid_schema_is_rejected_during_construction(tokenizer, invalid_schema):
    schema, diagnostic = invalid_schema

    with pytest.raises(ConstrainedDecodingError, match=diagnostic):
        build_json_logits_processor(_response_format(schema), tokenizer)


class _RecordingEngine:
    model_name = "test-model"
    is_mllm = False
    context_limit = None
    preserve_native_tool_format = False

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return GenerationOutput(
            text='{"ok":true}',
            new_text='{"ok":true}',
            finished=True,
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=5,
        )

    async def stream_chat(self, messages, **kwargs):
        yield await self.chat(messages, **kwargs)


@pytest.fixture
def client_and_engine(monkeypatch, tokenizer):
    engine = _RecordingEngine(tokenizer)
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_model_name", engine.model_name)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_default_thinking_token_budget", None)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    client = TestClient(server.app, raise_server_exceptions=False)
    yield client, engine
    client.close()


def _payload(schema, stream):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "return JSON"}],
        "max_tokens": 16,
        "stream": stream,
        "response_format": _response_format(schema),
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/messages"])
def test_invalid_schema_never_reaches_generation(
    client_and_engine, invalid_schema, stream, endpoint
):
    client, engine = client_and_engine
    schema, diagnostic = invalid_schema

    response = client.post(endpoint, json=_payload(schema, stream))

    assert response.status_code == 422, response.text
    assert diagnostic in response.text
    assert engine.calls == 0
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_schema_does_not_poison_valid_requests(
    client_and_engine, invalid_schema, stream
):
    client, engine = client_and_engine
    schema, _ = invalid_schema

    before = client.post("/v1/chat/completions", json=_payload(_VALID_SCHEMA, stream))
    assert before.status_code == 200, before.text
    assert engine.calls == 1

    rejected = client.post("/v1/chat/completions", json=_payload(schema, stream))
    assert rejected.status_code == 422, rejected.text
    assert engine.calls == 1

    after = client.post("/v1/chat/completions", json=_payload(_VALID_SCHEMA, stream))
    assert after.status_code == 200, after.text
    assert engine.calls == 2
    if stream:
        events = [
            line.removeprefix("data: ")
            for line in after.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(event) for event in events[:-1]]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )
        assert json.loads(content) == {"ok": True}
    else:
        assert after.json()["choices"][0]["message"]["content"] == '{"ok": true}'


@pytest.mark.anyio
async def test_invalid_schema_leaves_inflight_requests_running(
    client_and_engine, invalid_schema, monkeypatch
):
    _, engine = client_and_engine
    schema, _ = invalid_schema
    entered = asyncio.Queue()
    release = asyncio.Event()
    original_chat = engine.chat

    async def hold_valid_request(messages, **kwargs):
        output = await original_chat(messages, **kwargs)
        if messages[-1]["content"] == "held valid request":
            entered.put_nowait(None)
            await release.wait()
        return output

    monkeypatch.setattr(engine, "chat", hold_valid_request)
    transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        valid = _payload(_VALID_SCHEMA, False)
        valid["messages"][0]["content"] = "held valid request"
        pending = [
            asyncio.create_task(client.post("/v1/chat/completions", json=valid))
            for _ in range(3)
        ]
        try:
            for _ in pending:
                await asyncio.wait_for(entered.get(), timeout=5)
            rejected = await asyncio.wait_for(
                client.post("/v1/chat/completions", json=_payload(schema, False)),
                timeout=5,
            )
            assert rejected.status_code == 422, rejected.text
            assert engine.calls == 3
            assert all(not request.done() for request in pending)
        finally:
            release.set()
            completed = await asyncio.gather(*pending)

    assert all(response.status_code == 200 for response in completed)
    assert all(
        json.loads(response.json()["choices"][0]["message"]["content"]) == {"ok": True}
        for response in completed
    )
