# SPDX-License-Identifier: Apache-2.0
"""Tests for chat template kwargs forwarding."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.engine.base import GenerationOutput


def test_chat_completion_request_preserves_chat_template_kwargs():
    request = srv.ChatCompletionRequest(
        model="test-model",
        messages=[srv.Message(role="user", content="Hello")],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert request.chat_template_kwargs == {"enable_thinking": False}


def test_batched_engine_applies_chat_template_kwargs():
    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("test-model")
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "prompt"

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        assert prompt == "prompt"
        engine._tokenizer.apply_chat_template.assert_called_once()
        assert (
            engine._tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"]
            is False
        )


def test_chat_completion_endpoint_forwards_chat_template_kwargs():
    captured = {}

    class FakeEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False

        async def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return GenerationOutput(
                text="ORBIT",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason="stop",
            )

    client = TestClient(srv.app)
    original_engine = srv._engine
    original_model_name = srv._model_name
    srv._engine = FakeEngine()
    srv._model_name = "test-model"
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply with ORBIT."}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name

    assert response.status_code == 200
    assert captured["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.json()["choices"][0]["message"]["content"] == "ORBIT"
