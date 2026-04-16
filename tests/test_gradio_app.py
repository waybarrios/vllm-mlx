# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gradio chat app request payload behavior."""

import sys

from vllm_mlx import gradio_app


class DummyResponse:
    """Minimal mock response object for requests.post."""

    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_create_chat_function_uses_default_served_model_name(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse("ok")

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)

    chat_fn = gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=128,
        temperature=0.3,
    )
    output = chat_fn({"text": "hello", "files": []}, history=[])

    assert output == "ok"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["model"] == "default"


def test_create_chat_function_uses_configured_served_model_name(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse("ok")

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)

    chat_fn = gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=128,
        temperature=0.3,
        served_model_name="my-served-model",
    )
    output = chat_fn({"text": "hello", "files": []}, history=[])

    assert output == "ok"
    assert captured["json"]["model"] == "my-served-model"


def test_main_text_only_uses_configured_served_model_name(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse("ok")

    class FakeChatInterface:
        def __init__(self, fn, **kwargs):
            self.fn = fn

        def launch(self, server_port, share):
            captured["server_port"] = server_port
            captured["share"] = share
            captured["chat_output"] = self.fn("hello", [])

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)
    monkeypatch.setattr(gradio_app.gr, "ChatInterface", FakeChatInterface)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm-mlx-chat",
            "--text-only",
            "--served-model-name",
            "my-served-model",
        ],
    )

    gradio_app.main()

    assert captured["chat_output"] == "ok"
    assert captured["json"]["model"] == "my-served-model"
