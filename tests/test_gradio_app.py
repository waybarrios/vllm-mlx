# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gradio chat app request payload behavior."""

import sys

import pytest

from vllm_mlx import gradio_app


class DummyResponse:
    """Minimal mock response object for requests.post."""

    def __init__(self, content: str, usage: dict | None = None):
        self._content = content
        self._usage = usage

    def raise_for_status(self):
        return None

    def json(self):
        body = {"choices": [{"message": {"content": self._content}}]}
        if self._usage is not None:
            body["usage"] = self._usage
        return body


class HealthResponse:
    """Minimal mock of GET /health."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


MLLM_HEALTH = {"model_loaded": True, "model_type": "mllm", "model_name": "vl"}
TEXT_ONLY_HEALTH = {"model_loaded": True, "model_type": "llm", "model_name": "lm"}


@pytest.fixture(autouse=True)
def _multimodal_server(monkeypatch):
    """Answer /health as a loaded multimodal server unless a test says not."""
    monkeypatch.setattr(
        gradio_app.requests, "get", lambda *a, **k: HealthResponse(MLLM_HEALTH)
    )


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


def test_chat_function_does_not_print_prompt_or_file_details(monkeypatch, capsys):
    monkeypatch.setattr(
        gradio_app.requests, "post", lambda *args, **kwargs: DummyResponse("ok")
    )
    chat_fn = gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=16,
        temperature=0.0,
    )

    assert chat_fn({"text": "private prompt", "files": []}, history=[]) == "ok"
    captured = capsys.readouterr()
    assert "private prompt" not in captured.out


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


def test_create_chat_function_does_not_replay_historical_media(monkeypatch):
    captured_requests = []

    def fake_post(url, json, timeout):
        captured_requests.append(json)
        return DummyResponse("ok")

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)
    monkeypatch.setattr(
        gradio_app,
        "build_media_items",
        lambda files: [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/quicktime;base64,AAAA"},
            }
        ],
    )

    chat_fn = gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=128,
        temperature=0.3,
    )
    chat_fn(
        {"text": "Identify the instrument.", "files": ["clip.mov"]},
        history=[],
    )
    chat_fn(
        {"text": "What pattern is on the strap?", "files": []},
        history=[
            _user_turn("Identify the instrument.", "clip.mov"),
            {"role": "assistant", "content": "An electric guitar."},
        ],
    )

    first_messages = captured_requests[0]["messages"]
    assert first_messages[0]["content"][1]["type"] == "video_url"

    follow_up_messages = captured_requests[1]["messages"]
    assert all(isinstance(message["content"], str) for message in follow_up_messages)
    assert gradio_app.OMITTED_MEDIA_NOTE in follow_up_messages[0]["content"]
    assert "data:video/" not in str(follow_up_messages)


def _user_turn(text: str, *files: str) -> dict:
    """A user turn as Gradio 6 hands it to the chat function."""
    parts = [{"type": "file", "file": {"path": f}, "alt_text": None} for f in files]
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def _video_chat_fn(monkeypatch, captured_requests: list):
    def fake_post(url, json, timeout):
        captured_requests.append(json)
        return DummyResponse("ok")

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)
    monkeypatch.setattr(
        gradio_app,
        "build_media_items",
        lambda files: [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/quicktime;base64,AAAA"},
            }
        ],
    )
    return gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=128,
        temperature=0.3,
    )


def test_interleaved_sessions_do_not_share_media_markers(monkeypatch):
    """One chat function serves every browser session (review on #733).

    Session A attaches a video; session B is text-only; a third session starts
    a new chat in between. Each follow-up must see markers for its own history
    only - B never inherits A's marker, and B's new chat cannot erase A's.
    """
    captured = []
    chat_fn = _video_chat_fn(monkeypatch, captured)
    a_history = [
        _user_turn("Identify the instrument.", "clip.mov"),
        {"role": "assistant", "content": "A guitar."},
    ]
    b_history = [_user_turn("Hello"), {"role": "assistant", "content": "Hi."}]

    chat_fn({"text": "Identify the instrument.", "files": ["clip.mov"]}, history=[])
    chat_fn({"text": "A: strap pattern?", "files": []}, history=a_history)
    chat_fn({"text": "B: anything?", "files": []}, history=b_history)
    chat_fn({"text": "C: new chat", "files": []}, history=[])
    chat_fn({"text": "A: and the body?", "files": []}, history=a_history)
    chat_fn({"text": "B: still nothing?", "files": []}, history=b_history)

    _, a1, b1, _, a2, b2 = (str(r["messages"]) for r in captured)
    note = gradio_app.OMITTED_MEDIA_NOTE
    assert note in a1 and note in a2
    assert note not in b1 and note not in b2


def test_new_chat_carries_no_media_marker(monkeypatch):
    captured = []
    chat_fn = _video_chat_fn(monkeypatch, captured)
    chat_fn({"text": "Old chat", "files": ["clip.mov"]}, history=[])
    chat_fn(
        {"text": "Follow up", "files": []},
        history=[_user_turn("New chat"), {"role": "assistant", "content": "ok"}],
    )
    assert gradio_app.OMITTED_MEDIA_NOTE not in str(captured[1]["messages"])


@pytest.mark.parametrize(
    "content, has_media",
    [
        (
            [
                {"type": "file", "file": {"path": "a.mov"}},
                {"type": "text", "text": "x"},
            ],
            True,
        ),
        ([{"type": "text", "text": "x"}], False),
        ({"path": "a.png"}, True),  # Gradio 4/5 file message
        (("a.png", None), True),  # Gradio 4 tuple
        ("plain text", False),
        ([{"path": "a.png"}, "caption"], True),
    ],
)
def test_history_message_has_media_shapes(content, has_media):
    assert gradio_app._history_message_has_media(content) is has_media


def test_file_message_path_never_reaches_the_prompt(monkeypatch):
    captured = []
    chat_fn = _video_chat_fn(monkeypatch, captured)
    chat_fn(
        {"text": "Follow up", "files": []},
        history=[
            {"role": "user", "content": {"path": "/tmp/gradio/secret-name.mov"}},
            {"role": "assistant", "content": "ok"},
        ],
    )
    sent = str(captured[0]["messages"])
    assert "secret-name" not in sent
    assert gradio_app.OMITTED_MEDIA_NOTE in sent


def _small_image_chat_fn(monkeypatch, prompt_tokens: int):
    def fake_post(url, json, timeout):
        return DummyResponse(
            "Red.",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": 2},
        )

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)
    monkeypatch.setattr(
        gradio_app,
        "build_media_items",
        lambda files: [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
        ],
    )
    return gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=16,
        temperature=0.0,
    )


@pytest.mark.parametrize("prompt_tokens", [1, 74, 199])
def test_small_image_answer_is_not_flagged_as_dropped(monkeypatch, prompt_tokens):
    # A 56x56 image is four visual tokens on Qwen2-VL; the whole prompt can sit
    # well under any fixed floor and still be a correctly processed image.
    chat_fn = _small_image_chat_fn(monkeypatch, prompt_tokens)

    output = chat_fn({"text": "What color?", "files": ["dot.png"]}, history=[])

    assert output == "Red."


def test_media_sent_to_text_only_server_is_flagged(monkeypatch):
    chat_fn = _small_image_chat_fn(monkeypatch, prompt_tokens=5000)
    monkeypatch.setattr(
        gradio_app.requests, "get", lambda *a, **k: HealthResponse(TEXT_ONLY_HEALTH)
    )

    output = chat_fn({"text": "What color?", "files": ["dot.png"]}, history=[])

    assert output.startswith("Red.")
    assert "WARNING: 1 attachment(s) (image)" in output
    assert "loaded as text-only" in output


def test_unreadable_health_is_not_reported_as_a_drop(monkeypatch):
    chat_fn = _small_image_chat_fn(monkeypatch, prompt_tokens=10)

    def unreachable(*args, **kwargs):
        raise gradio_app.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(gradio_app.requests, "get", unreachable)

    output = chat_fn({"text": "What color?", "files": ["dot.png"]}, history=[])

    assert output == "Red."


def test_text_only_turn_does_not_query_health(monkeypatch):
    def no_health(*args, **kwargs):
        raise AssertionError("health queried for a turn without media")

    monkeypatch.setattr(gradio_app.requests, "get", no_health)
    monkeypatch.setattr(
        gradio_app.requests, "post", lambda *a, **k: DummyResponse("ok")
    )
    chat_fn = gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=16,
        temperature=0.0,
    )

    assert chat_fn({"text": "hello", "files": []}, history=[]) == "ok"


@pytest.mark.parametrize(
    "health, warns",
    [
        (MLLM_HEALTH, False),
        (TEXT_ONLY_HEALTH, True),
        ({"model_loaded": False, "model_type": None}, False),
    ],
)
def test_text_only_model_warning(health, warns):
    assert (gradio_app.text_only_model_warning(health) is not None) is warns


def test_create_chat_function_clears_media_indexes_for_new_chat(monkeypatch):
    captured_requests = []

    def fake_post(url, json, timeout):
        captured_requests.append(json)
        return DummyResponse("ok")

    monkeypatch.setattr(gradio_app.requests, "post", fake_post)
    monkeypatch.setattr(
        gradio_app,
        "build_media_items",
        lambda files: [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/quicktime;base64,AAAA"},
            }
        ],
    )

    chat_fn = gradio_app.create_chat_function(
        server_url="http://localhost:8000",
        max_tokens=128,
        temperature=0.3,
    )
    chat_fn({"text": "Old chat", "files": ["clip.mov"]}, history=[])
    chat_fn({"text": "New chat", "files": []}, history=[])
    chat_fn(
        {"text": "Follow up", "files": []},
        history=[
            {"role": "user", "content": "New chat"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    new_chat_follow_up = captured_requests[2]["messages"]
    assert gradio_app.OMITTED_MEDIA_NOTE not in str(new_chat_follow_up)


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
