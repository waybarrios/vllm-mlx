# SPDX-License-Identifier: Apache-2.0
"""Refuse a prompt longer than the model context before any engine work."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as server
from vllm_mlx.engine.base import BaseEngine, GenerationOutput
from vllm_mlx.utils.tokenizer import count_prompt_tokens

CONTEXT = 40


class _WordTokenizer:
    """One token per whitespace-separated word, no special tokens."""

    bos_token = None
    model_max_length = int(1e30)

    def encode(self, text, add_special_tokens=True):
        return text.split()


class _BosTokenizer(_WordTokenizer):
    """Like a tokenizer whose encode prepends BOS unless told not to."""

    bos_token = "<s>"

    def encode(self, text, add_special_tokens=True):
        words = text.split()
        return ["<s>"] + words if add_special_tokens else words


class _FakeEngine:
    """Engine double that counts the words of a rendered chat prompt."""

    model_name = "test-model"
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self, context_limit=CONTEXT, tokenizer=None):
        self.context_limit = context_limit
        self.tokenizer = tokenizer or _WordTokenizer()
        self.template_calls = []
        self.prompt_tokens_unknown = False
        self.generations = 0

    def count_chat_prompt_tokens(
        self, messages, tools=None, chat_template_kwargs=None, enable_thinking=None
    ):
        self.template_calls.append(
            {
                "tools": tools,
                "chat_template_kwargs": chat_template_kwargs,
                "enable_thinking": enable_thinking,
            }
        )
        if self.prompt_tokens_unknown:
            return None
        extra = (chat_template_kwargs or {}).get("padding", "")
        prompt = " ".join([m["content"] for m in messages] + [extra]).strip()
        return count_prompt_tokens(self.tokenizer, prompt)

    async def generate(self, prompt, **kwargs):
        self.generations += 1
        return _output()

    async def chat(self, messages, **kwargs):
        self.generations += 1
        return _output()

    async def stream_generate(self, prompt, **kwargs):
        self.generations += 1
        yield _output()

    async def stream_chat(self, messages, **kwargs):
        self.generations += 1
        yield _output()


def _output():
    return GenerationOutput(
        text="ok",
        new_text="ok",
        finished=True,
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=1,
    )


def _words(count):
    return " ".join(["w"] * count)


@pytest.fixture
def make_client(monkeypatch):
    clients = []

    def make(engine):
        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_model_manager", None)
        monkeypatch.setattr(server, "_model_name", engine.model_name)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_default_thinking_token_budget", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)
        client = TestClient(server.app, raise_server_exceptions=False)
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


def _chat(prompt_words, stream=False, **extra):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": _words(prompt_words)}],
        "max_tokens": 1,
        "stream": stream,
        **extra,
    }


def _completion(prompt, stream=False, **extra):
    return {
        "model": "test-model",
        "prompt": prompt,
        "max_tokens": 1,
        "stream": stream,
        **extra,
    }


def _assert_refusal(response, prompt_tokens, param):
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    detail = response.json()["detail"]
    assert detail["code"] == "context_length_exceeded"
    assert detail["type"] == "invalid_request_error"
    assert detail["param"] == param
    assert detail["prompt_tokens"] == prompt_tokens
    assert detail["max_model_len"] == CONTEXT
    assert str(CONTEXT) in detail["message"]
    assert "data:" not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_chat_prompt_over_context_is_refused(make_client, stream):
    engine = _FakeEngine()
    client = make_client(engine)

    response = client.post("/v1/chat/completions", json=_chat(CONTEXT + 1, stream))

    _assert_refusal(response, CONTEXT + 1, "messages")
    assert engine.generations == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("words", [CONTEXT - 1, CONTEXT])
def test_chat_prompt_within_context_is_answered(make_client, stream, words):
    engine = _FakeEngine()
    client = make_client(engine)

    response = client.post("/v1/chat/completions", json=_chat(words, stream))

    assert response.status_code == 200, response.text
    assert engine.generations == 1


def test_chat_prompt_is_counted_as_the_engine_renders_it(make_client):
    """The count covers the template output, not the raw message text."""
    engine = _FakeEngine()
    client = make_client(engine)
    tool = {
        "type": "function",
        "function": {"name": "f", "parameters": {"type": "object"}},
    }
    body = _chat(
        CONTEXT - 5,
        tools=[tool],
        chat_template_kwargs={"padding": _words(10)},
        enable_thinking=False,
    )

    response = client.post("/v1/chat/completions", json=body)

    _assert_refusal(response, CONTEXT + 5, "messages")
    assert engine.generations == 0
    (call,) = engine.template_calls
    assert call["enable_thinking"] is False
    assert call["chat_template_kwargs"] == {"padding": _words(10)}
    assert [t["function"]["name"] for t in call["tools"]] == ["f"]


def test_unknown_context_limit_is_accepted(make_client):
    engine = _FakeEngine(context_limit=None)
    client = make_client(engine)

    chat = client.post("/v1/chat/completions", json=_chat(CONTEXT * 10))
    completion = client.post("/v1/completions", json=_completion(_words(CONTEXT * 10)))

    assert chat.status_code == 200, chat.text
    assert completion.status_code == 200, completion.text
    assert engine.generations == 2
    assert engine.template_calls == []


def test_chat_prompt_the_engine_cannot_count_is_left_to_the_engine(make_client):
    engine = _FakeEngine()
    engine.prompt_tokens_unknown = True
    client = make_client(engine)

    response = client.post("/v1/chat/completions", json=_chat(CONTEXT * 10))

    assert response.status_code == 200, response.text
    assert engine.generations == 1
    assert len(engine.template_calls) == 1


def test_chat_template_error_is_the_engines_own_error(make_client):
    """A template that fails to render fails as it does in engine.chat."""
    engine = _FakeEngine()

    def broken_count(*args, **kwargs):
        raise ValueError("template failed")

    engine.count_chat_prompt_tokens = broken_count
    client = make_client(engine)

    response = client.post("/v1/chat/completions", json=_chat(1))

    assert response.status_code == 500
    assert engine.generations == 0


def test_max_tokens_over_server_limit_is_reported_first(make_client):
    engine = _FakeEngine()
    client = make_client(engine)
    body = _chat(CONTEXT + 1, max_tokens=server._max_request_tokens + 1)

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"max_tokens exceeds server limit ({server._max_request_tokens})"
    )


def test_prompt_plus_max_tokens_over_context_is_not_refused(make_client):
    engine = _FakeEngine()
    client = make_client(engine)

    response = client.post(
        "/v1/chat/completions", json=_chat(CONTEXT, max_tokens=CONTEXT)
    )

    assert response.status_code == 200, response.text
    assert engine.generations == 1


@pytest.mark.parametrize("stream", [False, True])
def test_completion_prompt_over_context_is_refused(make_client, stream):
    engine = _FakeEngine()
    client = make_client(engine)

    response = client.post(
        "/v1/completions", json=_completion(_words(CONTEXT + 1), stream)
    )

    _assert_refusal(response, CONTEXT + 1, "prompt")
    assert engine.generations == 0


@pytest.mark.parametrize("stream", [False, True])
def test_completion_prompt_at_context_is_answered(make_client, stream):
    engine = _FakeEngine()
    client = make_client(engine)

    response = client.post("/v1/completions", json=_completion(_words(CONTEXT), stream))

    assert response.status_code == 200, response.text
    assert engine.generations == 1


def test_completion_list_refuses_when_any_prompt_is_over_context(make_client):
    engine = _FakeEngine()
    client = make_client(engine)
    prompts = ["short", _words(CONTEXT + 1)]

    response = client.post("/v1/completions", json=_completion(prompts))

    _assert_refusal(response, CONTEXT + 1, "prompt")
    assert engine.generations == 0


def test_completion_list_stream_checks_only_the_prompt_it_runs(make_client):
    engine = _FakeEngine()
    client = make_client(engine)
    prompts = ["short", _words(CONTEXT + 1)]

    response = client.post("/v1/completions", json=_completion(prompts, stream=True))

    assert response.status_code == 200, response.text
    assert engine.generations == 1


def test_completion_list_stream_refuses_an_over_context_first_prompt(make_client):
    engine = _FakeEngine()
    client = make_client(engine)
    prompts = [_words(CONTEXT + 1), "short"]

    response = client.post("/v1/completions", json=_completion(prompts, stream=True))

    _assert_refusal(response, CONTEXT + 1, "prompt")
    assert engine.generations == 0


def test_refusal_releases_the_engine_for_the_next_request(make_client):
    engine = _FakeEngine()
    client = make_client(engine)

    refused = client.post("/v1/chat/completions", json=_chat(CONTEXT + 1))
    answered = client.post("/v1/chat/completions", json=_chat(1))

    _assert_refusal(refused, CONTEXT + 1, "messages")
    assert answered.status_code == 200, answered.text
    assert engine.generations == 1


class _ModelWithArgs:
    def __init__(self, max_position_embeddings):
        self.args = SimpleNamespace(max_position_embeddings=max_position_embeddings)


class _ModelWithConfig:
    def __init__(self, max_position_embeddings):
        self.config = SimpleNamespace(max_position_embeddings=max_position_embeddings)


class _BareModel:
    pass


class _TemplateTokenizer(_BosTokenizer):
    """Renders a chat as its words, then the ``padding`` template kwarg."""

    def __init__(self, model_max_length=int(1e30), prefix=""):
        self.model_max_length = model_max_length
        self.prefix = prefix
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        words = [m["content"] for m in messages] + [kwargs.get("padding", "")]
        return self.prefix + " ".join(words).strip()


def _simple_engine(model, tokenizer, mllm=False, loaded=True):
    from vllm_mlx.engine.simple import SimpleEngine

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=mllm):
        engine = SimpleEngine("test-model")
    engine._loaded = loaded
    engine._is_mllm = mllm
    if loaded:
        engine._model = SimpleNamespace(model=model, tokenizer=tokenizer)
    return engine


def _batched_engine(model, tokenizer, mllm=False, loaded=True):
    from vllm_mlx.engine.batched import BatchedEngine

    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=mllm):
        engine = BatchedEngine("test-model")
    engine._loaded = loaded
    engine._is_mllm = mllm
    if loaded:
        engine._model = model
        engine._tokenizer = tokenizer
    return engine


@pytest.fixture(params=[_simple_engine, _batched_engine], ids=["simple", "batched"])
def build_engine(request):
    return request.param


def test_engine_context_limit_reads_model_args(build_engine):
    engine = build_engine(_ModelWithArgs(40960), _TemplateTokenizer())
    assert engine.context_limit == 40960


def test_engine_context_limit_reads_model_config(build_engine):
    engine = build_engine(_ModelWithConfig(8192), _TemplateTokenizer())
    assert engine.context_limit == 8192


def test_engine_context_limit_ignores_a_finite_tokenizer_limit(build_engine):
    """Config declares 131072 and the tokenizer 16384: generation is bounded
    by the config only (the DeepSeek-R1-Distill-Qwen-1.5B shape)."""
    engine = build_engine(_ModelWithArgs(131072), _TemplateTokenizer(16384))
    assert engine.context_limit == 131072


def test_engine_context_limit_is_none_without_a_declared_context(build_engine):
    """A finite tokenizer limit alone does not become a generation limit."""
    engine = build_engine(_BareModel(), _TemplateTokenizer(4096))
    assert engine.context_limit is None


def test_multimodal_engine_reports_no_context_limit(build_engine):
    engine = build_engine(_ModelWithArgs(40960), _TemplateTokenizer(), mllm=True)
    assert engine.context_limit is None


def test_unloaded_engine_reports_no_context_limit(build_engine):
    engine = build_engine(None, None, loaded=False)
    assert engine.context_limit is None


def test_base_engine_reports_no_context_limit_by_default():
    assert BaseEngine.context_limit.fget(object()) is None


def test_base_engine_cannot_count_a_chat_prompt_by_default():
    assert BaseEngine.count_chat_prompt_tokens(object(), []) is None


def test_engine_counts_the_prompt_it_renders(build_engine):
    """Words of the messages plus the template kwarg, with the request's own
    tools, template kwargs and thinking flag reaching the template."""
    tokenizer = _TemplateTokenizer()
    engine = build_engine(_ModelWithArgs(40960), tokenizer)
    tool = {"type": "function", "function": {"name": "f"}}

    count = engine.count_chat_prompt_tokens(
        [{"role": "user", "content": _words(5)}],
        tools=[tool],
        chat_template_kwargs={"padding": _words(3)},
        enable_thinking=False,
    )

    assert count == 8 + 1  # the tokenizer adds a BOS the prompt does not carry
    (call,) = tokenizer.calls
    assert call["enable_thinking"] is False
    assert call["padding"] == _words(3)
    assert call["tools"] == [tool]
    assert call["add_generation_prompt"] is True


def test_engine_counts_a_prompt_that_starts_with_bos_once(build_engine):
    engine = build_engine(_ModelWithArgs(40960), _TemplateTokenizer(prefix="<s> "))

    count = engine.count_chat_prompt_tokens([{"role": "user", "content": _words(5)}])

    assert count == 1 + 5


def test_multimodal_engine_cannot_count_a_chat_prompt(build_engine):
    engine = build_engine(_ModelWithArgs(40960), _TemplateTokenizer(), mllm=True)
    assert engine.count_chat_prompt_tokens([{"role": "user", "content": "hi"}]) is None


def test_unloaded_engine_cannot_count_a_chat_prompt(build_engine):
    engine = build_engine(None, None, loaded=False)
    assert engine.count_chat_prompt_tokens([{"role": "user", "content": "hi"}]) is None


def test_engine_does_not_swallow_a_template_error(build_engine):
    class _Broken(_TemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("no template")

    engine = build_engine(_ModelWithArgs(40960), _Broken())
    with pytest.raises(ValueError, match="no template"):
        engine.count_chat_prompt_tokens([{"role": "user", "content": "hi"}])


def test_refusal_body_is_json_with_the_documented_keys():
    with pytest.raises(server.HTTPException) as caught:
        server._reject_prompt_over_context(CONTEXT, CONTEXT + 3, "messages")

    assert caught.value.status_code == 400
    assert json.loads(json.dumps(caught.value.detail)) == {
        "message": caught.value.detail["message"],
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
        "prompt_tokens": CONTEXT + 3,
        "max_model_len": CONTEXT,
    }


def test_raw_completion_prompt_counts_the_bos_the_engine_adds(make_client):
    engine = _FakeEngine(tokenizer=_BosTokenizer())
    client = make_client(engine)

    response = client.post("/v1/completions", json=_completion(_words(CONTEXT)))

    _assert_refusal(response, CONTEXT + 1, "prompt")
    assert engine.generations == 0
