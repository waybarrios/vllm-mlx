# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM assistant-drafter speculative wiring."""

import sys
from types import SimpleNamespace

import pytest


def test_mllm_chat_forwards_configured_assistant_drafter(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    draft_model = SimpleNamespace(accept_lens=[99])

    def fake_generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["accept_lens_at_call"] = list(draft_model.accept_lens)
        draft_model.accept_lens = [1, 2]
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=2)

    fake_cache = SimpleNamespace(make_prompt_cache=lambda *args, **kwargs: ["cache"])
    fake_models = SimpleNamespace(cache=fake_cache)
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(get_chat_template=lambda *args, **kwargs: "rendered prompt"),
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", fake_models)

    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    processor = SimpleNamespace(tokenizer=tokenizer)
    target = SimpleNamespace(language_model=object())

    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._loaded = True
    model.model = target
    model.processor = processor
    model.config = {}
    model._draft_model = draft_model
    model._cache_manager = None

    output = model.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        mllm_draft=True,
    )

    assert output.text == "ok"
    assert output.mtp_drafts == 6
    assert output.mtp_accepted == 3
    assert captured["accept_lens_at_call"] == []
    assert captured["kwargs"]["draft_model"] is draft_model
    assert captured["kwargs"]["draft_kind"] == "mtp"
    assert captured["kwargs"]["draft_block_size"] == 4


def test_mllm_draft_metrics_use_recorded_draft_counts():
    from vllm_mlx.models.mllm import MLXMultimodalLM

    draft_model = SimpleNamespace(
        accept_lens=[1, 0],
        _vllm_mlx_draft_counts=[2, 1],
        config=SimpleNamespace(block_size=4),
    )
    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._draft_model = draft_model

    assert model._draft_metrics_since(0) == {
        "mtp_drafts": 3,
        "mtp_accepted": 1,
    }


def test_mllm_chat_uses_configured_drafter_over_call_kwargs(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    configured_draft = SimpleNamespace(accept_lens=[], _vllm_mlx_draft_counts=[])
    caller_draft = SimpleNamespace()

    def fake_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(get_chat_template=lambda *args, **kwargs: "rendered prompt"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models",
        SimpleNamespace(cache=SimpleNamespace(make_prompt_cache=lambda *a, **k: None)),
    )

    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._loaded = True
    model.model = SimpleNamespace(language_model=object())
    model.processor = SimpleNamespace(tokenizer=tokenizer)
    model.config = {}
    model._draft_model = configured_draft
    model._cache_manager = None

    output = model.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        mllm_draft=True,
        draft_model=caller_draft,
        draft_kind="other",
        draft_block_size=99,
    )

    assert output.text == "ok"
    assert captured["kwargs"]["draft_model"] is configured_draft
    assert captured["kwargs"]["draft_kind"] == "mtp"
    assert captured["kwargs"]["draft_block_size"] == 4


def test_mllm_chat_requires_request_draft_opt_in(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    configured_draft = SimpleNamespace(accept_lens=[], _vllm_mlx_draft_counts=[])

    def fake_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        SimpleNamespace(get_chat_template=lambda *args, **kwargs: "rendered prompt"),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models",
        SimpleNamespace(cache=SimpleNamespace(make_prompt_cache=lambda *a, **k: None)),
    )

    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        draft_block_size=4,
    )
    model._loaded = True
    model.model = SimpleNamespace(language_model=object())
    model.processor = SimpleNamespace(tokenizer=tokenizer)
    model.config = {}
    model._draft_model = configured_draft
    model._cache_manager = None

    output = model.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        draft_model=object(),
        draft_kind="other",
        draft_block_size=99,
    )

    assert output.text == "ok"
    assert "mllm_draft" not in captured["kwargs"]
    assert "draft_model" not in captured["kwargs"]
    assert "draft_kind" not in captured["kwargs"]
    assert "draft_block_size" not in captured["kwargs"]
    assert output.mtp_drafts == 0
    assert output.mtp_accepted == 0


def test_simple_engine_text_route_stays_default_when_mllm_drafter_configured():
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=4,
    )
    engine._loaded = True
    engine._text_model = object()

    assert engine._should_route_text_through_text_model() is True
    assert (
        engine._should_route_text_through_text_model(mllm_draft_requested=True) is False
    )


def test_chat_request_passes_mllm_draft_opt_in():
    from vllm_mlx.server import (
        ChatCompletionRequest,
        Message,
        _prepare_chat_completion_invocation,
    )

    class Engine:
        is_mllm = True
        preserve_native_tool_format = False

    request = ChatCompletionRequest(
        model="gemma4",
        messages=[Message(role="user", content="hello")],
        mllm_draft=True,
    )

    prepared = _prepare_chat_completion_invocation(Engine(), request, 16)

    assert prepared.chat_kwargs["mllm_draft"] is True


@pytest.mark.asyncio
async def test_simple_engine_forwards_mllm_draft_opt_in_to_mllm_path():
    from vllm_mlx.engine.simple import SimpleEngine

    captured = {}

    class FakeMLLM:
        def stream_chat(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            yield SimpleNamespace(
                text="ok",
                finish_reason="stop",
                prompt_tokens=3,
                mtp_drafts=2,
                mtp_accepted=1,
            )

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=4,
    )
    engine._loaded = True
    engine._is_mllm = True
    engine._text_model = object()
    engine._model = FakeMLLM()

    outputs = [
        output
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=8,
            temperature=0.0,
            mllm_draft=True,
        )
    ]

    assert captured["kwargs"]["mllm_draft"] is True
    assert outputs[-1].mtp_drafts == 2
    assert outputs[-1].mtp_accepted == 1
