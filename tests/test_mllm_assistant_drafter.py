# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM assistant-drafter speculative wiring."""

import sys
from types import SimpleNamespace


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
    )

    assert output.text == "ok"
    assert output.mtp_drafts == 6
    assert output.mtp_accepted == 3
    assert captured["accept_lens_at_call"] == []
    assert captured["kwargs"]["draft_model"] is draft_model
    assert captured["kwargs"]["draft_kind"] == "mtp"
    assert captured["kwargs"]["draft_block_size"] == 4


def test_simple_engine_disables_text_routing_when_mllm_drafter_configured():
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

    assert engine._should_route_text_through_text_model() is False
