# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM assistant-drafter speculative wiring."""

import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_eagle3_loader(monkeypatch, *, resolved_kind="eagle3", drafter=None):
    drafter = drafter or _upstream_eagle3_drafter()
    calls = []
    speculative = ModuleType("mlx_vlm.speculative")

    def load_drafter(path, kind):
        calls.append((path, kind))
        return drafter, resolved_kind

    speculative.load_drafter = load_drafter
    drafters = ModuleType("mlx_vlm.speculative.drafters")
    drafters.validate_drafter_compatibility = lambda *args: calls.append(args)
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.__path__ = []
    mlx_vlm.speculative = speculative
    speculative.drafters = drafters
    utils = ModuleType("mlx_vlm.utils")
    utils.load = lambda path: pytest.fail("Eagle3 must not load a drafter processor")
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", drafters)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return calls


def _upstream_eagle3_drafter(
    *,
    capture_layer_ids=(0, 1, 2),
    target_layer_ids=(0, 1, 2),
    target_hidden_size=8,
    target_vocab_size=10,
    internal_hidden_size=4,
):
    """Match mlx-vlm 0.6.5 Eagle3's config and model metadata layout."""
    return SimpleNamespace(
        config=SimpleNamespace(
            capture_layer_ids=capture_layer_ids,
            target_layer_ids=target_layer_ids,
            target_hidden_size=target_hidden_size,
            transformer_layer_config=SimpleNamespace(
                hidden_size=internal_hidden_size,
                vocab_size=target_vocab_size,
            ),
        ),
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
    )


def _eagle3_target(*, rollback=True, layer_count=4, hidden_size=8, vocab_size=10):
    language_model = SimpleNamespace(
        layers=[object()] * layer_count,
        config=SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size),
    )
    if rollback:
        language_model.rollback_speculative_cache = lambda *args: None
    return SimpleNamespace(language_model=language_model)


def test_eagle3_loader_uses_model_only_drafter_and_retains_resolved_kind(monkeypatch):
    """A separate processor load would wrongly bind Eagle3 to a second tokenizer."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    calls = _install_eagle3_loader(monkeypatch)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target()

    assert model._load_draft_model() is not None
    assert model.draft_kind == "eagle3"
    assert calls[0] == ("eagle3-drafter", "eagle3")
    assert calls[1][0] is model.model
    assert calls[1][2] == "eagle3"


def test_eagle3_preflight_reads_capture_layers_from_upstream_drafter(monkeypatch):
    """Gemma targets do not retain capture layers after their forward call."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    drafter = _upstream_eagle3_drafter(capture_layer_ids=(0, 1, 3))
    _install_eagle3_loader(monkeypatch, drafter=drafter)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target(layer_count=4)

    assert model._load_draft_model() is not None


def test_eagle3_preflight_falls_back_to_drafter_target_layers(monkeypatch):
    """Drafters without explicit capture layers use their target-layer fallback."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    drafter = _upstream_eagle3_drafter(
        capture_layer_ids=None, target_layer_ids=(0, 1, 3)
    )
    _install_eagle3_loader(monkeypatch, drafter=drafter)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target(layer_count=4)

    assert model._load_draft_model() is not None


def test_eagle3_explicit_kind_mismatch_fails_at_startup(monkeypatch):
    """Serving a resolved algorithm other than the requested algorithm is unsafe."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    _install_eagle3_loader(monkeypatch, resolved_kind="other")
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target()

    with pytest.raises(
        ValueError, match="requested 'eagle3'.*resolved 'other'.*eagle3-drafter"
    ):
        model._load_draft_model()


@pytest.mark.parametrize("resolved_kind", ["mtp", "dflash"])
def test_eagle3_kind_mismatch_keeps_requested_kind_for_retry(
    monkeypatch, resolved_kind
):
    """A failed resolution must retry the Eagle3 loader, never another kind."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    calls = _install_eagle3_loader(monkeypatch, resolved_kind=resolved_kind)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target()

    for _ in range(2):
        with pytest.raises(
            ValueError, match=f"requested 'eagle3'.*resolved '{resolved_kind}'"
        ):
            model._load_draft_model()
        assert model.draft_kind == "eagle3"

    assert calls == [
        ("eagle3-drafter", "eagle3"),
        ("eagle3-drafter", "eagle3"),
    ]


def test_eagle3_preflight_rejects_missing_rollback(monkeypatch):
    """Eagle3 must fail before generation without target cache rollback."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    _install_eagle3_loader(monkeypatch)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target(rollback=False)

    with pytest.raises(ValueError, match="rollback_speculative_cache"):
        model._load_draft_model()


@pytest.mark.parametrize(
    ("drafter", "property"),
    [
        (_upstream_eagle3_drafter(capture_layer_ids=()), "capture_layer_ids"),
        (_upstream_eagle3_drafter(capture_layer_ids=(0, 0)), "capture_layer_ids"),
        (_upstream_eagle3_drafter(capture_layer_ids=(-1,)), "capture_layer_ids"),
        (_upstream_eagle3_drafter(capture_layer_ids=(4,)), "capture_layer_ids"),
        (_upstream_eagle3_drafter(capture_layer_ids=(0, 1)), "capture_layer_ids"),
        (
            _upstream_eagle3_drafter(capture_layer_ids=(0, 1, 2, 3)),
            "capture_layer_ids",
        ),
    ],
)
def test_eagle3_preflight_rejects_invalid_target_contract(
    monkeypatch, drafter, property
):
    """Eagle3 must fail before generation when its target contract is incomplete."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    _install_eagle3_loader(monkeypatch, drafter=drafter)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = _eagle3_target()

    with pytest.raises(ValueError, match=property):
        model._load_draft_model()


@pytest.mark.parametrize(
    ("target", "drafter", "property"),
    [
        (
            _eagle3_target(hidden_size=8),
            _upstream_eagle3_drafter(target_hidden_size=9),
            "hidden_size",
        ),
        (
            _eagle3_target(vocab_size=10),
            _upstream_eagle3_drafter(target_vocab_size=11),
            "vocab_size",
        ),
    ],
)
def test_eagle3_preflight_rejects_incompatible_drafter_metadata(
    monkeypatch, target, drafter, property
):
    """Comparable target and drafter metadata must agree before Eagle3 starts."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    _install_eagle3_loader(monkeypatch, drafter=drafter)
    model = MLXMultimodalLM("target", draft_model="eagle3-drafter", draft_kind="eagle3")
    model.model = target

    with pytest.raises(ValueError, match=property):
        model._load_draft_model()


def test_eagle3_generation_kwargs_forward_resolved_kind():
    """Generation must use mlx-vlm's resolved Eagle3 kind and configured block size."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    model = MLXMultimodalLM(
        "target", draft_model="eagle3-drafter", draft_kind="eagle3", draft_block_size=4
    )
    model._draft_model = SimpleNamespace(accept_lens=[])

    assert model._draft_generation_kwargs({"mllm_draft": True}) == {
        "draft_model": model._draft_model,
        "draft_kind": "eagle3",
        "draft_block_size": 4,
    }


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


def test_simple_engine_defaults_configured_drafter_on_but_allows_opt_out():
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=4,
        default_mllm_draft=True,
    )

    assert engine._default_mllm_draft is True
    assert (
        engine._should_route_text_through_text_model(mllm_draft_requested=True) is False
    )
    assert (
        engine._should_route_text_through_text_model(mllm_draft_requested=False) is True
    )


def test_mllm_drafter_defaults_on_and_request_can_opt_out():
    from vllm_mlx.models.mllm import MLXMultimodalLM

    model = MLXMultimodalLM(
        "target",
        draft_model="assistant",
        draft_kind="mtp",
        default_draft_enabled=True,
    )
    model._draft_model = SimpleNamespace(accept_lens=[])

    assert model._draft_generation_kwargs() == {
        "draft_model": model._draft_model,
        "draft_kind": "mtp",
    }
    assert model._draft_generation_kwargs({"mllm_draft": False}) == {}


def test_simple_engine_reports_configured_mllm_drafter_status():
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=4,
        default_mllm_draft=True,
    )

    assert engine.get_stats()["mtp"] == {
        "enabled": True,
        "implementation": "mlx_vlm_assistant",
        "draft_model": "assistant",
        "draft_kind": "mtp",
        "draft_block_size": 4,
        "default_enabled": True,
        "continuous_batching_supported": True,
    }


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


def test_completion_request_preserves_mllm_draft_opt_out():
    from vllm_mlx.api.models import CompletionRequest

    request = CompletionRequest(
        model="gemma4",
        prompt="hello",
        mllm_draft=False,
    )

    assert request.mllm_draft is False


@pytest.mark.anyio
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


@pytest.mark.anyio
async def test_simple_engine_forwards_mllm_draft_opt_out_to_media_path():
    from vllm_mlx.engine.simple import SimpleEngine

    captured = {}

    class FakeMLLM:
        def stream_chat(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            yield SimpleNamespace(
                text="ok",
                finish_reason="stop",
                prompt_tokens=3,
                mtp_drafts=0,
                mtp_accepted=0,
            )

    engine = SimpleEngine(
        "gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        default_mllm_draft=True,
    )
    engine._loaded = True
    engine._text_model = object()
    engine._model = FakeMLLM()

    outputs = [
        output
        async for output in engine.stream_chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ],
            max_tokens=8,
            temperature=0.0,
            mllm_draft=False,
        )
    ]

    assert captured["kwargs"]["mllm_draft"] is False
    assert outputs[-1].text == "ok"
