# SPDX-License-Identifier: Apache-2.0
"""MLLM streams should preserve backend token counts and finish reasons."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.models.mllm import MLXMultimodalLM

GENERATED = ["Red", " square", "."]


def _stream_generate(*args, max_tokens=256, **kwargs):
    hit_eos = len(GENERATED) < max_tokens
    tokens = GENERATED[:max_tokens]
    for n, text in enumerate(tokens, 1):
        yield SimpleNamespace(text=text, prompt_tokens=3, generation_tokens=n)
    total = len(tokens) + (1 if hit_eos else 0)
    yield SimpleNamespace(
        text="",
        prompt_tokens=3,
        generation_tokens=total,
        finish_reason="stop" if hit_eos else "length",
    )


@pytest.fixture
def model(monkeypatch):
    vlm = ModuleType("mlx_vlm")
    vlm.stream_generate = _stream_generate
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *args, **kwargs: "prompt"
    prompt_utils.get_chat_template = lambda *args, **kwargs: "prompt"
    cache = ModuleType("mlx_vlm.models.cache")
    cache.make_prompt_cache = lambda *args, **kwargs: []
    models = ModuleType("mlx_vlm.models")
    models.cache = cache
    for module in (vlm, prompt_utils, models, cache):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    wrapper = MLXMultimodalLM("test-model", enable_cache=False)
    wrapper._loaded = True
    wrapper.model = SimpleNamespace(language_model=object())
    wrapper.processor = SimpleNamespace(encode=lambda text: [1, 2, 3])
    wrapper.config = {}
    monkeypatch.setattr(wrapper, "_prepare_images", lambda images: ["prepared-image"])
    return wrapper


IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "test-image"}},
        ],
    }
]


@pytest.mark.parametrize(
    ("max_tokens", "stop", "expected_tokens", "expected_reason"),
    [
        (256, None, len(GENERATED) + 1, "stop"),
        (len(GENERATED) + 1, None, len(GENERATED) + 1, "stop"),
        (2, None, 2, "length"),
        (2, ["green"], 2, "length"),
    ],
    ids=["eos", "eos-at-cap", "length", "unmatched-stop-at-cap"],
)
def test_stream_chat_reports_generated_token_count(
    model, max_tokens, stop, expected_tokens, expected_reason
):
    chunks = list(
        model.stream_chat(messages=IMAGE_MESSAGES, max_tokens=max_tokens, stop=stop)
    )

    assert "".join(chunk.text for chunk in chunks) == "".join(GENERATED[:max_tokens])
    assert chunks[-1].completion_tokens == expected_tokens
    assert chunks[-1].prompt_tokens == 3
    assert chunks[-1].finish_reason == expected_reason


def test_stream_chat_prioritizes_custom_stop_over_backend_length(model, monkeypatch):
    def stream_generate(*args, **kwargs):
        yield SimpleNamespace(text="Red", prompt_tokens=3, generation_tokens=1)
        yield SimpleNamespace(
            text=" square", prompt_tokens=3, generation_tokens=2, finish_reason="length"
        )

    monkeypatch.setattr("mlx_vlm.stream_generate", stream_generate)

    chunks = list(
        model.stream_chat(messages=IMAGE_MESSAGES, max_tokens=2, stop=["Red square"])
    )

    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].completion_tokens == 2
