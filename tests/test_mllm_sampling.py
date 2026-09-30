# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters must reach mlx-vlm on every public MLLM route."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.models.mllm import MLXMultimodalLM


@pytest.fixture
def model(monkeypatch):
    # Keep the wrapper real and replace only model inference and media I/O.
    # mlx-vlm accepts temperature, not temp, including at our 0.6.5 floor.
    def generate(*args, temperature=0.0, top_p=1.0, top_k=0, **kwargs):
        return SimpleNamespace(
            text=f"{temperature}:{top_p}:{top_k}",
            prompt_tokens=3,
            generation_tokens=1,
        )

    vlm = ModuleType("mlx_vlm")
    vlm.generate = generate
    vlm.stream_generate = lambda *args, **kwargs: iter([generate(*args, **kwargs)])
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


@pytest.mark.parametrize(
    "route", ["generate", "stream_generate", "chat", "stream_chat"]
)
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_sampling_parameters_reach_mlx_vlm(model, route, temperature):
    kwargs = {"temperature": temperature, "top_p": 0.37, "top_k": 17}
    if "chat" in route:
        kwargs["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "test-image"}},
                ],
            }
        ]
    else:
        kwargs.update(prompt="Describe this image", images=["test-image"])

    result = getattr(model, route)(**kwargs)
    text = (
        "".join(chunk.text for chunk in result)
        if route.startswith("stream")
        else result.text
    )

    assert text == f"{temperature}:0.37:17"
