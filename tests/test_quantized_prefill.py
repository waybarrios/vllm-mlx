# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.models import llama, qwen3
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache
from mlx_lm.models.llama import Model, ModelArgs

from vllm_mlx.quantized_prefill import layerwise_prefill_model


@pytest.fixture(autouse=True)
def local_generation_stream(monkeypatch):
    from importlib import import_module

    module = import_module("mlx_lm.generate")
    monkeypatch.setattr(module, "generation_stream", mx.new_stream(mx.default_device()))


@pytest.mark.parametrize("model_module", [llama, qwen3])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 17, 19, 33])
@pytest.mark.parametrize("bits", [4, 8])
def test_exact_deferred_quantization_parity(model_module, dtype, length, bits):
    mx.random.seed(731)
    args = model_module.ModelArgs(
        model_type="llama" if model_module is llama else "qwen3",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=384,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000,
        head_dim=64,
        tie_word_embeddings=False,
    )
    model = model_module.Model(args)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    tokens = mx.array([(i * 7 + 3) % 128 for i in range(length)])

    def run(model, step_size, **options):
        cache = make_prompt_cache(model)
        histories = []

        def record_history(history, logits):
            histories.append(history.tolist())
            return logits

        outputs = list(
            generate_step(
                tokens,
                model,
                max_tokens=4,
                prompt_cache=cache,
                prefill_step_size=step_size,
                logits_processors=[record_history],
                **options,
            )
        )
        return outputs, cache, histories

    reference, reference_cache, reference_histories = run(
        model, 8, kv_bits=bits, quantized_kv_start=length
    )
    request_model = layerwise_prefill_model(model, tokens, bits, prefill_step_size=8)
    assert request_model.layers[0] is model.layers[0]
    actual, cache, histories = run(request_model, length)
    assert histories == reference_histories
    for (token, logprobs), (expected, expected_logprobs) in zip(
        actual, reference, strict=True
    ):
        assert token == expected
        assert logprobs.dtype == expected_logprobs.dtype
        assert mx.array_equal(logprobs, expected_logprobs).item()
    for a, b in zip(cache, reference_cache, strict=True):
        assert isinstance(a, QuantizedKVCache)
        assert a.offset == b.offset == length + 4
        for (key, x), (other_key, y) in zip(
            tree_flatten(a.state), tree_flatten(b.state), strict=True
        ):
            assert key == other_key and x.dtype == y.dtype
            assert mx.array_equal(x, y).item()


@pytest.fixture
def tiny_wrapper():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from vllm_mlx.models.llm import MLXLanguageModel

    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=128,
    )
    mx.random.seed(3)
    model = Model(args)
    mx.eval(model.parameters())
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 128)}}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
    )
    wrapper = MLXLanguageModel("tiny-model")
    wrapper.model, wrapper.tokenizer, wrapper._loaded = model, tokenizer, True
    return wrapper


def test_request_decoder_is_isolated_on_close_and_error(tiny_wrapper):
    model = tiny_wrapper.model
    decoder = model.model
    prompt = " ".join(str(i % 127 + 1) for i in range(160))
    baseline = list(tiny_wrapper.stream_generate(prompt, max_tokens=3, temperature=0))
    stream = tiny_wrapper.stream_generate(
        prompt, max_tokens=20, temperature=0, kv_cache_bits=4
    )
    next(stream)
    assert model.model is decoder
    # A normal request can use the same weights while the quantized stream is open.
    ordinary = list(tiny_wrapper.stream_generate(prompt, max_tokens=3, temperature=0))
    assert [c.token for c in ordinary] == [c.token for c in baseline]
    stream.close()

    def fail(tokens, logits):
        raise RuntimeError("processor failed")

    with pytest.raises(RuntimeError, match="processor failed"):
        list(
            tiny_wrapper.stream_generate(
                prompt, kv_cache_bits=4, logits_processors=[fail]
            )
        )
    assert model.model is decoder
    streamed = list(
        tiny_wrapper.stream_generate(
            prompt, max_tokens=3, temperature=0, kv_cache_bits=4
        )
    )
    generated = tiny_wrapper.generate(
        prompt, max_tokens=3, temperature=0, kv_cache_bits=4
    )
    assert generated.text == "".join(c.text for c in streamed)
    with pytest.raises(ValueError, match="fresh"):
        list(tiny_wrapper.stream_generate(prompt, prompt_cache=[], kv_cache_bits=4))


def test_single_token_prompt(tiny_wrapper):
    output = list(
        tiny_wrapper.stream_generate([1], max_tokens=2, temperature=0, kv_cache_bits=4)
    )
    assert len(output) == 2
    assert output[0].prompt_tokens == 1


def test_engine_prefill_step_size_reaches_each_layer(tiny_wrapper, monkeypatch):
    from vllm_mlx.engine.simple import SimpleEngine
    from vllm_mlx.models.llm import MLXLanguageModel

    monkeypatch.setattr("vllm_mlx.engine.simple.is_mllm_model", lambda _: False)
    monkeypatch.setattr(
        MLXLanguageModel, "load", lambda m: setattr(m, "model", tiny_wrapper.model)
    )
    engine = SimpleEngine("tiny-model", prefill_step_size=8)
    engine.prepare_for_start()
    model, _, options = engine._model._generation_inputs(list(range(17)), 8)
    assert model.model.step_size == 8
    assert options["prefill_step_size"] == 17


@pytest.fixture
def tiny_engine(tiny_wrapper, monkeypatch):
    from vllm_mlx.engine.simple import SimpleEngine

    monkeypatch.setattr("vllm_mlx.engine.simple.is_mllm_model", lambda _: False)
    engine = SimpleEngine("tiny-model", prefix_trie_cache=True)
    engine._model, engine._loaded = tiny_wrapper, True
    engine._supports_system_kv_cache = True
    return engine


@pytest.mark.anyio
async def test_chat_cancellation_releases_caches_and_restores_model(
    tiny_engine, monkeypatch
):
    # Earlier generators' destructor errors must not pin this request's frames.
    gc.collect()
    refs = []
    factory = KVCache.to_quantized

    def capture(*args, **kwargs):
        cache = factory(*args, **kwargs)
        refs.append(weakref.ref(cache))
        return cache

    monkeypatch.setattr(KVCache, "to_quantized", capture)
    model = tiny_engine._model.model
    decoder = model.model
    messages = [
        {"role": "system", "content": "1 2 3"},
        {"role": "user", "content": " ".join(str(i % 127 + 1) for i in range(160))},
    ]
    stream = tiny_engine.stream_chat(
        messages, temperature=0, max_tokens=20, kv_cache_bits=4
    )
    try:
        first = await anext(stream)
        assert first.prompt_tokens == 163
        assert refs and all(isinstance(r(), QuantizedKVCache) for r in refs)
        await stream.aclose()
        gc.collect()
        assert model.model is decoder
        assert all(r() is None for r in refs)
        assert not tiny_engine._system_kv_cache
        assert tiny_engine._prefix_trie_cache is None
        output = await tiny_engine.chat(
            messages, temperature=0, max_tokens=3, kv_cache_bits=4
        )
        assert output.completion_tokens == 3
        before = len(refs)
        await tiny_engine.generate("1 2 3", temperature=0, max_tokens=3)
        assert len(refs) == before
    finally:
        await stream.aclose()
        await tiny_engine.stop()


@pytest.mark.parametrize(
    "setting", ["_is_mllm", "_mtp", "_max_kv_size", "_draft_model"]
)
def test_incompatible_engine_features_are_rejected(tiny_engine, setting):
    setattr(tiny_engine, setting, True)
    with pytest.raises(ValueError, match="requires text-only"):
        tiny_engine.validate_quantized_prefill()


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError, match="dense Llama and Qwen3"):
        layerwise_prefill_model(object(), [1], bits=4)


def test_http_options_and_media_validation(tiny_engine):
    from fastapi import HTTPException
    from pydantic import ValidationError

    from vllm_mlx.server import (
        ChatCompletionRequest,
        CompletionRequest,
        _prepare_chat_completion_invocation,
    )

    messages = [{"role": "user", "content": "1 2 3"}]
    for cls, payload in [
        (CompletionRequest, {"prompt": "1 2 3"}),
        (ChatCompletionRequest, {"messages": messages}),
    ]:
        for invalid in (0, 1, 3, 5, 16, True, "4"):
            with pytest.raises(ValidationError):
                cls(model="tiny-model", kv_cache_bits=invalid, **payload)
    request = ChatCompletionRequest(
        model="tiny-model", kv_cache_bits=4, messages=messages
    )
    prepared = _prepare_chat_completion_invocation(tiny_engine, request, 4)
    assert prepared.chat_kwargs["kv_cache_bits"] == 4
    media_request = ChatCompletionRequest(
        model="tiny-model",
        kv_cache_bits=4,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }
                ],
            }
        ],
    )
    with pytest.raises(HTTPException, match="text-only") as exc:
        _prepare_chat_completion_invocation(tiny_engine, media_request, 4)
    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_completion_forwards_option_in_both_modes(tiny_engine, monkeypatch):
    from types import SimpleNamespace

    import vllm_mlx.server as server

    captured = []

    async def generate(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            text="ok",
            new_text="ok",
            finished=True,
            finish_reason="length",
            completion_tokens=1,
            prompt_tokens=3,
        )

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    monkeypatch.setattr(tiny_engine, "generate", generate)
    monkeypatch.setattr(tiny_engine, "stream_generate", stream_generate)
    monkeypatch.setattr(server, "_model_name", "tiny-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)
    monkeypatch.setattr(server, "get_engine", lambda: tiny_engine)
    request = server.CompletionRequest(
        model="tiny-model", prompt="1 2 3", kv_cache_bits=4
    )
    await server.create_completion(request, raw_request=None)
    chunks = [
        c
        async for c in server.stream_completion(tiny_engine, request.prompt, request, 1)
    ]
    assert chunks
    assert len(captured) == 2
    assert all(c["kv_cache_bits"] == 4 for c in captured)


def test_unsupported_http_mode_fails_before_sse_headers(monkeypatch):
    from fastapi.testclient import TestClient

    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_model_name", "tiny-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)
    monkeypatch.setattr(server, "get_engine", lambda: object())
    client = TestClient(server.app)
    for endpoint, payload in [
        ("/v1/completions", {"prompt": "1 2 3"}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "1 2 3"}]}),
    ]:
        for stream in (False, True):
            response = client.post(
                endpoint,
                json={
                    "model": "tiny-model",
                    "stream": stream,
                    "kv_cache_bits": 4,
                    **payload,
                },
            )
            assert response.status_code == 400
            assert response.headers["content-type"] == "application/json"
