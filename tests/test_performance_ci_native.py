"""Real tiny-model worker smoke, requiring MLX but no downloaded weights."""

import asyncio
import math

import pytest

mx = pytest.importorskip("mlx.core")

from scripts.performance_ci.worker import eos_processor, make_prompts, measure


class TinyTokenizer:
    eos_token_id = 31
    eos_token_ids = {31}
    clean_up_tokenization_spaces = False

    def encode(self, text, **kwargs):
        return [ord(c) % 30 for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(65 + t % 26) for t in tokens)


def test_prompts_have_exact_lengths_and_distinct_requests():
    prompts = make_prompts(TinyTokenizer(), 16, 4)
    assert len(prompts) == 4
    assert all(len(p) == 16 for p in prompts)
    assert len({tuple(p) for p in prompts}) == 4
    assert prompts == make_prompts(TinyTokenizer(), 16, 4)


def test_eos_mask_preserves_other_logits_and_supports_batched_shape():
    processor = eos_processor({1, 3})
    logits = mx.array([[1.0, 100.0, 3.0, 100.0]])
    result = processor(mx.array([0]), logits).tolist()[0]
    assert result[0] == 1
    assert result[2] == 3
    assert result[1] == result[3] == -math.inf


@pytest.mark.parametrize("prefix_cache,concurrency", [(False, 2), (True, 1)])
def test_worker_runs_fixed_length_first_and_warm_batches(prefix_cache, concurrency):
    from mlx_lm.models.qwen2 import Model, ModelArgs

    def loader():
        mx.random.seed(42)
        model = Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=2,
                rms_norm_eps=1e-5,
                vocab_size=32,
                num_key_value_heads=2,
            )
        )
        mx.eval(model.parameters())
        return model, TinyTokenizer()

    case = {
        "prompt_tokens": 128,
        "output_tokens": 8,
        "concurrency": concurrency,
        "prefix_cache": prefix_cache,
    }
    result = asyncio.run(measure(loader, case))
    assert result["model_load_seconds"] > 0
    assert len(result["prompt_sha256"]) == 64
    assert set(result["phases"]) == {"first", "warm"}
    for phase in result["phases"].values():
        assert phase["output_tokens"] == 8 * concurrency
        assert phase["batch_seconds"] > 0
        assert phase["ttft_seconds"] > 0
        assert phase["tokens_per_second"] > 0
        assert len(phase["output_sha256"]) == 64
    assert result["phases"]["first"]["cached_tokens"] == 0
    if prefix_cache:
        assert result["phases"]["warm"]["cached_tokens"] > 0
    else:
        assert result["phases"]["warm"]["cached_tokens"] == 0
