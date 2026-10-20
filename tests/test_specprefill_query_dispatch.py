from types import SimpleNamespace

import mlx.nn as nn
import pytest

from vllm_mlx import specprefill


def test_empty_rope_module_is_preserved():
    rope = nn.RoPE(16)
    assert not rope
    assert specprefill._get_rope(SimpleNamespace(rope=rope)) is rope


@pytest.mark.parametrize("identity", ["model_type", "config"])
def test_score_tokens_selects_qwen_gated_queries(monkeypatch, identity):
    attention = SimpleNamespace(
        rope=nn.RoPE(16), num_attention_heads=2, num_key_value_heads=1
    )
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    if identity == "config":
        model.config = SimpleNamespace(model_type="qwen3_5")
    else:
        model.model_type = "qwen3_5"
    monkeypatch.setattr(
        specprefill, "_find_attention_layers", lambda _: [(0, model.layers[0])]
    )
    monkeypatch.setattr(specprefill, "make_prompt_cache", lambda _: [])
    monkeypatch.setattr(specprefill, "_prefill_draft", lambda *args, **kwargs: None)

    class CaptureReached(Exception):
        pass

    def capture(model, query_buffer, query_extractor):
        assert query_extractor is specprefill._qwen35_extract_queries
        raise CaptureReached

    monkeypatch.setattr(specprefill, "_patch_attention_for_capture", capture)
    with pytest.raises(CaptureReached):
        specprefill.score_tokens(model, [1, 2, 3])
