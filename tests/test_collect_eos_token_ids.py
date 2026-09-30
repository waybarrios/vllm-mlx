# SPDX-License-Identifier: Apache-2.0
"""Tests for collect_eos_token_ids — the shared EOS/stop-token-set helper.

Regression context: the SimpleEngine MLLM text route passed the raw HF
processor tokenizer to mlx_lm.stream_generate, which defaults its stop set
to {tokenizer.eos_token_id} — dropping turn terminators that chat models
declare only in the config EOS list (Gemma 4: <turn|>=106,
<|tool_response>=50). Generation then ran straight through end-of-turn
until max_tokens on every request.
"""

import json

from vllm_mlx.utils.tokenizer import collect_eos_token_ids


class FakeTokenizer:
    def __init__(self, eos_token_id=None, eos_token_ids=None, name_or_path=None):
        if eos_token_id is not None:
            self.eos_token_id = eos_token_id
        if eos_token_ids is not None:
            self.eos_token_ids = eos_token_ids
        if name_or_path is not None:
            self.name_or_path = name_or_path


def _write_config(path, name, eos):
    (path / name).write_text(json.dumps({"eos_token_id": eos}))


def test_tokenizer_int_eos_only():
    assert collect_eos_token_ids(FakeTokenizer(eos_token_id=2)) == {2}


def test_tokenizer_list_eos():
    assert collect_eos_token_ids(FakeTokenizer(eos_token_id=[2, 7])) == {2, 7}


def test_wrapper_eos_token_ids_attr():
    tok = FakeTokenizer(eos_token_id=2, eos_token_ids={2, 5})
    assert collect_eos_token_ids(tok) == {2, 5}


def test_no_eos_anywhere_returns_empty():
    assert collect_eos_token_ids(FakeTokenizer()) == set()


def test_gemma4_config_eos_list_unioned(tmp_path):
    """The bug: <turn|>=106 / <|tool_response>=50 live only in the config."""
    _write_config(tmp_path, "config.json", [1, 106, 50])
    tok = FakeTokenizer(eos_token_id=1)
    assert collect_eos_token_ids(tok, tmp_path) == {1, 106, 50}


def test_generation_config_eos_unioned(tmp_path):
    _write_config(tmp_path, "generation_config.json", [1, 106])
    tok = FakeTokenizer(eos_token_id=1)
    assert collect_eos_token_ids(tok, tmp_path) == {1, 106}


def test_both_configs_unioned(tmp_path):
    _write_config(tmp_path, "config.json", [1, 106])
    _write_config(tmp_path, "generation_config.json", [1, 50])
    tok = FakeTokenizer(eos_token_id=1)
    assert collect_eos_token_ids(tok, tmp_path) == {1, 50, 106}


def test_model_path_falls_back_to_name_or_path(tmp_path):
    _write_config(tmp_path, "config.json", [1, 106, 50])
    tok = FakeTokenizer(eos_token_id=1, name_or_path=str(tmp_path))
    assert collect_eos_token_ids(tok) == {1, 106, 50}


def test_scalar_config_eos(tmp_path):
    _write_config(tmp_path, "generation_config.json", 7)
    assert collect_eos_token_ids(FakeTokenizer(), tmp_path) == {7}


def test_corrupt_config_ignored(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    tok = FakeTokenizer(eos_token_id=2)
    assert collect_eos_token_ids(tok, tmp_path) == {2}


def test_missing_model_path_ok():
    tok = FakeTokenizer(eos_token_id=2)
    assert collect_eos_token_ids(tok, "/nonexistent/path") == {2}


def test_mllm_scheduler_batched_path_pins_gemma4_stop_set(tmp_path):
    """Pin the Gemma 4 stop set {1, 106, 50} through the batched path.

    Behavioral change vs the old MLLMScheduler._get_stop_tokens: that
    implementation read only generation_config.json, while the shared
    helper also reads config.json — Gemma 4 (MLX export) declares
    eos_token_id [1, 106, 50] in config.json only. Models that list extra
    ids (e.g. pad/bos) in the config EOS set now stop on tokens the
    batched path previously ignored; that union is the point of the fix.
    """
    from types import SimpleNamespace

    from vllm_mlx.mllm_scheduler import MLLMScheduler

    _write_config(tmp_path, "config.json", [1, 106, 50])
    tokenizer = FakeTokenizer(eos_token_id=1, name_or_path=str(tmp_path))

    # Processor wrapping a tokenizer (the usual MLLM shape)
    scheduler = SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer))
    assert MLLMScheduler._get_stop_tokens(scheduler) == {1, 106, 50}

    # Bare tokenizer as processor (no .tokenizer attribute)
    scheduler = SimpleNamespace(processor=tokenizer)
    assert MLLMScheduler._get_stop_tokens(scheduler) == {1, 106, 50}
