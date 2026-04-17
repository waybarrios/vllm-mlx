# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.bench_serve — prompt loading and sweep expansion."""

import json
from pathlib import Path
from typing import Optional

import pytest

from vllm_mlx.bench_serve import (
    BenchServeResult,
    SweepConfig,
    expand_sweep,
    load_prompt_set,
)

# ---------------------------------------------------------------------------
# TestPromptLoading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    """Tests for load_prompt_set()."""

    def test_load_short(self):
        prompts = load_prompt_set("short")
        assert isinstance(prompts, list)
        assert len(prompts) == 5
        for p in prompts:
            assert "role" in p
            assert p["role"] == "user"
            assert "content" in p
            assert len(p["content"]) > 0

    def test_load_medium(self):
        prompts = load_prompt_set("medium")
        assert isinstance(prompts, list)
        assert len(prompts) == 5
        for p in prompts:
            assert "role" in p
            assert "content" in p

    def test_load_long(self):
        prompts = load_prompt_set("long")
        assert isinstance(prompts, list)
        assert len(prompts) == 3
        for p in prompts:
            assert "role" in p
            assert "content" in p
            # Long prompts should actually be long
            assert len(p["content"]) > 1000

    def test_load_thinking(self):
        prompts = load_prompt_set("thinking")
        assert isinstance(prompts, list)
        assert len(prompts) == 3
        for p in prompts:
            assert "role" in p
            assert "content" in p

    def test_all_builtins_are_lists_of_dicts(self):
        for name in ("short", "medium", "long", "thinking"):
            prompts = load_prompt_set(name)
            assert isinstance(prompts, list), f"{name}: expected list"
            assert len(prompts) > 0, f"{name}: expected non-empty list"
            for item in prompts:
                assert isinstance(item, dict), f"{name}: items must be dicts"

    def test_unknown_name_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt_set("nonexistent_set")

    def test_unknown_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt_set("/tmp/does_not_exist_bench_serve_test.json")

    def test_custom_file_via_tmp_path(self, tmp_path: Path):
        custom = [
            {"role": "user", "content": "Hello, world!"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        custom_file = tmp_path / "custom_prompts.json"
        custom_file.write_text(json.dumps(custom))

        loaded = load_prompt_set(str(custom_file))
        assert loaded == custom

    def test_custom_file_returns_exact_data(self, tmp_path: Path):
        payload = [{"role": "user", "content": "test", "extra": 42}]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(payload))

        result = load_prompt_set(str(p))
        assert result == payload

    def test_short_prompts_are_user_role(self):
        prompts = load_prompt_set("short")
        roles = {p["role"] for p in prompts}
        assert roles == {"user"}

    def test_thinking_prompts_contain_reasoning_keywords(self):
        prompts = load_prompt_set("thinking")
        combined = " ".join(p["content"] for p in prompts).lower()
        # At least one reasoning-heavy keyword should appear
        keywords = ["step", "deduc", "logic", "proof", "weighing", "clue"]
        assert any(kw in combined for kw in keywords)


# ---------------------------------------------------------------------------
# TestExpandSweep
# ---------------------------------------------------------------------------


class TestExpandSweep:
    """Tests for expand_sweep()."""

    def test_single_values_single_repetition(self):
        result = expand_sweep(["short"], [1], [None], [""], 1)
        assert result == [("short", 1, None, "", 0)]

    def test_two_prompt_sets_one_rep(self):
        result = expand_sweep(["short", "long"], [1], [None], [""], 1)
        assert len(result) == 2
        prompt_sets = [r[0] for r in result]
        assert "short" in prompt_sets
        assert "long" in prompt_sets

    def test_combinatorial_2x2x1x1x3(self):
        # 2 prompt_sets × 2 concurrencies × 1 thinking × 1 extra_body × 3 reps = 12
        result = expand_sweep(
            ["short", "medium"],
            [1, 4],
            [None],
            [""],
            3,
        )
        assert len(result) == 12

    def test_thinking_sweep(self):
        # 1 × 1 × 3 thinking × 1 × 1 = 3
        result = expand_sweep(["short"], [1], [None, True, False], [""], 1)
        assert len(result) == 3
        thinking_vals = [r[2] for r in result]
        assert None in thinking_vals
        assert True in thinking_vals
        assert False in thinking_vals

    def test_extra_body_sweep(self):
        bodies = ["", '{"top_p": 0.9}', '{"top_k": 50}']
        result = expand_sweep(["short"], [1], [None], bodies, 1)
        assert len(result) == 3
        extra_vals = [r[3] for r in result]
        assert set(extra_vals) == set(bodies)

    def test_repetition_indices_are_zero_based(self):
        result = expand_sweep(["short"], [1], [None], [""], 5)
        rep_indices = [r[4] for r in result]
        assert rep_indices == list(range(5))

    def test_repetition_indices_with_multiple_combos(self):
        result = expand_sweep(["short", "medium"], [1], [None], [""], 3)
        # 2 combos × 3 reps = 6 entries; each combo should have reps 0,1,2
        short_reps = [r[4] for r in result if r[0] == "short"]
        medium_reps = [r[4] for r in result if r[0] == "medium"]
        assert sorted(short_reps) == [0, 1, 2]
        assert sorted(medium_reps) == [0, 1, 2]

    def test_full_cartesian_3x2x2x2x2(self):
        # 3 × 2 × 2 × 2 × 2 = 48
        result = expand_sweep(
            ["short", "medium", "long"],
            [1, 4],
            [None, True],
            ["", '{"top_p": 0.9}'],
            2,
        )
        assert len(result) == 48

    def test_result_elements_are_sweep_config_tuples(self):
        result = expand_sweep(["short"], [2], [True], ['{"a":1}'], 1)
        assert len(result) == 1
        config = result[0]
        assert isinstance(config, tuple)
        assert len(config) == 5
        prompt_set, concurrency, thinking, extra_body, rep_idx = config
        assert prompt_set == "short"
        assert concurrency == 2
        assert thinking is True
        assert extra_body == '{"a":1}'
        assert rep_idx == 0

    def test_empty_inputs_return_empty(self):
        assert expand_sweep([], [1], [None], [""], 1) == []
        assert expand_sweep(["short"], [], [None], [""], 1) == []
        assert expand_sweep(["short"], [1], [], [""], 1) == []
        assert expand_sweep(["short"], [1], [None], [], 1) == []

    def test_zero_repetitions_return_empty(self):
        result = expand_sweep(["short"], [1], [None], [""], 0)
        assert result == []

    def test_output_type_is_list(self):
        result = expand_sweep(["short"], [1], [None], [""], 1)
        assert isinstance(result, list)

    def test_all_prompt_sets_appear_in_output(self):
        sets = ["short", "medium", "long", "thinking"]
        result = expand_sweep(sets, [1], [None], [""], 1)
        assert len(result) == 4
        output_sets = {r[0] for r in result}
        assert output_sets == set(sets)

    def test_concurrency_values_appear_in_output(self):
        result = expand_sweep(["short"], [1, 4, 16], [None], [""], 1)
        concurrencies = {r[1] for r in result}
        assert concurrencies == {1, 4, 16}

    def test_sweep_config_type_alias(self):
        # SweepConfig should be constructable as a 5-tuple
        config: SweepConfig = ("short", 1, None, "", 0)
        assert config[0] == "short"
        assert config[1] == 1
        assert config[2] is None
        assert config[3] == ""
        assert config[4] == 0


# ---------------------------------------------------------------------------
# TestBenchServeResult
# ---------------------------------------------------------------------------


class TestBenchServeResult:
    """Basic sanity checks for the BenchServeResult dataclass."""

    def test_default_instantiation(self):
        r = BenchServeResult()
        assert r.run_id == ""
        assert r.concurrency == 1
        assert r.max_tokens == 256
        assert r.enable_thinking is None
        assert r.validated is True

    def test_field_assignment(self):
        r = BenchServeResult(
            run_id="abc123",
            model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
            concurrency=4,
            ttft_ms=42.5,
            gen_tps=123.4,
            cache_hit_rate=0.85,
            validated=False,
        )
        assert r.run_id == "abc123"
        assert r.model_id == "mlx-community/Llama-3.2-1B-Instruct-4bit"
        assert r.concurrency == 4
        assert r.ttft_ms == 42.5
        assert r.gen_tps == 123.4
        assert r.cache_hit_rate == 0.85
        assert r.validated is False

    def test_enable_thinking_can_be_none_or_bool(self):
        r_none = BenchServeResult(enable_thinking=None)
        r_true = BenchServeResult(enable_thinking=True)
        r_false = BenchServeResult(enable_thinking=False)
        assert r_none.enable_thinking is None
        assert r_true.enable_thinking is True
        assert r_false.enable_thinking is False

    def test_has_all_required_fields(self):
        r = BenchServeResult()
        # Identity
        assert hasattr(r, "run_id")
        assert hasattr(r, "timestamp")
        assert hasattr(r, "tag")
        # Hardware
        assert hasattr(r, "chip")
        assert hasattr(r, "gpu_cores")
        assert hasattr(r, "memory_gb")
        assert hasattr(r, "bandwidth_gbs")
        assert hasattr(r, "os_version")
        # Runtime
        assert hasattr(r, "model_id")
        assert hasattr(r, "model_type")
        assert hasattr(r, "engine_type")
        assert hasattr(r, "mtp_enabled")
        assert hasattr(r, "specprefill")
        assert hasattr(r, "kv_quant")
        assert hasattr(r, "cache_type")
        # Config
        assert hasattr(r, "prompt_set")
        assert hasattr(r, "concurrency")
        assert hasattr(r, "max_tokens")
        assert hasattr(r, "enable_thinking")
        assert hasattr(r, "extra_body")
        assert hasattr(r, "repetition")
        assert hasattr(r, "prompt_tokens")
        # Latency
        assert hasattr(r, "ttft_ms")
        assert hasattr(r, "tpot_ms")
        assert hasattr(r, "e2e_latency_ms")
        # Throughput
        assert hasattr(r, "gen_tps")
        assert hasattr(r, "prompt_tps")
        assert hasattr(r, "throughput_tps")
        assert hasattr(r, "requests_per_s")
        # Memory
        assert hasattr(r, "metal_active_gb")
        assert hasattr(r, "metal_peak_gb")
        assert hasattr(r, "metal_cache_gb")
        # Cache
        assert hasattr(r, "cache_hits")
        assert hasattr(r, "cache_misses")
        assert hasattr(r, "cache_hit_rate")
        assert hasattr(r, "tokens_saved")
        # Validation
        assert hasattr(r, "validated")
