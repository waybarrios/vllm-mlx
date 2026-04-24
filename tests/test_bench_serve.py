# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.bench_serve — prompt loading and sweep expansion."""

import asyncio
import csv
import json
import os
from pathlib import Path

import pytest

from vllm_mlx.bench_serve import (
    RESULT_COLUMNS,
    BenchServeResult,
    SweepConfig,
    Workload,
    WorkloadCase,
    compute_request_metrics,
    compute_summary_stats,
    detect_hardware_fingerprint,
    expand_sweep,
    format_csv,
    format_json,
    format_sql,
    format_table,
    format_workload_csv,
    format_workload_payload,
    format_workload_sql,
    load_prompt_set,
    load_workload,
    parse_health_response,
    parse_metrics_text,
    parse_sse_line,
    parse_status_response,
    run_bench_serve_workload,
    run_workload_case,
    summarize_workload_results,
    validate_quality_checks,
    validate_response,
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
        for msgs in prompts:
            assert isinstance(msgs, list) and msgs
            first = msgs[0]
            assert first["role"] == "user"
            assert "content" in first and len(first["content"]) > 0

    def test_load_medium(self):
        prompts = load_prompt_set("medium")
        assert isinstance(prompts, list)
        assert len(prompts) == 5
        for msgs in prompts:
            assert isinstance(msgs, list) and msgs
            assert "role" in msgs[0]
            assert "content" in msgs[0]

    def test_load_long(self):
        prompts = load_prompt_set("long")
        assert isinstance(prompts, list)
        assert len(prompts) == 3
        for msgs in prompts:
            assert isinstance(msgs, list) and msgs
            assert "role" in msgs[0]
            assert "content" in msgs[0]
            # Long prompts should actually be long (sum across all messages)
            assert sum(len(m["content"]) for m in msgs) > 1000

    def test_load_thinking(self):
        prompts = load_prompt_set("thinking")
        assert isinstance(prompts, list)
        assert len(prompts) == 3
        for msgs in prompts:
            assert isinstance(msgs, list) and msgs
            assert "role" in msgs[0]
            assert "content" in msgs[0]

    def test_all_builtins_return_lists_of_message_lists(self):
        for name in ("short", "medium", "long", "thinking"):
            prompts = load_prompt_set(name)
            assert isinstance(prompts, list), f"{name}: expected list"
            assert len(prompts) > 0, f"{name}: expected non-empty list"
            for msgs in prompts:
                assert (
                    isinstance(msgs, list) and msgs
                ), f"{name}: items must be non-empty lists"
                for m in msgs:
                    assert isinstance(m, dict), f"{name}: messages must be dicts"

    def test_unknown_name_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt_set("nonexistent_set")

    def test_unknown_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt_set("/tmp/does_not_exist_bench_serve_test.json")

    def test_custom_file_flat_format(self, tmp_path: Path):
        """Legacy flat format — list of dicts — still works, normalised to list-of-lists."""
        custom = [
            {"role": "user", "content": "Hello, world!"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        custom_file = tmp_path / "custom_prompts.json"
        custom_file.write_text(json.dumps(custom))

        loaded = load_prompt_set(str(custom_file))
        assert loaded == [[m] for m in custom]

    def test_custom_file_multi_message_format(self, tmp_path: Path):
        """New format — list of message lists — used for system+user scenarios."""
        custom = [
            [
                {"role": "system", "content": "You are a code assistant."},
                {"role": "user", "content": "Hi"},
            ],
            [
                {"role": "system", "content": "You are a code assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]
        custom_file = tmp_path / "multi.json"
        custom_file.write_text(json.dumps(custom))

        loaded = load_prompt_set(str(custom_file))
        assert loaded == custom

    def test_custom_file_preserves_extra_fields(self, tmp_path: Path):
        payload = [{"role": "user", "content": "test", "extra": 42}]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(payload))

        result = load_prompt_set(str(p))
        # Flat format → normalised to [[msg]]
        assert result == [[payload[0]]]

    def test_invalid_top_level_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}')
        with pytest.raises(ValueError, match="non-empty JSON list"):
            load_prompt_set(str(p))

    def test_empty_list_raises(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="non-empty JSON list"):
            load_prompt_set(str(p))

    def test_invalid_entry_type_raises(self, tmp_path: Path):
        p = tmp_path / "weird.json"
        p.write_text('["not a dict or list"]')
        with pytest.raises(ValueError, match="must be dict or list"):
            load_prompt_set(str(p))

    def test_short_prompts_are_user_role(self):
        prompts = load_prompt_set("short")
        roles = {msgs[0]["role"] for msgs in prompts}
        assert roles == {"user"}

    def test_thinking_prompts_contain_reasoning_keywords(self):
        prompts = load_prompt_set("thinking")
        combined = " ".join(m["content"] for msgs in prompts for m in msgs).lower()
        # At least one reasoning-heavy keyword should appear
        keywords = ["step", "deduc", "logic", "proof", "weighing", "clue"]
        assert any(kw in combined for kw in keywords)


class TestWorkloadLoading:
    """Tests for declarative contract workload loading."""

    def test_load_workload_with_defaults(self, tmp_path: Path):
        workload_file = tmp_path / "workload.json"
        workload_file.write_text(
            json.dumps(
                {
                    "name": "writing-contract",
                    "defaults": {
                        "max_tokens": 128,
                        "enable_thinking": True,
                        "policy_timeout_ms": 180000,
                        "checks": {"forbidden_regex": ["<think>"]},
                    },
                    "cases": [
                        {
                            "id": "case-a",
                            "messages": [
                                {"role": "user", "content": "Write a short note."}
                            ],
                            "tags": ["quality"],
                        }
                    ],
                }
            )
        )

        workload = load_workload(workload_file)

        assert workload.name == "writing-contract"
        assert len(workload.cases) == 1
        case = workload.cases[0]
        assert case.case_id == "case-a"
        assert case.max_tokens == 128
        assert case.enable_thinking is True
        assert case.policy_timeout_ms == 180000
        assert case.checks == {"forbidden_regex": ["<think>"]}
        assert case.tags == ("quality",)

    def test_load_workload_rejects_missing_messages(self, tmp_path: Path):
        workload_file = tmp_path / "workload.json"
        workload_file.write_text(json.dumps({"cases": [{"id": "bad"}]}))

        with pytest.raises(ValueError, match="messages"):
            load_workload(workload_file)

    def test_load_workload_case_from_request_path(self, tmp_path: Path):
        request_file = tmp_path / "request.json"
        request_file.write_text(
            json.dumps(
                {
                    "model": "ignored-by-workload-runner",
                    "messages": [{"role": "user", "content": "Write the artifact."}],
                    "stream": True,
                    "max_tokens": 32768,
                    "enable_thinking": True,
                    "thinking_token_budget": 8192,
                    "temperature": 0.6,
                }
            )
        )
        workload_file = tmp_path / "workload.json"
        workload_file.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "id": "resume",
                            "request_path": "request.json",
                            "extra_body": {"top_p": 0.95},
                        }
                    ]
                }
            )
        )

        workload = load_workload(workload_file)
        case = workload.cases[0]

        assert case.messages == [{"role": "user", "content": "Write the artifact."}]
        assert case.request_path == "request.json"
        assert case.max_tokens == 32768
        assert case.enable_thinking is True
        assert case.extra_body == {
            "thinking_token_budget": 8192,
            "temperature": 0.6,
            "top_p": 0.95,
        }


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


# ---------------------------------------------------------------------------
# TestAutoDetectionParsing  (Task 3)
# ---------------------------------------------------------------------------


class TestAutoDetectionParsing:
    """Unit tests for server response parsers and hardware fingerprint."""

    def test_parse_health_response(self):
        data = {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "model_type": "llm",
        }
        result = parse_health_response(data)
        assert result["model_name"] == "mlx-community/Llama-3.2-1B-Instruct-4bit"
        assert result["model_type"] == "llm"

    def test_parse_health_mllm(self):
        data = {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "mlx-community/gemma-4-27b",
            "model_type": "mllm",
        }
        result = parse_health_response(data)
        assert result["model_type"] == "mllm"
        assert result["model_name"] == "mlx-community/gemma-4-27b"

    def test_parse_status_response(self):
        data = {
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "metal": {
                "active_memory_gb": 12.5,
                "peak_memory_gb": 14.0,
                "cache_memory_gb": 2.0,
            },
            "cache": {"type": "paged"},
        }
        result = parse_status_response(data)
        assert result["model"] == "mlx-community/Llama-3.2-1B-Instruct-4bit"
        assert result["metal_active_gb"] == pytest.approx(12.5)
        assert result["metal_peak_gb"] == pytest.approx(14.0)
        assert result["metal_cache_gb"] == pytest.approx(2.0)
        assert result["cache_type"] == "paged"

    def test_parse_status_response_accepts_legacy_metal_keys(self):
        data = {
            "model": "legacy-server",
            "metal": {
                "active_gb": 12.5,
                "peak_gb": 14.0,
                "cache_gb": 2.0,
            },
        }
        result = parse_status_response(data)
        assert result["metal_active_gb"] == pytest.approx(12.5)
        assert result["metal_peak_gb"] == pytest.approx(14.0)
        assert result["metal_cache_gb"] == pytest.approx(2.0)

    def test_parse_status_no_metal(self):
        data = {"model": "some-model"}
        result = parse_status_response(data)
        assert result["metal_active_gb"] == pytest.approx(0.0)
        assert result["metal_peak_gb"] == pytest.approx(0.0)
        assert result["metal_cache_gb"] == pytest.approx(0.0)
        assert result["cache_type"] == ""

    def test_parse_metrics_text_with_cache_stats(self):
        text = (
            "# HELP vllm_prefix_cache_hits_total Total prefix cache hits\n"
            "# TYPE vllm_prefix_cache_hits_total counter\n"
            "vllm_prefix_cache_hits_total 42\n"
            "# HELP vllm_prefix_cache_misses_total Total prefix cache misses\n"
            "# TYPE vllm_prefix_cache_misses_total counter\n"
            "vllm_prefix_cache_misses_total 8\n"
            "# HELP vllm_prefix_cache_tokens_saved_total Tokens saved\n"
            "# TYPE vllm_prefix_cache_tokens_saved_total counter\n"
            "vllm_prefix_cache_tokens_saved_total 1024\n"
        )
        result = parse_metrics_text(text)
        assert result["cache_hits"] == 42
        assert result["cache_misses"] == 8
        assert result["tokens_saved"] == 1024

    def test_parse_metrics_empty(self):
        result = parse_metrics_text("")
        assert result["cache_hits"] == 0
        assert result["cache_misses"] == 0
        assert result["tokens_saved"] == 0

    def test_detect_hardware_fingerprint(self):
        result = detect_hardware_fingerprint()
        assert isinstance(result, dict)
        assert "chip" in result
        assert "gpu_cores" in result
        assert "memory_gb" in result
        assert "bandwidth_gbs" in result
        assert "os_version" in result
        assert isinstance(result["os_version"], str)
        assert len(result["os_version"]) > 0


# ---------------------------------------------------------------------------
# TestSSEParsing  (Task 4)
# ---------------------------------------------------------------------------


class TestSSEParsing:
    """Unit tests for parse_sse_line()."""

    def _make_line(self, delta_content=None, finish_reason=None, usage=None):
        """Build a synthetic SSE data line."""
        chunk: dict = {
            "choices": [
                {
                    "delta": (
                        {"content": delta_content} if delta_content is not None else {}
                    ),
                    "finish_reason": finish_reason,
                }
            ]
        }
        if usage is not None:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}"

    def test_parse_data_line(self):
        line = self._make_line(delta_content="Hello")
        result = parse_sse_line(line)
        assert result is not None
        assert result["content"] == "Hello"
        assert result["finish_reason"] is None
        assert result["usage"] is None

    def test_parse_done(self):
        assert parse_sse_line("data: [DONE]") is None

    def test_parse_empty_line(self):
        assert parse_sse_line("") is None

    def test_parse_comment_line(self):
        assert parse_sse_line(": keep-alive") is None

    def test_parse_no_content(self):
        line = self._make_line()  # delta has no content key
        result = parse_sse_line(line)
        assert result is not None
        assert result["content"] == ""

    def test_parse_with_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        line = self._make_line(delta_content=None, usage=usage)
        result = parse_sse_line(line)
        assert result is not None
        assert result["usage"] == usage

    def test_parse_with_finish_reason(self):
        line = self._make_line(finish_reason="stop")
        result = parse_sse_line(line)
        assert result is not None
        assert result["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# TestRequestMetrics  (Task 4)
# ---------------------------------------------------------------------------


class TestRequestMetrics:
    """Unit tests for compute_request_metrics()."""

    def test_basic_metrics(self):
        # Simulate: request starts at 0, first token at 0.1s, then 4 more tokens
        # every 0.02s.
        t_start = 0.0
        t_first = 0.1
        token_times = [t_first + i * 0.02 for i in range(5)]
        t_end = token_times[-1] + 0.001  # tiny extra after last token
        prompt_tokens = 20
        completion_tokens = 5

        metrics = compute_request_metrics(
            t_start=t_start,
            t_first_token=t_first,
            token_times=token_times,
            t_end=t_end,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # TTFT should be ~100 ms
        assert metrics["ttft_ms"] == pytest.approx(100.0, abs=1.0)
        # TPOT should be ~20 ms (inter-token interval)
        assert metrics["tpot_ms"] == pytest.approx(20.0, abs=1.0)
        # E2E latency > TTFT
        assert metrics["e2e_latency_ms"] > metrics["ttft_ms"]
        # gen_tps > 0
        assert metrics["gen_tps"] > 0.0
        # prompt_tps > 0
        assert metrics["prompt_tps"] > 0.0

    def test_single_token(self):
        t_start = 0.0
        t_first = 0.05
        token_times = [t_first]
        t_end = t_first + 0.001
        metrics = compute_request_metrics(
            t_start=t_start,
            t_first_token=t_first,
            token_times=token_times,
            t_end=t_end,
            prompt_tokens=10,
            completion_tokens=1,
        )
        # Single token → no inter-token interval → TPOT = 0.0
        assert metrics["tpot_ms"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestValidation  (Task 5)
# ---------------------------------------------------------------------------


class TestValidation:
    """Unit tests for validate_response()."""

    def test_valid_response(self):
        is_valid, msg = validate_response(
            finish_reason="stop", content="Hello world", status_code=200
        )
        assert is_valid is True
        assert msg == ""

    def test_empty_content(self):
        is_valid, msg = validate_response(
            finish_reason="stop", content="", status_code=200
        )
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_length_truncation(self):
        is_valid, msg = validate_response(
            finish_reason="length", content="partial text", status_code=200
        )
        assert is_valid is False
        assert "length" in msg.lower()

    def test_missing_finish_reason(self):
        is_valid, msg = validate_response(
            finish_reason=None, content="some text", status_code=200
        )
        assert is_valid is False
        assert msg != ""

    def test_http_error(self):
        is_valid, msg = validate_response(
            finish_reason="stop", content="error body", status_code=500
        )
        assert is_valid is False
        assert "500" in msg


class TestQualityChecks:
    """Unit tests for workload quality checks."""

    def test_required_and_forbidden_regex(self):
        ok, issues = validate_quality_checks(
            "stop",
            "Dear team,\nThis is a clean response.\nSincerely",
            {
                "required_regex": ["Dear team", "Sincerely"],
                "forbidden_regex": ["<think>", "unsupported claim"],
                "min_chars": 20,
                "finish_reason": "stop",
            },
        )

        assert ok is True
        assert issues == []

    def test_forbidden_regex_failure(self):
        ok, issues = validate_quality_checks(
            "stop",
            "Visible answer\n<think>hidden plan</think>",
            {"forbidden_regex": ["<think>"]},
        )

        assert ok is False
        assert any("forbidden_regex matched" in issue for issue in issues)

    def test_json_check(self):
        ok, issues = validate_quality_checks(
            "stop",
            '{"title": "Engineer", "priority": 1}',
            {"json": True},
        )

        assert ok is True
        assert issues == []

    def test_json_check_failure(self):
        ok, issues = validate_quality_checks("stop", "not json", {"json": True})

        assert ok is False
        assert any("not valid JSON" in issue for issue in issues)


class TestWorkloadSummary:
    """Unit tests for workload summary aggregation."""

    def test_summarize_workload_results_tracks_quality_and_policy(self):
        results = [
            {
                "ok": True,
                "case_id": "case-a",
                "repetition": 0,
                "policy": {"within_timeout": True},
                "quality": {"ok": True, "content_chars": 120},
                "metrics": {
                    "e2e_latency_ms": 100.0,
                    "ttft_ms": 10.0,
                    "gen_tps": 20.0,
                },
            },
            {
                "ok": True,
                "case_id": "case-a",
                "repetition": 1,
                "policy": {"within_timeout": False},
                "quality": {"ok": True, "content_chars": 160},
                "metrics": {
                    "e2e_latency_ms": 200.0,
                    "ttft_ms": 20.0,
                    "gen_tps": 10.0,
                },
            },
        ]

        summary = summarize_workload_results(results)

        assert summary["passed"] is True
        assert summary["quality_passed"] is True
        assert summary["policy_timeout_passed"] is False
        assert summary["failure_rate"] == pytest.approx(0.0)
        assert summary["latency_ms"]["p50"] == pytest.approx(150.0)
        assert summary["unique_case_count"] == 1
        assert summary["repetition_count"] == 2
        assert summary["case_summaries"]["case-a"]["repetitions"] == [0, 1]
        assert summary["case_summaries"]["case-a"]["latency_ms"]["min"] == 100.0
        assert summary["case_summaries"]["case-a"]["latency_ms"]["max"] == 200.0


class TestWorkloadRunner:
    """Unit tests for contract workload execution records."""

    def test_run_workload_case_records_metrics_policy_and_quality(self, monkeypatch):
        metrics_responses = iter(
            [
                {"cache_hits": 10, "cache_misses": 3, "tokens_saved": 100},
                {"cache_hits": 12, "cache_misses": 4, "tokens_saved": 140},
            ]
        )

        async def fake_scrape_metrics(client, base_url):
            return next(metrics_responses)

        async def fake_stream_chat_completion(**kwargs):
            assert kwargs["max_tokens"] == 64
            assert kwargs["enable_thinking"] is True
            assert kwargs["extra_body"] == {"temperature": 0.6}
            return {
                "ttft_ms": 25.0,
                "tpot_ms": 3.0,
                "e2e_latency_ms": 2500.0,
                "gen_tps": 12.5,
                "prompt_tps": 500.0,
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "finish_reason": "stop",
                "content": "Dear team,\nA clean benchmark artifact.\nSincerely",
            }

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "cache": {"type": "paged"},
                    "metal": {
                        "active_memory_gb": 42.0,
                        "peak_memory_gb": 45.0,
                        "cache_memory_gb": 4.0,
                    },
                }

        class FakeClient:
            async def get(self, url):
                return FakeResponse()

        monkeypatch.setattr("vllm_mlx.bench_serve.scrape_metrics", fake_scrape_metrics)
        monkeypatch.setattr(
            "vllm_mlx.bench_serve.stream_chat_completion",
            fake_stream_chat_completion,
        )

        workload = Workload(
            name="writing-contract",
            description="",
            defaults={"max_tokens": 64},
            cases=[],
        )
        case = WorkloadCase(
            case_id="resume-smoke",
            messages=[{"role": "user", "content": "Write the artifact."}],
            request_path="/tmp/request.json",
            max_tokens=64,
            enable_thinking=True,
            extra_body={"temperature": 0.6},
            policy_timeout_ms=1800,
            checks={
                "finish_reason": "stop",
                "required_regex": ["Dear team", "Sincerely"],
                "forbidden_regex": ["<think>"],
            },
            tags=("resume",),
        )

        record = asyncio.run(
            run_workload_case(
                FakeClient(),
                "http://server",
                workload=workload,
                case=case,
                model="test-model",
                runtime={"engine_type": "mllm"},
                hardware={"chip": "test"},
                run_id="run123",
                timestamp="2026-04-23T00:00:00+00:00",
                repetition=2,
                scrape=True,
                include_content=True,
            )
        )

        assert record["ok"] is True
        assert record["quality"]["ok"] is True
        assert record["quality"]["issues"] == []
        assert record["quality"]["content"].startswith("Dear team")
        assert record["repetition"] == 2
        assert record["request"]["request_path"] == "/tmp/request.json"
        assert record["policy"]["within_timeout"] is False
        assert record["metrics"]["cache_hits"] == 2
        assert record["metrics"]["cache_misses"] == 1
        assert record["metrics"]["tokens_saved"] == 40
        assert record["metrics"]["metal"]["metal_active_gb"] == pytest.approx(42.0)
        assert record["metrics"]["metal"]["cache_type"] == "paged"

    def test_run_bench_serve_workload_repeats_each_case(self, tmp_path, monkeypatch):
        workload_file = tmp_path / "workload.json"
        workload_file.write_text(
            json.dumps(
                {
                    "name": "repeat-contract",
                    "cases": [
                        {
                            "id": "case-a",
                            "messages": [{"role": "user", "content": "A"}],
                        },
                        {
                            "id": "case-b",
                            "messages": [{"role": "user", "content": "B"}],
                        },
                    ],
                }
            )
        )
        observed = []

        async def fake_auto_detect_runtime(client, url):
            return {"model_id": "test-model"}

        def fake_detect_hardware_fingerprint():
            return {"chip": "test"}

        async def fake_run_workload_case(*args, **kwargs):
            observed.append((kwargs["case"].case_id, kwargs["repetition"]))
            return {
                "run_id": kwargs["run_id"],
                "timestamp": kwargs["timestamp"],
                "workload": kwargs["workload"].name,
                "case_id": kwargs["case"].case_id,
                "repetition": kwargs["repetition"],
                "tags": [],
                "model_id": kwargs["model"],
                "runtime": kwargs["runtime"],
                "hardware": kwargs["hardware"],
                "request": {},
                "policy": {"within_timeout": None},
                "metrics": {
                    "e2e_latency_ms": 100.0 + kwargs["repetition"],
                    "ttft_ms": 10.0,
                    "gen_tps": 20.0,
                },
                "quality": {"ok": True, "content_chars": 20},
                "ok": True,
            }

        monkeypatch.setattr(
            "vllm_mlx.bench_serve.auto_detect_runtime", fake_auto_detect_runtime
        )
        monkeypatch.setattr(
            "vllm_mlx.bench_serve.detect_hardware_fingerprint",
            fake_detect_hardware_fingerprint,
        )
        monkeypatch.setattr(
            "vllm_mlx.bench_serve.run_workload_case", fake_run_workload_case
        )

        payload = asyncio.run(
            run_bench_serve_workload(
                url="http://server",
                workload_path=str(workload_file),
                output_path=str(tmp_path / "results.json"),
                output_format="json",
                scrape=False,
                request_timeout_s=None,
                repetitions=3,
            )
        )

        assert observed == [
            ("case-a", 0),
            ("case-b", 0),
            ("case-a", 1),
            ("case-b", 1),
            ("case-a", 2),
            ("case-b", 2),
        ]
        assert len(payload["results"]) == 6
        assert payload["workload"]["repetitions"] == 3
        assert payload["summary"]["case_summaries"]["case-a"]["sample_count"] == 3


# ---------------------------------------------------------------------------
# TestSummaryStats  (Task 5)
# ---------------------------------------------------------------------------


class TestSummaryStats:
    """Unit tests for compute_summary_stats()."""

    def test_basic_summary(self):
        stats = compute_summary_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert stats["mean"] == pytest.approx(30.0)
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(50.0)
        assert stats["p50"] == pytest.approx(30.0)

    def test_single_value(self):
        stats = compute_summary_stats([42.0])
        assert stats["mean"] == pytest.approx(42.0)
        assert stats["stddev"] == pytest.approx(0.0)
        assert stats["min"] == pytest.approx(42.0)
        assert stats["max"] == pytest.approx(42.0)
        assert stats["p50"] == pytest.approx(42.0)
        assert stats["p95"] == pytest.approx(42.0)
        assert stats["p99"] == pytest.approx(42.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_summary_stats([])


# ---------------------------------------------------------------------------
# Formatter helpers  (Task 6)
# ---------------------------------------------------------------------------


def _make_sample_result(**overrides) -> BenchServeResult:
    """Return a BenchServeResult with realistic defaults, accepting overrides."""
    defaults = dict(
        run_id="run-abc123",
        timestamp="2026-04-17T10:00:00Z",
        tag="ci",
        chip="Apple M3 Max",
        gpu_cores=40,
        memory_gb=128.0,
        bandwidth_gbs=400.0,
        os_version="macOS 15.4",
        model_id="mlx-community/gemma-3-4b-it-4bit",
        model_type="llm",
        engine_type="vllm-mlx",
        mtp_enabled=False,
        specprefill=False,
        kv_quant="",
        cache_type="paged",
        prompt_set="short",
        concurrency=4,
        max_tokens=256,
        enable_thinking=None,
        extra_body="",
        repetition=0,
        prompt_tokens=32,
        ttft_ms=85.3,
        tpot_ms=12.4,
        e2e_latency_ms=420.7,
        gen_tps=80.6,
        prompt_tps=310.2,
        throughput_tps=75.1,
        requests_per_s=2.4,
        metal_active_gb=12.5,
        metal_peak_gb=14.0,
        metal_cache_gb=2.0,
        cache_hits=10,
        cache_misses=2,
        cache_hit_rate=0.833,
        tokens_saved=320,
        validated=True,
    )
    defaults.update(overrides)
    return BenchServeResult(**defaults)


def _make_sample_workload_payload() -> dict:
    return {
        "run_id": "run123",
        "timestamp": "2026-04-23T00:00:00+00:00",
        "summary": {"passed": True},
        "results": [
            {
                "run_id": "run123",
                "timestamp": "2026-04-23T00:00:00+00:00",
                "workload": "writing-contract",
                "case_id": "resume-smoke",
                "repetition": 0,
                "tags": ["resume", "quality"],
                "model_id": "test-model",
                "runtime": {
                    "engine_type": "mllm",
                    "model_type": "mllm",
                    "mtp_enabled": False,
                    "specprefill": False,
                    "kv_quant": "",
                    "cache_type": "paged",
                },
                "hardware": {
                    "chip": "M4 Ultra",
                    "memory_gb": 256.0,
                    "os_version": "macOS-test",
                },
                "request": {
                    "max_tokens": 32768,
                    "enable_thinking": True,
                    "extra_body": {"temperature": 0.6},
                },
                "policy": {"timeout_ms": 180000, "within_timeout": True},
                "metrics": {
                    "ttft_ms": 25.0,
                    "tpot_ms": 3.0,
                    "e2e_latency_ms": 2500.0,
                    "gen_tps": 12.5,
                    "prompt_tps": 500.0,
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "cache_hits": 2,
                    "cache_misses": 1,
                    "tokens_saved": 40,
                    "metal": {
                        "metal_active_gb": 42.0,
                        "metal_peak_gb": 45.0,
                        "metal_cache_gb": 4.0,
                    },
                },
                "quality": {
                    "ok": True,
                    "issues": [],
                    "finish_reason": "stop",
                    "content_chars": 512,
                    "content_preview": "Clean artifact",
                },
                "ok": True,
            }
        ],
    }


# ---------------------------------------------------------------------------
# TestFormatters  (Task 6)
# ---------------------------------------------------------------------------


class TestFormatters:
    """Unit tests for output formatter functions."""

    def test_format_table_not_empty(self):
        r = _make_sample_result()
        output = format_table([r])
        assert len(output) > 0
        assert r.model_id in output or str(r.ttft_ms) in output or "85" in output

    def test_format_json_roundtrip(self):
        r1 = _make_sample_result(run_id="r1")
        r2 = _make_sample_result(run_id="r2", concurrency=8)
        output = format_json([r1, r2])
        parsed = json.loads(output)
        assert len(parsed) == 2
        assert parsed[0]["run_id"] == "r1"
        assert parsed[1]["run_id"] == "r2"

    def test_format_csv_parseable(self):
        r = _make_sample_result()
        output = format_csv([r])
        reader = csv.DictReader(output.splitlines())
        rows = list(reader)
        assert len(rows) == 1
        assert "model_id" in rows[0]
        assert "ttft_ms" in rows[0]
        assert rows[0]["model_id"] == r.model_id

    def test_format_sql_valid(self):
        r = _make_sample_result()
        output = format_sql([r])
        assert "CREATE TABLE" in output
        assert "INSERT" in output
        assert "bench_serve" in output

    def test_format_sql_escapes_quotes(self):
        r = _make_sample_result(tag="it's a test")
        output = format_sql([r])
        assert "it''s a test" in output

    def test_format_sql_handles_nan_inf(self):
        r = _make_sample_result(ttft_ms=float("nan"), gen_tps=float("inf"))
        output = format_sql([r])
        # NaN and Inf should become NULL, not invalid SQL literals
        assert "nan" not in output.lower().split("'")[-1]  # not outside strings
        assert "inf" not in output.lower().split("'")[-1]
        assert "NULL" in output

    def test_result_columns_match_dataclass(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(BenchServeResult)}
        assert set(RESULT_COLUMNS) == field_names

    def test_format_workload_csv_parseable(self):
        output = format_workload_csv(_make_sample_workload_payload())
        rows = list(csv.DictReader(output.splitlines()))
        assert len(rows) == 1
        assert rows[0]["case_id"] == "resume-smoke"
        assert rows[0]["repetition"] == "0"
        assert rows[0]["model_id"] == "test-model"
        assert rows[0]["request_extra_body"] == '{"temperature": 0.6}'

    def test_format_workload_sql_valid(self):
        output = format_workload_sql(_make_sample_workload_payload())
        assert "CREATE TABLE IF NOT EXISTS bench_serve_workload" in output
        assert "case_id TEXT, repetition INTEGER, tags TEXT" in output
        assert "INSERT INTO bench_serve_workload" in output
        assert "resume-smoke" in output

    def test_format_workload_payload_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="Unsupported workload output format"):
            format_workload_payload(_make_sample_workload_payload(), "xml")


# ---------------------------------------------------------------------------
# TestBenchServeIntegration  (Task 8)
# ---------------------------------------------------------------------------


BENCH_SERVE_URL = os.environ.get("BENCH_SERVE_TEST_URL")


@pytest.mark.skipif(
    BENCH_SERVE_URL is None,
    reason="Set BENCH_SERVE_TEST_URL to run integration tests",
)
class TestBenchServeIntegration:
    """Integration tests requiring a running vllm-mlx server."""

    def test_smoke_run(self):
        """End-to-end: run bench-serve with minimal config against a real server."""
        from vllm_mlx.bench_serve import run_bench_serve

        results = asyncio.run(
            run_bench_serve(
                url=BENCH_SERVE_URL,
                prompt_sets=["short"],
                concurrencies=[1],
                repetitions=1,
                warmup=0,
                max_tokens=32,
                fmt="json",
                scrape=False,
            )
        )
        assert len(results) == 1
        r = results[0]
        assert r.ttft_ms > 0
        assert r.gen_tps > 0
        assert r.model_id != ""
        assert r.validated is True

    def test_sql_output_is_valid(self):
        """Verify SQL output contains CREATE TABLE and INSERT."""
        from vllm_mlx.bench_serve import format_sql, run_bench_serve

        results = asyncio.run(
            run_bench_serve(
                url=BENCH_SERVE_URL,
                prompt_sets=["short"],
                concurrencies=[1],
                repetitions=1,
                warmup=0,
                max_tokens=16,
                fmt="table",
                scrape=False,
            )
        )
        sql = format_sql(results)
        assert "CREATE TABLE IF NOT EXISTS bench_serve" in sql
        assert "INSERT INTO bench_serve" in sql
