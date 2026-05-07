# SPDX-License-Identifier: Apache-2.0
"""
Targeted regression tests for the bench-serve workload runner.

The existing ``test_bench_serve.py`` already covers happy paths for workload
loading, sweep expansion, formatters, runner end-to-end, and basic quality
checks. Issue #499 highlighted that the load/validate/run/report code paths
in ``vllm_mlx/bench_serve.py`` are dense and worth pinning further. This
module fills the remaining corners around the areas the issue calls out:

- ``load_workload`` validation errors. Every guard clause that rejects a
  malformed workload JSON, plus tag-string normalisation.
- ``load_workload`` default merging. ``max_tokens``, ``enable_thinking``,
  and the workload-name fallback to filename stem.
- Streaming tool-call accumulation. ``accumulate_tool_calls`` and
  ``finalize_tool_calls`` chunk-boundary behaviour and ordering.
- ``validate_quality_checks`` diagnostics. Tool-call argument validation
  edge cases (invalid JSON, non-object, missing keys, count mismatch) and
  the combination of ``no_tool_calls`` with content-length checks.
- Artifact schema compatibility. ``summarize_workload_results``,
  ``format_workload_json``, ``format_workload_csv``, and
  ``format_workload_table`` are pinned to a stable core set of keys and
  columns so an accidental rename or deletion during refactor is caught.

Brittleness note: assertions on ``match=`` strings and diagnostic
substrings deliberately use short, stable anchors instead of full
sentences. The contract is the diagnostic *intent* (which subject the
error names, which check failed), not the exact wording.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vllm_mlx.bench_serve import (
    accumulate_tool_calls,
    finalize_tool_calls,
    format_workload_csv,
    format_workload_json,
    format_workload_table,
    load_workload,
    summarize_workload_results,
    validate_quality_checks,
)

# ---------------------------------------------------------------------------
# load_workload — validation guard clauses
# ---------------------------------------------------------------------------


class TestLoadWorkloadValidation:
    def _write(self, tmp_path: Path, payload) -> Path:
        f = tmp_path / "workload.json"
        f.write_text(json.dumps(payload))
        return f

    def test_root_must_be_object(self, tmp_path: Path):
        f = tmp_path / "workload.json"
        f.write_text(json.dumps(["not", "an", "object"]))
        with pytest.raises(ValueError, match=r"root.*JSON|JSON.*root"):
            load_workload(f)

    def test_empty_cases_list_rejected(self, tmp_path: Path):
        f = self._write(tmp_path, {"cases": []})
        with pytest.raises(ValueError, match="cases"):
            load_workload(f)

    def test_missing_cases_key_rejected(self, tmp_path: Path):
        f = self._write(tmp_path, {"defaults": {}})
        with pytest.raises(ValueError, match="cases"):
            load_workload(f)

    def test_defaults_must_be_object(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {
                "defaults": "not-an-object",
                "cases": [{"id": "a", "messages": [{"role": "user", "content": "hi"}]}],
            },
        )
        with pytest.raises(ValueError, match="defaults"):
            load_workload(f)

    def test_case_must_be_object(self, tmp_path: Path):
        f = self._write(tmp_path, {"cases": ["not-an-object"]})
        # ``case`` alone would also match the list-level "non-empty cases"
        # error; pin to the per-item form.
        with pytest.raises(ValueError, match=r"case.*must be"):
            load_workload(f)

    def test_extra_body_invalid_type_rejected(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {
                "cases": [
                    {
                        "id": "a",
                        "messages": [{"role": "user", "content": "hi"}],
                        "extra_body": "not-an-object",
                    }
                ]
            },
        )
        # ``dict.update`` on a string raises ValueError with a "sequence"
        # message before reaching the explicit ``extra_body must be an
        # object`` check. Anchor on either path so the test still catches
        # a regression that takes the explicit branch.
        with pytest.raises((ValueError, TypeError), match=r"extra_body|sequence|dict"):
            load_workload(f)

    def test_tags_string_is_normalised_to_list(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {
                "cases": [
                    {
                        "id": "a",
                        "messages": [{"role": "user", "content": "hi"}],
                        "tags": "single-tag",
                    }
                ]
            },
        )
        workload = load_workload(f)
        assert workload.cases[0].tags == ("single-tag",)

    def test_tags_invalid_type_rejected(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {
                "cases": [
                    {
                        "id": "a",
                        "messages": [{"role": "user", "content": "hi"}],
                        "tags": 42,
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match="tags"):
            load_workload(f)

    def test_checks_non_dict_truthy_rejected_as_value_error(self, tmp_path: Path):
        # Before this PR, a non-dict truthy ``checks`` value (string or
        # list) crashed with a cryptic ``AttributeError`` from
        # ``checks.items()`` deep inside the loader. Now ``_merge_case_checks``
        # validates the type and raises a case-scoped ``ValueError``.
        f = self._write(
            tmp_path,
            {
                "cases": [
                    {
                        "id": "a",
                        "messages": [{"role": "user", "content": "hi"}],
                        "checks": "not-an-object",
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match="checks"):
            load_workload(f)


# ---------------------------------------------------------------------------
# load_workload — default merging
# ---------------------------------------------------------------------------


class TestLoadWorkloadDefaultMerging:
    """Pin the default-merging behaviours not already covered by
    ``tests/test_bench_serve.py``. The existing suite covers basic
    ``max_tokens`` / ``enable_thinking`` / ``policy_timeout_ms`` propagation
    from defaults (see ``test_load_workload_with_defaults``) and
    ``checks`` list-merging. The corners left open are: (a) case-level
    values winning over defaults, and (b) workload name falling back to
    the filename stem when no ``name`` field is set.
    """

    def _write(self, tmp_path: Path, payload, *, name: str = "workload.json") -> Path:
        f = tmp_path / name
        f.write_text(json.dumps(payload))
        return f

    def test_case_value_wins_over_default_max_tokens(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {
                "defaults": {"max_tokens": 64},
                "cases": [
                    {
                        "id": "a",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 200,
                    }
                ],
            },
        )
        workload = load_workload(f)
        assert workload.cases[0].max_tokens == 200

    def test_workload_name_defaults_to_filename_stem(self, tmp_path: Path):
        f = self._write(
            tmp_path,
            {"cases": [{"id": "a", "messages": [{"role": "user", "content": "hi"}]}]},
            name="my-suite.json",
        )
        workload = load_workload(f)
        assert workload.name == "my-suite"


# ---------------------------------------------------------------------------
# Streaming tool-call accumulation
# ---------------------------------------------------------------------------


class TestStreamingToolCallAccumulation:
    def test_concatenates_name_and_arguments_across_deltas(self):
        acc: dict[int, dict] = {}
        # First delta: id + name fragment.
        accumulate_tool_calls(
            acc,
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_", "arguments": '{"city":'},
                }
            ],
        )
        # Second delta: rest of name + rest of arguments.
        accumulate_tool_calls(
            acc,
            [
                {
                    "index": 0,
                    "function": {"name": "weather", "arguments": '"Tokyo"}'},
                }
            ],
        )
        finalised = finalize_tool_calls(acc)
        assert len(finalised) == 1
        tc = finalised[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}

    def test_finalize_returns_index_sorted_even_when_inserted_out_of_order(self):
        acc: dict[int, dict] = {}
        # Indices arrive out of order: 2, 0, 1.
        accumulate_tool_calls(
            acc,
            [{"index": 2, "id": "c", "function": {"name": "third"}}],
        )
        accumulate_tool_calls(
            acc,
            [{"index": 0, "id": "a", "function": {"name": "first"}}],
        )
        accumulate_tool_calls(
            acc,
            [{"index": 1, "id": "b", "function": {"name": "second"}}],
        )
        finalised = finalize_tool_calls(acc)
        assert [tc["function"]["name"] for tc in finalised] == [
            "first",
            "second",
            "third",
        ]

    def test_id_set_on_first_delta_is_preserved_when_later_delta_omits_id(self):
        acc: dict[int, dict] = {}
        accumulate_tool_calls(
            acc,
            [{"index": 0, "id": "call_X", "function": {"name": "f"}}],
        )
        # Later delta omits the id (only sends argument fragment).
        accumulate_tool_calls(
            acc,
            [{"index": 0, "function": {"arguments": "{}"}}],
        )
        assert acc[0]["id"] == "call_X"

    def test_default_index_is_zero_when_omitted(self):
        """OpenAI's spec says ``index`` is required, but mlx-lm style streams
        sometimes omit it on the first delta. Accumulator must default to 0
        rather than raise."""
        acc: dict[int, dict] = {}
        accumulate_tool_calls(
            acc,
            [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
        )
        assert 0 in acc
        assert acc[0]["function"]["name"] == "f"


# ---------------------------------------------------------------------------
# validate_quality_checks — diagnostics for tool-call argument checks
# ---------------------------------------------------------------------------


class TestQualityCheckDiagnostics:
    def test_tool_call_count_mismatch_reports_actual_count(self):
        ok, issues = validate_quality_checks(
            finish_reason="stop",
            content="ignored",
            checks={"tool_call_count": 2},
            tool_calls=[{"function": {"name": "f", "arguments": "{}"}}],
        )
        assert not ok
        # Anchor on the check name and the two relevant counts. We don't
        # pin the exact wording around "expected"/"got" so a future
        # cleanup of the diagnostic phrasing doesn't break the test.
        assert any(
            "tool_call_count" in issue and "1" in issue and "2" in issue
            for issue in issues
        )

    def test_tool_call_args_invalid_json_reports_issue(self):
        ok, issues = validate_quality_checks(
            finish_reason="stop",
            content="ignored",
            checks={"tool_call_args_required_keys": {"f": ["x"]}},
            tool_calls=[{"function": {"name": "f", "arguments": "{not-json"}}],
        )
        assert not ok
        assert any("invalid JSON" in issue for issue in issues)

    def test_tool_call_args_non_object_reports_issue(self):
        ok, issues = validate_quality_checks(
            finish_reason="stop",
            content="ignored",
            checks={"tool_call_args_required_keys": {"f": ["x"]}},
            tool_calls=[{"function": {"name": "f", "arguments": "[1, 2, 3]"}}],
        )
        assert not ok
        assert any("not an object" in issue for issue in issues)

    def test_tool_call_args_missing_keys_lists_what_is_missing(self):
        ok, issues = validate_quality_checks(
            finish_reason="stop",
            content="ignored",
            checks={"tool_call_args_required_keys": {"f": ["a", "b", "c"]}},
            tool_calls=[{"function": {"name": "f", "arguments": '{"a": 1, "x": 2}'}}],
        )
        assert not ok
        # The diagnostic must name the missing keys so an operator can fix
        # the prompt or the check, not just say "something is missing".
        # Find the issue that talks about missing keys, then assert b/c are
        # listed and the present key 'a' is not. Quoting matters: the keys
        # are emitted as Python repr so we anchor on "'a'" to avoid
        # matching the letter 'a' inside other words.
        missing_issue = next((i for i in issues if "missing" in i), None)
        assert missing_issue is not None, f"no missing-keys diagnostic in {issues}"
        assert "'b'" in missing_issue and "'c'" in missing_issue
        assert "'a'" not in missing_issue

    def test_no_tool_calls_combines_with_other_checks(self):
        ok, issues = validate_quality_checks(
            finish_reason="stop",
            content="hi",
            checks={"no_tool_calls": True, "min_chars": 100},
            tool_calls=[{"function": {"name": "f", "arguments": "{}"}}],
        )
        assert not ok
        # Both checks fail; both must surface so the operator sees the full
        # picture rather than chasing them one at a time.
        joined = " ".join(issues)
        assert "no_tool_calls" in joined
        assert "min_chars" in joined

    def test_finish_reason_list_accepts_any_member(self):
        # finish_reason="length" is treated as truncation by the basic check
        # in validate_response, so we exercise a non-truncation alternative
        # (tool_calls) that the test should accept when present in the list.
        ok, issues = validate_quality_checks(
            finish_reason="tool_calls",
            content="",
            checks={"finish_reason": ["stop", "tool_calls"]},
            tool_calls=[{"function": {"name": "f", "arguments": "{}"}}],
        )
        assert ok, issues

    def test_finish_reason_string_form_rejects_others(self):
        # Single-string form: only "stop" is allowed; finish_reason="tool_calls"
        # must surface as an explicit issue, not a generic basic-check failure.
        ok, issues = validate_quality_checks(
            finish_reason="tool_calls",
            content="",
            checks={"finish_reason": "stop"},
            tool_calls=[{"function": {"name": "f", "arguments": "{}"}}],
        )
        assert not ok
        # Anchor on the check name plus the rejected value. Wording around
        # how the rejection is phrased ("not in", "not allowed", etc.) is
        # not part of the contract.
        assert any(
            "finish_reason" in issue and "tool_calls" in issue for issue in issues
        )


# ---------------------------------------------------------------------------
# Artifact schema compatibility
# ---------------------------------------------------------------------------


def _make_record(*, case_id: str = "a", quality_ok: bool = True) -> dict:
    """Minimal but realistic workload record for schema/format tests.

    Mirrors the shape ``run_workload_case`` produces (see
    ``vllm_mlx.bench_serve.run_workload_case`` for the source-of-truth
    record builder). Kept local to this module so changes to the runner
    don't accidentally hide schema drift in the formatters.
    """
    return {
        "run_id": "run-x",
        "timestamp": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:00Z",
        "workload": "demo",
        "case_id": case_id,
        "repetition": 0,
        "tags": [],
        "model_id": "test/model",
        "runtime": {
            "engine_type": "test",
            "model_type": "tiny",
            "mtp_enabled": False,
            "specprefill": False,
            "kv_quant": "",
            "cache_type": "paged",
        },
        "hardware": {
            "chip": "M0",
            "memory_gb": 1.0,
            "os_version": "darwin-test",
        },
        "request": {
            "max_tokens": 128,
            "request_path": None,
            "enable_thinking": False,
            "extra_body": {},
            "message_count": 1,
        },
        "policy": {"timeout_ms": None, "within_timeout": None},
        "cache_reset": {"attempted": False},
        "metrics": {
            "ttft_ms": 1.0,
            "tpot_ms": 1.0,
            "e2e_latency_ms": 10.0,
            "gen_tps": 10.0,
            "prompt_tps": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0,
            "metal": {
                "metal_active_gb": 0.0,
                "metal_peak_gb": 0.0,
                "metal_cache_gb": 0.0,
            },
        },
        "quality": {
            "ok": quality_ok,
            "issues": [],
            "finish_reason": "stop",
            "content_chars": 10,
            "content_preview": "hi",
        },
        "tool_calls": None,
        "ok": quality_ok,
    }


class TestArtifactSchemaCompatibility:
    """Pin a stable core set of keys/columns in the workload artifacts so a
    refactor cannot silently drop a field that downstream operators or
    release qualification dashboards depend on. The contract is
    *core-keys-must-exist*, not *exact match*: adding new fields is fine,
    deleting or renaming a core field breaks the suite.
    """

    def test_summarize_results_has_core_top_level_keys(self):
        summary = summarize_workload_results([_make_record()])
        core = {
            "case_count",
            "unique_case_count",
            "repetition_count",
            "passed",
            "failure_count",
            "failure_rate",
            "quality_passed",
            "quality_failure_count",
            "policy_timeout_passed",
            "policy_timeout_failure_count",
            "latency_ms",
            "ttft_ms",
            "gen_tps",
            "case_summaries",
        }
        missing = core - summary.keys()
        assert not missing, f"summary lost core keys: {sorted(missing)}"

    def test_summarize_results_per_case_has_core_keys(self):
        summary = summarize_workload_results([_make_record(case_id="resume")])
        case = summary["case_summaries"]["resume"]
        core = {
            "sample_count",
            "repetitions",
            "passed",
            "failure_count",
            "failure_rate",
            "policy_timeout_passed",
            "policy_timeout_failure_count",
            "latency_ms",
            "ttft_ms",
            "gen_tps",
            "content_chars",
        }
        missing = core - case.keys()
        assert not missing, f"per-case summary lost core keys: {sorted(missing)}"

    def test_workload_table_renders_core_columns(self):
        payload = {"results": [_make_record(case_id="resume")]}
        rendered = format_workload_table(payload)
        # Anchor on the column headers that downstream qualification
        # dashboards rely on. New columns are fine; deleting any of these
        # is a contract break.
        for column in (
            "case_id",
            "repetition",
            "ttft_ms",
            "gen_tps",
            "e2e_latency_ms",
            "quality_ok",
        ):
            assert column in rendered, f"table missing column: {column}"
        assert "resume" in rendered

    def test_workload_csv_rows_have_core_columns(self):
        payload = {"results": [_make_record(case_id="resume")]}
        output = format_workload_csv(payload)
        rows = list(csv.DictReader(output.splitlines()))
        assert len(rows) == 1
        for column in (
            "run_id",
            "timestamp",
            "workload",
            "case_id",
            "repetition",
            "model_id",
            "ttft_ms",
            "gen_tps",
            "e2e_latency_ms",
            "quality_ok",
            "finish_reason",
        ):
            assert column in rows[0], f"CSV missing column: {column}"
        assert rows[0]["case_id"] == "resume"

    def test_workload_json_passes_through_top_level_keys(self):
        # ``format_workload_json`` is a thin dump, so the contract is that
        # any top-level keys callers attach (workload metadata, summary,
        # results) round-trip unchanged. Pin that explicitly so a future
        # change that rewraps the payload doesn't go unnoticed.
        payload = {
            "workload": "demo",
            "summary": {"passed": True},
            "results": [_make_record()],
        }
        parsed = json.loads(format_workload_json(payload))
        assert parsed.keys() >= {"workload", "summary", "results"}
        assert parsed["results"][0]["case_id"] == "a"
        assert parsed["summary"]["passed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
