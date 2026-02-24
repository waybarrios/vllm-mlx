# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.cli_args module."""

import argparse
import warnings

import pytest

from vllm_mlx.cli_args import (
    add_all_serve_args,
    build_scheduler_config,
    rebuild_server_args_from_namespace,
    resolve_prefix_cache,
    validate_serve_args,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser(*, positional_model: bool = False):
    """Create a fresh parser with the standard serve args."""
    parser = argparse.ArgumentParser()
    add_all_serve_args(parser, positional_model=positional_model)
    return parser


def _parse(argv, *, positional_model: bool = False):
    """Parse *argv* through a standard parser and return the namespace."""
    parser = _make_parser(positional_model=positional_model)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# build_scheduler_config
# ---------------------------------------------------------------------------


def test_build_scheduler_config_defaults():
    """Default CLI args should produce a SchedulerConfig with documented defaults."""
    args = _parse(["--model", "test-model"])
    cfg = build_scheduler_config(args)

    assert cfg.max_num_seqs == 256
    assert cfg.prefill_batch_size == 8
    assert cfg.completion_batch_size == 32
    assert cfg.enable_prefix_cache is True
    assert cfg.prefix_cache_size == 100
    assert cfg.use_memory_aware_cache is True
    assert cfg.cache_memory_mb is None
    assert cfg.cache_memory_percent == 0.20
    assert cfg.kv_cache_quantization is False
    assert cfg.use_paged_cache is False
    assert cfg.chunked_prefill_tokens == 0
    assert cfg.speculative_method is None


def test_build_scheduler_config_custom():
    """Custom CLI values should propagate into SchedulerConfig fields."""
    args = _parse(
        [
            "--model",
            "X",
            "--continuous-batching",
            "--max-num-seqs",
            "128",
            "--prefill-batch-size",
            "16",
            "--completion-batch-size",
            "64",
            "--no-prefix-cache",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            "4",
            "--use-paged-cache",
            "--paged-cache-block-size",
            "128",
            "--chunked-prefill-tokens",
            "512",
            "--speculative-method",
            "ngram",
            "--num-speculative-tokens",
            "5",
        ]
    )
    cfg = build_scheduler_config(args)

    assert cfg.max_num_seqs == 128
    assert cfg.prefill_batch_size == 16
    assert cfg.completion_batch_size == 64
    assert cfg.enable_prefix_cache is False
    assert cfg.kv_cache_quantization is True
    assert cfg.kv_cache_quantization_bits == 4
    assert cfg.use_paged_cache is True
    assert cfg.paged_cache_block_size == 128
    assert cfg.chunked_prefill_tokens == 512
    assert cfg.speculative_method == "ngram"
    assert cfg.num_speculative_tokens == 5


# ---------------------------------------------------------------------------
# resolve_prefix_cache
# ---------------------------------------------------------------------------


def test_resolve_prefix_cache_default():
    """Without any prefix-cache flags the default should be True."""
    args = _parse(["--model", "m"])
    assert resolve_prefix_cache(args) is True


def test_resolve_prefix_cache_no_prefix():
    """--no-prefix-cache should disable prefix caching."""
    args = _parse(["--model", "m", "--no-prefix-cache"])
    assert resolve_prefix_cache(args) is False


def test_resolve_prefix_cache_disable_legacy():
    """--disable-prefix-cache should disable caching and emit a DeprecationWarning."""
    args = _parse(["--model", "m", "--disable-prefix-cache"])
    with pytest.warns(DeprecationWarning, match="--no-prefix-cache"):
        result = resolve_prefix_cache(args)
    assert result is False


def test_resolve_prefix_cache_no_prefix_overrides_enable():
    """--no-prefix-cache should win over --enable-prefix-cache."""
    args = _parse(["--model", "m", "--no-prefix-cache", "--enable-prefix-cache"])
    assert resolve_prefix_cache(args) is False


# ---------------------------------------------------------------------------
# validate_serve_args — tool calling
# ---------------------------------------------------------------------------


def test_validate_tool_choice_without_parser():
    """--enable-auto-tool-choice without --tool-call-parser should exit."""
    args = _parse(["--model", "m", "--enable-auto-tool-choice"])
    with pytest.raises(SystemExit):
        validate_serve_args(args)


def test_validate_tool_parser_without_choice():
    """--tool-call-parser without --enable-auto-tool-choice should warn."""
    args = _parse(["--model", "m", "--tool-call-parser", "mistral"])
    with pytest.warns(match="no effect"):
        validate_serve_args(args)


# ---------------------------------------------------------------------------
# validate_serve_args — speculative decoding
# ---------------------------------------------------------------------------


def test_validate_speculative_without_batching():
    """--speculative-method without --continuous-batching should exit."""
    args = _parse(["--model", "m", "--speculative-method", "ngram"])
    with pytest.raises(SystemExit):
        validate_serve_args(args)


# ---------------------------------------------------------------------------
# validate_serve_args — cache / timeout boundaries
# ---------------------------------------------------------------------------


def test_validate_cache_memory_percent_invalid():
    """--cache-memory-percent outside (0, 1] should exit."""
    args_zero = _parse(["--model", "m", "--cache-memory-percent", "0"])
    with pytest.raises(SystemExit):
        validate_serve_args(args_zero)

    args_over = _parse(["--model", "m", "--cache-memory-percent", "1.5"])
    with pytest.raises(SystemExit):
        validate_serve_args(args_over)


def test_validate_timeout_invalid():
    """--timeout <= 0 should exit."""
    args_zero = _parse(["--model", "m", "--timeout", "0"])
    with pytest.raises(SystemExit):
        validate_serve_args(args_zero)

    args_neg = _parse(["--model", "m", "--timeout", "-1"])
    with pytest.raises(SystemExit):
        validate_serve_args(args_neg)


# ---------------------------------------------------------------------------
# rebuild_server_args_from_namespace
# ---------------------------------------------------------------------------


def test_rebuild_server_args_roundtrip():
    """Rebuilt args, when re-parsed, should produce the same values."""
    original_argv = [
        "--model",
        "test-model",
        "--continuous-batching",
        "--max-num-seqs",
        "128",
        "--kv-cache-quantization",
        "--speculative-method",
        "ngram",
        "--api-key",
        "secret",
        "--port",
        "9000",
    ]
    args1 = _parse(original_argv)
    rebuilt = rebuild_server_args_from_namespace(args1)

    # Re-parse with a fresh parser (positional_model=False for --model style)
    args2 = _parse(rebuilt)

    assert args2.model == args1.model
    assert args2.continuous_batching == args1.continuous_batching
    assert args2.max_num_seqs == args1.max_num_seqs
    assert args2.kv_cache_quantization == args1.kv_cache_quantization
    assert args2.speculative_method == args1.speculative_method
    assert args2.api_key == args1.api_key
    assert args2.port == args1.port


def test_rebuild_forwards_legacy_disable_prefix_cache():
    """Legacy --disable-prefix-cache must be forwarded as --no-prefix-cache."""
    args = _parse(["--model", "m", "--disable-prefix-cache"])
    rebuilt = rebuild_server_args_from_namespace(args)

    assert "--no-prefix-cache" in rebuilt
    assert "--disable-prefix-cache" not in rebuilt

    # Re-parse and verify prefix cache is disabled
    args2 = _parse(rebuilt)
    assert args2.no_prefix_cache is True


# ---------------------------------------------------------------------------
# Parser help / structure
# ---------------------------------------------------------------------------


def test_add_all_serve_args_help_output():
    """The help text should contain key argument names."""
    parser = _make_parser(positional_model=False)
    help_text = parser.format_help()

    for expected in (
        "--model",
        "--continuous-batching",
        "--kv-cache-quantization",
        "--speculative-method",
        "--no-prefix-cache",
        "--max-num-seqs",
        "--api-key",
    ):
        assert expected in help_text, f"{expected!r} not found in parser help"


def test_server_and_cli_parser_equivalence():
    """Server-style (--model) and CLI-style (positional model) parsers should agree."""
    # Server-style parser (--model flag)
    args_server = _parse(
        [
            "--model",
            "test-model",
            "--continuous-batching",
            "--max-num-seqs",
            "64",
            "--kv-cache-quantization",
        ],
        positional_model=False,
    )

    # CLI-style parser (positional model)
    args_cli = _parse(
        [
            "test-model",
            "--continuous-batching",
            "--max-num-seqs",
            "64",
            "--kv-cache-quantization",
        ],
        positional_model=True,
    )

    assert args_server.model == args_cli.model == "test-model"
    assert args_server.continuous_batching == args_cli.continuous_batching is True
    assert args_server.max_num_seqs == args_cli.max_num_seqs == 64
    assert args_server.kv_cache_quantization == args_cli.kv_cache_quantization is True
