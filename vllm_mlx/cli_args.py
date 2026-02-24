# SPDX-License-Identifier: Apache-2.0
"""
Shared CLI argument definitions for vllm-mlx.

All entry points (cli.py, server.py) use these functions to register
argparse arguments, ensuring consistency.
"""

import argparse
import sys
import warnings
from typing import Optional

# ---------------------------------------------------------------------------
# Argument builder functions
# ---------------------------------------------------------------------------


def add_model_args(
    parser: argparse.ArgumentParser, *, positional: bool = False
) -> None:
    """Add model-related arguments."""
    if positional:
        parser.add_argument("model", type=str, help="Model to serve")
    else:
        parser.add_argument(
            "--model",
            type=str,
            default="mlx-community/Llama-3.2-3B-Instruct-4bit",
            help="Model to serve (default: mlx-community/Llama-3.2-3B-Instruct-4bit)",
        )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )


def add_server_args(parser: argparse.ArgumentParser) -> None:
    """Add host/port arguments."""
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")


def add_batching_args(parser: argparse.ArgumentParser) -> None:
    """Add continuous-batching and scheduler tuning arguments."""
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences (default: 256)",
    )
    parser.add_argument(
        "--prefill-batch-size",
        type=int,
        default=8,
        help="Prefill batch size (default: 8)",
    )
    parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=32,
        help="Completion batch size (default: 32)",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help=(
            "Max prefill tokens per scheduler step (0=disabled). "
            "Prevents starvation of active requests during long prefills."
        ),
    )


def add_cache_args(parser: argparse.ArgumentParser) -> None:
    """Add prefix-cache and KV-cache arguments."""
    # New standard flag
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    # Legacy flags (hidden)
    parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Maximum number of cached prefix entries (default: 100)",
    )
    parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Fixed cache memory budget in MB (default: auto-detect)",
    )
    parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache when auto-detecting (default: 0.20)",
    )
    parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache eviction",
    )

    # KV cache quantization
    parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Enable KV cache quantization for prefix cache memory reduction",
    )
    parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bit width (default: 8)",
    )
    parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum token count before quantizing a cache entry (default: 256)",
    )

    # Paged cache (experimental)
    parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged (block-aware) prefix cache (experimental)",
    )
    parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per paged-cache block (default: 64)",
    )
    parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of paged-cache blocks (default: 1000)",
    )


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    """Add generation-limit arguments."""
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum tokens per response (default: 32768)",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default sampling temperature (overrides per-request if set)",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top-p sampling value (overrides per-request if set)",
    )


def add_security_args(parser: argparse.ArgumentParser) -> None:
    """Add authentication / rate-limit / timeout arguments."""
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (default: disabled)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300.0)",
    )


def add_tool_calling_args(parser: argparse.ArgumentParser) -> None:
    """Add tool-calling arguments."""
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable automatic tool choice in chat completions",
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=[
            "auto",
            "mistral",
            "qwen",
            "qwen3_coder",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
            "glm47",
        ],
        help="Tool call parser to use (requires --enable-auto-tool-choice)",
    )


def add_reasoning_args(parser: argparse.ArgumentParser) -> None:
    """Add reasoning-parser arguments (choices loaded dynamically)."""
    from .reasoning import list_parsers

    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode. The model generates <think>...</think> "
        "reasoning before its response. A reasoning parser is auto-activated "
        "to separate thinking from content in the API response.",
    )
    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices if reasoning_choices else None,
        help="Explicit reasoning parser (implies --enable-thinking)",
    )


def add_speculative_decoding_args(parser: argparse.ArgumentParser) -> None:
    """Add speculative decoding arguments."""
    parser.add_argument(
        "--speculative-method",
        type=str,
        default=None,
        choices=["ngram", "draft_model", "mtp"],
        help="Speculative decoding method (default: disabled)",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="Number of draft tokens per speculative step (default: 3)",
    )
    parser.add_argument(
        "--spec-decode-disable-batch-size",
        type=int,
        default=None,
        help="Disable speculative decoding when batch size exceeds this value",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Path to draft model for speculative decoding (required with --speculative-method draft_model)",
    )
    parser.add_argument(
        "--mtp-model",
        type=str,
        default=None,
        help=(
            "Model to load MTP weights from (default: same as main model). "
            "Use this to load MTP weights from the original HuggingFace "
            "checkpoint when the main model is a quantized MLX conversion "
            "that doesn't include MTP weights."
        ),
    )
    parser.add_argument(
        "--spec-decode-auto-disable-threshold",
        type=float,
        default=0.4,
        help=(
            "Auto-disable speculative decoding when the recent acceptance rate "
            "drops below this threshold (default: 0.4). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--spec-decode-auto-disable-window",
        type=int,
        default=50,
        help=(
            "Number of recent speculation rounds over which to evaluate the "
            "acceptance rate for auto-disable decisions (default: 50)."
        ),
    )


def add_extra_server_args(parser: argparse.ArgumentParser) -> None:
    """Add miscellaneous server-only arguments."""
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP server configuration file",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model for /v1/embeddings endpoint",
    )


def add_all_serve_args(
    parser: argparse.ArgumentParser, *, positional_model: bool = True
) -> None:
    """Register every argument group needed by the ``serve`` command."""
    add_model_args(parser, positional=positional_model)
    add_server_args(parser)
    add_batching_args(parser)
    add_cache_args(parser)
    add_generation_args(parser)
    add_security_args(parser)
    add_tool_calling_args(parser)
    add_reasoning_args(parser)
    add_speculative_decoding_args(parser)
    add_extra_server_args(parser)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def resolve_prefix_cache(args) -> bool:
    """Resolve the effective prefix-cache enablement from CLI flags.

    Priority: ``--no-prefix-cache`` > ``--disable-prefix-cache`` > default (True).
    """
    if getattr(args, "no_prefix_cache", False):
        return False

    if getattr(args, "disable_prefix_cache", False):
        warnings.warn(
            "Use --no-prefix-cache instead of --disable-prefix-cache",
            DeprecationWarning,
            stacklevel=2,
        )
        return False

    # Default: prefix cache enabled (matches --enable-prefix-cache default=True)
    return True


def build_scheduler_config(args):
    """Build a :class:`SchedulerConfig` from parsed CLI arguments."""
    from .scheduler import SchedulerConfig

    enable_prefix_cache = resolve_prefix_cache(args)

    return SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        prefill_batch_size=args.prefill_batch_size,
        completion_batch_size=args.completion_batch_size,
        enable_prefix_cache=enable_prefix_cache,
        prefix_cache_size=args.prefix_cache_size,
        use_memory_aware_cache=not args.no_memory_aware_cache,
        cache_memory_mb=args.cache_memory_mb,
        cache_memory_percent=args.cache_memory_percent,
        kv_cache_quantization=args.kv_cache_quantization,
        kv_cache_quantization_bits=args.kv_cache_quantization_bits,
        kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
        kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        use_paged_cache=args.use_paged_cache,
        paged_cache_block_size=args.paged_cache_block_size,
        max_cache_blocks=args.max_cache_blocks,
        chunked_prefill_tokens=args.chunked_prefill_tokens,
        speculative_method=getattr(args, "speculative_method", None),
        num_speculative_tokens=getattr(args, "num_speculative_tokens", 3),
        spec_decode_disable_batch_size=getattr(
            args, "spec_decode_disable_batch_size", None
        ),
        draft_model_name=getattr(args, "draft_model", None),
        spec_decode_auto_disable_threshold=getattr(
            args, "spec_decode_auto_disable_threshold", 0.4
        ),
        spec_decode_auto_disable_window=getattr(
            args, "spec_decode_auto_disable_window", 50
        ),
        model_name=getattr(args, "model", None),
        mtp_model_name=getattr(args, "mtp_model", None),
    )


def validate_serve_args(args) -> None:
    """Validate parsed serve arguments.

    Raises :class:`SystemExit` on hard errors and emits :mod:`warnings` for
    non-fatal issues.
    """
    # Tool-calling checks
    if getattr(args, "enable_auto_tool_choice", False) and not getattr(
        args, "tool_call_parser", None
    ):
        sys.exit("Error: --enable-auto-tool-choice requires --tool-call-parser")

    if getattr(args, "tool_call_parser", None) and not getattr(
        args, "enable_auto_tool_choice", False
    ):
        warnings.warn(
            "--tool-call-parser has no effect without --enable-auto-tool-choice"
        )

    # Speculative decoding requires continuous batching
    if getattr(args, "speculative_method", None) and not getattr(
        args, "continuous_batching", False
    ):
        sys.exit("Error: --speculative-method requires --continuous-batching")

    # Draft model method requires --draft-model
    if getattr(args, "speculative_method", None) == "draft_model" and not getattr(
        args, "draft_model", None
    ):
        sys.exit("Error: --speculative-method draft_model requires --draft-model")

    # --draft-model without draft_model method is a warning
    if (
        getattr(args, "draft_model", None)
        and getattr(args, "speculative_method", None) != "draft_model"
    ):
        warnings.warn(
            "--draft-model has no effect without --speculative-method draft_model"
        )

    # Cache memory percent must be in (0, 1]
    cache_pct = getattr(args, "cache_memory_percent", 0.20)
    if not (0 < cache_pct <= 1):
        sys.exit(f"Error: --cache-memory-percent must be in (0, 1], got {cache_pct}")

    # Timeout must be positive
    timeout = getattr(args, "timeout", 300.0)
    if timeout <= 0:
        sys.exit(f"Error: --timeout must be > 0, got {timeout}")


def rebuild_server_args_from_namespace(args) -> list[str]:
    """Convert a parsed ``Namespace`` back to a CLI argument list."""
    result: list[str] = []

    # Model (positional -> --model)
    if hasattr(args, "model") and args.model:
        result.extend(["--model", args.model])

    # Server args (only if non-default)
    if getattr(args, "host", "0.0.0.0") != "0.0.0.0":
        result.extend(["--host", args.host])
    if getattr(args, "port", 8000) != 8000:
        result.extend(["--port", str(args.port)])

    # Boolean flags (store_true) — include only when True
    _BOOL_FLAGS = {
        "mllm": "--mllm",
        "continuous_batching": "--continuous-batching",
        "no_prefix_cache": "--no-prefix-cache",
        "no_memory_aware_cache": "--no-memory-aware-cache",
        "kv_cache_quantization": "--kv-cache-quantization",
        "use_paged_cache": "--use-paged-cache",
        "enable_auto_tool_choice": "--enable-auto-tool-choice",
        "enable_thinking": "--enable-thinking",
    }
    for attr, flag in _BOOL_FLAGS.items():
        if getattr(args, attr, False):
            result.append(flag)

    # Legacy --disable-prefix-cache → forward as --no-prefix-cache
    if (
        getattr(args, "disable_prefix_cache", False)
        and "--no-prefix-cache" not in result
    ):
        result.append("--no-prefix-cache")

    # Valued args — include only when they differ from the default
    _VALUED_ARGS: list[tuple[str, str, object]] = [
        ("max_num_seqs", "--max-num-seqs", 256),
        ("prefill_batch_size", "--prefill-batch-size", 8),
        ("completion_batch_size", "--completion-batch-size", 32),
        ("stream_interval", "--stream-interval", 1),
        ("chunked_prefill_tokens", "--chunked-prefill-tokens", 0),
        ("prefix_cache_size", "--prefix-cache-size", 100),
        ("cache_memory_percent", "--cache-memory-percent", 0.20),
        ("kv_cache_quantization_bits", "--kv-cache-quantization-bits", 8),
        ("kv_cache_quantization_group_size", "--kv-cache-quantization-group-size", 64),
        ("kv_cache_min_quantize_tokens", "--kv-cache-min-quantize-tokens", 256),
        ("paged_cache_block_size", "--paged-cache-block-size", 64),
        ("max_cache_blocks", "--max-cache-blocks", 1000),
        ("max_tokens", "--max-tokens", 32768),
        ("rate_limit", "--rate-limit", 0),
        ("timeout", "--timeout", 300.0),
        ("num_speculative_tokens", "--num-speculative-tokens", 3),
        (
            "spec_decode_auto_disable_threshold",
            "--spec-decode-auto-disable-threshold",
            0.4,
        ),
        ("spec_decode_auto_disable_window", "--spec-decode-auto-disable-window", 50),
    ]
    for attr, flag, default in _VALUED_ARGS:
        val = getattr(args, attr, default)
        if val != default:
            result.extend([flag, str(val)])

    # Optional string/int args — include only when not None
    _OPTIONAL_ARGS: list[tuple[str, str]] = [
        ("cache_memory_mb", "--cache-memory-mb"),
        ("api_key", "--api-key"),
        ("tool_call_parser", "--tool-call-parser"),
        ("reasoning_parser", "--reasoning-parser"),
        ("speculative_method", "--speculative-method"),
        ("spec_decode_disable_batch_size", "--spec-decode-disable-batch-size"),
        ("draft_model", "--draft-model"),
        ("mtp_model", "--mtp-model"),
        ("mcp_config", "--mcp-config"),
        ("embedding_model", "--embedding-model"),
        ("default_temperature", "--default-temperature"),
        ("default_top_p", "--default-top-p"),
    ]
    for attr, flag in _OPTIONAL_ARGS:
        val = getattr(args, attr, None)
        if val is not None:
            result.extend([flag, str(val)])

    return result
