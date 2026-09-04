from types import SimpleNamespace

import pytest


def _serve_args(**overrides):
    args = {
        "api_key": None,
        "auto_unload_idle_seconds": 0.0,
        "cache_memory_mb": None,
        "cache_memory_percent": 0.2,
        "chunked_prefill_tokens": 0,
        "continuous_batching": False,
        "default_min_p": None,
        "default_presence_penalty": None,
        "default_repetition_penalty": None,
        "default_temperature": None,
        "default_top_k": None,
        "default_top_p": None,
        "disable_prefix_cache": False,
        "download_retries": 0,
        "download_timeout": 1,
        "embedding_model": None,
        "embedding_max_length": None,
        "embedding_overflow_policy": "truncate",
        "enable_auto_tool_choice": False,
        "enable_metrics": False,
        "enable_mtp": False,
        "enable_prefix_cache": True,
        "gpu_memory_utilization": 0.9,
        "host": "127.0.0.1",
        "kv_cache_min_quantize_tokens": 256,
        "kv_cache_quantization": False,
        "kv_cache_quantization_bits": 8,
        "kv_cache_quantization_group_size": 64,
        "max_cache_blocks": 1000,
        "max_num_seqs": 32,
        "max_tokens": 16,
        "memory_budget_gb": None,
        "mcp_config": None,
        "mllm_prefill_step_size": None,
        "mllm": False,
        "model": "local-test-model",
        "models_config": None,
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
        "no_memory_aware_cache": False,
        "offline": True,
        "paged_cache_block_size": 64,
        "port": 8000,
        "prefill_batch_size": 8,
        "prefill_step_size": 512,
        "prefix_cache_size": 100,
        "prefix_trie_cache": False,
        "prefix_trie_cache_size": 32,
        "prefix_trie_cache_memory_mb": None,
        "rate_limit": 0,
        "reasoning_parser": None,
        "served_model_name": None,
        "specprefill": False,
        "specprefill_backbone_pct": 0.0,
        "specprefill_draft_model": None,
        "specprefill_keep_pct": 0.3,
        "specprefill_threshold": 8192,
        "stream_interval": 1,
        "tool_call_parser": None,
        "timeout": 300,
        "lazy_load_model": False,
        "use_paged_cache": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_serve_command_propagates_all_sampling_defaults(monkeypatch):
    from vllm_mlx import cli, server
    from vllm_mlx.utils import download

    loaded = {}

    monkeypatch.setattr(
        download, "ensure_model_downloaded", lambda *args, **kwargs: "local-test-model"
    )
    monkeypatch.setattr(
        server,
        "load_model",
        lambda *args, **kwargs: loaded.update({"args": args, "kwargs": kwargs}),
    )
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    for attr in (
        "_default_temperature",
        "_default_top_p",
        "_default_top_k",
        "_default_min_p",
        "_default_presence_penalty",
        "_default_repetition_penalty",
    ):
        monkeypatch.setattr(server, attr, None)

    cli.serve_command(
        _serve_args(
            default_temperature=0.6,
            default_top_p=0.95,
            default_top_k=20,
            default_min_p=0.0,
            default_presence_penalty=0.0,
            default_repetition_penalty=1.0,
            specprefill_backbone_pct=0.25,
        )
    )

    assert server._default_temperature == 0.6
    assert server._default_top_p == 0.95
    assert server._default_top_k == 20
    assert server._default_min_p == 0.0
    assert server._default_presence_penalty == 0.0
    assert server._default_repetition_penalty == 1.0
    assert loaded["kwargs"]["specprefill_backbone_pct"] == 0.25


def test_serve_parser_accepts_registered_step3p5_tool_parser():
    from vllm_mlx import cli

    parser = cli.create_parser()

    args = parser.parse_args(
        [
            "serve",
            "--model",
            "local-test-model",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "step3p5",
        ]
    )

    assert args.tool_call_parser == "step3p5"


def test_memory_budget_help_describes_scope_and_limitations(capsys):
    from vllm_mlx.cli import create_parser

    with pytest.raises(SystemExit):
        create_parser().parse_args(["serve", "--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert "registry manager model-weight residency budget" in help_text
    assert "not a total runtime-memory limit" in help_text
    assert "does not guarantee prevention of Metal/MLX OOM" in help_text


def test_serve_command_rejects_memory_budget_outside_registry_mode(capsys):
    from vllm_mlx import cli

    with pytest.raises(SystemExit):
        cli.serve_command(_serve_args(memory_budget_gb=8.0))

    assert "--memory-budget-gb requires --models-config" in capsys.readouterr().out


def test_serve_command_applies_memory_budget_override_end_to_end(tmp_path, monkeypatch):
    from vllm_mlx import cli, server

    config_path = tmp_path / "models.yaml"
    config_path.write_text("""
manager:
  memory_budget_gb: 4
models:
  - name: test
    path: /tmp/test-model
""".strip())
    args = cli.create_parser().parse_args(
        [
            "serve",
            "--models-config",
            str(config_path),
            "--memory-budget-gb",
            "6.5",
            "--offline",
        ]
    )
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    cli.serve_command(args)

    assert server._model_manager is not None
    assert server._model_manager.memory_budget_bytes == int(6.5 * (1024**3))
