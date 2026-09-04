# SPDX-License-Identifier: Apache-2.0
"""No-MLX regression coverage for Eagle3 server validation."""

import sys
import types
from importlib.machinery import ModuleSpec
from unittest.mock import patch

import pytest


def _install_mlx_stubs(monkeypatch) -> None:
    """Install the import-time MLX surface needed by the engine modules."""
    core = types.ModuleType("mlx.core")
    core.__spec__ = ModuleSpec("mlx.core", loader=None)

    class Array:
        pass

    core.array = Array
    core.Stream = type("Stream", (), {})
    mlx = types.ModuleType("mlx")
    mlx.__spec__ = ModuleSpec("mlx", loader=None, is_package=True)
    mlx.__path__ = []
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__spec__ = ModuleSpec("mlx_lm", loader=None, is_package=True)
    mlx_lm.__path__ = []
    generate = types.ModuleType("mlx_lm.generate")
    generate.BatchGenerator = type("BatchGenerator", (), {})
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_logits_processors = lambda *args, **kwargs: []
    sample_utils.make_sampler = lambda *args, **kwargs: None
    tokenizer_utils = types.ModuleType("mlx_lm.tokenizer_utils")
    tokenizer_utils.NaiveStreamingDetokenizer = type(
        "NaiveStreamingDetokenizer", (), {}
    )
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.ArraysCache = type("ArraysCache", (), {})
    mlx_lm.generate = generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.tokenizer_utils", tokenizer_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)


def test_load_model_rejects_eagle3_continuous_batching(monkeypatch):
    """The server must defend against Eagle3 batching outside the CLI."""
    _install_mlx_stubs(monkeypatch)
    from vllm_mlx import server

    with pytest.raises(ValueError, match="Eagle3 uses SimpleEngine"):
        server.load_model(
            "gemma4",
            use_batching=True,
            force_mllm=True,
            mllm_draft_model="eagle3-drafter",
            mllm_draft_kind="eagle3",
        )


def test_load_model_rejects_eagle3_block_size_below_two(monkeypatch):
    """Eagle3 block size one must fail before SimpleEngine construction."""
    _install_mlx_stubs(monkeypatch)
    from vllm_mlx import server

    class FakeSimpleEngine:
        is_mllm = True

        async def start(self):
            pass

    with patch.object(
        server, "SimpleEngine", return_value=FakeSimpleEngine()
    ) as simple_engine:
        with pytest.raises(ValueError, match="Eagle3.*at least 2"):
            server.load_model(
                "gemma4",
                force_mllm=True,
                mllm_draft_model="eagle3-drafter",
                mllm_draft_kind="eagle3",
                mllm_draft_block_size=1,
            )

    simple_engine.assert_not_called()


def test_load_model_accepts_eagle3_block_size_two(monkeypatch):
    """Eagle3's smallest usable block size reaches SimpleEngine unchanged."""
    _install_mlx_stubs(monkeypatch)
    from vllm_mlx import server

    class FakeSimpleEngine:
        is_mllm = True

        async def start(self):
            pass

    fake_engine = FakeSimpleEngine()
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_lifespan_active", False)

    with patch.object(
        server, "SimpleEngine", return_value=fake_engine
    ) as simple_engine:
        server.load_model(
            "gemma4",
            force_mllm=True,
            mllm_draft_model="eagle3-drafter",
            mllm_draft_kind="eagle3",
            mllm_draft_block_size=2,
        )

    assert simple_engine.call_args.kwargs["mllm_draft_block_size"] == 2


def test_load_model_keeps_mtp_block_size_one_valid(monkeypatch):
    """MTP retains its existing positive block-size contract."""
    _install_mlx_stubs(monkeypatch)
    from vllm_mlx import server

    class FakeSimpleEngine:
        is_mllm = True

        async def start(self):
            pass

    fake_engine = FakeSimpleEngine()
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_lifespan_active", False)

    with patch.object(
        server, "SimpleEngine", return_value=fake_engine
    ) as simple_engine:
        server.load_model(
            "gemma4",
            force_mllm=True,
            mllm_draft_model="assistant",
            mllm_draft_kind="mtp",
            mllm_draft_block_size=1,
        )

    assert simple_engine.call_args.kwargs["mllm_draft_block_size"] == 1


def test_simple_engine_uses_resolved_eagle3_kind_and_reports_speculative(monkeypatch):
    """SimpleEngine status must expose the loaded Eagle3 algorithm, not only config."""
    _install_mlx_stubs(monkeypatch)
    from vllm_mlx.engine.simple import SimpleEngine
    from vllm_mlx.models import mllm

    class FakeMLLM:
        def __init__(self, *args, **kwargs):
            self.draft_kind = "eagle3"

        def load(self):
            pass

    monkeypatch.setattr(mllm, "MLXMultimodalLM", FakeMLLM)
    engine = SimpleEngine(
        "target",
        force_mllm=True,
        mllm_draft_model="eagle3-drafter",
        mllm_draft_kind="eagle3",
        mllm_draft_block_size=4,
        default_mllm_draft=True,
    )
    engine.prepare_for_start()

    assert engine._mllm_draft_kind == "eagle3"
    assert engine.get_stats()["speculative"] == {
        "enabled": True,
        "implementation": "mlx_vlm_eagle3",
        "draft_model": "eagle3-drafter",
        "draft_kind": "eagle3",
        "draft_block_size": 4,
        "default_enabled": True,
        "continuous_batching_supported": False,
    }


@pytest.mark.anyio
async def test_status_projects_eagle3_speculative_configuration(monkeypatch):
    """The public status payload must retain the active Eagle3 configuration."""
    _install_mlx_stubs(monkeypatch)
    from vllm_mlx import server

    class Eagle3Engine:
        def get_stats(self):
            return {
                "running": True,
                "uptime_seconds": 1,
                "steps_executed": 0,
                "num_running": 0,
                "num_waiting": 0,
                "num_requests_processed": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "requests": [],
                "speculative": {
                    "enabled": True,
                    "implementation": "mlx_vlm_eagle3",
                    "draft_model": "eagle3-drafter",
                    "draft_kind": "eagle3",
                    "draft_block_size": 2,
                    "default_enabled": True,
                    "continuous_batching_supported": False,
                },
            }

    monkeypatch.setattr(server, "_engine", Eagle3Engine())
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_model_name", "target")
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)

    result = await server.status()

    assert result["mtp"] == {"enabled": False}
    assert result["speculative"]["draft_kind"] == "eagle3"
    assert result["speculative"]["draft_block_size"] == 2
