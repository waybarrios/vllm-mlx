"""BatchedEngine MLLM scheduler configuration wiring tests.

These tests avoid model loading; importing the real loader may import MLX.
They validate that CLI-level
SchedulerConfig fields survive the BatchedEngine -> MLLMSchedulerConfig bridge.
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_start_mllm_forwards_prefix_cache_disable_to_mllm_scheduler(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine

    captured = {}

    class FakeMLXMultimodalLM:
        def __init__(self, model_name, trust_remote_code=True, **kwargs):
            self.model_name = model_name
            self.model = object()
            self.processor = object()

        def load(self):
            return None

    class FakeMLLMSchedulerConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs
            self.__dict__.update(kwargs)

    class FakeMLLMScheduler:
        def __init__(self, model, processor, config):
            captured["scheduler_config"] = config

        async def start(self):
            return None

    import vllm_mlx.engine.batched as batched_mod

    fake_mllm_scheduler = types.ModuleType("vllm_mlx.mllm_scheduler")
    fake_mllm_scheduler.MLLMScheduler = FakeMLLMScheduler
    fake_mllm_scheduler.MLLMSchedulerConfig = FakeMLLMSchedulerConfig
    fake_mllm_model = types.ModuleType("vllm_mlx.models.mllm")
    fake_mllm_model.MLXMultimodalLM = FakeMLXMultimodalLM
    monkeypatch.setitem(sys.modules, "vllm_mlx.mllm_scheduler", fake_mllm_scheduler)
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.mllm", fake_mllm_model)
    monkeypatch.setattr(
        batched_mod.BatchedEngine, "_inject_mtp_mllm", lambda self: None
    )

    engine = BatchedEngine(
        model_name="fake-qwen",
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            prefill_batch_size=4,
            completion_batch_size=8,
            prefill_step_size=256,
            mllm_prefill_step_size=None,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            cache_memory_mb=123,
            enable_mtp=False,
            mtp_num_draft_tokens=1,
            kv_cache_quantization=False,
            kv_cache_quantization_bits=8,
            kv_cache_quantization_group_size=64,
            chunked_prefill_tokens=0,
            max_kv_size=0,
        ),
        force_mllm=True,
    )

    asyncio.run(engine._start_mllm())

    assert captured["config_kwargs"]["enable_prefix_cache"] is False
    assert captured["config_kwargs"]["use_memory_aware_cache"] is False
    assert captured["config_kwargs"]["cache_memory_mb"] == 123
    assert captured["config_kwargs"]["prefix_cache_memory_mb"] == 123
    # SSD fields default to (None, 10.0) when absent from SchedulerConfig.
    assert captured["config_kwargs"]["ssd_cache_dir"] is None
    assert captured["config_kwargs"]["ssd_cache_max_gb"] == 10.0


def _run_start_mllm_with_external_drafter(monkeypatch, loaded_drafter):
    from vllm_mlx.engine.batched import BatchedEngine
    import vllm_mlx.models.mllm as mllm_mod

    captured = {}

    def load_released_gemma_fixture(model_path):
        captured["loader_path"] = model_path
        return loaded_drafter

    monkeypatch.setattr(
        mllm_mod, "load_gemma4_assistant_drafter", load_released_gemma_fixture
    )

    class FakeMLXMultimodalLM:
        _load_draft_model = mllm_mod.MLXMultimodalLM._load_draft_model

        def __init__(self, model_name, trust_remote_code=True, **kwargs):
            captured["model_kwargs"] = kwargs
            self.model = object()
            self.processor = object()
            self.draft_kind = kwargs["draft_kind"]
            self.draft_model_path = kwargs["draft_model"]

        def load(self):
            self._draft_model = self._load_draft_model()

    class FakeMLLMSchedulerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeMLLMScheduler:
        def __init__(self, model, processor, config, **kwargs):
            captured["scheduler_kwargs"] = kwargs

        async def start(self):
            return None

        def get_stats(self):
            return {}

    import vllm_mlx.engine.batched as batched_mod

    fake_scheduler = types.ModuleType("vllm_mlx.mllm_scheduler")
    fake_scheduler.MLLMScheduler = FakeMLLMScheduler
    fake_scheduler.MLLMSchedulerConfig = FakeMLLMSchedulerConfig
    fake_model = types.ModuleType("vllm_mlx.models.mllm")
    fake_model.MLXMultimodalLM = FakeMLXMultimodalLM
    monkeypatch.setitem(sys.modules, "vllm_mlx.mllm_scheduler", fake_scheduler)
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.mllm", fake_model)
    monkeypatch.setattr(
        batched_mod.BatchedEngine, "_inject_mtp_mllm", lambda self: None
    )

    engine = BatchedEngine(
        model_name="gemma4",
        scheduler_config=_base_scheduler_config(enable_mtp=False),
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=6,
    )
    asyncio.run(engine._start_mllm())
    captured["stats"] = engine.get_stats()
    return captured


def test_start_mllm_forwards_external_assistant_drafter(monkeypatch):
    # Released mlx-vlm v0.6.5 (84f43753) and v0.6.6 (c9e27b08):
    # gemma4_assistant.Gemma4AssistantDraftModel has no capability marker.
    # This models that attribute contract only; no weights are loaded and no
    # numerical Gemma support is certified by this fixture.
    class ReleasedGemmaDrafter:
        pass

    loaded_drafter = ReleasedGemmaDrafter()
    captured = _run_start_mllm_with_external_drafter(monkeypatch, loaded_drafter)

    assert captured["loader_path"] == "assistant"
    assert captured["model_kwargs"]["draft_model"] == "assistant"
    assert captured["model_kwargs"]["draft_kind"] == "mtp"
    assert captured["model_kwargs"]["draft_block_size"] == 6
    assert captured["scheduler_kwargs"] == {
        "draft_model": loaded_drafter,
        "draft_kind": "mtp",
        "draft_block_size": 6,
    }
    assert captured["stats"]["mtp"]["continuous_batching_supported"] is None
    assert not hasattr(loaded_drafter, "supports_continuous_batching")


@pytest.mark.parametrize(
    "loaded_drafter",
    [
        None,
        object(),
        SimpleNamespace(supports_continuous_batching=None),
        SimpleNamespace(supports_continuous_batching="false"),
        SimpleNamespace(supports_continuous_batching=False),
    ],
)
def test_capability_reporting_does_not_change_startup_admission(
    monkeypatch, loaded_drafter
):
    captured = _run_start_mllm_with_external_drafter(monkeypatch, loaded_drafter)
    assert captured["scheduler_kwargs"]["draft_model"] is loaded_drafter


def _generation_output():
    return SimpleNamespace(
        output_text="ok",
        output_token_ids=[1],
        prompt_tokens=2,
        completion_tokens=1,
        finish_reason="stop",
        mtp_drafts=1,
        mtp_accepted=1,
    )


def _batched_mllm_engine(scheduler, *, default_mllm_draft=False):
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine(
        model_name="gemma4",
        force_mllm=True,
        mllm_draft_model="assistant",
        mllm_draft_kind="mtp",
        mllm_draft_block_size=6,
        default_mllm_draft=default_mllm_draft,
    )
    engine._loaded = True
    engine._mllm_scheduler = scheduler
    engine._mllm_instance = SimpleNamespace(
        _draft_model=SimpleNamespace(supports_continuous_batching=True)
    )
    return engine


def test_generate_forwards_mllm_draft_opt_in():
    scheduler = SimpleNamespace(generate=AsyncMock(return_value=_generation_output()))
    engine = _batched_mllm_engine(scheduler)

    asyncio.run(engine.generate("hello", mllm_draft=True))

    assert scheduler.generate.await_args.kwargs["mllm_draft"] is True


def test_generate_uses_configured_mllm_draft_default():
    scheduler = SimpleNamespace(generate=AsyncMock(return_value=_generation_output()))
    engine = _batched_mllm_engine(
        scheduler,
        default_mllm_draft=True,
    )

    asyncio.run(engine.generate("hello"))

    assert scheduler.generate.await_args.kwargs["mllm_draft"] is True


def test_generate_allows_explicit_mllm_draft_opt_out():
    scheduler = SimpleNamespace(generate=AsyncMock(return_value=_generation_output()))
    engine = _batched_mllm_engine(
        scheduler,
        default_mllm_draft=True,
    )

    asyncio.run(engine.generate("hello", mllm_draft=False))

    assert scheduler.generate.await_args.kwargs["mllm_draft"] is False


def test_stream_generate_uses_configured_mllm_draft_default():
    async def stream_outputs(_request_id):
        yield SimpleNamespace(
            output_text="ok",
            new_text="ok",
            prompt_tokens=2,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            mtp_drafts=1,
            mtp_accepted=1,
        )

    scheduler = SimpleNamespace(
        add_request_async=AsyncMock(return_value="request-1"),
        stream_outputs=stream_outputs,
    )
    engine = _batched_mllm_engine(
        scheduler,
        default_mllm_draft=True,
    )

    async def consume_stream():
        return [output async for output in engine.stream_generate("hello")]

    outputs = asyncio.run(consume_stream())

    assert outputs[0].text == "ok"
    assert scheduler.add_request_async.await_args.kwargs["mllm_draft"] is True


def test_idle_stats_report_configured_mllm_draft_default():
    engine = _batched_mllm_engine(
        SimpleNamespace(get_stats=lambda: {}),
        default_mllm_draft=True,
    )

    stats = engine.get_stats()

    assert stats["mtp"] == {
        "enabled": True,
        "implementation": "external_assistant",
        "draft_model": "assistant",
        "draft_kind": "mtp",
        "draft_block_size": 6,
        "default_enabled": True,
        "continuous_batching_supported": True,
    }


@pytest.mark.parametrize("reported", [False, None, "false"])
def test_batched_stats_preserve_scheduler_batch_capability(reported):
    engine = _batched_mllm_engine(
        SimpleNamespace(
            get_stats=lambda: {"mtp": {"continuous_batching_supported": reported}}
        )
    )

    assert engine.get_stats()["mtp"]["continuous_batching_supported"] is (
        reported if type(reported) is bool else None
    )


def _run_start_mllm(monkeypatch, scheduler_config):
    """Run BatchedEngine._start_mllm with fakes, return captured kwargs."""
    from vllm_mlx.engine.batched import BatchedEngine

    captured = {}

    class FakeMLXMultimodalLM:
        def __init__(self, model_name, trust_remote_code=True, **kwargs):
            self.model = object()
            self.processor = object()

        def load(self):
            return None

    class FakeMLLMSchedulerConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs
            self.__dict__.update(kwargs)

    class FakeMLLMScheduler:
        def __init__(self, model, processor, config):
            pass

        async def start(self):
            return None

    import vllm_mlx.engine.batched as batched_mod

    fake_mllm_scheduler = types.ModuleType("vllm_mlx.mllm_scheduler")
    fake_mllm_scheduler.MLLMScheduler = FakeMLLMScheduler
    fake_mllm_scheduler.MLLMSchedulerConfig = FakeMLLMSchedulerConfig
    fake_mllm_model = types.ModuleType("vllm_mlx.models.mllm")
    fake_mllm_model.MLXMultimodalLM = FakeMLXMultimodalLM
    monkeypatch.setitem(sys.modules, "vllm_mlx.mllm_scheduler", fake_mllm_scheduler)
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.mllm", fake_mllm_model)
    monkeypatch.setattr(
        batched_mod.BatchedEngine, "_inject_mtp_mllm", lambda self: None
    )

    engine = BatchedEngine(
        model_name="fake-qwen",
        scheduler_config=scheduler_config,
        force_mllm=True,
    )
    asyncio.run(engine._start_mllm())
    return captured


def _base_scheduler_config(**overrides):
    cfg = dict(
        max_num_seqs=16,
        prefill_batch_size=4,
        completion_batch_size=8,
        prefill_step_size=256,
        mllm_prefill_step_size=None,
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        cache_memory_mb=None,
        enable_mtp=False,
        mtp_num_draft_tokens=1,
        kv_cache_quantization=False,
        kv_cache_quantization_bits=8,
        kv_cache_quantization_group_size=64,
        chunked_prefill_tokens=0,
        max_kv_size=0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_start_mllm_forwards_ssd_cache_fields(monkeypatch):
    captured = _run_start_mllm(
        monkeypatch,
        _base_scheduler_config(ssd_cache_dir="/tmp/ssd-kv", ssd_cache_max_gb=42.0),
    )
    assert captured["config_kwargs"]["ssd_cache_dir"] == "/tmp/ssd-kv"
    assert captured["config_kwargs"]["ssd_cache_max_gb"] == 42.0


def test_start_mllm_forwards_prefix_cache_memory_percent(monkeypatch):
    """The MLLM prefix cache must receive the configured percentage limit."""
    captured = _run_start_mllm(
        monkeypatch,
        _base_scheduler_config(cache_memory_percent=0.35),
    )

    assert captured["config_kwargs"]["prefix_cache_memory_percent"] == 0.35
