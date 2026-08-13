"""BatchedEngine MLLM scheduler configuration wiring tests.

These tests avoid model loading and MLX imports. They validate that CLI-level
SchedulerConfig fields survive the BatchedEngine -> MLLMSchedulerConfig bridge.
"""

import asyncio
import sys
import types
from types import SimpleNamespace


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
        def __init__(self, model, processor, config, **kwargs):
            captured["scheduler_config"] = config
            captured["scheduler_kwargs"] = kwargs

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


def _run_start_mllm(monkeypatch, scheduler_config, **engine_kwargs):
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
        def __init__(self, model, processor, config, **kwargs):
            captured["scheduler_kwargs"] = kwargs

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
        **engine_kwargs,
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


def test_start_mllm_forwards_specprefill_configuration(monkeypatch):
    captured = _run_start_mllm(
        monkeypatch,
        _base_scheduler_config(),
        specprefill_enabled=True,
        specprefill_threshold=4096,
        specprefill_keep_pct=0.25,
        specprefill_backbone_pct=0.1,
    )

    assert captured["config_kwargs"]["specprefill_enabled"] is True
    assert captured["config_kwargs"]["specprefill_threshold"] == 4096
    assert captured["config_kwargs"]["specprefill_keep_pct"] == 0.25
    assert captured["config_kwargs"]["specprefill_backbone_pct"] == 0.1
    assert captured["scheduler_kwargs"]["specprefill_draft_model"] is None


def test_prepare_mllm_loads_specprefill_draft_after_target_is_prepared(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine

    import vllm_mlx.engine.batched as batched_mod

    draft = object()
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: (draft, object())
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    fake_mllm_model = types.ModuleType("vllm_mlx.models.mllm")
    fake_mllm_model.MLXMultimodalLM = object
    monkeypatch.setitem(sys.modules, "vllm_mlx.models.mllm", fake_mllm_model)

    engine = BatchedEngine(
        model_name="fake-qwen",
        force_mllm=True,
        specprefill_enabled=True,
        specprefill_draft_model="draft-model",
    )
    engine._model = object()
    engine._processor = object()
    monkeypatch.setattr(
        batched_mod.BatchedEngine, "_inject_mtp_mllm", lambda self: None
    )

    engine._prepare_mllm_model()

    assert engine._specprefill_draft_model is draft
