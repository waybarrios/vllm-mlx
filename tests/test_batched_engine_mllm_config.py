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
