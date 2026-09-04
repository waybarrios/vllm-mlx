"""BatchedEngine MLLM scheduler configuration wiring tests.

These tests avoid model loading and MLX imports. They validate that CLI-level
SchedulerConfig fields survive the BatchedEngine -> MLLMSchedulerConfig bridge.
"""

import asyncio
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest


def _install_mlx_stubs() -> bool:
    """Make the configuration bridge importable on non-Apple CI runners."""
    try:  # pragma: no cover - depends on the environment
        import mlx.core  # noqa: F401

        return False
    except Exception:
        pass

    core = types.ModuleType("mlx.core")
    core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)

    class _Array:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _Stream:
        def __init__(self, idx: int = 0) -> None:
            self.idx = idx

    core.array = _Array
    core.Stream = _Stream
    core.clear_cache = lambda: None
    core.default_device = lambda: "gpu"
    core.new_stream = lambda *args, **kwargs: _Stream()
    core.set_default_stream = lambda *args, **kwargs: None
    core.metal = SimpleNamespace(
        is_available=lambda: False,
        get_active_memory=lambda: 0,
        get_peak_memory=lambda: 0,
        get_cache_memory=lambda: 0,
    )
    mlx = types.ModuleType("mlx")
    mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None, is_package=True)
    mlx.__path__ = []
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    return True


@pytest.fixture(scope="module", autouse=True)
def _scoped_mlx_stubs():
    """Keep non-Apple import stubs confined to this test module."""
    missing = object()
    prefixes = ("mlx", "vllm_mlx")
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    child_links = (
        ("mlx", "core"),
        ("vllm_mlx", "engine"),
        ("vllm_mlx.engine", "batched"),
    )
    saved_child_attrs = {
        (parent_name, child_name): getattr(
            saved_modules.get(parent_name), child_name, missing
        )
        for parent_name, child_name in child_links
    }
    installed = _install_mlx_stubs()
    yield
    if not installed:
        return

    for name in list(sys.modules):
        if (
            any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
            and name not in saved_modules
        ):
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)

    for (parent_name, child_name), saved_attr in saved_child_attrs.items():
        parent = saved_modules.get(parent_name)
        if parent is None:
            continue
        if saved_attr is missing:
            parent.__dict__.pop(child_name, None)
        else:
            setattr(parent, child_name, saved_attr)


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
    assert captured["scheduler_kwargs"]["specprefill_draft_model"] is None


def test_start_mllm_forwards_external_assistant_drafter(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine

    captured = {}
    loaded_drafter = object()

    class FakeMLXMultimodalLM:
        def __init__(self, model_name, trust_remote_code=True, **kwargs):
            captured["model_kwargs"] = kwargs
            self.model = object()
            self.processor = object()
            self._draft_model = loaded_drafter

        def load(self):
            return None

    class FakeMLLMSchedulerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeMLLMScheduler:
        def __init__(self, model, processor, config, **kwargs):
            captured["scheduler_kwargs"] = kwargs

        async def start(self):
            return None

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

    assert captured["model_kwargs"]["draft_model"] == "assistant"
    assert captured["model_kwargs"]["draft_kind"] == "mtp"
    assert captured["model_kwargs"]["draft_block_size"] == 6
    assert captured["scheduler_kwargs"] == {
        "specprefill_draft_model": None,
        "draft_model": loaded_drafter,
        "draft_kind": "mtp",
        "draft_block_size": 6,
    }


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
