"""Regression coverage for total-context admission limits (#795)."""

import asyncio
import importlib.util
import inspect
import sys
import time
from collections import deque
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests import _mlx_stub

_mlx_stub.install_if_unavailable()


def _server_module_or_skip():
    if any(
        importlib.util.find_spec(package) is None
        for package in ("uvicorn", "prometheus_client")
    ):
        pytest.skip("server extras are unavailable")
    from vllm_mlx import server

    return server


def test_total_context_accepts_exact_limit():
    """Changing the comparison from ``>`` to ``>=`` would break this boundary."""
    from vllm_mlx.context_limits import validate_context_length

    validate_context_length(
        prompt_tokens=49_152,
        max_tokens=16_384,
        max_model_len=65_536,
    )


def test_total_context_rejects_one_token_over_limit():
    """Removing combined prompt/output admission would admit 65,537 tokens."""
    from vllm_mlx.context_limits import ContextLengthExceeded
    from vllm_mlx.context_limits import validate_context_length

    with pytest.raises(ContextLengthExceeded) as exc_info:
        validate_context_length(
            prompt_tokens=49_153,
            max_tokens=16_384,
            max_model_len=65_536,
        )

    error = exc_info.value
    assert error.prompt_tokens == 49_153
    assert error.max_tokens == 16_384
    assert error.max_model_len == 65_536
    assert error.total_tokens == 65_537


def test_prompt_encoding_does_not_duplicate_existing_bos_token():
    """Admission must count the BOS-prefixed tokens generation actually uses."""
    from vllm_mlx.context_limits import encode_prompt

    class Tokenizer:
        bos_token = "<s>"

        def encode(self, _prompt, add_special_tokens=True):
            return [0, 1, 2] if add_special_tokens else [1, 2]

    assert encode_prompt(Tokenizer(), "<s>hello") == [1, 2]


def test_scheduler_configs_share_the_context_limit_default():
    """Omitting either config field would leave one batching route unprotected."""
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig
    from vllm_mlx.scheduler import SchedulerConfig

    assert SchedulerConfig().max_model_len == 65_536
    assert MLLMSchedulerConfig().max_model_len == 65_536


def test_scheduler_configs_accept_context_limit_override():
    """Ignoring an operator override would enforce the wrong deployment budget."""
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig
    from vllm_mlx.scheduler import SchedulerConfig

    assert SchedulerConfig(max_model_len=131_072).max_model_len == 131_072
    assert MLLMSchedulerConfig(max_model_len=131_072).max_model_len == 131_072


def test_text_scheduler_rejects_before_prefix_cache_lookup():
    """Moving admission below cache fetch would allocate work for a rejected request."""
    from vllm_mlx.context_limits import ContextLengthExceeded
    from vllm_mlx.request import Request, SamplingParams
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    class Tokenizer:
        def encode(self, _prompt):
            return [1, 2, 3]

    class CacheMustNotRun:
        def fetch_cache(self, *_args):
            raise AssertionError("prefix cache consulted before context admission")

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(max_model_len=4)
    scheduler.tokenizer = Tokenizer()
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.block_aware_cache = CacheMustNotRun()
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None

    request = Request(
        request_id="too-long",
        prompt="abc",
        sampling_params=SamplingParams(max_tokens=2),
    )

    with pytest.raises(ContextLengthExceeded):
        scheduler.add_request(request)

    assert scheduler.requests == {}
    assert list(scheduler.waiting) == []


def test_mllm_generator_counts_media_expanded_input_ids():
    """Admission must use the expanded MLLM prompt, not its short source text."""
    from vllm_mlx.context_limits import ContextLengthExceeded
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator
    from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

    class ExpandedInputIds:
        size = 3

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.max_model_len = 4
    request = MLLMBatchRequest(
        uid=1,
        request_id="too-long-mllm",
        prompt="x",
        max_tokens=2,
        input_ids=ExpandedInputIds(),
    )

    with pytest.raises(ContextLengthExceeded):
        generator._validate_context_length(request)


def test_partially_initialized_mllm_generator_uses_default_context_limit():
    """Lightweight generator construction must keep the production default."""
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator
    from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    request = MLLMBatchRequest(
        uid=1,
        request_id="within-default-limit",
        prompt="x",
        max_tokens=1,
        input_ids=SimpleNamespace(size=1),
    )

    generator._validate_context_length(request)


def test_mllm_generator_rejects_before_prefix_or_kv_work():
    """Oversized expanded prompts fail only their request before KV creation."""
    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator
    from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

    class ExpandedInputIds:
        size = 3

    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.max_model_len = 4
    generator._preprocess_request = lambda request: setattr(
        request, "input_ids", ExpandedInputIds()
    )
    generator._pending_error_responses = []
    request = MLLMBatchRequest(
        uid=1,
        request_id="too-long-before-kv",
        prompt="x",
        max_tokens=2,
    )

    assert generator._process_prompts([request]) is None
    assert len(generator._pending_error_responses) == 1
    assert generator._pending_error_responses[0].request_id == request.request_id
    assert generator._pending_error_responses[0].finish_reason == "error"


def test_simple_mllm_counter_uses_processor_expanded_input_ids(monkeypatch):
    """SimpleEngine admission must count media tokens emitted by the processor."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    captured = {}
    fake_utils = ModuleType("mlx_vlm.utils")

    def prepare_inputs(processor, **kwargs):
        captured["processor"] = processor
        captured.update(kwargs)
        return {"input_ids": SimpleNamespace(size=7)}

    fake_utils.prepare_inputs = prepare_inputs
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_utils)

    model = MLXMultimodalLM.__new__(MLXMultimodalLM)
    model.processor = object()
    model.model = SimpleNamespace(config=SimpleNamespace(image_token_index=42))

    assert model._count_formatted_prompt_tokens("prompt", images=["image"]) == 7
    assert captured["prompts"] == "prompt"
    assert captured["images"] == ["image"]
    assert captured["image_token_index"] == 42


def test_simple_mllm_rejects_before_generation(monkeypatch):
    """Direct MLLM callers must not enter generation after an oversized preflight."""
    import vllm_mlx.engine.simple as simple_module
    from vllm_mlx.context_limits import ContextLengthExceeded

    class Model:
        def count_chat_prompt_tokens(self, _messages, **_kwargs):
            return 3

        def stream_chat(self, **_kwargs):
            raise AssertionError("generation entered before context admission")
            yield

    engine = simple_module.SimpleEngine(
        "test-model",
        force_mllm=True,
        max_model_len=4,
    )
    engine._model = Model()
    engine._loaded = True

    async def run_inline(func, *args, **kwargs):
        kwargs.pop("request_id", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(engine, "_run_blocking_serialized", run_inline)

    async def consume():
        async for _ in engine.stream_chat(
            [{"role": "user", "content": "x"}],
            max_tokens=2,
        ):
            pass

    with pytest.raises(ContextLengthExceeded):
        asyncio.run(consume())


def test_batched_preflight_matches_scheduler_bos_tokenization(monkeypatch):
    """A BOS-prefixed prompt must have the same count before and during admission."""
    import vllm_mlx.engine.batched as batched_module

    class Tokenizer:
        bos_token = "<s>"

        def encode(self, _prompt, add_special_tokens=True):
            return [0, 1, 2] if add_special_tokens else [1, 2]

    monkeypatch.setattr(batched_module, "is_mllm_model", lambda _model: False)
    engine = batched_module.BatchedEngine("test-model", max_model_len=4)
    engine._tokenizer = Tokenizer()
    engine._loaded = True

    assert asyncio.run(engine.validate_generate_context("<s>x", max_tokens=2)) == 2


def test_simple_streaming_preflight_uses_harmony_rendering(monkeypatch):
    """Harmony requests must be admitted from the prompt generation will use."""
    import vllm_mlx.engine.simple as simple_module
    from vllm_mlx.context_limits import ContextLengthExceeded
    from vllm_mlx.utils import harmony_render

    class Tokenizer:
        def apply_chat_template(self, _messages, **_kwargs):
            return "short"

        def encode(self, prompt, **_kwargs):
            return [1, 2, 3] if prompt == "harmony prompt" else [1]

    monkeypatch.setattr(simple_module, "is_mllm_model", lambda _model: False)
    monkeypatch.setattr(
        harmony_render,
        "render_messages",
        lambda *_args, **_kwargs: "harmony prompt",
    )
    engine = simple_module.SimpleEngine("test-model", max_model_len=4)
    engine._model = SimpleNamespace(tokenizer=Tokenizer())
    engine._loaded = True
    engine.use_harmony_rendering = True

    with pytest.raises(ContextLengthExceeded):
        asyncio.run(
            engine.validate_chat_context(
                [{"role": "user", "content": "x"}],
                max_tokens=2,
                streaming=True,
            )
        )


def test_temp_file_scope_cleans_only_files_registered_inside(monkeypatch):
    """Count-only media artifacts must be cleaned without touching older files."""
    from vllm_mlx.models.mllm import TempFileManager

    manager = TempFileManager()
    cleaned = []
    monkeypatch.setattr(manager, "cleanup", lambda path: cleaned.append(path) or True)
    manager.register("older-file")

    with manager.cleanup_scope():
        manager.register("preflight-file")

    assert cleaned == ["preflight-file"]


def test_engines_and_lifecycle_spec_keep_context_limit_override(monkeypatch):
    """Every engine construction path must retain the operator's limit."""
    import vllm_mlx.engine.batched as batched_module
    import vllm_mlx.engine.simple as simple_module
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.engine.simple import SimpleEngine
    from vllm_mlx.lifecycle import ModelSpec

    monkeypatch.setattr(simple_module, "is_mllm_model", lambda _model: False)
    monkeypatch.setattr(batched_module, "is_mllm_model", lambda _model: False)

    simple = SimpleEngine("test-model", max_model_len=131_072)
    batched = BatchedEngine("test-model", max_model_len=131_072)
    spec = ModelSpec(
        model_key="test",
        model_name="test-model",
        max_model_len=131_072,
    )

    assert simple.max_model_len == 131_072
    assert batched.max_model_len == 131_072
    assert spec.max_model_len == 131_072


def test_new_context_limit_preserves_existing_positional_arguments(monkeypatch):
    """Adding max_model_len must not reinterpret existing Python calls."""
    import vllm_mlx.engine.simple as simple_module
    from vllm_mlx.engine.simple import SimpleEngine
    from vllm_mlx.lifecycle import ModelSpec
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig
    from vllm_mlx.scheduler import SchedulerConfig

    server = _server_module_or_skip()
    monkeypatch.setattr(simple_module, "is_mllm_model", lambda _model: False)

    bound = inspect.signature(server.load_model).bind(
        "test-model", False, None, 1, 8, 8, True
    )
    spec = ModelSpec("key", "test-model", False, None, 1, 8, True)
    engine = SimpleEngine(
        "test-model",
        False,
        True,
        False,
        False,
        1,
        2048,
        False,
        8192,
        0.3,
        0.0,
        None,
        0,
        "assistant-model",
    )

    assert bound.arguments.get("force_mllm") is True
    assert "max_model_len" not in bound.arguments
    assert spec.force_mllm is True
    assert spec.max_model_len == 65_536
    assert engine._mllm_draft_model_path == "assistant-model"
    assert engine.max_model_len == 65_536
    assert list(inspect.signature(SchedulerConfig).parameters)[-1] == "max_model_len"
    assert (
        list(inspect.signature(MLLMSchedulerConfig).parameters)[-1] == "max_model_len"
    )


def test_batched_mllm_preflight_does_not_block_event_loop():
    """CPU-heavy MLLM token counting must not freeze unrelated requests."""
    from vllm_mlx.engine.batched import BatchedEngine

    events = []

    class Scheduler:
        def validate_context(self, **_kwargs):
            events.append("preflight-start")
            time.sleep(0.05)
            events.append("preflight-end")
            return 1

    engine = BatchedEngine("test-model", force_mllm=True)
    engine._loaded = True
    engine._mllm_scheduler = Scheduler()

    async def tick():
        await asyncio.sleep(0.01)
        events.append("event-loop-tick")

    async def run():
        await asyncio.gather(
            engine.validate_generate_context("hello", max_tokens=1),
            tick(),
        )

    asyncio.run(run())

    assert events.index("event-loop-tick") < events.index("preflight-end")


def test_batched_mllm_preflight_cleans_media_without_caching(monkeypatch):
    """Count-only MLLM preprocessing must leave no temp or vision-cache state."""
    import vllm_mlx.models.mllm as mllm_module
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    events = []
    captured = {}

    @contextmanager
    def cleanup_scope():
        events.append("cleanup-enter")
        try:
            yield
        finally:
            events.append("cleanup-exit")

    class BatchGenerator:
        def _preprocess_request(self, request, *, populate_vision_cache=True):
            captured["populate_vision_cache"] = populate_vision_cache
            request.input_ids = SimpleNamespace(size=3)

        def _validate_context_length(self, _request):
            events.append("validated")

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = BatchGenerator()
    scheduler._ensure_batch_generator = lambda: None
    monkeypatch.setattr(mllm_module._temp_manager, "cleanup_scope", cleanup_scope)

    prompt_tokens = scheduler.validate_context(
        prompt="describe",
        images=["image"],
        max_tokens=2,
    )

    assert prompt_tokens == 3
    assert captured["populate_vision_cache"] is False
    assert events == ["cleanup-enter", "validated", "cleanup-exit"]


@pytest.mark.parametrize("engine_kind", ["simple", "batched"])
def test_text_engine_preflight_rejects_one_token_over(engine_kind, monkeypatch):
    """Engine-owned admission protects direct callers outside the HTTP server."""
    import vllm_mlx.engine.batched as batched_module
    import vllm_mlx.engine.simple as simple_module
    from vllm_mlx.context_limits import ContextLengthExceeded

    class Tokenizer:
        bos_token = None

        def encode(self, _prompt, **_kwargs):
            return [1, 2, 3]

    if engine_kind == "simple":
        monkeypatch.setattr(simple_module, "is_mllm_model", lambda _model: False)
        engine = simple_module.SimpleEngine("test-model", max_model_len=4)
        engine._model = type("Model", (), {"tokenizer": Tokenizer()})()
    else:
        monkeypatch.setattr(batched_module, "is_mllm_model", lambda _model: False)
        engine = batched_module.BatchedEngine("test-model", max_model_len=4)
        engine._tokenizer = Tokenizer()

    engine._loaded = True

    with pytest.raises(ContextLengthExceeded):
        asyncio.run(engine.validate_generate_context("abc", max_tokens=2))


def test_simple_engine_rejects_before_generation_iterator(monkeypatch):
    """Direct SimpleEngine streams must not reach model generation when oversized."""
    import vllm_mlx.engine.simple as simple_module
    from vllm_mlx.context_limits import ContextLengthExceeded

    class Tokenizer:
        bos_token = None

        def encode(self, _prompt, **_kwargs):
            return [1, 2, 3]

    class Model:
        tokenizer = Tokenizer()

        def stream_generate(self, **_kwargs):
            raise AssertionError("generation entered before context admission")

    monkeypatch.setattr(simple_module, "is_mllm_model", lambda _model: False)
    engine = simple_module.SimpleEngine("test-model", max_model_len=4)
    engine._model = Model()
    engine._loaded = True

    async def consume():
        async for _ in engine.stream_generate("abc", max_tokens=2):
            pass

    with pytest.raises(ContextLengthExceeded):
        asyncio.run(consume())


def test_server_preflight_maps_context_error_to_http_400():
    """Streaming routes must reject before returning a successful response."""
    from fastapi import HTTPException

    from vllm_mlx.context_limits import ContextLengthExceeded

    server = _server_module_or_skip()

    class RejectingEngine:
        async def validate_generate_context(self, *_args, **_kwargs):
            raise ContextLengthExceeded(
                prompt_tokens=3,
                max_tokens=2,
                max_model_len=4,
            )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server._preflight_generate_context(
                RejectingEngine(),
                "abc",
                max_tokens=2,
            )
        )

    error = exc_info.value
    assert error.status_code == 400
    assert error.detail["code"] == "context_length_exceeded"
    assert error.detail["prompt_tokens"] == 3
    assert error.detail["max_tokens"] == 2
    assert error.detail["max_model_len"] == 4


def test_server_preflight_ignores_dynamic_non_async_mock_hook():
    """Duck-typed engines without a real validator must retain old behavior."""
    server = _server_module_or_skip()

    result = asyncio.run(server._preflight_chat_context(MagicMock(), [], max_tokens=1))

    assert result == 0


def test_chat_preflight_obeys_request_timeout(monkeypatch):
    """Slow admission must consume the same timeout budget as generation."""
    from fastapi import HTTPException

    server = _server_module_or_skip()
    released = []

    class SlowEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False
        use_harmony_rendering = False
        tokenizer = None

        async def validate_chat_context(self, *_args, **_kwargs):
            await asyncio.sleep(0.2)
            return 1

        async def chat(self, *_args, **_kwargs):
            raise AssertionError("generation entered after the request deadline")

    async def acquire(*_args, **_kwargs):
        return SlowEngine()

    async def release(*_args, **_kwargs):
        released.append(True)

    monkeypatch.setattr(server, "_model_name", "test-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_engine_for_request", release)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_tool_call_parser", None)

    request = server.ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1,
        timeout=0.01,
    )
    started = time.monotonic()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.create_chat_completion(request, SimpleNamespace()))

    assert exc_info.value.status_code == 504
    assert time.monotonic() - started < 0.1
    assert released == [True]


def test_standalone_server_parser_matches_primary_cli():
    """The legacy server entry point must expose the same bounded default."""
    server = _server_module_or_skip()
    parser = server.create_parser()

    assert parser.parse_args([]).max_model_len == 65_536
    assert parser.parse_args(["--max-model-len", "131072"]).max_model_len == 131_072


def test_load_model_rejects_impossible_default_output_budget():
    """The server default cannot request more output than the total context."""
    server = _server_module_or_skip()

    with pytest.raises(ValueError, match="cannot exceed max model length"):
        server.load_model(
            "test-model",
            max_tokens=5,
            max_request_tokens=5,
            max_model_len=4,
        )


def test_streaming_chat_rejects_before_engine_stream(monkeypatch):
    """An oversized chat must return 400 before the initial role SSE frame."""
    from fastapi import HTTPException

    from vllm_mlx.context_limits import ContextLengthExceeded

    server = _server_module_or_skip()

    class RejectingEngine:
        is_mllm = False
        preserve_native_tool_format = False
        use_harmony_rendering = False
        tokenizer = None

        async def validate_chat_context(self, *_args, **_kwargs):
            raise ContextLengthExceeded(
                prompt_tokens=3,
                max_tokens=2,
                max_model_len=4,
            )

        async def stream_chat(self, *_args, **_kwargs):
            raise AssertionError("engine stream entered before context admission")
            yield

    async def acquire(*_args, **_kwargs):
        return RejectingEngine()

    async def release(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "_model_name", "test-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_engine_for_request", release)

    request = server.ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "abc"}],
        max_tokens=2,
        stream=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.create_chat_completion(request, object()))

    assert exc_info.value.status_code == 400
