"""Regression coverage for total-context admission limits (#795)."""

import asyncio
import importlib.util
import sys
from collections import deque
from types import ModuleType, SimpleNamespace

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
    from vllm_mlx.context_limits import ContextLengthExceeded

    class Tokenizer:
        bos_token = "<s>"

        def encode(self, _prompt, add_special_tokens=True):
            return [0, 1, 2] if add_special_tokens else [1, 2]

    monkeypatch.setattr(batched_module, "is_mllm_model", lambda _model: False)
    engine = batched_module.BatchedEngine("test-model", max_model_len=4)
    engine._tokenizer = Tokenizer()
    engine._loaded = True

    with pytest.raises(ContextLengthExceeded):
        asyncio.run(engine.validate_generate_context("<s>x", max_tokens=2))


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
