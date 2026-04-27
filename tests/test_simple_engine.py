# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

pytestmark = pytest.mark.anyio


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def anyio_backend(self):
        return "asyncio"

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            # Simulate some work
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)

        # stream_generate tracks concurrency the same way so tests that
        # exercise SimpleEngine.generate() (which is now an accumulator
        # over stream_generate) see the same serialization behavior.
        def stream_generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            chunk = MagicMock()
            chunk.text = "test response"
            chunk.tokens = [1, 2, 3]
            chunk.finished = True
            chunk.finish_reason = "stop"
            chunk.prompt_tokens = 3
            chunk.completion_tokens = 3
            yield chunk

        model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)
        return model

    @pytest.fixture
    def mock_llm_model(self):
        """Create a mock LLM model."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def chat_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.chat = MagicMock(side_effect=chat_side_effect)
        return model

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_generate(self, mock_model):
        """Test that the lock prevents concurrent generate calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            tasks = [
                engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.anyio
    async def test_lock_prevents_concurrent_chat(self, mock_llm_model):
        """Test that the lock prevents concurrent chat calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True

            # Launch multiple concurrent chat calls
            tasks = [
                engine.chat(
                    messages=[{"role": "user", "content": f"test {i}"}], max_tokens=10
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_llm_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_llm_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    async def test_chat_with_tools_aggregates_streaming_path(self, mock_llm_model):
        """Tool-enabled non-stream chat should use the streaming path."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="partial",
                tokens=[1],
                prompt_tokens=11,
                completion_tokens=1,
                finish_reason=None,
                finished=False,
            )
            yield MagicMock(
                text='<|im_end|><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                tokens=[7, 8, 9],
                prompt_tokens=11,
                completion_tokens=4,
                finish_reason="stop",
                finished=True,
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True
            engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

            output = await engine.chat(
                messages=[{"role": "user", "content": "run pwd"}],
                max_tokens=16,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            assert (
                output.text
                == '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
            )
            assert output.tokens == [7, 8, 9]
            assert output.prompt_tokens == 11
            assert output.completion_tokens == 4
            assert output.finish_reason == "stop"
            mock_llm_model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_lock_serializes_stream_generate(self, mock_model):
        """Test that stream_generate uses the same lock as other methods."""
        from vllm_mlx.engine.simple import SimpleEngine

        def stream_generate_side_effect(**kwargs):
            # Yield a few chunks
            for i in range(3):
                chunk = MagicMock()
                chunk.text = f"chunk{i}"
                chunk.prompt_tokens = 5
                chunk.finished = i == 2
                chunk.finish_reason = "stop" if i == 2 else None
                yield chunk

        mock_model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Test that stream_generate acquires the lock
            # by checking if it blocks when lock is already held
            lock_acquired = asyncio.Event()
            stream_started = asyncio.Event()

            async def hold_lock():
                async with engine._generation_lock:
                    lock_acquired.set()
                    # Wait until stream tries to start
                    await asyncio.sleep(0.1)

            async def try_stream():
                # Wait for lock to be held
                await lock_acquired.wait()
                stream_started.set()
                # This should block until hold_lock releases
                result = []
                async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                    result.append(chunk)
                return result

            # Start both tasks
            hold_task = asyncio.create_task(hold_lock())
            stream_task = asyncio.create_task(try_stream())

            # Wait a bit for stream to try to acquire lock
            await asyncio.sleep(0.05)

            # Stream should have started but be blocked on the lock
            assert stream_started.is_set(), "Stream should have attempted to start"

            # Stream task should not be done yet (blocked on lock)
            assert not stream_task.done(), "Stream should be blocked waiting for lock"

            # Let hold_lock finish
            await hold_task

            # Now stream should complete
            result = await stream_task
            assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    @pytest.mark.anyio
    async def test_engine_initialization_creates_lock(self):
        """Test that SimpleEngine creates a lock on initialization."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

            assert hasattr(engine, "_generation_lock")
            assert isinstance(engine._generation_lock, asyncio.Lock)

    @pytest.mark.anyio
    async def test_run_blocking_serialized_rebinds_worker_generation_streams(self):
        """Worker-thread MLX generation should get fresh thread-local streams."""
        import importlib

        from vllm_mlx.engine.simple import SimpleEngine

        mlx_lm_generate = importlib.import_module("mlx_lm.generate")
        sentinel_stream = object()

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.mlx_streams.mx.default_device", return_value="gpu"),
            patch(
                "vllm_mlx.mlx_streams.mx.new_stream",
                return_value=sentinel_stream,
            ),
            patch("vllm_mlx.mlx_streams.mx.set_default_stream"),
        ):
            engine = SimpleEngine("test-model")
            observed = await engine._run_blocking_serialized(
                lambda: mlx_lm_generate.generation_stream
            )

        assert observed is sentinel_stream

    @pytest.mark.anyio
    async def test_llm_stream_generate_stays_on_model_load_thread(self):
        """SimpleEngine must load and stream on the same thread for MLX streams."""
        from vllm_mlx.engine.simple import SimpleEngine

        class FakeLLMModel:
            def __init__(self, *_args, **_kwargs):
                self._load_thread = None
                self.tokenizer = MagicMock()
                self.tokenizer.encode.return_value = [1, 2, 3]

            def load(self):
                self._load_thread = threading.get_ident()

            def stream_generate(self, **_kwargs):
                if threading.get_ident() != self._load_thread:
                    raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
                yield SimpleNamespace(
                    text="ok",
                    prompt_tokens=3,
                    finished=True,
                    finish_reason="stop",
                )

        with (
            patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False),
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLLMModel),
        ):
            engine = SimpleEngine("test-model")

            outputs = [
                chunk
                async for chunk in engine.stream_generate(
                    prompt="hello",
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                )
            ]

        assert outputs
        assert outputs[-1].new_text == "ok"
        assert outputs[-1].finished is True

    @pytest.mark.anyio
    async def test_start_keeps_text_routing_for_mllm_without_mtp(self):
        """MLLM text-only routing must stay available when MTP is disabled."""
        from vllm_mlx.engine.simple import SimpleEngine

        text_model = MagicMock()
        text_model.mtp = None
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 42

        mock_mllm = MagicMock()
        mock_mllm.model = MagicMock()
        mock_mllm.get_tokenizer.return_value = tokenizer

        with (
            patch(
                "vllm_mlx.models.mllm.MLXMultimodalLM",
                return_value=mock_mllm,
            ),
            patch(
                "vllm_mlx.text_model_from_vlm.build_text_model",
                return_value=text_model,
            ),
        ):
            engine = SimpleEngine("qwen3.6-27b", force_mllm=True, mtp=False)
            await engine.start()

        assert engine._text_model is text_model
        assert engine._text_tokenizer is tokenizer

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_routes_without_mtp(self):
        """Non-stream text-only MLLM chat must aggregate the TextModel route."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_stream_chat(*args, **kwargs):
            yield MagicMock(
                text="Hello",
                tokens=[1],
                prompt_tokens=5,
                completion_tokens=1,
                finish_reason="stop",
                finished=True,
            )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._model = MagicMock()
        engine.stream_chat = fake_stream_chat  # type: ignore[method-assign]

        output = await engine.chat(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        )

        assert output.text == "Hello"
        assert output.tokens == [1]
        assert output.prompt_tokens == 5
        assert output.completion_tokens == 1
        assert output.finish_reason == "stop"
        engine._model.chat.assert_not_called()

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_without_text_model_uses_stream_path(self):
        """When TextModel is unavailable, text-only MLLM non-stream chat should
        aggregate stream_chat to avoid mlx_vlm chat thread-stream mismatches.
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def chat(self, **kwargs):
                raise RuntimeError("There is no Stream(gpu, 3) in current thread.")

            def stream_chat(self, **kwargs):
                yield SimpleNamespace(text="one, two, three", finish_reason="stop")

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = None
        engine._model = FakeMllmModel()

        output = await engine.chat(
            messages=[{"role": "user", "content": "Count: one, two, three"}],
            max_tokens=16,
        )

        assert output.text == "one, two, three"
        assert output.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_nonstream_text_only_without_text_model_keeps_stream_thread_owner(
        self,
    ):
        """MLLM text-only non-stream path must keep stream_chat on model thread.

        Regression: aggregate_stream_chat -> stream_chat used _run_blocking_serialized,
        which moved mlx_vlm stream generation to a worker thread and could raise
        "There is no Stream(gpu, N) in current thread".
        """
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        class FakeMllmModel:
            def __init__(self):
                self._owner_thread = threading.get_ident()

            def stream_chat(self, **kwargs):
                if threading.get_ident() != self._owner_thread:
                    raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
                yield SimpleNamespace(
                    text="one, two, three",
                    finish_reason="stop",
                    prompt_tokens=3,
                )

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = None
        engine._model = FakeMllmModel()

        output = await engine.chat(
            messages=[{"role": "user", "content": "Count: one, two, three"}],
            max_tokens=16,
        )

        assert output.text == "one, two, three"
        assert output.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_requests_complete_in_order(self, mock_model):
        """Test that concurrent requests complete (may be in any order due to lock)."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            results = await asyncio.gather(
                *[
                    engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                    for i in range(3)
                ]
            )

            # All requests should complete
            assert len(results) == 3
            for result in results:
                assert result.text == "test response"

    @pytest.mark.anyio
    async def test_generate_accumulates_over_stream_generate(self):
        """generate() should iterate stream_generate() and return the last
        yielded GenerationOutput, forwarding per-request kwargs (including
        SpecPrefill overrides) through so they reach _stream_generate_specprefill.
        """
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        async def fake_stream_generate(**kwargs):
            captured_kwargs.update(kwargs)
            # First chunk: mid-generation
            yield GenerationOutput(
                text="partial",
                new_text="partial",
                tokens=[1, 2],
                prompt_tokens=11,
                completion_tokens=2,
                finished=False,
                finish_reason=None,
            )
            # Final chunk: finished
            yield GenerationOutput(
                text="partial final",
                new_text=" final",
                tokens=[1, 2, 3],
                prompt_tokens=11,
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
            )

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine.stream_generate = fake_stream_generate  # type: ignore[method-assign]

            output = await engine.generate(
                prompt="say hi",
                max_tokens=16,
                temperature=0.6,
                top_p=0.95,
                specprefill=True,
                specprefill_keep_pct=0.2,
            )

        # Accumulator returns the last GenerationOutput's fields
        assert output.text == "partial final"
        assert output.tokens == [1, 2, 3]
        assert output.prompt_tokens == 11
        assert output.completion_tokens == 3
        assert output.finish_reason == "stop"
        assert output.finished is True

        # Per-request SpecPrefill overrides reach stream_generate
        assert captured_kwargs.get("prompt") == "say hi"
        assert captured_kwargs.get("max_tokens") == 16
        assert captured_kwargs.get("specprefill") is True
        assert captured_kwargs.get("specprefill_keep_pct") == 0.2

    @pytest.mark.anyio
    async def test_generate_empty_stream_returns_safe_default(self):
        """If stream_generate yields nothing, generate() returns an empty
        stop-reason GenerationOutput rather than raising.
        """
        from vllm_mlx.engine.simple import SimpleEngine

        async def empty_stream_generate(**kwargs):
            return
            yield  # unreachable; makes this a generator

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine.stream_generate = empty_stream_generate  # type: ignore[method-assign]

            output = await engine.generate(prompt="anything", max_tokens=5)

        assert output.text == ""
        assert output.finish_reason == "stop"

    def test_seed_logits_processors_prepends_prompt_tokens(self):
        """Continuation decode processors must see the original prompt prefix."""
        from vllm_mlx.engine.simple import _seed_logits_processors

        seen = {}

        def processor(tokens, logits):
            seen["tokens"] = tokens.tolist()
            return logits

        seeded = _seed_logits_processors(
            mx.array([10, 11], dtype=mx.uint32), [processor]
        )

        logits = mx.zeros((1, 8), dtype=mx.float32)
        seeded[0](mx.array([12, 13], dtype=mx.uint32), logits)

        assert seen["tokens"] == [10, 11, 12, 13]

    @pytest.mark.anyio
    async def test_specprefill_success_preserves_mtp_path(self):
        """Successful sparse prefill should continue through the normal MTP path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_make_sampler(**kwargs):
            captured["sampler_kwargs"] = kwargs

            def _sample(_logprobs):
                return mx.array([17], dtype=mx.uint32)

            return _sample

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured["prompt"] = prompt.tolist()
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="B", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 99
        tokenizer.encode.return_value = [5, 6, 7]
        tokenizer.decode.side_effect = lambda ids: "".join(
            {17: "A", 99: ""}.get(tok, f"<{tok}>") for tok in ids
        )

        text_model = MagicMock()
        text_model.mtp = object()
        text_model.make_mtp_cache.return_value = ["mtp-cache"]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
            specprefill_threshold=1,
        )
        engine._loaded = True
        engine._text_model = text_model
        engine._text_tokenizer = tokenizer
        engine._draft_model = object()

        with (
            patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                return_value=["backbone-cache"],
            ),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                return_value=[],
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch(
                "vllm_mlx.specprefill.score_tokens",
                return_value=mx.array([1.0, 0.9, 0.8], dtype=mx.float32),
            ),
            patch(
                "vllm_mlx.specprefill.select_chunks",
                return_value=mx.array([0, 1, 2], dtype=mx.int32),
            ),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 3, 32), dtype=mx.float32),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=4,
                    temperature=0.6,
                    top_p=0.95,
                )
            ]

        assert [chunk.new_text for chunk in outputs] == ["A", "B"]
        assert captured["sampler_kwargs"] == {
            "temp": 0.6,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
        }
        assert captured["prompt"] == [17]
        assert captured["kwargs"]["mtp"] is True
        assert captured["kwargs"]["prompt_cache"] == ["backbone-cache", "mtp-cache"]
        assert captured["kwargs"]["max_tokens"] == 3
        assert captured["kwargs"]["logits_processors"] is None

    @pytest.mark.anyio
    async def test_stream_generate_text_forwards_logits_processors_and_sampler_args(
        self,
    ):
        """Text routing must preserve request-local decoding controls."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        sampler_calls = []
        penalty_calls = []
        user_processor = MagicMock()
        penalty_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        def fake_make_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return MagicMock()

        def fake_make_logits_processors(**kwargs):
            penalty_calls.append(kwargs)
            return [penalty_processor]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer

        with (
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.sample_utils.make_sampler", side_effect=fake_make_sampler),
            patch(
                "mlx_lm.sample_utils.make_logits_processors",
                side_effect=fake_make_logits_processors,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    min_p=0.1,
                    presence_penalty=1.5,
                    repetition_penalty=1.2,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert sampler_calls == [{"temp": 0.3, "top_p": 0.8, "top_k": 40, "min_p": 0.1}]
        assert penalty_calls == [{"repetition_penalty": 1.2, "presence_penalty": 1.5}]
        assert captured_kwargs["logits_processors"] == [
            user_processor,
            penalty_processor,
        ]

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_when_logits_processors_active(
        self,
    ):
        """Custom logits processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}
        user_processor = MagicMock()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[user_processor],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is user_processor

    @pytest.mark.anyio
    async def test_stream_generate_text_disables_mtp_for_thinking_processor(
        self,
    ):
        """Thinking-budget processors must fail closed to non-MTP decoding."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine("test-model", force_mllm=True, mtp=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        thinking_proc = MagicMock()

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[thinking_proc],
                )
            ]

        assert outputs[-1].text == "Hello"
        assert "mtp" not in captured_kwargs
        assert captured_kwargs["logits_processors"][0] is thinking_proc

    @pytest.mark.anyio
    async def test_stream_generate_text_passes_num_draft_tokens(self):
        """Text routing should forward configured MTP draft depth."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        captured_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(text="Hello", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_tokenizer = tokenizer

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                )
            ]

        assert outputs[-1].text == "Hello"
        assert captured_kwargs["mtp"] is True
        assert captured_kwargs["num_draft_tokens"] == 4

    @pytest.mark.anyio
    async def test_stream_generate_text_reenables_mtp_after_retired_processor_when_enabled(
        self,
    ):
        """Retired thinking processor handoff is an explicit opt-in path."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            if len(calls) == 1:
                processor.is_retired = True
                yield SimpleNamespace(token=11, text="Hello", finish_reason=None)
            else:
                yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [11]

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
        )
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch("mlx_lm.models.cache.trim_prompt_cache"),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 2
        assert "mtp" not in calls[0]
        assert calls[0]["logits_processors"][0] is processor
        assert calls[1]["mtp"] is True
        assert calls[1]["num_draft_tokens"] == 4
        assert "logits_processors" not in calls[1]

    @pytest.mark.anyio
    async def test_stream_generate_text_specprefill_reenables_mtp_after_retirement(
        self,
    ):
        """SpecPrefill retirement-to-MTP continuation is explicit opt-in."""
        from types import SimpleNamespace

        from vllm_mlx.engine.simple import SimpleEngine

        calls = []

        class RetiringProcessor:
            def __init__(self):
                self.is_retired = False

            def __call__(self, tokens, logits):
                return logits

        processor = RetiringProcessor()

        def fake_stream_generate(model, tokenizer, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            yield SimpleNamespace(token=12, text=" world", finish_reason="stop")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|im_start|>user\nhello"
        tokenizer.bos_token = None
        tokenizer.eos_token_id = 42
        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.decode.side_effect = lambda toks: "Hello" if toks == [11] else ""

        engine = SimpleEngine(
            "test-model",
            force_mllm=True,
            mtp=True,
            mtp_num_draft_tokens=4,
            specprefill_enabled=True,
        )
        engine._loaded = True
        engine._draft_model = MagicMock()
        engine._text_model = MagicMock()
        engine._text_model.mtp = MagicMock()
        engine._text_model.make_mtp_cache.return_value = []
        engine._text_tokenizer = tokenizer

        def fake_sample(tokens, logits, sampler, logits_processors):
            processor.is_retired = True
            return mx.array(11, dtype=mx.uint32), logits

        with (
            patch.dict(
                "os.environ",
                {"VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME": "1"},
            ),
            patch("mlx_lm.stream_generate", side_effect=fake_stream_generate),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch(
                "vllm_mlx.specprefill.score_tokens", return_value=mx.array([0.1, 0.2])
            ),
            patch("vllm_mlx.specprefill.select_chunks", return_value=mx.array([0, 1])),
            patch(
                "vllm_mlx.specprefill.sparse_prefill",
                return_value=mx.zeros((1, 1, 32)),
            ),
            patch("vllm_mlx.specprefill.cleanup_rope"),
            patch(
                "vllm_mlx.engine.simple._sample_with_processors",
                side_effect=fake_sample,
            ),
        ):
            outputs = [
                chunk
                async for chunk in engine._stream_generate_text(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=16,
                    temperature=0.7,
                    top_p=0.9,
                    specprefill=True,
                    logits_processors=[processor],
                )
            ]

        assert outputs[-1].text == "Hello world"
        assert len(calls) == 1
        assert calls[0]["mtp"] is True
        assert calls[0]["num_draft_tokens"] == 4
        assert "logits_processors" not in calls[0]

    @pytest.mark.anyio
    async def test_cancellation_does_not_release_lock_before_worker_finishes(
        self, mock_llm_model
    ):
        """A cancelled blocking chat call must not overlap the next worker."""
        from threading import Event, Lock

        from vllm_mlx.engine.simple import SimpleEngine

        first_started = Event()
        release_workers = Event()
        call_count = 0
        call_lock = Lock()

        def chat_side_effect(**kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
                mock_llm_model._concurrent_count += 1
                mock_llm_model._max_concurrent = max(
                    mock_llm_model._max_concurrent,
                    mock_llm_model._concurrent_count,
                )
                if current_call == 1:
                    first_started.set()

            try:
                release_workers.wait(timeout=1.0)
                result = MagicMock()
                result.text = f"response-{current_call}"
                result.tokens = [1, 2, 3]
                result.finish_reason = "stop"
                return result
            finally:
                with call_lock:
                    mock_llm_model._concurrent_count -= 1

        mock_llm_model.chat = MagicMock(side_effect=chat_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True

            task1 = asyncio.create_task(
                engine.chat(
                    messages=[{"role": "user", "content": "first"}], max_tokens=8
                )
            )
            await asyncio.to_thread(first_started.wait, 1.0)

            task1.cancel()
            task2 = asyncio.create_task(
                engine.chat(
                    messages=[{"role": "user", "content": "second"}], max_tokens=8
                )
            )

            await asyncio.sleep(0.05)
            release_workers.set()

            with pytest.raises(asyncio.CancelledError):
                await task1
            result2 = await task2

            assert result2.text == "response-2"
            assert mock_llm_model._max_concurrent == 1

    @pytest.mark.anyio
    async def test_specprefill_path_does_not_prelock_serialized_runner(self):
        """SpecPrefill should let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            assert not engine._generation_lock.locked()
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._draft_model = MagicMock()
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_specprefill(
                prompt="hello",
                tokens=[1, 2, 3, 4],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].completion_tokens == 0

    @pytest.mark.anyio
    async def test_text_mtp_path_does_not_prelock_serialized_runner(self):
        """Text-only MTP path should let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            assert not engine._generation_lock.locked()
            return []

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._text_model = MagicMock()
            engine._text_model.make_mtp_cache = MagicMock(return_value=[])
            engine._text_tokenizer = MagicMock()
            engine._text_tokenizer.apply_chat_template = MagicMock(return_value="hello")
            engine._text_tokenizer.bos_token = None
            engine._draft_model = None
            engine._run_blocking_serialized = fake_serialized  # type: ignore[method-assign]

            outputs = []
            async for chunk in engine._stream_generate_text(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=4,
                temperature=0.7,
                top_p=0.9,
            ):
                outputs.append(chunk)

            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].completion_tokens == 0

    @pytest.mark.anyio
    async def test_specprefill_threads_same_cancel_check_to_helpers(self):
        """SpecPrefill worker should pass one cooperative cancel hook through both phases."""
        from vllm_mlx.engine.simple import SimpleEngine

        captured = {}

        def fake_score_tokens(*args, cancel_check=None, **kwargs):
            captured["score"] = cancel_check
            return mx.array([0.5], dtype=mx.float32)

        def fake_sparse_prefill(*args, cancel_check=None, **kwargs):
            captured["prefill"] = cancel_check
            return mx.zeros((1, 1, 8), dtype=mx.float32)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()
            engine._model.tokenizer.decode = MagicMock(return_value="A")
            engine._model.tokenizer.eos_token_id = 0

            outputs = []
            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "mlx_lm.sample_utils.make_sampler",
                    return_value=lambda logits: mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.score_tokens", side_effect=fake_score_tokens
                ),
                patch(
                    "vllm_mlx.specprefill.select_chunks",
                    return_value=mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.sparse_prefill",
                    side_effect=fake_sparse_prefill,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                async for chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    outputs.append(chunk.new_text)

        assert outputs == ["A"]
        assert callable(captured["score"])
        assert captured["score"] is captured["prefill"]

    @pytest.mark.anyio
    async def test_cancelling_specprefill_request_stops_during_scoring(self):
        """Cancelling SpecPrefill should signal the blocking scorer and exit without output."""
        import time
        from threading import Event

        from vllm_mlx.engine.simple import SimpleEngine, _SpecPrefillCancelled

        score_started = Event()
        score_cancelled = Event()

        def fake_score_tokens(*args, cancel_check=None, **kwargs):
            assert callable(cancel_check)
            score_started.set()
            while True:
                try:
                    cancel_check()
                except _SpecPrefillCancelled:
                    score_cancelled.set()
                    raise
                time.sleep(0.01)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()

            async def consume():
                async for _chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    pytest.fail("Cancelled SpecPrefill request should not emit output")

            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "vllm_mlx.specprefill.score_tokens",
                    side_effect=fake_score_tokens,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                task = asyncio.create_task(consume())
                assert await asyncio.to_thread(score_started.wait, 1.0)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task

        assert await asyncio.to_thread(score_cancelled.wait, 1.0)

    @pytest.mark.anyio
    async def test_cancelling_specprefill_request_stops_during_sparse_prefill(self):
        """Cancelling SpecPrefill should signal the sparse-prefill loop and exit without output."""
        import time
        from threading import Event

        from vllm_mlx.engine.simple import SimpleEngine, _SpecPrefillCancelled

        prefill_started = Event()
        prefill_cancelled = Event()

        def fake_sparse_prefill(*args, cancel_check=None, **kwargs):
            assert callable(cancel_check)
            prefill_started.set()
            while True:
                try:
                    cancel_check()
                except _SpecPrefillCancelled:
                    prefill_cancelled.set()
                    raise
                time.sleep(0.01)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._loaded = True
            engine._draft_model = MagicMock()
            engine._model = MagicMock()
            engine._model.model = MagicMock()
            engine._model.tokenizer = MagicMock()

            async def consume():
                async for _chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    pytest.fail("Cancelled SpecPrefill request should not emit output")

            with (
                patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
                patch(
                    "vllm_mlx.specprefill.score_tokens",
                    return_value=mx.array([0.5], dtype=mx.float32),
                ),
                patch(
                    "vllm_mlx.specprefill.select_chunks",
                    return_value=mx.array([0], dtype=mx.int32),
                ),
                patch(
                    "vllm_mlx.specprefill.sparse_prefill",
                    side_effect=fake_sparse_prefill,
                ),
                patch("vllm_mlx.specprefill.cleanup_rope"),
            ):
                task = asyncio.create_task(consume())
                assert await asyncio.to_thread(prefill_started.wait, 1.0)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task
        assert await asyncio.to_thread(prefill_cancelled.wait, 1.0)
