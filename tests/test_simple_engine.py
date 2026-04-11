# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
