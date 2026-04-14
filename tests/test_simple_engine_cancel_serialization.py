# SPDX-License-Identifier: Apache-2.0
"""Regression test for cancellation-safe SimpleEngine serialization."""

from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import MagicMock, patch


class SimpleEngineCancelSerializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_does_not_release_lock_before_worker_finishes(self):
        """A cancelled request must not let a second MLX worker overlap."""
        from vllm_mlx.engine.simple import SimpleEngine

        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        model._concurrent_count = 0
        model._max_concurrent = 0

        first_started = threading.Event()
        release_workers = threading.Event()
        call_count = 0
        call_lock = threading.Lock()

        def generate_side_effect(**kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
                model._concurrent_count += 1
                model._max_concurrent = max(
                    model._max_concurrent, model._concurrent_count
                )
                if current_call == 1:
                    first_started.set()

            release_workers.wait(timeout=1.0)

            with call_lock:
                model._concurrent_count -= 1

            result = MagicMock()
            result.text = f"response-{current_call}"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = model
            engine._loaded = True

            task1 = asyncio.create_task(engine.generate(prompt="first", max_tokens=8))
            await asyncio.to_thread(first_started.wait, 1.0)

            task1.cancel()
            task2 = asyncio.create_task(engine.generate(prompt="second", max_tokens=8))

            await asyncio.sleep(0.05)
            release_workers.set()

            with self.assertRaises(asyncio.CancelledError):
                await task1
            result2 = await task2

            self.assertEqual(result2.text, "response-2")
            self.assertEqual(
                model._max_concurrent,
                1,
                "cancellation released the generation lock before the first worker finished",
            )

    async def test_specprefill_path_does_not_prelock_serialized_runner(self):
        """Specprefill streaming must let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            self.assertFalse(engine._generation_lock.locked())
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

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].finished)
            self.assertEqual(outputs[0].completion_tokens, 0)

    async def test_text_mtp_path_does_not_prelock_serialized_runner(self):
        """Text-only MTP streaming must let _run_blocking_serialized own the lock."""
        from vllm_mlx.engine.simple import SimpleEngine

        async def fake_serialized(func, *args, **kwargs):
            self.assertFalse(engine._generation_lock.locked())
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

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].finished)
            self.assertEqual(outputs[0].completion_tokens, 0)


if __name__ == "__main__":
    unittest.main()
