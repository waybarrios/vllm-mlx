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


if __name__ == "__main__":
    unittest.main()
