# SPDX-License-Identifier: Apache-2.0
"""Regression test for cancellation-safe SimpleEngine serialization."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class SimpleEngineCancelSerializationTests(unittest.IsolatedAsyncioTestCase):
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
