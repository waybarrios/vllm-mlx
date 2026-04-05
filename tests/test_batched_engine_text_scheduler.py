# SPDX-License-Identifier: Apache-2.0
"""Focused regressions for BatchedEngine raw-output handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.request import RequestOutput


def _make_mllm_engine() -> BatchedEngine:
    engine = BatchedEngine("dummy-model", force_mllm=True)
    engine._loaded = True
    return engine


def test_generate_preserves_raw_output_for_mllm_scheduler():
    async def _run():
        engine = _make_mllm_engine()
        engine._mllm_scheduler = AsyncMock()
        engine._mllm_scheduler.generate.return_value = RequestOutput(
            request_id="req-1",
            output_text="<|channel>thought\nplan<channel|>Final answer",
            prompt_tokens=9,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )

        output = await engine.generate(prompt="hello", raw_output=True)

        assert output.text == "<|channel>thought\nplan<channel|>Final answer"

    asyncio.run(_run())


def test_stream_generate_preserves_raw_output_for_mllm_scheduler():
    async def _run():
        engine = _make_mllm_engine()
        engine._mllm_scheduler = MagicMock()
        engine._mllm_scheduler.add_request_async = AsyncMock(return_value="req-1")

        async def _stream_outputs(_request_id):
            yield RequestOutput(
                request_id="req-1",
                output_text="<|channel>thought\n",
                new_text="<|channel>thought\n",
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
            )
            yield RequestOutput(
                request_id="req-1",
                output_text="<|channel>thought\nplan<channel|>Final answer",
                new_text="plan<channel|>Final answer",
                prompt_tokens=4,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
            )

        engine._mllm_scheduler.stream_outputs = _stream_outputs

        outputs = []
        async for output in engine.stream_generate(prompt="hello", raw_output=True):
            outputs.append((output.text, output.new_text))

        assert outputs == [
            ("<|channel>thought\n", "<|channel>thought\n"),
            (
                "<|channel>thought\nplan<channel|>Final answer",
                "plan<channel|>Final answer",
            ),
        ]

    asyncio.run(_run())
