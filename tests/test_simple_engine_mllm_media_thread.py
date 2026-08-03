import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.anyio
async def test_mllm_media_request_keeps_stream_chat_on_current_thread_when_text_route_exists():
    """Media-bearing MLLM requests must not move mlx_vlm stream_chat to a
    generic worker when an auxiliary text route exists.
    """
    from vllm_mlx.engine.simple import SimpleEngine

    class FakeMllmModel:
        def __init__(self):
            self._owner_thread = threading.get_ident()

        def stream_chat(self, **kwargs):
            if threading.get_ident() != self._owner_thread:
                raise RuntimeError("There is no Stream(gpu, 3) in current thread.")
            yield SimpleNamespace(
                text="image described",
                finish_reason="stop",
                prompt_tokens=5,
            )

    async def fail_if_called(*_args, **_kwargs):
        raise RuntimeError("media requests should not use generic worker routing")

    engine = SimpleEngine("test-model", force_mllm=True, mtp=False)
    engine._loaded = True
    engine._text_model = MagicMock()
    engine._text_tokenizer = MagicMock()
    engine._model = FakeMllmModel()
    engine._run_blocking_serialized = fail_if_called  # type: ignore[method-assign]

    outputs = [
        chunk
        async for chunk in engine.stream_chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ],
            max_tokens=16,
        )
    ]

    assert outputs[-1].text == "image described"
    assert outputs[-1].finish_reason == "stop"
