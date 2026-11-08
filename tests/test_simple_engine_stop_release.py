"""SimpleEngine.stop() must free a cycle-held model before its final clear."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx.engine.simple import SimpleEngine

pytestmark = pytest.mark.anyio


async def test_stop_collects_cycle_held_model_before_clearing_cache():
    """mlx-vlm's load_processor sets ``tokenizer.stopping_criteria`` to a
    StoppingCriteria that points back at the tokenizer, so the processor (and
    the model wrapper holding it) is only freed by a cyclic GC pass. An idle
    server can go a long time without one, so after an idle unload the
    tokenizer stays live until some later collection. stop() must collect
    before its final clear_cache()."""

    class FakeTokenizer:
        pass

    class FakeStoppingCriteria:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tokenizer = FakeTokenizer()
    tokenizer.stopping_criteria = FakeStoppingCriteria(tokenizer)
    model = SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer))
    tokenizer_ref = weakref.ref(tokenizer)
    del tokenizer

    engine = SimpleEngine("test-model")
    engine._loaded = True
    engine._model = model
    del model

    alive_at_clear = []

    def fake_clear_cache():
        alive_at_clear.append(tokenizer_ref() is not None)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with patch("vllm_mlx.engine.simple.mx.clear_cache", fake_clear_cache):
            await engine.stop()
    finally:
        if gc_was_enabled:
            gc.enable()

    assert alive_at_clear == [False]
