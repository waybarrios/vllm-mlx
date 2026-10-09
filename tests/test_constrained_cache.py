"""Portable regressions for constrained-decoding tokenizer adaptation."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path


def _load_cache_module():
    path = Path(__file__).resolve().parents[1] / "vllm_mlx/constrained/cache.py"
    loader = importlib.machinery.SourceFileLoader(
        "constrained_cache_under_test", str(path)
    )
    spec = importlib.util.spec_from_loader("constrained_cache_under_test", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_get_vocab_size_resolves_vlm_processor_tokenizer():
    cache = _load_cache_module()

    class Tokenizer:
        all_special_ids: list[int] = []
        vocab_size = 42

    class Processor:
        tokenizer = Tokenizer()

    assert cache._get_vocab_size(Processor()) == 42
