# SPDX-License-Identifier: Apache-2.0
"""
Tests for chunked prefill compatibility with mlx-lm tuple format changes.

Regression test for issue #155: mlx-lm >= 0.31.0 added prompt_checkpoints
as a 7th element to BatchGenerator.unprocessed_prompts tuples. The chunked
prefill code in scheduler.py hardcoded a 6-element unpacking which crashed
with ValueError on the new format.
"""

from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx
    from mlx_lm.models import cache
    from mlx_lm.generate import BatchGenerator

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture
def batch_gen():
    """BatchGenerator with a minimal mock model.

    make_prompt_cache checks hasattr(model, 'make_cache') and MagicMock
    returns True for any attribute, so we delete it to fall through to
    the default KVCache-per-layer path.
    """
    model = MagicMock()
    model.layers = [MagicMock(), MagicMock()]
    del model.make_cache

    gen = BatchGenerator(
        model=model,
        max_tokens=100,
        prefill_batch_size=1,
        completion_batch_size=1,
    )
    return gen


class TestChunkedPrefillTupleCompat:
    """Verify chunked prefill handles varying tuple sizes from mlx-lm."""

    def test_7_element_tuples_unpack_correctly(self):
        """Issue #155: 7-element tuples from mlx-lm >= 0.31.0 must not crash."""
        mock_cache = MagicMock()
        mock_cache.empty.return_value = True
        batch_prompts = [
            (0, [1, 2, 3, 4, 5], 100, [mock_cache], None, [], -1),
            (1, [10, 20, 30], 50, [mock_cache], None, [], -1),
        ]

        (
            uids,
            inputs_raw,
            max_tokens_list,
            caches,
            samplers,
            logits_processors,
            *_extra,
        ) = zip(*batch_prompts)

        assert uids == (0, 1)
        assert inputs_raw == ([1, 2, 3, 4, 5], [10, 20, 30])
        assert max_tokens_list == (100, 50)
        assert len(_extra) == 1  # prompt_checkpoints
        assert _extra[0] == (-1, -1)

    def test_6_element_tuples_still_work(self):
        """Backward compat: old mlx-lm without prompt_checkpoints."""
        mock_cache = MagicMock()
        mock_cache.empty.return_value = True
        batch_prompts = [
            (0, [1, 2, 3], 100, [mock_cache], None, []),
        ]

        (
            uids,
            inputs_raw,
            max_tokens_list,
            caches,
            samplers,
            logits_processors,
            *_extra,
        ) = zip(*batch_prompts)

        assert uids == (0,)
        assert len(_extra) == 0

    def test_8_element_tuples_forward_compat(self):
        """Future-proofing: if mlx-lm adds more fields, still works."""
        mock_cache = MagicMock()
        mock_cache.empty.return_value = True
        batch_prompts = [
            (0, [1, 2, 3], 100, [mock_cache], None, [], -1, "future"),
        ]

        (
            uids,
            inputs_raw,
            max_tokens_list,
            caches,
            samplers,
            logits_processors,
            *_extra,
        ) = zip(*batch_prompts)

        assert uids == (0,)
        assert len(_extra) == 2

    def test_batch_generator_insert_creates_7_element_tuples(self, batch_gen):
        """Verify mlx-lm 0.31.x BatchGenerator.insert creates 7-element tuples."""
        prompt_cache = cache.make_prompt_cache(batch_gen.model)

        batch_gen.insert([[1, 2, 3, 4, 5]], max_tokens=[50], caches=[prompt_cache])

        assert len(batch_gen.unprocessed_prompts) == 1
        prompt_tuple = batch_gen.unprocessed_prompts[0]
        assert len(prompt_tuple) >= 7, (
            f"Expected >= 7 elements in prompt tuple, got {len(prompt_tuple)}. "
            f"mlx-lm may have changed tuple format again."
        )

    def test_chunked_prefill_with_7_element_tuples(self, batch_gen):
        """Integration: _install_chunked_prefill works with 7-element tuples."""
        from vllm_mlx.scheduler import _install_chunked_prefill

        prompt_cache = cache.make_prompt_cache(batch_gen.model)

        batch_gen.insert(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            max_tokens=[50],
            caches=[prompt_cache],
        )

        _install_chunked_prefill(batch_gen, budget=4)

        # Must NOT crash with "too many values to unpack".
        # Later errors (AttributeError, etc.) are expected since the mock
        # model doesn't do real inference — only the unpacking matters.
        try:
            batch_gen._next()
        except ValueError as e:
            if "unpack" in str(e).lower():
                pytest.fail(
                    f"Issue #155 regression: chunked prefill crashed on "
                    f"7-element tuple unpacking: {e}"
                )
        except Exception:
            pass  # Expected: mock model can't do real forward pass
