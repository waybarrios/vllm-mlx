# SPDX-License-Identifier: Apache-2.0
"""D15 regression: media requests must never touch the prompt prefix cache.

The prefix cache is keyed on token IDs alone, and every image/audio clip
occupies identical placeholder IDs — so a media request stored or fetched
under a token-only key replays a DIFFERENT image's KV for any later request
with the same prompt text. Found live: 67 distinct document pages through
an MLLM serve produced 2 distinct transcriptions.

These tests pin the fixed contract at the seams:
  1. ``_maybe_store_prefix_cache`` stores text-only requests and skips media.
  2. ``MLLMBatchRequest.is_text_only`` defaults False — an unpreprocessed request
     is treated as media, so the safe direction is cache bypass.
"""

from unittest.mock import MagicMock

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


def _bare_generator(prefix_cache) -> MLLMBatchGenerator:
    gen = object.__new__(MLLMBatchGenerator)
    gen.prefix_cache = prefix_cache
    gen._think_suffix_len = 0
    return gen


def _finished_batch(req: MLLMBatchRequest):
    batch = MagicMock()
    batch.requests = [req]
    batch.num_tokens = [4]
    batch.extract_cache.return_value = []
    return batch


def _request(*, text_only: bool, images=None) -> MLLMBatchRequest:
    req = MLLMBatchRequest(
        uid=1,
        request_id="r1",
        prompt="Transcribe this page.",
        images=images,
    )
    req.input_ids = mx.array([1, 2, 3, 4, 5])
    req.is_text_only = text_only
    return req


def test_media_request_is_never_stored():
    prefix_cache = MagicMock()
    gen = _bare_generator(prefix_cache)
    req = _request(text_only=False, images=["data:image/png;base64,AAAA"])

    gen._maybe_store_prefix_cache(_finished_batch(req), end_indices=[0])

    prefix_cache.store.assert_not_called()


def test_text_only_request_is_still_stored():
    prefix_cache = MagicMock()
    gen = _bare_generator(prefix_cache)
    req = _request(text_only=True)

    gen._maybe_store_prefix_cache(_finished_batch(req), end_indices=[0])

    assert prefix_cache.store.call_count == 1
    stored_key = prefix_cache.store.call_args.args[0]
    assert stored_key == [1, 2, 3, 4, 5]


def test_is_text_only_defaults_to_media():
    # Safe-direction check: before preprocessing runs, a request must count
    # as media so it can never be served another request's cached KV.
    req = MLLMBatchRequest(uid=2, request_id="r2", prompt="hello")
    assert req.is_text_only is False
