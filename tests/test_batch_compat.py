# SPDX-License-Identifier: Apache-2.0
"""Tests for mlx-lm batch compatibility helpers."""

from types import SimpleNamespace

from vllm_mlx.batch_compat import _sanitize_batch_generator_logits_processors


def test_sanitizer_aligns_native_batch_logits_processor_slots():
    """Native mlx-lm batches should keep one processor slot per UID."""
    processor = object()
    generation_batch = SimpleNamespace(uids=[], logits_processors=[[]])
    prompt_batch = SimpleNamespace(uids=[1, 2], logits_processors=[[processor], None])
    batch_generator = SimpleNamespace(
        _generation_batch=generation_batch,
        _prompt_batch=prompt_batch,
    )

    _sanitize_batch_generator_logits_processors(batch_generator)

    assert generation_batch.logits_processors == []
    assert prompt_batch.logits_processors == [[processor], []]


def test_sanitizer_normalizes_native_prompt_batch_extensions():
    """Plain requests should stay iterable when joining a strict batch."""
    processor = object()

    class FakePromptBatch:
        def __init__(self, uids, logits_processors):
            self.uids = uids
            self.logits_processors = logits_processors

        def extend(self, batch):
            processors = (
                batch.logits_processors
                if any(batch.logits_processors)
                else [None] * len(batch.uids)
            )
            self.uids.extend(batch.uids)
            self.logits_processors.extend(processors)

    prompt_batch = FakePromptBatch([1], [[processor]])
    batch_generator = SimpleNamespace(_prompt_batch=prompt_batch)
    _sanitize_batch_generator_logits_processors(batch_generator)
    wrapped_extend = prompt_batch.extend
    _sanitize_batch_generator_logits_processors(batch_generator)

    prompt_batch.extend(FakePromptBatch([2], [[]]))

    assert prompt_batch.extend is wrapped_extend
    assert prompt_batch.logits_processors == [[processor], []]


def test_sanitizer_preserves_legacy_mtp_state_support():
    """Legacy active and partial MTP state should remain normalized."""
    active_batch = SimpleNamespace(logits_processors=[None])
    partial = {"logits_processors": [None]}
    batch_generator = SimpleNamespace(active_batch=active_batch, _partial=partial)

    _sanitize_batch_generator_logits_processors(batch_generator)

    assert active_batch.logits_processors == [[]]
    assert partial["logits_processors"] == [[]]
