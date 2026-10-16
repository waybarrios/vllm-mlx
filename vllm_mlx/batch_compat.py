# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for mlx-lm 0.31.3 batch state.

That release can retain stale empty processor slots after filtering and can
turn a plain request's empty slot into ``None`` while extending a prompt batch.
Both behaviors were fixed upstream after 0.31.3 and can wedge mixed constrained
and unconstrained traffic (#784).
"""

from functools import wraps


def _normalize_logits_processors(logits_processors):
    """Normalize empty per-sequence processor slots to lists."""
    if logits_processors is None:
        return None
    return [processors or [] for processors in logits_processors]


def _sanitize_batch_logits_processors(batch) -> None:
    """Keep a native batch's processor slots aligned with its UIDs."""
    if batch is None or not hasattr(batch, "logits_processors"):
        return

    processors = _normalize_logits_processors(batch.logits_processors) or []
    uids = getattr(batch, "uids", None)
    if uids is not None:
        num_sequences = len(uids)
        if len(processors) > num_sequences:
            processors = processors[-num_sequences:] if num_sequences else []
        elif len(processors) < num_sequences:
            processors.extend([[] for _ in range(num_sequences - len(processors))])

    batch.logits_processors = processors


def _wrap_prompt_batch_extend(prompt_batch) -> None:
    """Normalize slots created inside mlx-lm's prompt-batch extension."""
    if prompt_batch is None or getattr(
        prompt_batch, "_vllm_mlx_logits_processor_compat", False
    ):
        return

    original_extend = getattr(prompt_batch, "extend", None)
    if not callable(original_extend):
        return

    @wraps(original_extend)
    def extend_and_normalize(*args, **kwargs):
        result = original_extend(*args, **kwargs)
        _sanitize_batch_logits_processors(prompt_batch)
        return result

    try:
        prompt_batch.extend = extend_and_normalize
        prompt_batch._vllm_mlx_logits_processor_compat = True
    except (AttributeError, TypeError):
        # Newer mlx-lm objects may prevent per-instance method wrapping. Their
        # native implementation already normalizes these transitions.
        return


def _sanitize_batch_generator_logits_processors(batch_generator) -> None:
    """Sanitize stale BatchGenerator processor state before decode."""
    active_batch = getattr(batch_generator, "active_batch", None)
    _sanitize_batch_logits_processors(active_batch)

    partial = getattr(batch_generator, "_partial", None)
    if isinstance(partial, dict) and "logits_processors" in partial:
        partial["logits_processors"] = _normalize_logits_processors(
            partial["logits_processors"]
        )

    generation_batch = getattr(batch_generator, "_generation_batch", None)
    _sanitize_batch_logits_processors(generation_batch)

    prompt_batch = getattr(batch_generator, "_prompt_batch", None)
    _sanitize_batch_logits_processors(prompt_batch)
    _wrap_prompt_batch_extend(prompt_batch)
