# SPDX-License-Identifier: Apache-2.0
"""Regression tests: --enable-mtp must not crash on mlx-lm >= 0.31.

The MTP monkey-patch wraps ``BatchGenerator._step``, which was refactored
away in mlx-lm 0.31.x (decode moved to ``GenerationBatch._step``).  On an
incompatible BatchGenerator the install must warn and no-op instead of
raising ``AttributeError`` at generator creation.
"""

import logging

from mlx_lm.generate import BatchGenerator

from vllm_mlx.scheduler import _install_mtp


class _ModernBatchGenerator:
    """Mimics the mlx-lm >= 0.31 BatchGenerator surface (no ``_step``)."""

    def insert(self, *args, **kwargs):
        raise NotImplementedError

    def remove(self, *args, **kwargs):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError


def test_install_mtp_skips_without_step_hook(caplog):
    bg = _ModernBatchGenerator()

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        _install_mtp(bg, model=None)

    assert not hasattr(bg, "_step")
    assert bg.__dict__.get("_next") is None
    assert any("[MTP] disabled" in rec.message for rec in caplog.records)


def test_installed_mlx_lm_batch_generator_lacks_step_hook():
    # Documents the incompatibility this guard exists for: if this ever
    # fails, mlx-lm regained a _step hook and the MTP patch (and this
    # guard) should be revisited.
    assert not hasattr(BatchGenerator, "_step")


def test_scheduler_enable_mtp_survives_modern_batch_generator(caplog):
    """Integration: the real construction path with enable_mtp=True.

    Builds a real Scheduler and calls _create_batch_generator, so the guard is
    exercised through the actual call path with the installed mlx-lm
    BatchGenerator (which lacks the _step hook).  Construction and request
    scheduling must continue, MTP must be absent from the generator, and the
    operator-visible warning must be emitted.
    """
    from types import SimpleNamespace

    from vllm_mlx.scheduler import (
        Request,
        SamplingParams,
        Scheduler,
        SchedulerConfig,
    )

    model = SimpleNamespace(mtp=object())  # has an MTP head: only the guard disables
    tokenizer = SimpleNamespace(
        encode=lambda text: list(range(len(text.split()))),
        decode=lambda ids: " ".join(str(i) for i in ids),
        eos_token_id=0,
        eos_token_ids={0},
    )
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(enable_prefix_cache=False, enable_mtp=True),
    )

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        bg = scheduler._create_batch_generator(SamplingParams())

    # Construction survived on the real, modern BatchGenerator.
    assert bg is not None
    assert not hasattr(bg, "_step")
    # MTP is absent: the install no-oped, so no MTP surface was attached.
    assert not hasattr(bg, "get_mtp_stats")
    assert any("[MTP] disabled" in rec.message for rec in caplog.records)

    # Request scheduling continues.
    scheduler.add_request(
        Request(
            request_id="mtp-guard-1",
            prompt="hello there",
            sampling_params=SamplingParams(max_tokens=4),
        )
    )
    assert scheduler.has_requests()
    assert scheduler.get_num_waiting() == 1
