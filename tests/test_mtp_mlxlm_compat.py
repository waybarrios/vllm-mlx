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
