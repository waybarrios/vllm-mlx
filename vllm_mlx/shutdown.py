# SPDX-License-Identifier: Apache-2.0
"""Process-exit policy for the serve entry points.

This module holds the *decision* of whether the process may exit at the end of
lifespan shutdown without running CPython finalization, separated from the
MLX-heavy modules so it can be tested on any platform. The mechanics of the
exit itself (``exit_without_finalizing``) live here too; only its optional
temp-file sweep imports an engine module, and lazily.

Background: finalization joins non-daemon threads that have touched MLX, and
MLX keeps its compile cache in thread-local storage whose destructor calls
``_Py_Dealloc`` without a thread state or the GIL — EXC_BAD_ACCESS at teardown,
after every request has already been served. The full write-up lives on
``exit_without_finalizing``'s original home, ``cli._exit_without_finalizing``,
which now delegates here.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

__all__ = [
    "clean_exit_enabled",
    "should_exit_without_finalizing",
    "exit_without_finalizing",
]


def clean_exit_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the early-exit workaround is enabled (VLLM_MLX_CLEAN_EXIT).

    Defaults to enabled; ``VLLM_MLX_CLEAN_EXIT=0`` restores normal interpreter
    finalization for debugging the teardown crash itself.
    """
    if env is None:
        env = os.environ
    return env.get("VLLM_MLX_CLEAN_EXIT", "1") != "0"


def should_exit_without_finalizing(
    primary_exc: BaseException | None,
    cleanup_exc: BaseException | None,
    exit_process_after_shutdown: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Decide whether lifespan shutdown may skip CPython finalization.

    Only a shutdown that raised nothing may skip finalization, and only when a
    serve entry point opted in. If either the request-serving phase or the
    cleanup phase failed, the caller has to see the exception and the process
    has to exit non-zero — exiting early would report a failed shutdown as a
    success. Embedded/library use never opts in
    (``exit_process_after_shutdown`` stays false), because exiting the host
    process would be indefensible.
    """
    if not exit_process_after_shutdown:
        return False
    if primary_exc is not None or cleanup_exc is not None:
        return False
    return clean_exit_enabled(env)


def exit_without_finalizing(status: int = 0) -> None:
    """Leave the process without running interpreter finalization.

    Runs the one cleanup we own (multimodal temp files), flushes stdio, and
    calls ``os._exit``. Honors ``VLLM_MLX_CLEAN_EXIT=0`` as a final guard for
    direct callers; the lifespan path has already consulted
    ``should_exit_without_finalizing``.
    """
    if not clean_exit_enabled():
        return
    try:
        from .models.mllm import cleanup_all_temp_files

        cleanup_all_temp_files()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
