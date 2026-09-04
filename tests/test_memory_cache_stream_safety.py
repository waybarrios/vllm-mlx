# SPDX-License-Identifier: Apache-2.0
"""MLX thread-locality regression tests for ``MemoryAwarePrefixCache.store``.

MLX streams are owned by the thread that created them, and every thread gets
its *own* default GPU stream.  A **lazy** array built on thread A aborts the
process when it is evaluated on thread B::

    libc++abi: terminating due to uncaught exception of type std::runtime_error:
    There is no Stream(gpu, 1) in current thread.

Through ``mx.eval`` that surfaces as a catchable ``RuntimeError``; through
numpy's buffer protocol (``np.array(arr)``, which the SSD spill path uses) the
C++ exception unwinds with no handler and hits ``std::terminate`` -> SIGABRT.

``_evict_lru`` structurally spills the *least-recently-used* entry, i.e. one
built by an earlier request on a different thread, so any cache left lazy at
``store()`` time is a latent uncatchable abort.  These tests pin the invariant
that ``store()`` materializes on the producing thread.

Upstream satisfies this invariant in ``_detach_cache_for_storage``, which
accumulates every leaf into one ``eval_targets`` list -- recursing into
``CacheList`` children -- and issues a single ``mx.eval`` on the calling
thread (``memory_cache.py``, the ``is_root`` branch).  These tests are
black-box on purpose: they pin the *invariant*, not that mechanism.

The cross-thread cases run in a subprocess because the failure mode is
``abort()``, which would take the whole pytest run with it.
"""

import subprocess
import sys
import textwrap

PRELUDE = """
import threading
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

mx.eval(mx.ones((2, 2)))  # bind the main thread's gpu stream

def make_kv(lazy_scale):
    layer = KVCache()
    # `* lazy_scale` keeps the graph UNEVALUATED, exactly like a real decode
    layer.keys = mx.ones((1, 2, 8, 64), dtype=mx.float32) * lazy_scale
    layer.values = mx.ones((1, 2, 8, 64), dtype=mx.float32) * lazy_scale
    layer.offset = 8
    return layer
"""


def _run_subprocess(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestStoreMaterializesOnProducerThread:
    """``store()`` must leave nothing lazy for a later thread to evaluate."""

    def test_cache_list_survives_numpy_on_another_thread(self):
        """GLM-5.2 DSA shape: CacheList layers, no KVCache trim path at all."""
        result = _run_subprocess("""
            from mlx_lm.models.cache import CacheList

            cache_obj = MemoryAwarePrefixCache(
                model=object(), config=MemoryCacheConfig(min_prefix_tokens=1)
            )
            tokens = list(range(16))

            def producer():
                layers = [CacheList(make_kv(3.0), make_kv(5.0)) for _ in range(2)]
                assert cache_obj.store(tokens, layers)

            t = threading.Thread(target=producer)
            t.start()
            t.join()

            # The spill path: numpy buffer protocol, on a DIFFERENT thread.
            entry = cache_obj._entries[tuple(tokens)]
            for layer in entry.cache:
                for inner in layer.caches:
                    assert np.array(inner.keys).sum() > 0
                    assert np.array(inner.values).sum() > 0
            print("SURVIVED")
            """)
        assert result.returncode == 0, (
            f"exit={result.returncode} (134 == SIGABRT)\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "SURVIVED" in result.stdout

    def test_quantized_cache_survives_numpy_on_another_thread(self):
        """`_quantize_cache` builds fresh lazy arrays AFTER the trim step."""
        result = _run_subprocess("""
            cache_obj = MemoryAwarePrefixCache(
                model=object(),
                config=MemoryCacheConfig(
                    min_prefix_tokens=1,
                    kv_quantize=True,
                    kv_min_quantize_tokens=1,
                    kv_group_size=64,
                ),
            )
            tokens = list(range(16))

            def producer():
                assert cache_obj.store(tokens, [make_kv(3.0), make_kv(5.0)])

            t = threading.Thread(target=producer)
            t.start()
            t.join()

            entry = cache_obj._entries[tuple(tokens)]
            for layer in entry.cache:
                for part in (*layer.keys, *layer.values):
                    np.array(part)
            print("SURVIVED")
            """)
        assert result.returncode == 0, (
            f"exit={result.returncode} (134 == SIGABRT)\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "SURVIVED" in result.stdout
