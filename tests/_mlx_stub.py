# SPDX-License-Identifier: Apache-2.0
"""Opt-in mlx/mlx_lm stub for test files that don't need real MLX behavior.

Deliberately NOT wired into conftest.py: conftest.py loads unconditionally
for every pytest invocation that touches this directory, so a stub
installed there would leak into ``sys.modules`` for *every* test file
collected in the same run -- including ones like test_memory_cache.py and
test_prefix_cache.py, which have their own working ``except ImportError``
fallback for real mlx_lm absence and break when that import instead
"succeeds" against a MagicMock (e.g. ``isinstance(x, mock_attr)`` raises
``TypeError``, not a clean not-mlx result).

Call ``install_if_unavailable()`` at the top of a test file, before any
``import vllm_mlx...`` line, and keep that file's CI selector in its own
pytest invocation, separate from files that have their own mlx-optional
handling (see .github/workflows/ci.yml's dedicated
"Run MLX-stubbed metrics/scheduler tests" step). A no-op wherever mlx is
actually installed (the Apple Silicon CI job, or a dev machine with mlx
present).
"""


def install_if_unavailable() -> None:
    try:
        import mlx.core  # noqa: F401

        return
    except ImportError:
        pass

    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    for name in (
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_lm",
        "mlx_lm.generate",
        "mlx_lm.models",
        "mlx_lm.models.cache",
        "mlx_lm.sample_utils",
        "mlx_lm.tokenizer_utils",
    ):
        if name in sys.modules:
            continue
        stub = MagicMock(name=name)
        # A real ModuleSpec, not None: importlib.util.find_spec() (used by
        # e.g. transformers' is_mlx_available()) raises ValueError on a
        # sys.modules entry whose __spec__ is falsy rather than a spec.
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        stub.__path__ = []  # let it satisfy submodule imports like a package
        sys.modules[name] = stub
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, stub)
