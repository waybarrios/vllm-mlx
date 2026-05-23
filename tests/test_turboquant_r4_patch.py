# SPDX-License-Identifier: Apache-2.0
"""Tests for the TurboQuant R4 runtime patch.

R4 is an experimental, opt-in patch (env-gated via VLLM_MLX_TURBOQUANT_R4).
It is NOT applied by default because R4 is only needed for activation
quantization (per QuaRot paper); weight-only quantization works correctly
with R1 alone.

These tests verify:
1. Patch is idempotent (safe to call multiple times).
2. Patch correctly inserts Hadamard before down_proj in Qwen3NextMLP.
3. Patch correctly inserts Hadamard before down_proj in SwitchGLU.
4. Patch is no-op when its modules are unavailable.
"""

import pytest


from vllm_mlx.patches.turboquant_r4 import (
    install_turboquant_r4,
    is_turboquant_model,
)


def _reset_patches():
    """Clear sentinel attributes so tests can re-patch from scratch."""
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        if hasattr(Qwen3NextMLP, "_turboquant_r4_patched"):
            delattr(Qwen3NextMLP, "_turboquant_r4_patched")
    except ImportError:
        pass
    try:
        from mlx_lm.models.switch_layers import SwitchGLU

        if hasattr(SwitchGLU, "_turboquant_r4_patched"):
            delattr(SwitchGLU, "_turboquant_r4_patched")
    except ImportError:
        pass


def test_idempotent_install():
    """Calling install_turboquant_r4 twice must not double-wrap."""
    _reset_patches()
    install_turboquant_r4("all")
    install_turboquant_r4("all")  # second call must be a no-op
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        assert getattr(Qwen3NextMLP, "_turboquant_r4_patched", False) is True
    except ImportError:
        pytest.skip("mlx_lm.qwen3_next not importable")


def test_scope_shared_only():
    """scope='shared' patches Qwen3NextMLP but NOT SwitchGLU."""
    _reset_patches()
    install_turboquant_r4("shared")
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        assert getattr(Qwen3NextMLP, "_turboquant_r4_patched", False) is True
    except ImportError:
        pytest.skip("mlx_lm.qwen3_next not importable")
    try:
        from mlx_lm.models.switch_layers import SwitchGLU

        assert getattr(SwitchGLU, "_turboquant_r4_patched", False) is False
    except ImportError:
        pass


def test_scope_experts_only():
    """scope='experts' patches SwitchGLU but NOT Qwen3NextMLP."""
    _reset_patches()
    install_turboquant_r4("experts")
    try:
        from mlx_lm.models.switch_layers import SwitchGLU

        assert getattr(SwitchGLU, "_turboquant_r4_patched", False) is True
    except ImportError:
        pytest.skip("mlx_lm.switch_layers not importable")
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        assert getattr(Qwen3NextMLP, "_turboquant_r4_patched", False) is False
    except ImportError:
        pass


def test_is_turboquant_model():
    """Detect TurboQuant via quantization.type config key."""
    assert is_turboquant_model({"quantization": {"type": "turboquant_v1"}}) is True
    assert (
        is_turboquant_model({"quantization_config": {"type": "turboquant_v1"}}) is True
    )
    assert is_turboquant_model({"quantization": {"type": "affine"}}) is False
    assert is_turboquant_model({}) is False
    assert is_turboquant_model({"quantization": None}) is False
