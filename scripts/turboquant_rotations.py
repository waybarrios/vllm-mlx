#!/usr/bin/env python3
"""TurboQuant rotation pipeline — R1, R2, R4 for outlier-free quantization.

Principle (QuaRot 2024 + JANGTQ generalization):

R1 — Hadamard rotation of the residual stream. Fused offline into:
    embed_tokens.weight  →  embed @ R1.T
    lm_head.weight       →  lm_head @ diag(1 + norm_final) @ R1.T
    every block.in_proj  →  W_in @ diag(1 + norm_block) @ R1.T
    every block.out_proj →  R1 @ W_out

R2 — per-head Hadamard rotation in attention head_dim. Fused offline into:
    q_proj / k_proj / v_proj  (output side: rotation per head)
    o_proj                    (input side: inverse rotation per head)
NOT applicable to "gated attention" (Qwen3.5/3.6) where output = o_proj(attn*sigmoid(gate))
because element-wise gate breaks the rotation cancellation.

R4 — Hadamard rotation of the intermediate activation, before down_proj.
    Fused offline: down_proj.weight @= R4_hadamard.T (input side rotation)
    Applied online (runtime patch): intermediate' = mx.hadamard_transform(intermediate)

Math:
    Pure RMSNorm: norm(R @ x) = R @ norm(x)  (rotation preserves norm)
    With +1.0 offset (qwen3.5/3.6): fold (1 + norm_weight) into downstream W first,
    then apply rotation, then set norm weight to 0 (so 1 + 0 = identity).

This module only contains the helpers and a self-test on a 64-dim toy model.
The actual model-walking conversion lives in turboquant_quantize.py.
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple


def hadamard_native_or_padded(x: mx.array, axis: int = -1) -> mx.array:
    """Apply Hadamard transform along `axis`. Falls back to padded RHT if size
    is not directly supported.

    mx.hadamard_transform supports n = m * 2^k for m in {1, 12, 20, 28}.
    For other sizes (rare), pad with zeros, transform, truncate.
    Returns array of same shape (or padded one if needed; caller must handle).
    """
    n = x.shape[axis]
    # Check native support.
    for m in (1, 12, 20, 28):
        if n % m == 0 and ((n // m) & (n // m - 1)) == 0:
            return mx.hadamard_transform(x)
    # No native support — try padded RHT (next-larger m·2^k).
    # For now raise; padding rarely needed for our target hidden sizes.
    raise NotImplementedError(
        f"Hadamard size {n} not natively supported; implement padded RHT"
    )


def random_hadamard_matrix(n: int, seed: int = 0) -> mx.array:
    """Build an n×n orthogonal matrix = diag(s) · H(n) / sqrt(n),
    where s ∈ {-1, +1}^n is sampled deterministically from `seed`.

    Used when we need an explicit R matrix (e.g. for offline weight fusion
    when we can't reshape the operand to apply mx.hadamard_transform directly).
    """
    rng = np.random.RandomState(seed)
    sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    # Build H(n) by applying mx.hadamard_transform to identity columns.
    I_n = mx.eye(n, dtype=mx.float32)  # noqa: E741 (matrix identity)
    H = mx.hadamard_transform(I_n)  # equivalent to H @ I = H
    # H is already 1/sqrt(n)-scaled by mx.hadamard_transform? Verify:
    # mx.hadamard_transform docs: applies (1/sqrt(n)) H_n. We want unitary so OK.
    R = H * mx.array(sign, dtype=mx.float32)[None, :]
    return R


# ---------------------------------------------------------------------------
# RMSNorm fold helpers — qwen3.5/3.6 store norm weight as offset (effective = 1+w).
# ---------------------------------------------------------------------------


def fold_rmsnorm_into_linear(
    norm_weight: mx.array,
    linear_weight: mx.array,
    has_offset: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Fold diag(scale) into a downstream Linear so the norm becomes identity.

    `scale = (1 + norm_weight) if has_offset else norm_weight`.
    `linear_weight` shape (out, in); we left-multiply by diag(scale) on input axis:
        W_new[o, i] = W_old[o, i] * scale[i]
    Returns (new_norm_weight=zeros_or_ones, new_linear_weight).
    """
    if has_offset:
        scale = 1.0 + norm_weight.astype(mx.float32)
        # New norm weight = 0 so that runtime computes (1 + 0) = identity.
        new_norm = mx.zeros_like(norm_weight)
    else:
        scale = norm_weight.astype(mx.float32)
        new_norm = mx.ones_like(norm_weight)
    new_lin = (linear_weight.astype(mx.float32) * scale[None, :]).astype(
        linear_weight.dtype
    )
    return new_norm, new_lin


# ---------------------------------------------------------------------------
# R1 fusion — global rotation of residual stream.
# ---------------------------------------------------------------------------


def apply_r1_to_in_projection(weight: mx.array, hadamard_axis_size: int) -> mx.array:
    """W_in @= R1.T equivalently: rotate input side via Hadamard on the in-axis.

    weight: (out, in). hadamard_axis_size: in.
    Returns rotated weight with same shape.
    """
    # mx.hadamard_transform applies on last axis: H(W^T) = W @ H.T (since H is orthogonal symmetric).
    # We want W @ R1.T which is identical to applying Hadamard to W (last axis).
    assert weight.shape[1] == hadamard_axis_size
    # Stay in fp32 after rotation — bf16 round-trip loses precision when
    # 2048-channel Hadamard accumulates ~sqrt(2048)*eps relative error.
    return mx.hadamard_transform(weight.astype(mx.float32))


def apply_r1_to_out_projection(weight: mx.array, hadamard_axis_size: int) -> mx.array:
    """W_out = R1 @ W_out, equivalently: rotate output side.

    weight: (out, in). hadamard_axis_size: out.
    Transpose, apply Hadamard on last axis, transpose back.
    """
    assert weight.shape[0] == hadamard_axis_size
    wT = mx.swapaxes(weight.astype(mx.float32), 0, 1)  # (in, out)
    wT_rot = mx.hadamard_transform(wT)  # H @ W^T = (W @ H)^T... careful
    # Actually mx.hadamard_transform applied to last axis of (in, out) gives:
    # result[i, :] = H @ wT[i, :] is INCORRECT — mx applies H to last dim as vector.
    # Equivalent to W^T @ H^T (i.e. left-multiplication on the transposed axis).
    # We want R1 @ W_out → (R1 @ W_out).T = W_out.T @ R1.T → apply Hadamard to last axis of W^T.
    rotated = mx.swapaxes(wT_rot, 0, 1)  # back to (out, in)
    return rotated  # stay fp32, no down-cast


# ---------------------------------------------------------------------------
# R4 fusion (offline part) — input-side rotation of down_proj.
# ---------------------------------------------------------------------------


def apply_r4_to_down_proj(weight: mx.array, intermediate_size: int) -> mx.array:
    """down_proj.weight @= R4.T  (input-axis rotation).

    weight: (hidden_size, intermediate_size). Same as apply_r1_to_in_projection.
    """
    return apply_r1_to_in_projection(weight, intermediate_size)


# ---------------------------------------------------------------------------
# Self-test — tiny model: norm → linear → activation → norm → linear.
# Validate forward equivalence pre/post-rotation.
# ---------------------------------------------------------------------------


class _ToyRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.random.normal((dim,)) * 0.1  # like Qwen3.5/3.6 offset

    def __call__(self, x):
        # Effective norm: (x / rms(x)) * (1 + weight)  -- offset semantics
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * (1.0 + self.weight)


class _ToyBlock(nn.Module):
    """Minimal block: norm → up_proj → SiLU → down_proj → residual."""

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.norm = _ToyRMSNorm(dim)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x):
        h = self.norm(x)
        h = self.up_proj(h)
        h = nn.silu(h)
        h = self.down_proj(h)
        return x + h


def _fold_block_norm(block: _ToyBlock):
    """Fold (1 + norm.weight) into up_proj input side, set norm to zero."""
    new_norm, new_up = fold_rmsnorm_into_linear(
        block.norm.weight, block.up_proj.weight, has_offset=True
    )
    block.norm.weight = new_norm
    block.up_proj.weight = new_up


def _fuse_r1(block: _ToyBlock, dim: int):
    """Fuse R1 (Hadamard) into block: up_proj input-side, down_proj output-side."""
    # In this toy, R1 acts on residual stream which is `dim`.
    # up_proj input has `dim` → rotate input.
    block.up_proj.weight = apply_r1_to_in_projection(block.up_proj.weight, dim)
    # down_proj output has `dim` → rotate output.
    block.down_proj.weight = apply_r1_to_out_projection(block.down_proj.weight, dim)


def _fuse_r4(block: _ToyBlock, intermediate: int):
    """Fuse R4 into block: down_proj input-side rotation. R4 must be applied
    online to the activation before down_proj (runtime patch)."""
    block.down_proj.weight = apply_r4_to_down_proj(block.down_proj.weight, intermediate)


class _ToyBlockWithR4(_ToyBlock):
    """Same as _ToyBlock but applies R4 online before down_proj."""

    def __call__(self, x):
        h = self.norm(x)
        h = self.up_proj(h)
        h = nn.silu(h)
        h = mx.hadamard_transform(h)  # R4 online
        h = self.down_proj(h)
        return x + h


def self_test():
    """Verify forward equivalence: R1 alone, R4 alone, R1+R4 combined."""
    mx.random.seed(42)
    dim = 64  # 64 = 1 * 2^6, native Hadamard
    intermediate = 256  # 256 = 1 * 2^8, native Hadamard
    batch, seq = 2, 4

    # Build reference block + input.
    ref = _ToyBlock(dim, intermediate)
    x_in = mx.random.normal((batch, seq, dim))
    y_ref = ref(x_in)
    mx.eval(y_ref)

    # Build rotated version (clone state, then fold + rotate).

    def clone_block(template):
        b = _ToyBlock(dim, intermediate)
        b.norm.weight = template.norm.weight + 0  # copy
        b.up_proj.weight = template.up_proj.weight + 0
        b.down_proj.weight = template.down_proj.weight + 0
        return b

    # === R1 alone ===
    b1 = clone_block(ref)
    _fold_block_norm(b1)
    _fuse_r1(b1, dim)
    # R1 changes residual stream basis: pre-rotate input, post-rotate output.
    x_rot = mx.hadamard_transform(x_in.astype(mx.float32))
    y_rot = b1(x_rot)
    y_back = mx.hadamard_transform(y_rot)  # H is involutive (H @ H = I), un-rotate
    diff_r1 = mx.abs(y_back - y_ref).max().item()
    print(f"  R1 only:  max abs diff = {diff_r1:.3e}  (expect < 1e-4)")

    # === R4 alone ===
    b4 = clone_block(ref)
    _fold_block_norm(b4)
    _fuse_r4(b4, intermediate)
    # R4 acts inside the block — output should be IDENTICAL to ref (no input change needed).
    b4_with_runtime = _ToyBlockWithR4(dim, intermediate)
    b4_with_runtime.norm.weight = b4.norm.weight + 0
    b4_with_runtime.up_proj.weight = b4.up_proj.weight + 0
    b4_with_runtime.down_proj.weight = b4.down_proj.weight + 0
    y_r4 = b4_with_runtime(x_in)
    diff_r4 = mx.abs(y_r4 - y_ref).max().item()
    print(f"  R4 only:  max abs diff = {diff_r4:.3e}  (expect < 1e-4)")

    # === R1 + R4 combined ===
    b14 = clone_block(ref)
    _fold_block_norm(b14)
    _fuse_r1(b14, dim)
    _fuse_r4(b14, intermediate)
    b14_with_runtime = _ToyBlockWithR4(dim, intermediate)
    b14_with_runtime.norm.weight = b14.norm.weight + 0
    b14_with_runtime.up_proj.weight = b14.up_proj.weight + 0
    b14_with_runtime.down_proj.weight = b14.down_proj.weight + 0
    x_rot = mx.hadamard_transform(x_in.astype(mx.float32))
    y_14 = b14_with_runtime(x_rot)
    y_14_back = mx.hadamard_transform(y_14)
    diff_14 = mx.abs(y_14_back - y_ref).max().item()
    print(f"  R1 + R4:  max abs diff = {diff_14:.3e}  (expect < 1e-4)")

    # Pass criterion: all rotations must give numerically equivalent forward.
    ok = max(diff_r1, diff_r4, diff_14) < 1e-3
    print(f"\n  {'✓ PASS' if ok else '✗ FAIL'} — rotation equivalence")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant rotation pipeline — toy block equivalence test")
    print("=" * 60)
    self_test()
