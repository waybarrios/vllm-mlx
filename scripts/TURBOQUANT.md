# TurboQuant — Hadamard-rotated weight quantization

QuaRot-style R1 (residual-stream Hadamard) rotation fused offline into model
weights, enabling 4-bit quantization on hybrid-attention MoE models like
Qwen3.5/3.6 and MiniMax-M2 without quality degradation.

## What's included

- `turboquant_rotations.py` — math primitives (Hadamard helpers, RMSNorm fold,
  R1/R2/R4 fusion).
- `turboquant_quantize.py` — main conversion script for Qwen3.5/3.6 family
  (BF16 source). Supports both dense (Qwen3.6-27B) and MoE (Qwen3.6-35B-A3B).
- `minimax_m2_convert.py` — conversion script for MiniMax-M2 family
  (FP8 block-128 source). Handles `mx.from_fp8` dequantization + per-block
  scaling + per-expert stacking (256 separate `block_sparse_moe.experts.<n>.w1/w2/w3`
  → batched `switch_mlp.{gate,down,up}_proj` per layer).
- `vllm_mlx/utils/tokenizer.py` — loader hook detecting `quantization.type ==
  "turboquant_v1"` and installing (optional) R4 runtime patch.
- `vllm_mlx/patches/turboquant_r4.py` — opt-in monkey-patch for online R4
  Hadamard (only useful with activation quantization).

## Recipe (default)

- `R1` (residual stream Hadamard): fused offline into embed, lm_head, and ALL
  in/out projections (q/k/v/o_proj, gate/up/down_proj per expert, etc).
- `R2` (head-space rotation): SKIPPED for Qwen3.5/3.6 because of their gated
  attention quirk (`output = o_proj(attn_out * sigmoid(query_gate))`) —
  element-wise gate breaks R2's offline cancellation. Applicable to MiniMax-M2
  and standard-attention models.
- `R4` (online Hadamard before down_proj): **NOT applied by default**. Per the
  QuaRot paper, R4 is only needed for activation quantization. For weight-only
  4-bit/8-bit affine schemes, R4 is redundant. The implementation is included
  as opt-in (`--enable-r4` + `VLLM_MLX_TURBOQUANT_R4=all`) for users who pair
  this with NVFP4/MXFP8 activation quantization downstream.
- RMSNorm fold: `(1 + raw_norm_weight)` (Qwen3.5/3.6 offset semantics) folded
  into downstream projections. Norm weights stored as 1.0 so runtime
  `weight * normalized = normalized` is identity.
- Quantization: 4-bit affine on routed `switch_mlp.*` experts; 8-bit affine
  on attention, linear_attn projections, `shared_expert`, embed_tokens,
  lm_head. Router gate (`mlp.gate`) and gating heads kept in fp16.

## Usage

### Qwen3.5/3.6 family (BF16 source)

```bash
# Verify BF16 source shards via SHA256 first (see mandatory rule).
python scripts/turboquant_quantize.py \
    --src /path/to/Qwen3.6-Model-BF16 \
    --dst /path/to/Qwen3.6-Model-mlx-turboquant
```

### MiniMax-M2 family (FP8 source)

```bash
python scripts/minimax_m2_convert.py \
    --src /path/to/MiniMax-M2-Model-FP8 \
    --dst /path/to/MiniMax-M2-Model-mlx-turboquant
```

Both produce mlx-loadable safetensors with `quantization.type == "turboquant_v1"`.

## Validated on

- Qwen3.6-35B-A3B (256 routed experts, MoE, 40 layers, BF16 source): 22 GB, 12/12.
- Qwen3.6-27B Heretic (dense, 64 layers, BF16 source): 19 GB, 12/12.
- MiniMax-M2.7 (256 routed experts, MoE, 62 layers, FP8 block-128 source):
  ~145 GB output (in progress).

## References

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (Ashkboos et al., 2024)](https://arxiv.org/abs/2404.00456)
- [AMD Quark — Rotation-based quantization with QuaRot](https://quark.docs.amd.com/release-0.9/pytorch/tutorial_quarot.html)
