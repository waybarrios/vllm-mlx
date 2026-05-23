#!/usr/bin/env python3
"""TurboQuant conversion: BF16 → R1+R4 rotated + 4-bit/8-bit affine quantized.

Operates DIRECTLY on safetensors shards without loading the full model into
memory. Per-tensor pipeline: read → fold norm → rotate → quantize → write.

Targets Qwen3.6-35B-A3B (qwen3_5_moe, hybrid attention) for Phase 0 PoC,
extendable to MiniMax-M2.7 (minimax_m2) in Phase 1.

Recipe:
  - R1 (residual stream Hadamard): fused offline into all in/out projections,
    embed_tokens, lm_head.
  - R4 (intermediate Hadamard before down_proj): pre-rotation fused offline;
    runtime patch applies mx.hadamard_transform online.
  - R2 (head-space rotation): SKIPPED for qwen3.5/3.6 (gated attention quirk).
  - RMSNorm fold: (1 + norm_weight) folded into downstream projections; norm
    weights set to 0 so runtime `(1 + 0) = 1` identity.
  - Quantization: 4-bit affine on switch_mlp experts, 8-bit affine on
    attention, shared_expert, embed_tokens, lm_head. Router gate fp16.
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

# Import rotation helpers from sibling module.
# Add this directory to path so sibling import works regardless of cwd.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent))

from turboquant_rotations import (
    apply_r1_to_in_projection,
    apply_r1_to_out_projection,
    apply_r4_to_down_proj,
)

# ---------------------------------------------------------------------------
# Architecture-specific module classification.
# ---------------------------------------------------------------------------

# Special key marker meaning "BF16 source has fused gate+up, split during processing".
GATE_UP_FUSED_SUFFIX = "switch_mlp.gate_up_proj.weight"


# Patterns that match weight tensor paths. (Qwen3.6-35B-A3B style.)
# Each entry: (regex, role)
QWEN36_PATTERNS = [
    # Embeddings / head
    (r"language_model\.model\.embed_tokens\.weight$", "embed"),
    (r"language_model\.lm_head\.weight$", "lm_head"),
    (r"language_model\.model\.norm\.weight$", "final_norm"),
    # Per-layer norms
    (r"language_model\.model\.layers\.(\d+)\.input_layernorm\.weight$", "input_norm"),
    (
        r"language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight$",
        "post_norm",
    ),
    # Full attention (layers where is_linear=False, every 4th)
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "self_q"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "self_k"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "self_v"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "self_o"),
    (
        r"language_model\.model\.layers\.(\d+)\.self_attn\.[qk]_norm\.weight$",
        "self_qk_norm",
    ),
    # GatedDeltaNet (linear_attn)
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$",
        "linear_in_qkv",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$",
        "linear_in_z",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight$",
        "linear_in_a",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight$",
        "linear_in_b",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$",
        "linear_out",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$",
        "linear_conv",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.linear_attn\.norm\.weight$",
        "linear_inner_norm",
    ),
    (r"language_model\.model\.layers\.(\d+)\.linear_attn\.dt_bias$", "linear_dt_bias"),
    (r"language_model\.model\.layers\.(\d+)\.linear_attn\.A_log$", "linear_A_log"),
    # MoE
    (r"language_model\.model\.layers\.(\d+)\.mlp\.gate\.weight$", "router_gate"),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight$",
        "shared_expert_gate",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj\.weight$",
        "shared_gate",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj\.weight$",
        "shared_up",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj\.weight$",
        "shared_down",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.gate_proj\.weight$",
        "expert_gate",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.up_proj\.weight$",
        "expert_up",
    ),
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.down_proj\.weight$",
        "expert_down",
    ),
    # Fused gate+up marker (will be split at processing time).
    (
        r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.gate_up_proj\.weight$",
        "expert_gate_up_fused",
    ),
    # Dense MLP (Qwen3.6-27B, model_type qwen3_5_text without MoE)
    # WARNING: order matters — match these AFTER the switch_mlp/shared_expert patterns above.
    (r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", "dense_gate"),
    (r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.weight$", "dense_up"),
    (r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight$", "dense_down"),
    # Vision tower (pass through unchanged)
    (r"^visual\.", "vision_passthrough"),
    (r"vision_tower\.", "vision_passthrough"),
    (r"mtp\.", "mtp_passthrough"),
]


def classify(key: str) -> Tuple[str, int]:
    """Return (role_name, layer_idx or -1)."""
    for pat, role in QWEN36_PATTERNS:
        m = re.search(pat, key)
        if m:
            idx = int(m.group(1)) if m.groups() else -1
            return role, idx
    return "unknown", -1


# ---------------------------------------------------------------------------
# Two-pass conversion: pass 1 collects norm weights per layer; pass 2 rewrites.
# ---------------------------------------------------------------------------


def sanitize_key(key: str) -> str | None:
    """Apply mlx_vlm/mlx_lm-style key rename. Returns None to drop the key.

    Matches sanitize() in mlx_vlm/models/qwen3_5/qwen3_5.py +
    qwen3_5_moe specifics (NOT including expert split — that's handled
    at processing time by EXPERT_SPLIT_SUFFIXES below).
    """
    if key.startswith("mtp."):
        return None
    # Rename top-level prefixes.
    if key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model")
    elif key.startswith("model.visual"):
        key = key.replace("model.visual", "vision_tower")
    elif key.startswith("lm_head"):
        key = "language_model." + key
    # Rename fused experts.gate_up_proj → marker, will be split at processing.
    if key.endswith(".mlp.experts.gate_up_proj"):
        return key.replace(
            ".mlp.experts.gate_up_proj", ".mlp.switch_mlp.gate_up_proj.weight"
        )
    if key.endswith(".mlp.experts.down_proj"):
        return key.replace(".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj.weight")
    return key


def load_all_keys(model_dir: Path) -> Dict[str, str]:
    """Return {sanitized_weight_key: (shard_filename, original_key)}."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        raw = idx["weight_map"]
    else:
        files = list(model_dir.glob("*.safetensors"))
        assert len(files) == 1, f"expected single shard, got {files}"
        data = mx.load(str(files[0]))
        raw = {k: files[0].name for k in data}
    # Apply sanitize, drop None.
    sanitized: Dict[str, Tuple[str, str]] = {}
    for orig, shard in raw.items():
        new_k = sanitize_key(orig)
        if new_k is None:
            continue
        sanitized[new_k] = (shard, orig)
    return sanitized


def collect_norm_weights(
    model_dir: Path, keys: Dict[str, Tuple[str, str]]
) -> Dict[str, mx.array]:
    """Pre-load all norm weights (small) for fold computation. Keys is dict of
    {sanitized_key: (shard_name, original_key)}. Returns dict by SANITIZED key."""
    norms: Dict[str, mx.array] = {}
    norm_keys = [
        k
        for k in keys
        if k.endswith("input_layernorm.weight")
        or k.endswith("post_attention_layernorm.weight")
        or k.endswith("model.norm.weight")
    ]
    by_shard: Dict[str, List[Tuple[str, str]]] = {}
    for k in norm_keys:
        shard, orig = keys[k]
        by_shard.setdefault(shard, []).append((k, orig))
    for shard, ks in by_shard.items():
        data = mx.load(str(model_dir / shard))
        for sanitized_k, orig_k in ks:
            norms[sanitized_k] = data[orig_k]
        del data
    return norms


def get_norm_for_layer(norms: Dict[str, mx.array], layer: int, role: str) -> mx.array:
    """Return the (1 + w) scale for fold given role."""
    if role == "input_norm":
        k = f"language_model.model.layers.{layer}.input_layernorm.weight"
    elif role == "post_norm":
        k = f"language_model.model.layers.{layer}.post_attention_layernorm.weight"
    elif role == "final_norm":
        k = "language_model.model.norm.weight"
    else:
        raise KeyError(role)
    return norms[k]


# Roles needing input-norm fold (uses input_layernorm of same layer).
INPUT_NORM_FOLD_ROLES = {
    "self_q",
    "self_k",
    "self_v",
    "linear_in_qkv",
    "linear_in_z",
    "linear_in_a",
    "linear_in_b",
}
# Roles needing post-norm fold (uses post_attention_layernorm of same layer).
POST_NORM_FOLD_ROLES = {
    "router_gate",
    "shared_expert_gate",
    "shared_gate",
    "shared_up",
    "expert_gate",
    "expert_up",
    "dense_gate",
    "dense_up",
}


def _is_sensitive_layer(layer: int, num_layers: int) -> bool:
    """mixed_4_8 sensitive layer heuristic: first 1/8, last 1/8, every 3rd middle.

    Sensitive layers get 8-bit (instead of 4-bit base) for the most precision-
    critical projections (down_proj, v_proj). Matches the predicate used by
    Qwen3.6-27B PROD mixed_4_8.
    """
    if layer < 0 or num_layers <= 0:
        return False  # global tensors (embed, lm_head) handled separately
    first_eighth = num_layers // 8
    return (
        layer < first_eighth
        or layer >= 7 * first_eighth
        or (layer - first_eighth) % 3 == 2
    )


def apply_fold_and_rotate(
    key: str,
    tensor: mx.array,
    role: str,
    layer: int,
    norms: Dict[str, mx.array],
    hidden_size: int,
    intermediate_size: int,
    skip_rotations: bool = False,
    fold_norms: bool = True,
    r1_roles: set | None = None,
    enable_r4: bool = True,
    expert_bits: int = 4,
    num_layers: int = 0,
) -> Tuple[mx.array, str]:
    """Apply RMSNorm fold + R1 + (R4 if applicable). Returns (new_tensor, suggested_quant).

    suggested_quant in {"q4", "q8", "fp16", "skip"}.
    skip_rotations: diagnostic — fold norms but DON'T rotate. Should produce
    a model equivalent to standard 4-bit/8-bit mix quantization.
    fold_norms: if False, keep (1+raw) in norm (matching PROD mixed_4_8 behavior).
    Required True for R1 rotation to make sense.
    """
    w = tensor  # already mx.array bf16
    orig_dtype = w.dtype

    # --- Norm folds ---
    if fold_norms:
        if role in INPUT_NORM_FOLD_ROLES:
            nw = get_norm_for_layer(norms, layer, "input_norm")
            scale = 1.0 + nw.astype(mx.float32)
            w = (w.astype(mx.float32) * scale[None, :]).astype(orig_dtype)
        elif role in POST_NORM_FOLD_ROLES:
            nw = get_norm_for_layer(norms, layer, "post_norm")
            scale = 1.0 + nw.astype(mx.float32)
            if role in ("expert_gate", "expert_up"):
                # 3D weight (E, out, in)
                w = (w.astype(mx.float32) * scale[None, None, :]).astype(orig_dtype)
            else:
                w = (w.astype(mx.float32) * scale[None, :]).astype(orig_dtype)
        elif role == "lm_head":
            nw = get_norm_for_layer(norms, 0, "final_norm")  # final norm is global
            scale = 1.0 + nw.astype(mx.float32)
            w = (w.astype(mx.float32) * scale[None, :]).astype(orig_dtype)

    # --- R1 rotation (skipped in diagnostic mode) ---
    # IMPORTANT: rotated weights stay in fp32 throughout the pipeline to avoid
    # bf16 down-cast precision loss after Hadamard transform. mx.quantize
    # accepts fp32 input and handles precision internally.
    in_axis_roles = INPUT_NORM_FOLD_ROLES | POST_NORM_FOLD_ROLES | {"lm_head", "embed"}
    out_axis_roles = {
        "self_o",
        "linear_out",
        "shared_down",
        "expert_down",
        "dense_down",
    }
    if skip_rotations:
        # Skip R1 + R4 — just fold norms + quantize. Diagnostic only.
        # Output format is "mlx" → sanitize() is skipped at load.
        # If fold_norms=True: norm absorbed into projections, store norm=1.0.
        # If fold_norms=False: store norm=(1+raw) matching PROD behavior.
        if role in ("input_norm", "post_norm", "final_norm"):
            if fold_norms:
                w = mx.ones_like(w)
            else:
                w = (w.astype(mx.float32) + 1.0).astype(orig_dtype)
        elif role == "self_qk_norm":
            w = (w.astype(mx.float32) + 1.0).astype(orig_dtype)
        # Determine quant mode and return early.
        if role in (
            "expert_gate",
            "expert_up",
            "expert_down",
            "dense_gate",
            "dense_up",
            "dense_down",
        ):
            return w, ("q4" if expert_bits == 4 else "q8")
        elif role in (
            "self_q",
            "self_k",
            "self_v",
            "self_o",
            "linear_in_qkv",
            "linear_in_z",
            "linear_in_a",
            "linear_in_b",
            "linear_out",
            "shared_gate",
            "shared_up",
            "shared_down",
            "embed",
            "lm_head",
        ):
            return w, "q8"
        elif role in (
            "router_gate",
            "shared_expert_gate",
            "self_qk_norm",
            "linear_inner_norm",
            "input_norm",
            "post_norm",
            "final_norm",
            "linear_dt_bias",
            "linear_A_log",
            "linear_conv",
        ):
            return w, "fp16"
        elif role in ("vision_passthrough", "mtp_passthrough"):
            return w, "skip"
        return w, "fp16"
    # R1 only if role is in selected set (bisect-friendly).
    apply_r1 = r1_roles is None or role in r1_roles
    if apply_r1:
        if role in in_axis_roles:
            if role in ("expert_gate", "expert_up"):
                # 3D — Hadamard on last axis (input). Stay fp32 after rotation.
                w = mx.hadamard_transform(w.astype(mx.float32))
            else:
                w = apply_r1_to_in_projection(w, hidden_size)
        elif role in out_axis_roles:
            if role == "expert_down":
                # 3D (E, hidden, intermediate) — rotate hidden axis (axis=1)
                wT = mx.swapaxes(w.astype(mx.float32), 1, 2)
                wT_rot = mx.hadamard_transform(wT)
                w = mx.swapaxes(wT_rot, 1, 2)
            else:
                w = apply_r1_to_out_projection(w, hidden_size)

    # --- R4 rotation (pre-rotate down_proj input side). Stay fp32 ---
    # enable_r4 can be "shared" / "experts" / "all" / False for bisect.
    r4_shared = enable_r4 is True or enable_r4 == "shared" or enable_r4 == "all"
    r4_experts = enable_r4 is True or enable_r4 == "experts" or enable_r4 == "all"
    if r4_shared and role == "shared_down":
        w = apply_r4_to_down_proj(w, intermediate_size)
    elif r4_experts and role == "expert_down":
        w = mx.hadamard_transform(w.astype(mx.float32))

    # --- Norm weights: format=mlx skips sanitize, so we must bake +1.0 here.
    if role in ("input_norm", "post_norm", "final_norm"):
        if fold_norms:
            w = mx.ones_like(w)  # folded into projections
        else:
            w = (w.astype(mx.float32) + 1.0).astype(orig_dtype)  # PROD-like
    elif role == "self_qk_norm":
        # q_norm / k_norm are NOT folded (internal head-space norms), but
        # they DO need the +1.0 sanitize offset baked in.
        w = (w.astype(mx.float32) + 1.0).astype(orig_dtype)

    # --- Quantization decision ---
    # MoE routed experts: 4-bit (or 8 if expert_bits=8 diagnostic).
    if role in ("expert_gate", "expert_up", "expert_down"):
        return w, ("q4" if expert_bits == 4 else "q8")

    # Dense MLP: mixed_4_8-style — 4-bit base + 8-bit on sensitive layers'
    # down_proj. Matches PROD predicate for Qwen3.6-27B for throughput parity.
    if role == "dense_down":
        return w, ("q8" if _is_sensitive_layer(layer, num_layers) else "q4")
    if role in ("dense_gate", "dense_up"):
        return w, "q4"

    # Attention: 4-bit base + 8-bit on v_proj of sensitive layers (mixed_4_8).
    if role == "self_v":
        return w, ("q8" if _is_sensitive_layer(layer, num_layers) else "q4")
    if role in ("self_q", "self_k", "self_o"):
        return w, "q4"

    # GatedDeltaNet projections: 4-bit base (no sensitive split in mixed_4_8).
    if role in (
        "linear_in_qkv",
        "linear_in_z",
        "linear_in_a",
        "linear_in_b",
        "linear_out",
    ):
        return w, "q4"

    # Shared expert (MoE only): 4-bit + 8-bit on sensitive down_proj.
    if role == "shared_down":
        return w, ("q8" if _is_sensitive_layer(layer, num_layers) else "q4")
    if role in ("shared_gate", "shared_up"):
        return w, "q4"

    # Embeddings + LM head: always 8-bit (high outlier sensitivity).
    if role in ("embed", "lm_head"):
        return w, "q8"
    # Router / norms / small fp16-passthrough tensors.
    if role in (
        "router_gate",
        "shared_expert_gate",
        "self_qk_norm",
        "linear_inner_norm",
        "input_norm",
        "post_norm",
        "final_norm",
        "linear_dt_bias",
        "linear_A_log",
        "linear_conv",
    ):
        return w, "fp16"
    if role in ("vision_passthrough", "mtp_passthrough"):
        return w, "skip"
    return w, "fp16"  # default: keep as is


def quantize_tensor(
    w: mx.array, mode: str, group_size: int = 64
) -> Dict[str, mx.array]:
    """Apply mx.quantize with given bits. Returns dict of tensors (weight, scales, biases).

    mode: "q4" → 4-bit affine, "q8" → 8-bit affine, "fp16" → keep as bf16 (no quant).
    """
    if mode == "q4":
        bits = 4
    elif mode == "q8":
        bits = 8
    else:
        return {"": w}  # raw (single key)

    # Quantize from fp32 if w is fp32 (preserves precision from rotation).
    # mx.quantize requires fp16/bf16/fp32; fp32 gives best scale fit.
    qin_dtype = mx.float32 if w.dtype == mx.float32 else mx.bfloat16

    if w.ndim == 2:
        if w.shape[-1] % group_size != 0:
            return {"": w.astype(mx.bfloat16)}
        qw, scales, biases = mx.quantize(
            w.astype(qin_dtype), group_size=group_size, bits=bits
        )
        return {".weight": qw, ".scales": scales, ".biases": biases}
    elif w.ndim == 3:
        E = w.shape[0]
        qws, scs, bss = [], [], []
        for e in range(E):
            qw, sc, bi = mx.quantize(
                w[e].astype(qin_dtype), group_size=group_size, bits=bits
            )
            qws.append(qw)
            scs.append(sc)
            bss.append(bi)
        return {
            ".weight": mx.stack(qws, axis=0),
            ".scales": mx.stack(scs, axis=0),
            ".biases": mx.stack(bss, axis=0),
        }
    else:
        return {"": w.astype(mx.bfloat16)}


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


def convert(
    src: Path,
    dst: Path,
    max_shard_gb: float = 5.0,
    layer_limit: int | None = None,
    skip_rotations: bool = False,
    fold_norms: bool = True,
    r1_roles: set | None = None,
    enable_r4: bool = False,  # R4 default OFF (not needed for weight-only quant per QuaRot)
    expert_bits: int = 4,
) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[main] src = {src}")
    print(f"[main] dst = {dst}")

    # Load config to find hidden_size, intermediate_size, layer count.
    cfg = json.loads((src / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    hidden = text_cfg.get("hidden_size") or text_cfg.get("dim")
    intermediate = text_cfg.get("moe_intermediate_size") or text_cfg.get(
        "intermediate_size"
    )
    num_layers = text_cfg.get("num_hidden_layers")
    num_experts = text_cfg.get("num_experts", 0)
    print(
        f"[main] hidden={hidden}, moe_intermediate={intermediate}, layers={num_layers}, experts={num_experts}"
    )
    if layer_limit is not None:
        print(f"[main] LAYER LIMIT for sanity test: {layer_limit}")

    # Map all keys to shards.
    keys = load_all_keys(src)
    print(f"[main] total tensors: {len(keys)}")

    # Collect norms (small, fits in RAM easily).
    print("[main] pre-loading norm weights...")
    norms = collect_norm_weights(src, keys)
    print(f"  collected {len(norms)} norm tensors")

    # Process tensors per shard, save in batches sized by max_shard_gb.
    out_shards: List[Dict[str, np.ndarray]] = [{}]
    cur_size = 0
    max_size = int(max_shard_gb * 1e9)
    out_weight_map: Dict[str, str] = {}

    quant_cfg = {
        "type": "turboquant_v1" if not skip_rotations else "turboquant_v1_norot_diag",
        "r1": "hadamard" if not skip_rotations else "none",
        "r4": "hadamard_intermediate" if (not skip_rotations and enable_r4) else "none",
        "r1_roles": list(r1_roles) if r1_roles is not None else "all",
        "expert_bits": expert_bits,
        "attn_bits": 8,
        "group_size": 64,
        "fold_offset_norms": True,
        "skip_r2": "qwen3_5_gated_attention",
    }
    quant_per_path: Dict[str, dict] = {}

    by_shard: Dict[str, List[Tuple[str, str]]] = {}
    for sanitized_k, (shard, orig_k) in keys.items():
        by_shard.setdefault(shard, []).append((sanitized_k, orig_k))

    # out_shards is a list of dicts of mx.arrays; written via mx.save_safetensors per shard
    out_shards_mx: List[Dict[str, mx.array]] = [{}]
    cur_size = 0

    t_start = time.time()
    tensor_count = 0
    skipped = 0

    def emit_quantized(out_key: str, w: mx.array, qmode: str):
        nonlocal cur_size
        parts = quantize_tensor(w, qmode, group_size=quant_cfg["group_size"])
        bits = 4 if qmode == "q4" else 8
        base_key = (
            out_key[: -len(".weight")] if out_key.endswith(".weight") else out_key
        )
        for suf, t in parts.items():
            ok = base_key + suf if suf else out_key
            mx.eval(t)
            sz = t.nbytes
            if cur_size + sz > max_size:
                out_shards_mx.append({})
                cur_size = 0
            out_shards_mx[-1][ok] = t
            cur_size += sz
        # Per-path key in config must match module path (no .weight suffix).
        config_key = base_key
        quant_per_path[config_key] = {
            "group_size": quant_cfg["group_size"],
            "bits": bits,
            "mode": "affine",
        }

    def emit_raw(out_key: str, t: mx.array):
        nonlocal cur_size
        mx.eval(t)
        sz = t.nbytes
        if cur_size + sz > max_size:
            out_shards_mx.append({})
            cur_size = 0
        out_shards_mx[-1][out_key] = t
        cur_size += sz

    for shard_name in sorted(by_shard.keys()):
        items = by_shard[shard_name]
        print(f"\n[shard] processing {shard_name} ({len(items)} tensors)...")
        data = mx.load(str(src / shard_name))
        for k, orig_k in items:
            role, layer = classify(k)
            if layer_limit is not None and layer >= 0 and layer >= layer_limit:
                skipped += 1
                continue
            tensor = data[orig_k]

            # Special case: fused gate+up needs to be split into two separate
            # weights BEFORE fold/rotation (each half gets identical treatment).
            if role == "expert_gate_up_fused":
                mid = tensor.shape[-2] // 2  # split out_dim in half
                gate_w = tensor[..., :mid, :]
                up_w = tensor[..., mid:, :]
                base_prefix = k.replace(
                    ".switch_mlp.gate_up_proj.weight", ".switch_mlp"
                )
                # Process each half as expert_gate / expert_up respectively.
                for half_w, half_role, half_name in [
                    (gate_w, "expert_gate", "gate_proj"),
                    (up_w, "expert_up", "up_proj"),
                ]:
                    new_w, qmode = apply_fold_and_rotate(
                        k,
                        half_w,
                        half_role,
                        layer,
                        norms,
                        hidden,
                        intermediate,
                        skip_rotations=skip_rotations,
                        fold_norms=fold_norms,
                        r1_roles=r1_roles,
                        enable_r4=enable_r4,
                        expert_bits=expert_bits,
                        num_layers=num_layers,
                    )
                    out_key = f"{base_prefix}.{half_name}.weight"
                    if qmode in ("q4", "q8"):
                        emit_quantized(out_key, new_w, qmode)
                    elif qmode == "fp16":
                        emit_raw(out_key, new_w)
                tensor_count += 2
                continue

            # Apply mlx_vlm/mlx_lm sanitize-style tensor transformations.
            # 1) conv1d.weight: moveaxis(2, 1) — matches qwen3_5.py sanitize.
            if "conv1d.weight" in k and tensor.ndim == 3 and tensor.shape[-1] != 1:
                tensor = mx.moveaxis(tensor, 2, 1)

            new_w, qmode = apply_fold_and_rotate(
                k,
                tensor,
                role,
                layer,
                norms,
                hidden,
                intermediate,
                skip_rotations=skip_rotations,
                fold_norms=fold_norms,
                r1_roles=r1_roles,
                enable_r4=enable_r4,
                expert_bits=expert_bits,
                num_layers=num_layers,
            )

            if qmode in ("q4", "q8"):
                emit_quantized(k, new_w, qmode)
            elif qmode == "fp16":
                emit_raw(k, new_w)
            elif qmode == "skip":
                # Vision sanitize transformations:
                # - patch_embed.proj.weight: PyTorch → MLX layout transpose
                # - position_ids: drop (not a real weight)
                if "patch_embed.proj.weight" in k:
                    if tensor.ndim == 5 and tensor.shape[-1] != 3:
                        tensor = mx.transpose(tensor, (0, 2, 3, 4, 1))
                if "position_ids" in k:
                    continue
                emit_raw(k, tensor)

            tensor_count += 1
            if tensor_count % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  ...{tensor_count} tensors, {elapsed:.0f}s elapsed")

        del data  # free original shard from memory

    elapsed = time.time() - t_start
    print(
        f"\n[main] processed {tensor_count} tensors in {elapsed:.1f}s "
        f"(skipped {skipped} due to --layer-limit)"
    )

    # Save shards via mx.save_safetensors.
    out_shards = out_shards_mx
    print(f"[main] writing {len(out_shards)} output shards...")
    for i, shard in enumerate(out_shards):
        name = f"model-{i+1:05d}-of-{len(out_shards):05d}.safetensors"
        mx.save_safetensors(str(dst / name), shard, metadata={"format": "mlx"})
        for k in shard:
            out_weight_map[k] = name
        total_bytes = sum(v.nbytes for v in shard.values())
        print(f"  wrote {name} ({len(shard)} tensors, {total_bytes/1e9:.2f} GB)")

    # Write index.
    total_size = sum(v.nbytes for s in out_shards for v in s.values())
    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": out_weight_map,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Copy config + add quantization metadata.
    out_cfg = dict(cfg)
    out_cfg["quantization"] = {
        **quant_cfg,
        "group_size": quant_cfg["group_size"],
        "bits": 4,
        "mode": "affine",
    }
    for k, v in quant_per_path.items():
        out_cfg["quantization"][k] = v
    out_cfg["quantization_config"] = out_cfg["quantization"]
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2))

    # Copy auxiliary files (tokenizer etc).
    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "configuration.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "README.md",
    ]:
        src_f = src / fname
        if src_f.exists():
            shutil.copy(src_f, dst / fname)

    # Disk size summary.
    import subprocess

    r = subprocess.run(["du", "-sh", str(dst)], capture_output=True, text=True)
    print(f"\n[main] DONE: {dst}")
    print(f"  total size: {r.stdout.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--max-shard-gb", default=5.0, type=float)
    ap.add_argument(
        "--layer-limit",
        type=int,
        default=None,
        help="Process only first N layers (for sanity test).",
    )
    ap.add_argument(
        "--skip-rotations",
        action="store_true",
        help="DIAGNOSTIC: skip R1/R4, only fold/keep norms + quantize.",
    )
    ap.add_argument(
        "--no-fold-norms",
        action="store_true",
        help="DIAGNOSTIC: keep (1+raw) in norm weight (PROD-like).",
    )
    ap.add_argument(
        "--r1-roles",
        default=None,
        help="DIAGNOSTIC bisect: comma-separated role names to apply R1 to. None=all in/out_axis roles.",
    )
    ap.add_argument(
        "--enable-r4",
        action="store_true",
        help="EXPERIMENTAL: enable R4 (online Hadamard before down_proj). "
        "NOT recommended — per QuaRot paper, R4 is only needed for "
        "ACTIVATION quantization. Weight-only quant works correctly with R1 only.",
    )
    ap.add_argument(
        "--no-r4",
        action="store_true",
        help="(deprecated, default — R4 disabled). Kept for backward compat.",
    )
    ap.add_argument(
        "--r4-scope",
        default="all",
        choices=["all", "shared", "experts"],
        help="DIAGNOSTIC: limit R4 to shared (only shared_expert.down_proj) or experts (only switch_mlp.down_proj). Default: all.",
    )
    ap.add_argument(
        "--expert-bits",
        type=int,
        default=4,
        help="Bits for routed MoE expert weights (default 4).",
    )
    args = ap.parse_args()
    r1_set = None
    if args.r1_roles:
        r1_set = set(r.strip() for r in args.r1_roles.split(",") if r.strip())
    # R4 default OFF (per QuaRot: R4 only needed for activation quantization, not weight-only).
    if args.enable_r4:
        r4_arg = args.r4_scope
    else:
        r4_arg = False
    convert(
        args.src,
        args.dst,
        args.max_shard_gb,
        args.layer_limit,
        args.skip_rotations,
        fold_norms=not args.no_fold_norms,
        r1_roles=r1_set,
        enable_r4=r4_arg,
        expert_bits=args.expert_bits,
    )


if __name__ == "__main__":
    main()
