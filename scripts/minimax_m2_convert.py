#!/usr/bin/env python3
"""TurboQuant conversion for MiniMax-M2 family (model_type = minimax_m2).

Differences from Qwen3.5/3.6 pipeline (turboquant_quantize.py):
  - Source is FP8 block-128 (uint8 weight + fp32 scale_inv per 128x128 block).
    Requires dequant via mx.from_fp8 + per-block scaling before any rotation/quant.
  - Experts stored per-expert (block_sparse_moe.experts.<n>.w1/w2/w3.weight),
    must be stacked into batched (E, out, in) form matching SwitchLinear layout
    (semantic: w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj).
  - Top-level prefix is `model.*` (NOT `language_model.model.*`).
  - No shared_expert / shared_expert_gate.
  - No GatedDeltaNet (full attention only — but interleaved SWA per model docs).
  - Per-head QK RMSNorm (q_norm, k_norm) with same +1.0 sanitize offset as Qwen.

Memory strategy: process layer-by-layer with shard-on-demand loading.
Per-layer peak: ~7-10 GB (256 experts × 3 projections × ~10 MB each in bf16).

Reuses turboquant_rotations primitives for R1 Hadamard fusion.
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from turboquant_rotations import (
    apply_r1_to_in_projection,
    apply_r1_to_out_projection,
)

# ----------------------------------------------------------------------------
# FP8 dequantization (DeepSeek-V3 / MiniMax block-128 format).
# ----------------------------------------------------------------------------


def dequant_fp8_block128(weight_u8: mx.array, scale_inv: mx.array) -> mx.array:
    """Dequantize FP8 weight via per-block-128 scale, matching mlx_lm/models/minimax.py.

    weight_u8: uint8 array of shape (M, N) — FP8 E4M3 storage.
    scale_inv: float32 array of shape (M/128, N/128) — per-block multipliers
               (despite the "inv" suffix, mlx_lm multiplies by these).
    Returns: bf16 array of shape (M, N).
    """
    weight = mx.from_fp8(weight_u8, dtype=mx.bfloat16)
    bs = 128
    m, n = weight.shape
    pad_bottom = (-m) % bs
    pad_side = (-n) % bs
    if pad_bottom or pad_side:
        weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
    weight = weight.reshape(((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs))
    weight = (weight * scale_inv[:, None, :, None]).reshape(
        m + pad_bottom, n + pad_side
    )
    return weight[:m, :n].astype(mx.bfloat16)


def _maybe_dequant(weight: mx.array, scale_inv: mx.array | None) -> mx.array:
    """Dequant if scale present (FP8 source), otherwise return as-is (bf16)."""
    if scale_inv is None:
        return weight
    return dequant_fp8_block128(weight, scale_inv)


# ----------------------------------------------------------------------------
# Shard indexing — map each layer to the source shards it needs.
# ----------------------------------------------------------------------------


def build_layer_shard_index(model_path: Path) -> Tuple[Dict[int, Set[str]], Set[str]]:
    """Returns ({layer_idx: {shard_names}}, {global_shards}).

    `global_shards` covers embed_tokens, lm_head, model.norm.
    """
    idx = json.loads((model_path / "model.safetensors.index.json").read_text())
    layer_shards: Dict[int, Set[str]] = defaultdict(set)
    global_shards: Set[str] = set()
    layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    for key, shard in idx["weight_map"].items():
        m = layer_re.match(key)
        if m:
            layer_shards[int(m.group(1))].add(shard)
        elif key in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ):
            global_shards.add(shard)
    return layer_shards, global_shards


# ----------------------------------------------------------------------------
# Collect + dequant + stack one layer's weights.
# ----------------------------------------------------------------------------


def collect_minimax_layer(
    model_path: Path,
    layer_idx: int,
    shards: Set[str],
    num_experts: int,
    shard_cache: Dict[str, dict],
) -> Dict[str, mx.array]:
    """Load needed shards (with LRU cache), dequant FP8 + stack experts.

    Returns dict of bf16 weights with sanitized minimax-native keys:
      model.layers.{L}.block_sparse_moe.switch_mlp.{gate,down,up}_proj.weight
      model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
      model.layers.{L}.{input_layernorm,post_attention_layernorm}.weight
      model.layers.{L}.block_sparse_moe.gate.weight
      ... + qk norms, biases, etc.
    """
    # Load shards on demand (simple LRU).
    for s in shards:
        if s not in shard_cache:
            shard_cache[s] = mx.load(str(model_path / s))

    # Aggregate all keys for this layer.
    prefix = f"model.layers.{layer_idx}"
    layer_keys: Dict[str, mx.array] = {}
    for s in shards:
        for k, v in shard_cache[s].items():
            if k.startswith(prefix + "."):
                layer_keys[k] = v

    out: Dict[str, mx.array] = {}

    # ---- Stack + dequant routed experts ----
    expert_mapping = [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]
    for orig_name, new_name in expert_mapping:
        first_key = f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight"
        if first_key not in layer_keys:
            continue
        weights = []
        for e in range(num_experts):
            wk = f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
            sk = f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight_scale_inv"
            if wk not in layer_keys:
                raise KeyError(f"missing expert weight: {wk}")
            w_u8 = layer_keys[wk]
            scale = layer_keys.get(sk)
            w_bf16 = _maybe_dequant(w_u8, scale)
            weights.append(w_bf16)
        stacked = mx.stack(weights, axis=0)
        out_key = f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
        out[out_key] = stacked.astype(mx.bfloat16)
        mx.eval(out[out_key])

    # ---- Attention (q/k/v/o_proj) — dequant FP8 ----
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        wk = f"{prefix}.self_attn.{proj}.weight"
        sk = f"{prefix}.self_attn.{proj}.weight_scale_inv"
        if wk not in layer_keys:
            continue
        out[wk] = _maybe_dequant(layer_keys[wk], layer_keys.get(sk))
        mx.eval(out[wk])

    # ---- Router gate (block_sparse_moe.gate) — usually fp32 bf16, no scale ----
    for tail in (
        "block_sparse_moe.gate.weight",
        "block_sparse_moe.e_score_correction_bias",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ):
        wk = f"{prefix}.{tail}"
        if wk in layer_keys:
            out[wk] = layer_keys[wk]

    return out


# ----------------------------------------------------------------------------
# Per-tensor classification + TurboQuant pipeline.
# ----------------------------------------------------------------------------


# Sensitive-layer heuristic (matches Qwen3.6 mixed_4_8).
def _is_sensitive_layer(layer: int, num_layers: int) -> bool:
    if layer < 0 or num_layers <= 0:
        return False
    first_eighth = num_layers // 8
    return (
        layer < first_eighth
        or layer >= 7 * first_eighth
        or (layer - first_eighth) % 3 == 2
    )


# Classify minimax key -> (role, layer_idx)
MINIMAX_PATTERNS = [
    # Stacked experts (post-sanitize names)
    (
        r"model\.layers\.(\d+)\.block_sparse_moe\.switch_mlp\.gate_proj\.weight$",
        "expert_gate",
    ),
    (
        r"model\.layers\.(\d+)\.block_sparse_moe\.switch_mlp\.up_proj\.weight$",
        "expert_up",
    ),
    (
        r"model\.layers\.(\d+)\.block_sparse_moe\.switch_mlp\.down_proj\.weight$",
        "expert_down",
    ),
    # Attention
    (r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "self_q"),
    (r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "self_k"),
    (r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "self_v"),
    (r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "self_o"),
    (r"model\.layers\.(\d+)\.self_attn\.[qk]_norm\.weight$", "self_qk_norm"),
    # Norms
    (r"model\.layers\.(\d+)\.input_layernorm\.weight$", "input_norm"),
    (r"model\.layers\.(\d+)\.post_attention_layernorm\.weight$", "post_norm"),
    # Router gate (small fp32, no quant)
    (r"model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$", "router_gate"),
    (
        r"model\.layers\.(\d+)\.block_sparse_moe\.e_score_correction_bias$",
        "router_bias",
    ),
    # Global
    (r"model\.embed_tokens\.weight$", "embed"),
    (r"lm_head\.weight$", "lm_head"),
    (r"model\.norm\.weight$", "final_norm"),
]


def classify(key: str) -> Tuple[str, int]:
    for pat, role in MINIMAX_PATTERNS:
        m = re.match(pat, key)
        if m:
            idx = int(m.group(1)) if m.groups() else -1
            return role, idx
    return "unknown", -1


INPUT_NORM_FOLD_ROLES = {"self_q", "self_k", "self_v"}
POST_NORM_FOLD_ROLES = {"router_gate", "expert_gate", "expert_up"}


def apply_turboquant_per_tensor(
    key: str,
    tensor: mx.array,
    role: str,
    layer: int,
    norms: Dict[str, mx.array],
    hidden_size: int,
    num_layers: int,
    enable_r1: bool = True,
) -> Tuple[mx.array, str]:
    """Apply RMSNorm fold + R1 rotation + quant-mode decision for a minimax tensor.

    Returns (transformed_tensor, qmode in {"q4","q8","fp16","skip"}).
    """
    w = tensor
    orig_dtype = w.dtype

    # ---- RMSNorm fold (Qwen3.5/3.6-style +1.0 offset semantics) ----
    if role in INPUT_NORM_FOLD_ROLES:
        nw = norms[f"model.layers.{layer}.input_layernorm.weight"]
        scale = 1.0 + nw.astype(mx.float32)
        w = (w.astype(mx.float32) * scale[None, :]).astype(orig_dtype)
    elif role in POST_NORM_FOLD_ROLES:
        nw = norms[f"model.layers.{layer}.post_attention_layernorm.weight"]
        scale = 1.0 + nw.astype(mx.float32)
        if role in ("expert_gate", "expert_up"):
            w = (w.astype(mx.float32) * scale[None, None, :]).astype(orig_dtype)
        else:
            w = (w.astype(mx.float32) * scale[None, :]).astype(orig_dtype)
    elif role == "lm_head":
        nw = norms["model.norm.weight"]
        scale = 1.0 + nw.astype(mx.float32)
        w = (w.astype(mx.float32) * scale[None, :]).astype(orig_dtype)

    # ---- R1 rotation (residual stream Hadamard) ----
    in_axis_roles = INPUT_NORM_FOLD_ROLES | POST_NORM_FOLD_ROLES | {"embed", "lm_head"}
    out_axis_roles = {"self_o", "expert_down"}
    if enable_r1:
        if role in in_axis_roles:
            if role in ("expert_gate", "expert_up"):
                w = mx.hadamard_transform(w.astype(mx.float32))
            else:
                w = apply_r1_to_in_projection(w, hidden_size)
        elif role in out_axis_roles:
            if role == "expert_down":
                wT = mx.swapaxes(w.astype(mx.float32), 1, 2)
                wT_rot = mx.hadamard_transform(wT)
                w = mx.swapaxes(wT_rot, 1, 2)
            else:
                w = apply_r1_to_out_projection(w, hidden_size)

    # ---- Norm weight post-process (sanitize +1 baked in for format=mlx output) ----
    if role in ("input_norm", "post_norm", "final_norm"):
        w = mx.ones_like(w)  # folded into projections; runtime weight * x = x
    elif role == "self_qk_norm":
        w = (w.astype(mx.float32) + 1.0).astype(orig_dtype)

    # ---- Quantization decision (mixed_4_8 with sensitive layer 8-bit) ----
    if role == "expert_down":
        return w, ("q8" if _is_sensitive_layer(layer, num_layers) else "q4")
    if role in ("expert_gate", "expert_up"):
        return w, "q4"
    if role == "self_v":
        return w, ("q8" if _is_sensitive_layer(layer, num_layers) else "q4")
    if role in ("self_q", "self_k", "self_o"):
        return w, "q4"
    if role in ("embed", "lm_head"):
        return w, "q8"
    if role in (
        "router_gate",
        "router_bias",
        "self_qk_norm",
        "input_norm",
        "post_norm",
        "final_norm",
    ):
        return w, "fp16"
    return w, "fp16"


def quantize(w: mx.array, mode: str, group_size: int = 64) -> Dict[str, mx.array]:
    """Run mx.quantize. Returns {".weight": qw, ".scales": sc, ".biases": bi} or {"": w}."""
    if mode == "fp16":
        return {"": w.astype(mx.bfloat16)}
    if mode == "skip":
        return {"": w}
    bits = 4 if mode == "q4" else 8
    # Always cast to bf16 BEFORE quantize so scales+biases come out as bf16
    # (mx.quantize matches output scale dtype to input). FP32 scales would
    # add ~25 GB of unnecessary overhead on MiniMax (256 experts × 62 layers).
    w_in = w.astype(mx.bfloat16)
    if w.ndim == 2:
        if w.shape[-1] % group_size != 0:
            return {"": w_in}
        qw, sc, bi = mx.quantize(w_in, group_size=group_size, bits=bits)
        return {".weight": qw, ".scales": sc, ".biases": bi}
    if w.ndim == 3:
        E = w.shape[0]
        qws, scs, bss = [], [], []
        for e in range(E):
            qw, sc, bi = mx.quantize(w_in[e], group_size=group_size, bits=bits)
            qws.append(qw)
            scs.append(sc)
            bss.append(bi)
        return {
            ".weight": mx.stack(qws, axis=0),
            ".scales": mx.stack(scs, axis=0),
            ".biases": mx.stack(bss, axis=0),
        }
    return {"": w_in}


# ----------------------------------------------------------------------------
# Main conversion driver — streaming layer-by-layer.
# ----------------------------------------------------------------------------


def convert(
    src: Path,
    dst: Path,
    max_shard_gb: float = 5.0,
    layer_limit: int | None = None,
    enable_r1: bool = True,
) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[main] src = {src}")
    print(f"[main] dst = {dst}")

    cfg = json.loads((src / "config.json").read_text())
    hidden = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    num_experts = cfg.get("num_local_experts", cfg.get("num_experts", 256))
    print(f"[main] hidden={hidden}, layers={num_layers}, experts={num_experts}")
    if layer_limit:
        print(f"[main] LAYER LIMIT (sanity test): {layer_limit}")

    print("[main] indexing shards...")
    layer_shards, global_shards = build_layer_shard_index(src)
    print(
        f"  {len(layer_shards)} layers, {sum(len(s) for s in layer_shards.values())} shard-references, {len(global_shards)} global shards"
    )

    # Phase 1: collect norms (needed for fold computations later)
    print("[main] pre-loading norms (input/post layernorm per layer + model.norm)...")
    norms: Dict[str, mx.array] = {}
    for L in range(num_layers):
        for s in layer_shards.get(L, set()):
            d = mx.load(str(src / s))
            for nk in (
                f"model.layers.{L}.input_layernorm.weight",
                f"model.layers.{L}.post_attention_layernorm.weight",
            ):
                if nk in d:
                    norms[nk] = d[nk]
    for s in global_shards:
        d = mx.load(str(src / s))
        if "model.norm.weight" in d:
            norms["model.norm.weight"] = d["model.norm.weight"]
    print(f"  collected {len(norms)} norms")

    # Output state
    out_shards: List[Dict[str, mx.array]] = [{}]
    cur_size = 0
    max_size = int(max_shard_gb * 1e9)
    out_weight_map: Dict[str, str] = {}
    quant_per_path: Dict[str, dict] = {}

    quant_cfg = {
        "type": "turboquant_v1",
        "r1": "hadamard" if enable_r1 else "none",
        "r4": "none",
        "expert_bits": 4,
        "attn_bits": 8,
        "group_size": 64,
        "fold_offset_norms": True,
        "skip_r2": "mixed_attention_safety",
    }
    GROUP = 64

    def emit_quantized(out_key: str, w: mx.array, mode: str):
        nonlocal cur_size
        parts = quantize(w, mode, group_size=GROUP)
        bits = 4 if mode == "q4" else 8
        base = out_key[: -len(".weight")] if out_key.endswith(".weight") else out_key
        for suf, t in parts.items():
            ok = base + suf if suf else out_key
            mx.eval(t)
            sz = t.nbytes
            if cur_size + sz > max_size:
                out_shards.append({})
                cur_size = 0
            out_shards[-1][ok] = t
            cur_size += sz
        quant_per_path[base] = {"group_size": GROUP, "bits": bits, "mode": "affine"}

    def emit_raw(out_key: str, t: mx.array):
        nonlocal cur_size
        mx.eval(t)
        sz = t.nbytes
        if cur_size + sz > max_size:
            out_shards.append({})
            cur_size = 0
        out_shards[-1][out_key] = t
        cur_size += sz

    # Phase 2: per-layer streaming
    shard_cache: Dict[str, dict] = {}
    t0 = time.time()
    for L in range(num_layers):
        if layer_limit is not None and L >= layer_limit:
            break
        shards = layer_shards.get(L, set())
        if not shards:
            print(f"  [L{L}] no shards — skip")
            continue
        t_layer = time.time()
        layer_tensors = collect_minimax_layer(src, L, shards, num_experts, shard_cache)
        t_collect = time.time() - t_layer

        # Apply turboquant per tensor
        for k, t in layer_tensors.items():
            role, lyr = classify(k)
            new_w, mode = apply_turboquant_per_tensor(
                k, t, role, lyr, norms, hidden, num_layers, enable_r1=enable_r1
            )
            if mode in ("q4", "q8"):
                emit_quantized(k, new_w, mode)
            else:
                emit_raw(k, new_w)

        # Evict shards that no future layer needs
        future_shards = set()
        for fut_L in range(L + 1, num_layers):
            future_shards |= layer_shards.get(fut_L, set())
        future_shards |= global_shards
        for s in list(shard_cache.keys()):
            if s not in future_shards:
                del shard_cache[s]

        elapsed_layer = time.time() - t_layer
        print(
            f"  [L{L:2d}] collect={t_collect:5.1f}s  total={elapsed_layer:5.1f}s  "
            f"shards_cached={len(shard_cache)}  out_shards={len(out_shards)}"
        )

    # Phase 3: global tensors (embed, lm_head, model.norm)
    print("\n[main] processing global tensors (embed, lm_head, model.norm)...")
    for s in global_shards:
        d = shard_cache.get(s) or mx.load(str(src / s))
        for k, t in d.items():
            if k not in (
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.norm.weight",
            ):
                continue
            role, lyr = classify(k)
            new_w, mode = apply_turboquant_per_tensor(
                k, t, role, lyr, norms, hidden, num_layers, enable_r1=enable_r1
            )
            if mode in ("q4", "q8"):
                emit_quantized(k, new_w, mode)
            else:
                emit_raw(k, new_w)

    print(f"\n[main] total time: {time.time() - t0:.1f}s")

    # Phase 4: write output
    print(f"[main] writing {len(out_shards)} output shards...")
    for i, shard in enumerate(out_shards):
        name = f"model-{i+1:05d}-of-{len(out_shards):05d}.safetensors"
        mx.save_safetensors(str(dst / name), shard, metadata={"format": "mlx"})
        for k in shard:
            out_weight_map[k] = name
        sz = sum(v.nbytes for v in shard.values())
        print(f"  wrote {name} ({len(shard)} tensors, {sz/1e9:.2f} GB)")

    total_size = sum(v.nbytes for s in out_shards for v in s.values())
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int(total_size)},
                "weight_map": out_weight_map,
            },
            indent=2,
        )
    )

    # Phase 5: copy aux + write quant config
    out_cfg = dict(cfg)
    out_cfg["quantization"] = {
        **quant_cfg,
        "group_size": GROUP,
        "bits": 4,
        "mode": "affine",
    }
    for k, v in quant_per_path.items():
        out_cfg["quantization"][k] = v
    out_cfg["quantization_config"] = out_cfg["quantization"]
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2))

    for f in src.glob("*"):
        if f.is_file() and f.suffix in (".json", ".jinja", ".py", ".md", ".txt"):
            if f.name not in ("config.json", "model.safetensors.index.json"):
                shutil.copy(f, dst / f.name)

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
        help="Process only first N layers (sanity test).",
    )
    ap.add_argument(
        "--no-r1",
        action="store_true",
        help="DIAGNOSTIC: skip R1 rotation (matches mixed_4_8 baseline).",
    )
    args = ap.parse_args()
    convert(
        args.src,
        args.dst,
        args.max_shard_gb,
        args.layer_limit,
        enable_r1=not args.no_r1,
    )


if __name__ == "__main__":
    sys.exit(main())
