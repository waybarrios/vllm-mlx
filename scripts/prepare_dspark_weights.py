#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Prepare NVIDIA DSpark draft-model weights for MLX (Nemotron 3.5 Lightning).

Downloads nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark (a single
~1.3 GB safetensors), dequantizes every NVFP4-packed tensor to BF16, and saves
<model-dir>/dspark/weights.safetensors + dspark/config.json. At serve time
vllm_mlx/patches/nemotron_dspark.py re-quantizes the drafter to 8-bit (DSpark
was trained at NVFP4, so 8-bit is finer than its native format).

If you have an MLX-native NVFP4 repack of the same checkpoint, drop it at
<model-dir>/dspark-native/model.safetensors instead; the loader prefers it
and runs the drafter at native precision. Only dspark/config.json is needed
from this script in that case (use --config-only).

NVFP4 (modelopt): weights packed 2 x fp4-e2m1 per uint8, per-16-element block
scales in fp8-e4m3 (`<name>_scale`), one global fp32 scale (`<name>_scale_2`).
dequant = fp4_value * block_scale * global_scale. The block scales are e4m3
BIT PATTERNS stored in a uint8 tensor — decode them, never cast them.

Usage:
    python prepare_dspark_weights.py --mlx-model-path <nemotron 6bit dir>
        [--source nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark]
        [--survey-only]      # just print tensor names/shapes and exit
        [--config-only]      # write dspark/config.json only
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

# Make the pure-Python decode tables importable when run from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_mlx.spec_utils import FP4_E2M1_VALUES, fp8_e4m3_lut  # noqa: E402

SOURCE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"


def dequantize_nvfp4(packed, scales, global_scale, mx):
    """Dequantize one modelopt NVFP4 tensor to BF16.

    packed: uint8 [rows, cols/2] (low nibble = even column, high = odd)
    scales: uint8 [rows, cols/16] holding fp8-e4m3 bit patterns
    global_scale: fp32 scalar
    """
    fp4_lut = mx.array(FP4_E2M1_VALUES, dtype=mx.float32)
    e4m3_lut = mx.array(fp8_e4m3_lut(), dtype=mx.float32)
    lo = (packed & 0x0F).astype(mx.uint32)
    hi = (packed >> 4).astype(mx.uint32)
    vals = mx.stack([fp4_lut[lo], fp4_lut[hi]], axis=-1)
    vals = vals.reshape(packed.shape[0], packed.shape[1] * 2)
    s = e4m3_lut[scales.astype(mx.uint32)]
    s = mx.repeat(s, 16, axis=-1)[:, : vals.shape[1]]
    return (vals * s * global_scale).astype(mx.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--mlx-model-path", required=True)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--survey-only", action="store_true")
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument("--download-dir", default=None)
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    model_dir = Path(args.mlx_model_path).expanduser()
    dl = Path(args.download_dir or (model_dir / "dspark-src"))
    dl.mkdir(parents=True, exist_ok=True)

    files = (
        ["config.json"] if args.config_only else ["model.safetensors", "config.json"]
    )
    for fn in files:
        dest = dl / fn
        if not dest.exists():
            url = f"https://huggingface.co/{args.source}/resolve/main/{fn}"
            print(f"downloading {fn} ...")
            r = subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest), url])
            if r.returncode != 0:
                sys.exit(f"download failed: {fn}")

    dst = model_dir / "dspark"
    dst.mkdir(exist_ok=True)
    (dst / "config.json").write_text((dl / "config.json").read_text())
    if args.config_only:
        print(f"wrote {dst / 'config.json'}")
        return 0

    raw = mx.load(str(dl / "model.safetensors"))
    print(f"{len(raw)} tensors in source")

    if args.survey_only:
        for k in sorted(raw):
            v = raw[k]
            print(f"  {k:<70} {tuple(v.shape)} {v.dtype}")
        return 0

    out: dict = {}
    n_dq = 0
    for k in sorted(raw):
        if k.endswith("_scale") or k.endswith("_scale_2"):
            continue  # consumed alongside their base tensor
        w = raw[k]
        sk, gk = f"{k}_scale", f"{k}_scale_2"
        if sk in raw:  # NVFP4-packed
            gscale = raw[gk].astype(mx.float32) if gk in raw else mx.array(1.0)
            w = dequantize_nvfp4(raw[k], raw[sk], gscale, mx)
            n_dq += 1
        elif w.dtype in (mx.float32, mx.float16):
            w = w.astype(mx.bfloat16)
        mx.eval(w)
        out[k] = w

    mx.save_safetensors(str(dst / "weights.safetensors"), out)
    total = sum(v.nbytes for v in out.values())
    print(
        f"wrote {len(out)} tensors ({n_dq} dequantized from NVFP4, "
        f"{total / 1e6:.0f} MB) -> {dst / 'weights.safetensors'}"
    )

    # quick sanity: no NaN/Inf, plausible magnitudes
    for k in list(out)[:3] + list(out)[-3:]:
        a = out[k].astype(mx.float32)
        mabs = float(mx.abs(a).max())
        if math.isnan(mabs) or math.isinf(mabs) or mabs > 1e4:
            print(f"  WARNING {k}: max|w| = {mabs}")
        else:
            print(f"  ok {k}: max|w| = {mabs:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
