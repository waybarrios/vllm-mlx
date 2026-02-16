#!/usr/bin/env python3
"""
Add MTP (Multi-Token Prediction) weights to the MLX Step-3.5-Flash-4bit model.

Step 3.5 Flash has 3 MTP layers (layers 45, 46, 47 in the original model)
that were stripped during the MLX community 4-bit conversion. This script:

1. Downloads the BF16 shards containing MTP weights from the original model
2. Extracts layers 45-47, remaps to mtp.layers.{0,1,2}.*
3. Quantizes to 4-bit (matching the MLX model's quantization)
4. Saves as model-mtp.safetensors
5. Updates model.safetensors.index.json and config.json

Usage:
    python add_mtp_weights_step3p5.py [--mlx-model-path PATH] [--download-dir DIR]

Requirements:
    pip install mlx
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Default paths
DEFAULT_MLX_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Step-3.5-Flash-4bit"
)
DEFAULT_SOURCE_MODEL = "stepfun-ai/Step-3.5-Flash"

# MTP layers are in these two shards (layer 45 shared_head.norm in shard 1, rest in shard 2)
MTP_SHARDS = {
    "model-00001.safetensors": "https://huggingface.co/{model}/resolve/main/model-00001.safetensors",
    "model-00002.safetensors": "https://huggingface.co/{model}/resolve/main/model-00002.safetensors",
}

# Source layer indices for MTP
MTP_SOURCE_LAYERS = [45, 46, 47]
NUM_MTP_LAYERS = 3


def find_snapshot_dir(model_path: str) -> Path:
    """Find the latest snapshot directory in HF cache structure."""
    snapshots_dir = Path(model_path) / "snapshots"
    if not snapshots_dir.exists():
        if (Path(model_path) / "config.json").exists():
            return Path(model_path)
        raise FileNotFoundError(f"No snapshots found in {model_path}")

    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {snapshots_dir}")
    return snapshots[-1]


def download_mtp_shards(dest_path: Path, source_model: str) -> list[Path]:
    """Download the BF16 shards containing MTP weights using curl with resume."""
    shard_paths = []
    for shard_name, url_template in MTP_SHARDS.items():
        url = url_template.format(model=source_model)
        shard_path = dest_path / shard_name

        if shard_path.exists():
            size_gb = shard_path.stat().st_size / 1e9
            print(f"Shard {shard_name} already exists ({size_gb:.2f} GB)")
            if size_gb < 1.0:
                print("  WARNING: File seems too small, re-downloading...")
            else:
                shard_paths.append(shard_path)
                continue

        print(f"Downloading {shard_name}...")
        print(f"  URL: {url}")
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(shard_path), url],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Download of {shard_name} failed with code {result.returncode}"
            )

        size_gb = shard_path.stat().st_size / 1e9
        print(f"  Downloaded: {size_gb:.2f} GB")
        shard_paths.append(shard_path)

    return shard_paths


def extract_and_quantize_mtp_weights(
    shard_paths: list[Path], snapshot_dir: Path, quantization_bits: int = 4
):
    """Extract MTP weights from BF16 shards, remap, quantize, and save."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    print(f"\nExtracting MTP weights from {len(shard_paths)} shards...")

    # Load and merge MTP weights from both shards
    mtp_weights = {}
    for shard_path in shard_paths:
        print(f"  Loading {shard_path.name}...")
        all_weights = mx.load(str(shard_path))
        for k, v in all_weights.items():
            # Only keep layers 45, 46, 47
            for src_layer in MTP_SOURCE_LAYERS:
                if k.startswith(f"model.layers.{src_layer}."):
                    mtp_weights[k] = v
        del all_weights

    print(f"Found {len(mtp_weights)} MTP weight keys")

    # Remap keys: model.layers.{45,46,47}.* -> mtp.layers.{0,1,2}.*
    remapped = {}
    for key, value in mtp_weights.items():
        new_key = key
        for i, src_layer in enumerate(MTP_SOURCE_LAYERS):
            src_prefix = f"model.layers.{src_layer}."
            if key.startswith(src_prefix):
                suffix = key[len(src_prefix) :]
                # transformer.shared_head.* -> shared_head.*
                if suffix.startswith("transformer.shared_head."):
                    suffix = suffix.replace("transformer.shared_head.", "shared_head.")
                new_key = f"mtp.layers.{i}.{suffix}"
                break
        remapped[new_key] = value
    del mtp_weights

    print(f"Remapped to {len(remapped)} keys with mtp.layers.* prefix")

    # Read quantization config
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization", {})
    bits = quant_config.get("bits", quantization_bits)
    group_size = quant_config.get("group_size", 64)
    print(f"Target quantization: {bits}-bit, group_size={group_size}")

    # Norm suffixes that need +1.0 adjustment (HF zero_centered RMSNorm -> MLX standard)
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".enorm.weight",
        ".hnorm.weight",
        ".shared_head.norm.weight",
    )

    # Keys to keep in FP (not quantize)
    def should_skip_quantize(key):
        skip_patterns = (
            "layernorm.weight",
            "q_norm.weight",
            "k_norm.weight",
            "enorm.weight",
            "hnorm.weight",
            "shared_head.norm.weight",
        )
        return any(p in key for p in skip_patterns)

    quantized_weights = {}
    for key in sorted(remapped.keys()):
        weight = remapped.pop(key)

        # Norm adjustment: +1.0 for zero-centered RMSNorm weights (HF -> MLX convention)
        if any(key.endswith(s) for s in norm_suffixes) and weight.ndim == 1:
            weight = weight + 1.0
            mx.eval(weight)
            print(f"  Adjusted norm: {key}")

        if should_skip_quantize(key):
            print(f"  Keep FP: {key} {weight.shape}")
            quantized_weights[key] = weight
        elif weight.ndim >= 2 and weight.shape[-1] >= group_size:
            q_w, q_s, q_b = mx.quantize(weight, group_size=group_size, bits=bits)
            mx.eval(q_w, q_s, q_b)
            print(f"  Quantize {bits}-bit: {key} {q_w.shape}")
            quantized_weights[key] = q_w
            quantized_weights[key.replace(".weight", ".scales")] = q_s
            quantized_weights[key.replace(".weight", ".biases")] = q_b
        else:
            print(f"  Keep FP (small): {key} {weight.shape}")
            quantized_weights[key] = weight
        del weight
    del remapped

    # Save MTP weights
    mtp_output_file = snapshot_dir / "model-mtp.safetensors"
    print(
        f"\nSaving {len(quantized_weights)} quantized MTP weights to {mtp_output_file}"
    )
    mx.save_safetensors(str(mtp_output_file), quantized_weights)

    total_bytes = sum(v.nbytes for v in quantized_weights.values())
    print(f"MTP weights size: {total_bytes / 1e6:.1f} MB (quantized)")

    return mtp_output_file, list(quantized_weights.keys())


def update_model_index(snapshot_dir: Path, mtp_keys: list):
    """Update model.safetensors.index.json to include MTP weight keys."""
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"WARNING: No index file found at {index_path}, skipping")
        return

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    for key in mtp_keys:
        weight_map[key] = "model-mtp.safetensors"

    index["weight_map"] = weight_map

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Updated {index_path} with {len(mtp_keys)} MTP weight entries")


def update_config(snapshot_dir: Path):
    """Ensure config.json has num_nextn_predict_layers=3."""
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config["num_nextn_predict_layers"] = NUM_MTP_LAYERS

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated {config_path}: num_nextn_predict_layers={NUM_MTP_LAYERS}")


def install_modeling_file(snapshot_dir: Path):
    """Copy modeling_step3p5_mtp.py into the snapshot directory as modeling_step3p5.py.

    The stock MLX community model lacks MTP support in its modeling file.
    This replaces it with the MTP-enabled version bundled alongside this script.

    Note: Once https://github.com/ml-explore/mlx-lm/pull/901 is merged upstream,
    this workaround will no longer be necessary.
    """
    script_dir = Path(__file__).resolve().parent
    src = script_dir / "modeling_step3p5_mtp.py"
    dst = snapshot_dir / "modeling_step3p5.py"

    if not src.exists():
        print(f"WARNING: {src} not found, skipping modeling file install")
        return

    import shutil

    shutil.copy2(src, dst)
    print(f"Installed MTP modeling file: {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Add MTP weights to MLX Step-3.5-Flash-4bit model"
    )
    parser.add_argument(
        "--mlx-model-path",
        type=str,
        default=DEFAULT_MLX_MODEL,
        help=f"Path to MLX model directory (default: {DEFAULT_MLX_MODEL})",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default=DEFAULT_SOURCE_MODEL,
        help=f"HuggingFace model for MTP shards (default: {DEFAULT_SOURCE_MODEL})",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to download MTP shards to (default: temp dir)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bits (default: 4, matching 4-bit model)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use existing shards)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MTP Weight Addition for Step-3.5-Flash MLX Model")
    print("=" * 60)

    # Find snapshot directory
    snapshot_dir = find_snapshot_dir(args.mlx_model_path)
    print(f"\nMLX model snapshot: {snapshot_dir}")

    # Verify config exists
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: No config.json found in {snapshot_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Hidden size: {config.get('hidden_size', '?')}")
    print(f"Num hidden layers: {config.get('num_hidden_layers', '?')}")
    print(f"MoE experts: {config.get('moe_num_experts', '?')}")

    # Check if MTP weights already exist
    existing_mtp = config.get("num_nextn_predict_layers", 0)
    if existing_mtp > 0:
        index_path = snapshot_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            mtp_keys = [k for k in index.get("weight_map", {}) if k.startswith("mtp.")]
            if mtp_keys:
                print(f"\nFound {len(mtp_keys)} existing MTP weight keys")
                print("MTP weights already added. Nothing to do.")
                sys.exit(0)

    # Download MTP shards
    download_dir = (
        Path(args.download_dir) if args.download_dir else Path(tempfile.mkdtemp())
    )
    print(f"\nDownload directory: {download_dir}")

    if not args.skip_download:
        shard_paths = download_mtp_shards(download_dir, args.source_model)
    else:
        shard_paths = [download_dir / name for name in MTP_SHARDS]
        for p in shard_paths:
            if not p.exists():
                print(f"ERROR: Shard not found at {p}")
                sys.exit(1)

    # Extract, quantize, and save
    mtp_file, mtp_keys = extract_and_quantize_mtp_weights(
        shard_paths, snapshot_dir, quantization_bits=args.bits
    )

    # Update model index
    update_model_index(snapshot_dir, mtp_keys)

    # Update config
    update_config(snapshot_dir)

    # Install MTP-enabled modeling file
    install_modeling_file(snapshot_dir)

    print("\n" + "=" * 60)
    print("SUCCESS! MTP weights added to MLX model.")
    print("=" * 60)
    print(f"\nMTP weight file: {mtp_file}")
    print(f"Total MTP keys: {len(mtp_keys)}")
    print("\nTo use MTP, start the server with --enable-mtp:")
    print("  vllm-mlx serve mlx-community/Step-3.5-Flash-4bit \\")
    print("      --enable-mtp --port 1340")


if __name__ == "__main__":
    main()
