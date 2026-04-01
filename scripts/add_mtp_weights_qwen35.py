#!/usr/bin/env python3
"""
Add MTP (Multi-Token Prediction) weights to an existing MLX Qwen3.5 model.

This script:
1. Fetches the safetensors index from the original BF16 HuggingFace model
2. Identifies shards containing MTP weights (mtp.* keys)
3. Downloads only those shards via curl -C -
4. Extracts MTP weights
5. For MoE models: stacks expert weights (256×) into switch_mlp format
6. Applies norm shift (HF weight → MLX weight+1.0) for RMSNorm keys
7. Quantizes to match the MLX model's quantization scheme
8. Saves as mtp/weights.safetensors (subdirectory avoids mlx_vlm glob)

Supports both:
- MoE models (Qwen3.5-122B-A10B, 35B-A3B): 256 experts, sparse MTP attention
- Dense models (Qwen3.5-27B): full MTP with k/v projections and norms

Usage:
    python add_mtp_weights_qwen35.py --mlx-model-path PATH --source-model MODEL

Requirements:
    pip install mlx
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Known model configurations
MODEL_CONFIGS = {
    "Qwen/Qwen3.5-122B-A10B": {
        "num_experts": 256,
        "hidden_size": 3072,
        "is_moe": True,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "num_experts": 256,
        "hidden_size": 2048,
        "is_moe": True,
    },
    "Qwen/Qwen3.5-27B": {
        "num_experts": 0,
        "hidden_size": 5120,
        "is_moe": False,
    },
}


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


def fetch_shard_index(source_model: str, download_dir: Path) -> dict:
    """Fetch model.safetensors.index.json from HuggingFace."""
    index_url = f"https://huggingface.co/{source_model}/resolve/main/model.safetensors.index.json"
    index_path = download_dir / "source_index.json"

    print(f"Fetching shard index from {source_model}...")
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "-o", str(index_path), index_url],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch index: return code {result.returncode}")

    with open(index_path) as f:
        return json.load(f)


def identify_mtp_shards(index: dict) -> tuple[dict[str, str], set[str]]:
    """Identify which shards contain MTP weights.

    Returns:
        Tuple of (mtp_key_to_shard mapping, set of shard filenames to download)
    """
    weight_map = index.get("weight_map", {})
    mtp_keys = {}
    shards_needed = set()

    for key, shard in weight_map.items():
        if key.startswith("mtp."):
            mtp_keys[key] = shard
            shards_needed.add(shard)

    return mtp_keys, shards_needed


def download_shards(
    shards: set[str], source_model: str, download_dir: Path
) -> dict[str, Path]:
    """Download required shards using curl with resume support."""
    shard_paths = {}
    for shard_name in sorted(shards):
        shard_url = f"https://huggingface.co/{source_model}/resolve/main/{shard_name}"
        shard_path = download_dir / shard_name

        if shard_path.exists():
            size_gb = shard_path.stat().st_size / 1e9
            print(f"  {shard_name}: exists ({size_gb:.2f} GB)")
            shard_paths[shard_name] = shard_path
            continue

        print(f"  Downloading {shard_name}...")
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(shard_path), shard_url],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Download failed for {shard_name}: code {result.returncode}"
            )

        size_gb = shard_path.stat().st_size / 1e9
        print(f"  {shard_name}: {size_gb:.2f} GB")
        shard_paths[shard_name] = shard_path

    return shard_paths


def extract_and_quantize_mtp_weights(
    mtp_keys: dict[str, str],
    shard_paths: dict[str, Path],
    snapshot_dir: Path,
    is_moe: bool,
    num_experts: int,
    no_quantize: bool = False,
):
    """Extract MTP weights from BF16 shards, optionally quantize, and save."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    # Read MLX model's quantization config
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    quant_config = text_config.get("quantization", config.get("quantization", {}))
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 64)
    if no_quantize:
        print("MTP weights will be saved in BF16 (no quantization)")
    else:
        print(f"Target quantization: {bits}-bit, group_size={group_size}")

    # Group MTP keys by shard for efficient I/O
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in mtp_keys.items():
        shard_to_keys.setdefault(shard, []).append(key)

    # Load all MTP weights
    print(f"\nExtracting MTP weights from {len(shard_paths)} shards...")
    all_mtp_weights: dict[str, mx.array] = {}

    for shard_name, keys in sorted(shard_to_keys.items()):
        shard_path = shard_paths[shard_name]
        print(f"  Loading {shard_name} ({len(keys)} MTP keys)...")
        shard_data = mx.load(str(shard_path))
        for key in keys:
            if key in shard_data:
                all_mtp_weights[key] = shard_data[key]
        del shard_data

    print(f"Loaded {len(all_mtp_weights)} MTP weight tensors")

    # Norm keys that need +1.0 shift (HF centered ~0 → MLX centered ~1)
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
        "mtp.norm.weight",
    )

    # Keys to keep as FP (not quantize)
    skip_quantize_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        ".shared_expert_gate.weight",
    )

    def _quantize_one(key: str, weight: mx.array) -> dict[str, mx.array]:
        """Quantize a single weight, apply norm adjustment."""
        # Norm shift: +1.0 for RMSNorm weights
        if any(key.endswith(s) for s in norm_suffixes) and weight.ndim == 1:
            weight = weight + 1.0
            mx.eval(weight)
            print(f"  Norm shift: {key}")

        if no_quantize:
            print(f"  BF16: {key} {weight.shape}")
            return {key: weight}
        elif any(key.endswith(s) for s in skip_quantize_suffixes):
            print(f"  Keep FP: {key} {weight.shape}")
            return {key: weight}
        elif weight.ndim >= 2 and weight.shape[-1] >= group_size:
            q_w, q_s, q_b = mx.quantize(weight, group_size=group_size, bits=bits)
            mx.eval(q_w, q_s, q_b)
            print(f"  Quantize {bits}-bit: {key} {q_w.shape}")
            return {
                key: q_w,
                key.replace(".weight", ".scales"): q_s,
                key.replace(".weight", ".biases"): q_b,
            }
        else:
            print(f"  Keep FP (small): {key} {weight.shape}")
            return {key: weight}

    quantized_weights: dict[str, mx.array] = {}

    if is_moe and num_experts > 0:
        # Stack expert weights ONE PROJECTION AT A TIME to minimize peak memory
        for proj in ["up_proj", "down_proj", "gate_proj"]:
            expert_keys = [
                f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"
                for e in range(num_experts)
            ]
            if all(k in all_mtp_weights for k in expert_keys):
                stacked = mx.stack([all_mtp_weights.pop(k) for k in expert_keys])
                mx.eval(stacked)
                stacked_key = f"mtp.layers.0.mlp.switch_mlp.{proj}.weight"
                print(f"  Stacked {num_experts} experts for {proj}: {stacked.shape}")
                quantized_weights.update(_quantize_one(stacked_key, stacked))
                del stacked
            else:
                present = sum(1 for k in expert_keys if k in all_mtp_weights)
                if present > 0:
                    print(f"  WARNING: Only {present}/{num_experts} experts for {proj}")

    # Quantize remaining non-expert weights
    for key in sorted(all_mtp_weights.keys()):
        weight = all_mtp_weights.pop(key)
        quantized_weights.update(_quantize_one(key, weight))
        del weight
    del all_mtp_weights

    # Save to mtp/ subdirectory (avoids mlx_vlm glob loading all *.safetensors)
    mtp_output_dir = snapshot_dir / "mtp"
    mtp_output_dir.mkdir(exist_ok=True)
    mtp_output_file = mtp_output_dir / "weights.safetensors"
    mode_str = "BF16" if no_quantize else "quantized"
    print(
        f"\nSaving {len(quantized_weights)} {mode_str} MTP weights to {mtp_output_file}"
    )
    mx.save_safetensors(str(mtp_output_file), quantized_weights)

    total_bytes = sum(v.nbytes for v in quantized_weights.values())
    print(f"MTP weights size: {total_bytes / 1e6:.1f} MB ({mode_str})")

    return mtp_output_file, list(quantized_weights.keys())


def update_model_index(snapshot_dir: Path, mtp_keys: list[str]):
    """Update model.safetensors.index.json to include MTP weight keys."""
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"WARNING: No index file found at {index_path}, skipping index update")
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
    """Update config.json to signal MTP availability.

    For Qwen3.5, mtp_num_hidden_layers already exists in text_config.
    We add num_nextn_predict_layers at top level for vllm-mlx compatibility.
    """
    config_path = snapshot_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    num_mtp = text_config.get("mtp_num_hidden_layers", 0)

    if num_mtp > 0:
        # Set num_nextn_predict_layers for vllm-mlx MTP detection
        config["num_nextn_predict_layers"] = num_mtp
        text_config["num_nextn_predict_layers"] = num_mtp
        if "text_config" in config:
            config["text_config"] = text_config

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Updated config: num_nextn_predict_layers={num_mtp}")
    else:
        print("WARNING: mtp_num_hidden_layers not found in config")


def main():
    parser = argparse.ArgumentParser(description="Add MTP weights to MLX Qwen3.5 model")
    parser.add_argument(
        "--mlx-model-path",
        type=str,
        required=True,
        help="Path to MLX model directory (HF cache or direct path)",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        required=True,
        help="HuggingFace BF16 model to download MTP shards from (e.g., Qwen/Qwen3.5-122B-A10B)",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to download shards to (default: temp dir)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use existing shards in download-dir)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Don't delete downloaded BF16 shards after extraction",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Save MTP weights in BF16 (no quantization). Required for correct MTP predictions.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MTP Weight Addition for Qwen3.5 MLX Model")
    print("=" * 60)

    # Find snapshot directory
    snapshot_dir = find_snapshot_dir(args.mlx_model_path)
    print(f"\nMLX model snapshot: {snapshot_dir}")

    # Read config
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: No config.json found in {snapshot_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    model_type = text_config.get("model_type", config.get("model_type", "unknown"))
    hidden_size = text_config.get("hidden_size", "?")
    num_experts = text_config.get("num_experts", 0)
    is_moe = num_experts > 0
    mtp_layers = text_config.get("mtp_num_hidden_layers", 0)

    print(f"Model type: {model_type}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num experts: {num_experts} ({'MoE' if is_moe else 'Dense'})")
    print(f"MTP layers: {mtp_layers}")

    if mtp_layers == 0:
        print("ERROR: Model has no MTP layers configured (mtp_num_hidden_layers=0)")
        sys.exit(1)

    # Check if MTP weights already exist
    mtp_file = snapshot_dir / "mtp" / "weights.safetensors"
    if mtp_file.exists():
        size_mb = mtp_file.stat().st_size / 1e6
        print(f"\nWARNING: mtp/weights.safetensors already exists ({size_mb:.1f} MB)")
        print("Delete it first if you want to regenerate.")
        sys.exit(0)

    # Setup download directory
    if args.download_dir:
        download_dir = Path(args.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
    else:
        download_dir = Path(tempfile.mkdtemp(prefix="qwen35_mtp_"))
    print(f"\nDownload dir: {download_dir}")

    # Fetch shard index and identify MTP shards
    source_index = fetch_shard_index(args.source_model, download_dir)
    mtp_key_map, shards_needed = identify_mtp_shards(source_index)

    print(
        f"\nFound {len(mtp_key_map)} MTP weight keys across {len(shards_needed)} shards:"
    )
    for shard in sorted(shards_needed):
        count = sum(1 for v in mtp_key_map.values() if v == shard)
        print(f"  {shard}: {count} keys")

    # Download shards
    if not args.skip_download:
        print(f"\nDownloading {len(shards_needed)} shards...")
        shard_paths = download_shards(shards_needed, args.source_model, download_dir)
    else:
        shard_paths = {}
        for shard_name in shards_needed:
            p = download_dir / shard_name
            if p.exists():
                shard_paths[shard_name] = p
            else:
                print(f"ERROR: Shard not found: {p}")
                sys.exit(1)

    # Extract, optionally quantize, and save MTP weights
    mtp_file, mtp_weight_keys = extract_and_quantize_mtp_weights(
        mtp_key_map,
        shard_paths,
        snapshot_dir,
        is_moe,
        num_experts,
        no_quantize=args.no_quantize,
    )

    # NOTE: Do NOT update model.safetensors.index.json — mlx_vlm's glob
    # would try to load MTP weights and fail with strict loading.
    # MTP weights are loaded separately by inject_mtp_support().

    # Update config
    update_config(snapshot_dir)

    # Cleanup downloaded shards
    if not args.keep_shards and not args.skip_download:
        print("\nCleaning up downloaded shards...")
        for shard_path in shard_paths.values():
            shard_path.unlink(missing_ok=True)
            print(f"  Deleted {shard_path.name}")

    print("\n" + "=" * 60)
    print("SUCCESS! MTP weights added to MLX model.")
    print("=" * 60)
    print(f"\nMTP weight file: {mtp_file}")
    print(f"Total MTP keys: {len(mtp_weight_keys)}")
    print("\nTo use MTP, start the server with --enable-mtp:")
    print(f"  vllm-mlx serve {args.mlx_model_path} --enable-mtp")


if __name__ == "__main__":
    main()
