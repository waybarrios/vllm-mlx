# SPDX-License-Identifier: Apache-2.0
"""
MTP (Multi-Token Prediction) module infrastructure.

Provides abstract base class for model-specific MTP implementations and
a utility to load MTP weights from safetensors files that were stripped
by mlx-lm's sanitize().
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.quantized import QuantizedLinear

logger = logging.getLogger(__name__)


def _extract_quantization_config(model: nn.Module) -> dict | None:
    """Extract quantization config from the first QuantizedLinear in *model*.

    Traverses all sub-modules of *model* (via ``named_modules()``) and returns
    the ``bits``, ``group_size`` and ``mode`` of the first
    :class:`QuantizedLinear` found.  Returns ``None`` if the model is not
    quantized.
    """
    for _name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            return {
                "bits": module.bits,
                "group_size": module.group_size,
                "mode": module.mode,
            }
    return None


@dataclass
class MTPModuleConfig:
    """Configuration for an MTP module.

    Attributes:
        hidden_size: Hidden dimension of the main model.
        num_hidden_layers: Number of hidden layers in the main model
            (MTP layers start at this index in safetensors).
        num_mtp_layers: Number of MTP prediction layers (typically 1).
        rms_norm_eps: Epsilon for RMSNorm layers.
    """

    hidden_size: int = 4096
    num_hidden_layers: int = 30
    num_mtp_layers: int = 1
    rms_norm_eps: float = 1e-6


class MTPModule(ABC, nn.Module):
    """Abstract base for model-specific MTP implementations.

    Subclasses implement a single-step MTP forward pass that takes the
    last hidden state from the target model and the most recently predicted
    token, then produces logits for the next token and an updated hidden state.
    """

    def __init__(self, config: MTPModuleConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def __call__(
        self,
        hidden_states: mx.array,
        token_ids: mx.array,
        embed_fn: Any,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """Single MTP forward step.

        Args:
            hidden_states: (1, seq_len, hidden_size) from target model or
                previous MTP step.
            token_ids: (1, seq_len) token IDs to embed.
            embed_fn: Callable (nn.Embedding) to map token IDs to embeddings.
            cache: Optional KV cache for the MTP decoder layer(s).

        Returns:
            new_hidden_states: (1, seq_len, hidden_size) for next MTP step.
            The caller applies lm_head to produce logits.
        """
        ...

    def make_cache(self) -> Any:
        """Create KV cache for the MTP module's decoder layers.

        Subclasses should override to return appropriate cache objects.
        Returns None by default (no caching).
        """
        return None


# Known MTP weight prefix patterns across model families
_MTP_PREFIX_PATTERNS = [
    "model.layers.{num_hidden_layers}.",  # DeepSeek V3/V3.2, GLM-5, GLM-4 MoE Lite
    "model.mtp_layers.",  # MiMo
    "model.mtp.",  # Kimi, MiMo V2 Flash
]


def detect_mtp_prefix(
    model_path: str,
    num_hidden_layers: int,
) -> tuple[str, list[str]]:
    """Auto-detect the MTP weight prefix from safetensors files.

    Scans the safetensors index (or single file) for known MTP patterns.

    Args:
        model_path: Path to the model directory.
        num_hidden_layers: Number of hidden layers in the main model.

    Returns:
        Tuple of (detected prefix, list of matching weight keys).

    Raises:
        ValueError: If no MTP weights are found.
    """
    model_dir = Path(model_path)

    # Get all weight keys from safetensors index or file
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        all_keys = list(index.get("weight_map", {}).keys())
    else:
        single_path = model_dir / "model.safetensors"
        if single_path.exists():
            all_keys = list(mx.load(str(single_path)).keys())
        else:
            raise ValueError(f"No safetensors files found in {model_path}")

    # Try each known pattern
    found_prefix = None
    found_matching = None
    for pattern in _MTP_PREFIX_PATTERNS:
        prefix = pattern.format(num_hidden_layers=num_hidden_layers)
        matching = [k for k in all_keys if k.startswith(prefix)]
        if matching:
            if found_prefix is None:
                found_prefix = prefix
                found_matching = matching
                logger.info(
                    "Auto-detected MTP prefix '%s' (%d matching keys)",
                    prefix,
                    len(matching),
                )
            else:
                logger.warning(
                    "Additional MTP prefix '%s' also matched (%d keys); using '%s'",
                    prefix,
                    len(matching),
                    found_prefix,
                )

    if found_prefix is not None:
        return found_prefix, found_matching

    raise ValueError(
        f"No MTP weights found in {model_path}. "
        f"Tried patterns: {[p.format(num_hidden_layers=num_hidden_layers) for p in _MTP_PREFIX_PATTERNS]}. "
        f"This model may not have MTP layers, or it may be a quantized MLX "
        f"conversion where MTP weights were stripped."
    )


def detect_mtp_style(mtp_keys: list[str], prefix: str) -> str:
    """Detect MTP architecture style from weight keys.

    Args:
        mtp_keys: List of MTP weight keys.
        prefix: The detected MTP prefix.

    Returns:
        "standard" if enorm/hnorm/eh_proj pattern found (DeepSeek-style),
        "simple" if only decoder block keys found (MiMo-style).
    """
    suffixes = [k[len(prefix) :] for k in mtp_keys]
    has_enorm = any(s.startswith("enorm.") for s in suffixes)
    has_hnorm = any(s.startswith("hnorm.") for s in suffixes)
    has_eh_proj = any(s.startswith("eh_proj.") for s in suffixes)

    if has_enorm and has_hnorm and has_eh_proj:
        return "standard"
    return "simple"


def _sanitize_mtp_weights(
    mtp_weights: dict[str, mx.array],
    model_config: Any,
) -> dict[str, mx.array]:
    """Apply the same transformations as deepseek_v32 sanitize() to MTP weights.

    Raw HuggingFace safetensors for DeepSeek V3/V3.2/GLM models need:
    1. FP8 dequantization (weight_scale_inv blocks → bfloat16)
    2. Expert stacking (per-expert weights → switch_mlp format)
    3. kv_b_proj splitting (→ embed_q + unembed_out)

    Keys should have the ``model.layers.N.`` prefix already stripped.

    Note: Quantized MLX models (e.g. 4-bit mlx-community conversions) do NOT
    include MTP weights at all — mlx-lm's sanitize() strips them during
    conversion.  MTP speculative decoding therefore requires either the
    original (FP8/BF16) HuggingFace checkpoint or a custom MLX conversion
    that preserves the MTP layer.
    """

    # Step 1: FP8 dequantization
    def _dequant_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
        weight = mx.from_fp8(weight, dtype=mx.bfloat16)
        bs = 128  # block size
        m, n = weight.shape
        pad_bottom = (-m) % bs
        pad_side = (-n) % bs
        weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
        weight = weight.reshape(((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs))
        weight = (weight * scale_inv[:, None, :, None]).reshape(
            m + pad_bottom, n + pad_side
        )
        return weight[:m, :n].astype(mx.bfloat16)

    new_weights: dict[str, mx.array] = {}
    for k, v in mtp_weights.items():
        if "weight_scale_inv" in k:
            scale_inv = v
            wk = k.replace("_scale_inv", "")
            if wk in mtp_weights:
                new_weights[wk] = _dequant_fp8(mtp_weights[wk], scale_inv)
        elif k not in new_weights:
            new_weights[k] = v
    weights = new_weights

    # Step 2: Expert stacking (per-expert → switch_mlp)
    n_routed_experts = getattr(model_config, "n_routed_experts", None)
    if n_routed_experts:
        for _n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            for k_type in ["weight", "scales", "biases"]:
                first_key = f"mlp.experts.0.{m}.{k_type}"
                if first_key in weights:
                    to_join = []
                    for e in range(n_routed_experts):
                        ek = f"mlp.experts.{e}.{m}.{k_type}"
                        if ek in weights:
                            to_join.append(weights.pop(ek))
                    if to_join:
                        weights[f"mlp.switch_mlp.{m}.{k_type}"] = mx.stack(to_join)

    # Step 3: kv_b_proj splitting (→ embed_q + unembed_out)
    kv_b_key = "self_attn.kv_b_proj.weight"
    if kv_b_key in weights:
        v = weights.pop(kv_b_key)
        num_heads = model_config.num_attention_heads
        qk_nope_head_dim = model_config.qk_nope_head_dim
        v_head_dim = model_config.v_head_dim
        head_dim = qk_nope_head_dim + v_head_dim

        quantized = "self_attn.kv_b_proj.scales" in weights
        if quantized:
            kv_lora_rank = model_config.kv_lora_rank
            scales = weights.pop("self_attn.kv_b_proj.scales")
            biases = weights.pop("self_attn.kv_b_proj.biases")
            bits = (v.shape[-1] * 32) // kv_lora_rank
            group_size = kv_lora_rank // scales.shape[-1]
            v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

        v = v.reshape(num_heads, head_dim, -1)
        wk = mx.contiguous(v[:, :qk_nope_head_dim, :].swapaxes(-1, -2))
        wv = mx.contiguous(v[:, qk_nope_head_dim:, :])

        if quantized:
            wk, wk_scales, wk_biases = mx.quantize(wk, bits=bits, group_size=group_size)
            wv, wv_scales, wv_biases = mx.quantize(wv, bits=bits, group_size=group_size)
            weights["self_attn.embed_q.scales"] = wk_scales
            weights["self_attn.unembed_out.scales"] = wv_scales
            weights["self_attn.embed_q.biases"] = wk_biases
            weights["self_attn.unembed_out.biases"] = wv_biases

        weights["self_attn.embed_q.weight"] = wk
        weights["self_attn.unembed_out.weight"] = wv

    return weights


def load_mtp_weights(
    model_path: str,
    num_hidden_layers: int,
    mtp_module: nn.Module,
    model_config: Any = None,
    mtp_prefix: str | None = None,
    main_model: nn.Module | None = None,
) -> None:
    """Load MTP layer weights from safetensors, bypassing mlx-lm sanitize().

    mlx-lm's sanitize() removes MTP weights (layers >= num_hidden_layers).
    This function reads them directly from the safetensors files, applies
    the same sanitize transformations that mlx-lm would apply (FP8 dequant,
    expert stacking, kv_b_proj splitting), and loads them into the MTP module
    with key remapping.

    When *main_model* is given and is quantized, the MTP module is quantized
    to the same bit-width / group-size after weight loading so that the two
    modules operate at the same precision.

    Args:
        model_path: Path to the model directory containing safetensors files.
        num_hidden_layers: Number of hidden layers in the main model.
        mtp_module: The MTP module to load weights into.
        model_config: Model configuration (e.g. ``model.args``).  When
            provided the raw weights are run through
            :func:`_sanitize_mtp_weights` before key remapping.
        mtp_prefix: Override auto-detected MTP weight prefix.
        main_model: The main (target) model.  When provided and quantized,
            the MTP module will be quantized to the same configuration.
    """
    model_dir = Path(model_path)

    # Find safetensors index or single file
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        # Single safetensors file - load all and filter
        weight_map = None

    # Collect MTP weight keys and their source files
    if mtp_prefix is None:
        mtp_prefix, _ = detect_mtp_prefix(model_path, num_hidden_layers)
    mtp_weights: dict[str, mx.array] = {}

    if weight_map is not None:
        # Sharded safetensors: find files containing MTP weights
        files_to_load: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith(mtp_prefix):
                files_to_load.setdefault(filename, []).append(key)

        for filename, keys in files_to_load.items():
            filepath = model_dir / filename
            shard = mx.load(str(filepath))
            for key in keys:
                if key in shard:
                    mtp_weights[key] = shard[key]
            del shard
    else:
        # Single file: load and filter
        single_path = model_dir / "model.safetensors"
        if single_path.exists():
            all_weights = mx.load(str(single_path))
            for key, value in all_weights.items():
                if key.startswith(mtp_prefix):
                    mtp_weights[key] = value
            del all_weights

    if not mtp_weights:
        raise ValueError(
            f"No MTP weights found with prefix '{mtp_prefix}' in {model_path}. "
            f"This model may not have MTP layers, or it may be a quantized MLX "
            f"conversion where mlx-lm's sanitize() stripped MTP weights during "
            f"conversion.  MTP speculative decoding requires the original "
            f"HuggingFace checkpoint or a custom conversion that preserves MTP "
            f"weights."
        )

    logger.info(
        "Found %d MTP weight tensors with prefix '%s'",
        len(mtp_weights),
        mtp_prefix,
    )

    # Apply sanitize only when raw HF weights need transformation (FP8 dequant, expert stacking, etc.)
    has_fp8 = any("weight_scale_inv" in k for k in mtp_weights)
    has_experts = any("experts.0." in k for k in mtp_weights)
    has_kv_b_proj = any("kv_b_proj" in k for k in mtp_weights)
    needs_sanitize = has_fp8 or has_experts or has_kv_b_proj
    if model_config is not None and needs_sanitize:
        stripped = {}
        for key, value in mtp_weights.items():
            suffix = key[len(mtp_prefix) :]
            stripped[suffix] = value
        stripped = _sanitize_mtp_weights(stripped, model_config)
        mtp_weights = {f"{mtp_prefix}{k}": v for k, v in stripped.items()}
        logger.info(
            "Applied sanitize transforms to MTP weights (%d tensors after sanitize)",
            len(mtp_weights),
        )

    # Remap keys: strip prefix and map to module structure
    # Detect style to determine remapping strategy
    style = detect_mtp_style(list(mtp_weights.keys()), mtp_prefix)

    remapped: dict[str, mx.array] = {}
    for key, value in mtp_weights.items():
        suffix = key[len(mtp_prefix) :]  # Remove prefix

        if style == "standard":
            # DeepSeek-style: enorm/hnorm/eh_proj at top level, decoder components under decoder_layer.*
            if (
                suffix.startswith("enorm.")
                or suffix.startswith("hnorm.")
                or suffix.startswith("eh_proj.")
            ):
                remapped[suffix] = value
            elif (
                suffix.startswith("self_attn.")
                or suffix.startswith("mlp.")
                or suffix.startswith("input_layernorm.")
                or suffix.startswith("post_attention_layernorm.")
            ):
                remapped[f"decoder_layer.{suffix}"] = value
            elif suffix.startswith("shared_head.norm."):
                new_suffix = suffix.replace("shared_head.norm.", "shared_head_norm.")
                remapped[new_suffix] = value
            elif suffix.startswith("embed_tokens.") or suffix.startswith(
                "shared_head.head."
            ):
                logger.info(
                    "Skipping MTP weight '%s' (using main model's shared weights)", key
                )
            else:
                remapped[suffix] = value
                logger.warning("Unknown MTP weight key pattern: %s", key)
        else:
            # Simple-style (MiMo, Kimi): decoder components go under layers.{idx}.*
            import re

            layer_match = re.match(r"^(\d+)\.(.*)", suffix)
            if layer_match:
                # Multi-layer MTP (e.g., model.mtp_layers.0.*, model.mtp_layers.1.*)
                layer_idx = layer_match.group(1)
                rest = layer_match.group(2)
                remapped[f"layers.{layer_idx}.{rest}"] = value
            elif suffix.startswith("embed_tokens.") or suffix.startswith("lm_head."):
                logger.info(
                    "Skipping MTP weight '%s' (using main model's shared weights)", key
                )
            elif (
                suffix.startswith("self_attn.")
                or suffix.startswith("mlp.")
                or suffix.startswith("input_layernorm.")
                or suffix.startswith("post_attention_layernorm.")
            ):
                # Single-layer MTP from model.layers.{N}.* prefix (no layer index)
                remapped[f"layers.0.{suffix}"] = value
            elif suffix.startswith("norm."):
                # Final norm — load into module (not shared with main model)
                remapped[suffix] = value
            else:
                remapped[suffix] = value
                logger.warning("Unknown simple MTP weight key pattern: %s", key)

    # Load into MTP module
    mtp_module.load_weights(list(remapped.items()))
    logger.info("Loaded %d MTP weight tensors into MTP module", len(remapped))

    # Apply quantization to match main model if quantized
    if main_model is not None:
        quant_config = _extract_quantization_config(main_model)
        if quant_config is not None:
            from mlx.utils import tree_flatten

            pre_size = sum(v.nbytes for _, v in tree_flatten(mtp_module.parameters()))
            nn.quantize(
                mtp_module,
                group_size=quant_config["group_size"],
                bits=quant_config["bits"],
                mode=quant_config["mode"],
            )
            post_size = sum(v.nbytes for _, v in tree_flatten(mtp_module.parameters()))
            saved_mb = (pre_size - post_size) / (1024 * 1024)
            logger.info(
                "Quantized MTP module to %d-bit (group_size=%d, mode=%s): "
                "%.0f MB -> %.0f MB (saved %.0f MB)",
                quant_config["bits"],
                quant_config["group_size"],
                quant_config["mode"],
                pre_size / (1024 * 1024),
                post_size / (1024 * 1024),
                saved_mb,
            )
