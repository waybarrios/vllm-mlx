# SPDX-License-Identifier: Apache-2.0
"""
Runtime MTP (Multi-Token Prediction) support for Qwen3.5 models.

Qwen3.5 models may include a built-in MTP head that predicts token n+2
from hidden states + token n+1.  MTP weights are added to the quantized
MLX model via scripts/add_mtp_weights_qwen35.py.

Since mlx_lm's qwen3_5.py does NOT define MTP module/methods, this
module provides:
  - inject_mtp_support(): dynamically creates MTP module, loads weights,
    and monkey-patches the model class with return_hidden, mtp_forward,
    and make_mtp_cache
  - validate_mtp_support(): checks whether a loaded model has working MTP

Supports both Dense (27B) and MoE (122B-A10B, 35B-A3B) architectures.

The actual MTP scheduling logic lives in:
  - vllm_mlx/scheduler.py  (_install_mtp, _mtp_step, _mtp_next)
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _fixup_moe_mtp(mtp, inner_model, loaded_keys: set, mx) -> None:
    """Fix missing weights in MoE MTP module.

    MoE MTP checkpoints (122B, 35B) only contain: fc, q_proj, o_proj,
    shared_expert.*, and per-expert weights.  Missing:
    - k_proj, v_proj → zero out (attention becomes no-op)
    - gate, shared_expert_gate → copy from main model's last full-attn layer
    - norms → already at identity (weight=1.0), no action needed
    """
    import mlx.utils

    mtp_layer = mtp.layers[0]

    # Find last full-attention layer in main model for gate weights
    last_fa_layer = None
    for layer in reversed(inner_model.layers):
        if not layer.is_linear:
            last_fa_layer = layer
            break

    if last_fa_layer is None:
        logger.warning("[MTP fixup] No full-attention layer found in main model")
        return

    # Copy expert routing gate if not in checkpoint
    if "layers.0.mlp.gate.weight" not in loaded_keys:
        src = getattr(last_fa_layer.mlp, "gate", None)
        dst = getattr(mtp_layer.mlp, "gate", None)
        if src is not None and dst is not None:
            src_params = mlx.utils.tree_flatten(src.parameters())
            dst.load_weights(src_params)
            mx.eval(dst.parameters())
            logger.info("[MTP fixup] Copied mlp.gate from main model last layer")

    # Copy shared_expert_gate if not in checkpoint
    if "layers.0.mlp.shared_expert_gate.weight" not in loaded_keys:
        src = getattr(last_fa_layer.mlp, "shared_expert_gate", None)
        dst = getattr(mtp_layer.mlp, "shared_expert_gate", None)
        if src is not None and dst is not None:
            src_params = mlx.utils.tree_flatten(src.parameters())
            dst.load_weights(src_params)
            mx.eval(dst.parameters())
            logger.info(
                "[MTP fixup] Copied shared_expert_gate from main model last layer"
            )

    # Zero out k_proj and v_proj → attention becomes no-op
    attn = getattr(mtp_layer, "self_attn", None)
    if attn is None:
        return

    for proj_name in ("k_proj", "v_proj"):
        key = f"layers.0.self_attn.{proj_name}.weight"
        if key not in loaded_keys:
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue
            # For quantized layers: zero scales+biases → dequantized = 0
            if hasattr(proj, "scales"):
                proj.scales = mx.zeros_like(proj.scales)
                proj.biases = mx.zeros_like(proj.biases)
            else:
                proj.weight = mx.zeros_like(proj.weight)
            mx.eval(proj.parameters())
            logger.info(f"[MTP fixup] Zeroed {proj_name} (not in checkpoint)")


def inject_mtp_support(model: Any, model_path, config: dict) -> bool:
    """Inject MTP module into a loaded Qwen3.5 model.

    mlx_lm's qwen3_5.py does not define MTP layers, so we:
    1. Create MTP module matching the weight structure
    2. Quantize it to match the base model
    3. Load MTP weights from model-mtp.safetensors
    4. Monkey-patch Model with return_hidden, mtp_forward, make_mtp_cache

    Args:
        model: A model loaded via mlx_lm (strict=False, MTP weights ignored)
        model_path: Path to model directory (contains model-mtp.safetensors)
        config: Parsed config.json dict

    Returns:
        True if MTP was successfully injected, False otherwise.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Navigate nested config: text_config for VLM wrappers
    text_config = config.get("text_config", config)
    num_mtp_layers = text_config.get("mtp_num_hidden_layers", 0)
    if num_mtp_layers == 0:
        # Fallback: check flat config for num_nextn_predict_layers
        num_mtp_layers = text_config.get(
            "num_nextn_predict_layers",
            config.get("num_nextn_predict_layers", 0),
        )
    if num_mtp_layers == 0:
        logger.info("[MTP inject] No MTP layers configured, skipping")
        return False

    model_path = Path(model_path)
    # Look for MTP weights in mtp/ subdirectory first (avoids mlx_vlm glob),
    # then fall back to model-mtp.safetensors in model dir.
    mtp_file = model_path / "mtp" / "weights.safetensors"
    if not mtp_file.exists():
        mtp_file = model_path / "model-mtp.safetensors"
    if not mtp_file.exists():
        logger.warning(f"[MTP inject] MTP weights not found in {model_path}")
        return False

    # Get model args — navigate VLM wrapper if needed
    # Model hierarchy: Model → language_model (TextModel) → model (Qwen3_5TextModel)
    text_model = model
    if hasattr(model, "language_model"):
        text_model = model.language_model

    args = text_model.args

    # When loaded via mlx_vlm, args may be a TextConfig object missing fields
    # that mlx_lm's TextModelArgs defines (rope_theta, partial_rotary_factor,
    # rope_scaling, etc.). Build a proper TextModelArgs from the config dict.
    from mlx_lm.models.qwen3_5 import TextModelArgs

    if not isinstance(args, TextModelArgs):
        logger.info("[MTP inject] Building TextModelArgs from config dict")
        args = TextModelArgs.from_dict(text_config)

    # Detect MoE vs Dense from args
    num_experts = getattr(args, "num_experts", 0)
    is_moe = num_experts > 0

    # Import model components
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import DecoderLayer

    logger.info(
        f"[MTP inject] Creating MTP module ({num_mtp_layers} layers, "
        f"{'MoE' if is_moe else 'Dense'})"
    )

    # MTP decoder uses full attention (not GatedDeltaNet).
    # layer_idx = full_attention_interval - 1 ensures is_linear=False.
    fa_idx = args.full_attention_interval - 1

    class _MTPModule(nn.Module):
        def __init__(self, args, n_layers):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                DecoderLayer(args, layer_idx=fa_idx) for _ in range(n_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    mtp = _MTPModule(args, num_mtp_layers)

    # --- Load MTP weights in BF16 (no quantization) ---
    # MTP head is extremely sensitive to quantization — even 4-bit destroys
    # prediction quality (0% acceptance).  Keep MTP in full precision.
    # See: https://github.com/vllm-project/vllm/issues/36331
    quant_config = text_config.get("quantization", config.get("quantization", {}))
    bits = quant_config.get("bits", 4) if quant_config else 4
    group_size = quant_config.get("group_size", 64) if quant_config else 64

    logger.info(
        f"[MTP inject] Loading weights from {mtp_file.name} (BF16, no quantization)"
    )
    raw = mx.load(str(mtp_file))
    raw_mtp = {
        k.removeprefix("mtp."): v for k, v in raw.items() if k.startswith("mtp.")
    }
    del raw

    # Dequantize any quantized weight triplets (weight + scales + biases)
    mtp_weights: dict[str, mx.array] = {}
    processed = set()
    for key in sorted(raw_mtp.keys()):
        if key in processed:
            continue
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        scales_key = key.replace(".weight", ".scales")
        biases_key = key.replace(".weight", ".biases")

        if scales_key in raw_mtp and biases_key in raw_mtp:
            # Quantized triplet → dequantize to BF16
            dq = mx.dequantize(
                raw_mtp[key],
                raw_mtp[scales_key],
                raw_mtp[biases_key],
                group_size=group_size,
                bits=bits,
            )
            mtp_weights[key] = dq
            processed.update([key, scales_key, biases_key])
        else:
            # Already FP (norms, fc, shared_expert_gate)
            mtp_weights[key] = raw_mtp[key]
            processed.add(key)
    del raw_mtp

    mtp.load_weights(list(mtp_weights.items()), strict=False)
    mx.eval(mtp.parameters())

    dq_count = sum(1 for k in mtp_weights if not k.endswith((".scales", ".biases")))
    has_quantized = any(k.endswith(".scales") for k in processed)
    mode = "dequantized from quantized" if has_quantized else "native BF16"
    logger.info(f"[MTP inject] Loaded {dq_count} MTP weight tensors ({mode})")

    # --- Step 4: Fix missing MoE MTP weights ---
    # MoE checkpoints lack: k_proj, v_proj, gate, shared_expert_gate, norms.
    # Norms default to identity (weight=1.0) which is correct.
    # k_proj/v_proj: zero out → attention becomes no-op, MLP does prediction.
    # gate/shared_expert_gate: copy from main model's last full-attention layer.
    if is_moe:
        loaded_key_set = set(mtp_weights.keys())
        _fixup_moe_mtp(mtp, text_model.model, loaded_key_set, mx)

    # --- Attach MTP and monkey-patch model class ---
    text_model.mtp = mtp

    original_class = text_model.__class__

    class _Qwen3_5MTP(original_class):
        """Qwen3.5 with MTP support (injected at runtime)."""

        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            **kwargs,
        ):
            inner = self.model
            if input_embeddings is not None:
                hidden_states = input_embeddings
            else:
                hidden_states = inner.embed_tokens(inputs)

            if cache is None:
                cache = [None] * len(inner.layers)

            fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])

            for layer, c in zip(inner.layers, cache):
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=c)

            normed = inner.norm(hidden_states)

            if self.args.tie_word_embeddings:
                out = inner.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)

            if return_hidden:
                return out, normed  # post-norm hidden states (MTP expects post-norm)
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
        ):
            """Run MTP head: predict token n+2 from hidden states + token n+1."""
            input_embeds = self.model.embed_tokens(next_token_ids)
            e = self.mtp.pre_fc_norm_embedding(input_embeds)
            h = self.mtp.pre_fc_norm_hidden(hidden_states)
            x = self.mtp.fc(mx.concatenate([e, h], axis=-1))

            layer = self.mtp.layers[0]
            c = mtp_cache[0] if mtp_cache else None
            mask = create_attention_mask(x, c)
            x = layer(x, mask=mask, cache=c)

            x = self.mtp.norm(x)

            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(x)
            return self.lm_head(x)

        def make_mtp_cache(self):
            """Create KV cache for MTP layers."""
            if self.mtp is None:
                return None
            return [KVCache() for _ in self.mtp.layers]

    text_model.__class__ = _Qwen3_5MTP
    logger.info("[MTP inject] Model class patched with MTP support")

    # If we patched the inner language_model, also expose MTP on the outer Model
    if hasattr(model, "language_model") and model.language_model is text_model:
        model.mtp = mtp

    return True


def validate_mtp_support(model: Any) -> bool:
    """Validate that a loaded model has working MTP support.

    Checks:
    1. model.mtp exists and is not None
    2. model.mtp has layers with loaded weights
    3. model has return_hidden support in __call__
    4. model has mtp_forward method
    5. model has make_mtp_cache method

    Args:
        model: A model loaded via mlx_lm.load()

    Returns:
        True if MTP is fully functional, False otherwise.
    """
    # Navigate to text model if VLM wrapper
    text_model = model
    if hasattr(model, "language_model"):
        text_model = model.language_model

    mtp = getattr(text_model, "mtp", None)
    if mtp is None:
        args = getattr(text_model, "args", None)
        if args is not None:
            num_mtp = getattr(args, "mtp_num_hidden_layers", 0)
            if num_mtp == 0:
                num_mtp = getattr(args, "num_nextn_predict_layers", 0)
            if num_mtp > 0:
                logger.warning(
                    "[MTP] Model config has MTP layers=%d but model.mtp is None. "
                    "Run scripts/add_mtp_weights_qwen35.py to add weights.",
                    num_mtp,
                )
        return False

    mtp_layers = getattr(mtp, "layers", [])
    if not mtp_layers:
        logger.warning("[MTP] model.mtp exists but has no layers.")
        return False

    import inspect

    call_sig = inspect.signature(type(text_model).__call__)
    if "return_hidden" not in call_sig.parameters:
        logger.warning("[MTP] Model.__call__ does not accept return_hidden parameter.")
        return False

    if not hasattr(text_model, "mtp_forward") or not callable(text_model.mtp_forward):
        logger.warning("[MTP] Model does not have mtp_forward() method.")
        return False

    if not hasattr(text_model, "make_mtp_cache") or not callable(
        text_model.make_mtp_cache
    ):
        logger.warning("[MTP] Model does not have make_mtp_cache() method.")
        return False

    logger.info(
        "[MTP] Qwen3.5 model has working MTP support: %d MTP layer(s)",
        len(mtp_layers),
    )
    return True
