# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import json
import logging
from pathlib import Path

from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

# Models that require tokenizer fallback
FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def _needs_tokenizer_fallback(model_name: str) -> bool:
    """Check if model needs tokenizer fallback."""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


def _needs_strict_false(model_name: str) -> bool:
    """Check if model needs strict=False loading (VLM models with extra weights).

    VLM models (e.g., Qwen3.5) have vision_tower weights that don't match
    the text-only model class.  Loading with strict=True fails and wastes
    memory by loading all weights (~100 GB) before raising ValueError.
    Detect these models up-front to avoid the double-load penalty.
    """
    from mlx_lm.utils import _download, load_config

    try:
        model_path = _download(model_name)
        config = load_config(model_path)
    except Exception:
        return False
    # VLM models have vision_config or text_config with a separate model_type
    if "vision_config" in config and "text_config" in config:
        return True
    return False


def load_model_with_fallback(model_name: str, tokenizer_config: dict = None):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from mlx_lm import load

    tokenizer_config = tokenizer_config or {}

    # Check if model needs fallback (e.g., Nemotron)
    if _needs_tokenizer_fallback(model_name):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name)

    # VLM models (e.g., Qwen3.5) have extra vision weights that cause
    # strict=True to fail.  Skip the first load attempt to avoid loading
    # ~100 GB of weights twice (which can cause OOM on 256 GB systems).
    if _needs_strict_false(model_name):
        logger.info(
            f"Model {model_name} detected as VLM, loading directly with strict=False"
        )
        return _load_strict_false(model_name, tokenizer_config)

    try:
        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
    except ValueError as e:
        # Fallback for models with non-standard tokenizers
        if "TokenizersBackend" in str(e) or "Tokenizer class" in str(e):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            return _load_with_tokenizer_fallback(model_name)
        # Fallback for models with extra weights (e.g., vision tower, MTP layers).
        # Retry with strict=False to discard extra weights.
        elif "parameters not in model" in str(e):
            logger.warning(
                f"Extra parameters found (e.g., vision tower / MTP weights), "
                f"retrying with strict=False: {e}"
            )
            # Clear traceback references to free memory from the failed first load.
            # Without this, large models (200GB+) cause OOM during retry because
            # the traceback holds references to the first load's weight tensors.
            e.__traceback__ = None
            del e
            import gc

            gc.collect()
            return _load_strict_false(model_name, tokenizer_config)
        else:
            raise

    # After successful load, check if MTP weights exist but were stripped by sanitize()
    _try_inject_mtp_post_load(model, model_name)
    return model, tokenizer


def _load_strict_false(model_name: str, tokenizer_config: dict = None):
    """Load model with strict=False to discard extra weights.

    Handles models with extra parameters that the text-only model class
    doesn't define (e.g., vision tower weights in VLM models like Qwen3.5,
    or MTP layers).  The model's own sanitize() handles key remapping
    (e.g., language_model.* prefix), and strict=False silently drops
    unmatched keys.
    """
    import mlx.core as mx
    from mlx_lm.utils import _download, load_model, load_tokenizer

    model_path = _download(model_name)
    model, config = load_model(model_path, strict=False)

    # Verify weights loaded correctly
    from mlx.utils import tree_flatten

    params = tree_flatten(model.parameters())
    total_params = len(params)
    zero_params = sum(1 for _, v in params if mx.all(v == 0).item())
    logger.info(
        f"[strict=False] Loaded {total_params} parameters, "
        f"{zero_params} all-zero tensors"
    )
    # Spot-check embedding weights
    if hasattr(model, "language_model"):
        emb = model.language_model.model.embed_tokens.weight
        logger.info(
            f"[strict=False] embed_tokens: shape={emb.shape}, "
            f"dtype={emb.dtype}, mean={mx.mean(emb.astype(mx.float32)).item():.4f}"
        )

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id", None),
    )
    _try_inject_mtp(model, model_path, config)
    return model, tokenizer


def _try_inject_mtp(model, model_path, config):
    """Inject MTP support if model has MTP config + weights."""
    # Qwen3-Next: flat num_nextn_predict_layers
    if config.get("num_nextn_predict_layers", 0) > 0:
        # Detect Qwen3.5 vs Qwen3-Next by checking text_config or model_type
        text_config = config.get("text_config", config)
        model_type = text_config.get("model_type", config.get("model_type", ""))
        if "qwen3_5" in model_type:
            from ..patches.qwen3_5_mtp import inject_mtp_support
        else:
            from ..patches.qwen3_next_mtp import inject_mtp_support
        inject_mtp_support(model, model_path, config)
        return

    # Qwen3.5: mtp_num_hidden_layers in text_config
    text_config = config.get("text_config", config)
    num_mtp = text_config.get("mtp_num_hidden_layers", 0)
    if num_mtp > 0:
        from ..patches.qwen3_5_mtp import inject_mtp_support

        inject_mtp_support(model, model_path, config)


def _try_inject_mtp_post_load(model, model_name):
    """Check if MTP weights exist but were stripped by sanitize(), and inject."""
    import json

    from mlx_lm.utils import _download

    model_path = _download(model_name)
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    # Check for MTP in flat config and nested text_config
    text_config = config.get("text_config", {})
    num_mtp = config.get("num_nextn_predict_layers", 0)
    if num_mtp == 0:
        num_mtp = text_config.get("num_nextn_predict_layers", 0)
    if num_mtp == 0:
        num_mtp = text_config.get("mtp_num_hidden_layers", 0)
    # Also check mtp attribute on language_model for VLM wrappers
    check_model = model
    if hasattr(model, "language_model"):
        check_model = model.language_model
    if num_mtp > 0 and getattr(check_model, "mtp", None) is None:
        mtp_file = Path(model_path) / "mtp" / "weights.safetensors"
        if not mtp_file.exists():
            mtp_file = Path(model_path) / "model-mtp.safetensors"
        if mtp_file.exists():
            logger.info(
                f"[MTP] Found MTP config (layers={num_mtp}) and weights, injecting..."
            )
            _try_inject_mtp(model, model_path, config)
        else:
            logger.info(
                f"[MTP] Config has num_nextn_predict_layers={num_mtp} "
                "but MTP weights not found, skipping MTP."
            )


def _load_with_tokenizer_fallback(model_name: str):
    """Load model with fallback tokenizer for non-standard models like Nemotron."""
    from mlx_lm.utils import load_model

    logger.info("Loading with tokenizer fallback...")

    # Get model path - use local path if it exists, otherwise download from Hub
    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    # Load model
    model, _ = load_model(model_path)

    # Try to load tokenizer from tokenizer.json directly
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading tokenizer from tokenizer.json")
        base_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        # Read tokenizer_config.json for special tokens and chat template
        tokenizer_config_path = model_path / "tokenizer_config.json"
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                unk_token = config.get("unk_token", unk_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token="<pad>",
        )

        # Set chat template if available
        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template loaded from tokenizer_config.json")
        elif _needs_tokenizer_fallback(model_name):
            # Use official Nemotron chat template with thinking support
            tokenizer.chat_template = NEMOTRON_CHAT_TEMPLATE
            logger.info("Using official Nemotron chat template with thinking support")
        else:
            # Default simple ChatML format for other models
            tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
            logger.info("Using default ChatML chat template")

        logger.info("Tokenizer loaded via fallback successfully")
        return model, tokenizer
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
