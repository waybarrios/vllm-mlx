# SPDX-License-Identifier: Apache-2.0
"""
MLX forward pass for BERT-family sequence classification models.

Implements a from-weights forward pass for cross-encoder rerankers
that use the standard BERT/XLM-RoBERTa architecture with a
classification head. This avoids pulling in the full transformers
modeling stack at inference time — only the tokenizer is needed
from transformers.
"""

import mlx.core as mx
import mlx.nn as nn


def classifier_forward(
    input_ids: mx.array,
    attention_mask: mx.array,
    weights: dict[str, mx.array],
    config: dict,
) -> mx.array:
    """
    Run a BERT-family classifier forward pass on MLX.

    Args:
        input_ids: (batch, seq_len) token IDs.
        attention_mask: (batch, seq_len) attention mask (1=attend, 0=pad).
        weights: Dict mapping weight name -> mx.array.
        config: Model config dict (from config.json).

    Returns:
        logits: (batch, num_labels) classification logits.
    """
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    num_labels = config.get("num_labels", 1)
    eps = config.get("layer_norm_eps", 1e-12)

    head_dim = hidden_size // num_heads

    # Detect weight prefix (bert.* vs roberta.* vs xlm-roberta.*)
    prefix = _detect_prefix(weights)

    # --- Embeddings ---
    word_emb = weights[f"{prefix}.embeddings.word_embeddings.weight"]
    pos_emb = weights[f"{prefix}.embeddings.position_embeddings.weight"]
    tok_type_emb = weights[f"{prefix}.embeddings.token_type_embeddings.weight"]
    ln_w = weights[f"{prefix}.embeddings.LayerNorm.weight"]
    ln_b = weights[f"{prefix}.embeddings.LayerNorm.bias"]

    batch_size, seq_len = input_ids.shape
    position_ids = mx.arange(seq_len)[None, :]  # (1, seq_len)
    token_type_ids = mx.zeros_like(input_ids)

    hidden = word_emb[input_ids] + pos_emb[position_ids] + tok_type_emb[token_type_ids]
    hidden = _layer_norm(hidden, ln_w, ln_b, eps)

    # --- Encoder layers ---
    # Build causal-free attention mask: (batch, 1, 1, seq_len)
    if attention_mask is not None:
        ext_mask = attention_mask[:, None, None, :].astype(mx.float32)
        ext_mask = (1.0 - ext_mask) * -1e9
    else:
        ext_mask = None

    for i in range(num_layers):
        lp = f"{prefix}.encoder.layer.{i}"
        hidden = _encoder_layer(
            hidden, ext_mask, weights, lp, num_heads, head_dim, eps, config
        )

    # --- Pooler (CLS token) ---
    cls_hidden = hidden[:, 0, :]  # (batch, hidden_size)
    pooler_w = weights.get(f"{prefix}.pooler.dense.weight")
    pooler_b = weights.get(f"{prefix}.pooler.dense.bias")
    if pooler_w is not None:
        pooled = mx.tanh(cls_hidden @ pooler_w.T + pooler_b)
    else:
        pooled = cls_hidden

    # --- Classifier head ---
    clf_w = weights["classifier.weight"]
    clf_b = weights["classifier.bias"]
    logits = pooled @ clf_w.T + clf_b  # (batch, num_labels)

    return logits


def _detect_prefix(weights: dict) -> str:
    """Detect the model weight prefix (bert, roberta, xlm-roberta)."""
    for key in weights:
        if key.startswith("bert."):
            return "bert"
        if key.startswith("roberta."):
            return "roberta"
        if key.startswith("xlm-roberta."):
            return "xlm-roberta"
    # Default to bert
    return "bert"


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    """Apply layer normalization."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / mx.sqrt(var + eps) + bias


def _encoder_layer(
    hidden: mx.array,
    ext_mask: mx.array | None,
    weights: dict,
    prefix: str,
    num_heads: int,
    head_dim: int,
    eps: float,
    config: dict,
) -> mx.array:
    """Run one BERT encoder layer (self-attention + FFN)."""
    hidden_size = num_heads * head_dim

    # --- Self-attention ---
    q_w = weights[f"{prefix}.attention.self.query.weight"]
    q_b = weights[f"{prefix}.attention.self.query.bias"]
    k_w = weights[f"{prefix}.attention.self.key.weight"]
    k_b = weights[f"{prefix}.attention.self.key.bias"]
    v_w = weights[f"{prefix}.attention.self.value.weight"]
    v_b = weights[f"{prefix}.attention.self.value.bias"]

    batch_size, seq_len, _ = hidden.shape

    q = (
        (hidden @ q_w.T + q_b)
        .reshape(batch_size, seq_len, num_heads, head_dim)
        .transpose(0, 2, 1, 3)
    )
    k = (
        (hidden @ k_w.T + k_b)
        .reshape(batch_size, seq_len, num_heads, head_dim)
        .transpose(0, 2, 1, 3)
    )
    v = (
        (hidden @ v_w.T + v_b)
        .reshape(batch_size, seq_len, num_heads, head_dim)
        .transpose(0, 2, 1, 3)
    )

    scale = head_dim**-0.5
    attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (batch, heads, seq, seq)

    if ext_mask is not None:
        attn_scores = attn_scores + ext_mask

    attn_probs = mx.softmax(attn_scores, axis=-1)
    attn_out = (
        (attn_probs @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
    )

    # Attention output projection + residual + LayerNorm
    ao_w = weights[f"{prefix}.attention.output.dense.weight"]
    ao_b = weights[f"{prefix}.attention.output.dense.bias"]
    ao_ln_w = weights[f"{prefix}.attention.output.LayerNorm.weight"]
    ao_ln_b = weights[f"{prefix}.attention.output.LayerNorm.bias"]

    attn_out = attn_out @ ao_w.T + ao_b
    hidden = _layer_norm(hidden + attn_out, ao_ln_w, ao_ln_b, eps)

    # --- FFN ---
    inter_w = weights[f"{prefix}.intermediate.dense.weight"]
    inter_b = weights[f"{prefix}.intermediate.dense.bias"]
    out_w = weights[f"{prefix}.output.dense.weight"]
    out_b = weights[f"{prefix}.output.dense.bias"]
    out_ln_w = weights[f"{prefix}.output.LayerNorm.weight"]
    out_ln_b = weights[f"{prefix}.output.LayerNorm.bias"]

    intermediate = hidden @ inter_w.T + inter_b
    intermediate = _gelu(intermediate)
    ffn_out = intermediate @ out_w.T + out_b
    hidden = _layer_norm(hidden + ffn_out, out_ln_w, out_ln_b, eps)

    return hidden


def _gelu(x: mx.array) -> mx.array:
    """GELU activation (exact form)."""
    return nn.gelu(x)
