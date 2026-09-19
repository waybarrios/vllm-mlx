# SPDX-License-Identifier: Apache-2.0
"""
Qwen2 / Qwen2.5 implementation with fused QKV and Gate-Up projections.

Decode fuses QKV and gate/up matrix multiplications. Prefill uses views of the
same weights to avoid larger live activations. SwiGLU is compiled into one
pointwise operation with the same rounding as MLX's reference implementation.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.rope_utils import initialize_rope


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fuse the activation while preserving MLX dtype rounding."""
    return nn.silu(gate) * up


def project(x: mx.array, linear: nn.Module, splits: list[int]) -> list[mx.array]:
    """Fuse decode; use weight views in prefill to avoid larger live activations."""
    if x.shape[-2] == 1:
        return mx.split(linear(x), splits, axis=-1)
    outputs = []
    boundaries = [0, *splits, linear.weight.shape[0]]
    for start, end in zip(boundaries, boundaries[1:]):
        weight = linear.weight[start:end]
        if isinstance(linear, nn.QuantizedLinear):
            biases = linear.get("biases")
            y = mx.quantized_matmul(
                x,
                weight,
                scales=linear.scales[start:end],
                biases=None if biases is None else biases[start:end],
                transpose=True,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
            if "bias" in linear:
                y = y + linear.bias[start:end]
        elif "bias" in linear:
            y = mx.addmm(linear.bias[start:end], x, weight.T)
        else:
            y = x @ weight.T
        outputs.append(y)
    return outputs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    quantization: Optional[dict[str, Any]] = None
    quantization_config: Optional[dict[str, Any]] = None
    quantize_activations: bool = False

    def __post_init__(self):
        # The loader quantizes fused modules using the global configuration.
        # Reject layouts it cannot represent, so the standard loader can handle them.
        if self.quantize_activations or (
            self.quantization_config and not self.quantization
        ):
            raise ValueError(
                "Native Qwen2 requires floating-point or MLX weight quantization"
            )
        unfused = (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj")
        if any(key.endswith(unfused) for key in (self.quantization or {})):
            raise ValueError(
                "Native Qwen2 does not support per-projection quantization overrides"
            )


class FusedQwen2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // self.n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_dim = self.n_heads * head_dim
        self.kv_dim = self.n_kv_heads * head_dim
        self.split_indices = [self.q_dim, self.q_dim + self.kv_dim]

        # Unified QKV projection (1 GEMM instead of 3)
        self.qkv_proj = nn.Linear(dim, self.q_dim + 2 * self.kv_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = project(x, self.qkv_proj, self.split_indices)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class FusedQwen2MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Unified Gate-Up projection (1 GEMM instead of 2)
        self.hidden_dim = hidden_dim
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = project(x, self.gate_up_proj, [self.hidden_dim])
        return self.down_proj(swiglu(gate, up))


class FusedTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = FusedQwen2Attention(args)
        self.mlp = FusedQwen2MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class FusedQwen2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            FusedTransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        # Cache-specific masks also matter for single-token padded batches.
        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class Qwen2Model(nn.Module):
    """
    Qwen2 causal language model with fused QKV and Gate-Up projections.
    Compatible with mlx_lm generation, KVCache, and engine batch generators.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FusedQwen2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        """Expose layers property for cache construction and batch generators."""
        return self.model.layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """
        Transform checkpoint weights into fused QKV and Gate-Up representations.
        Supports both floating-point and quantized (scales/biases) formats.
        """
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for key in list(weights):
            if "self_attn.rotary_emb.inv_freq" in key:
                del weights[key]
        return self._fuse_weights(weights, num_layers=self.args.num_hidden_layers)

    @classmethod
    def _fuse_weights(
        cls, weights: dict[str, mx.array], num_layers: int
    ) -> dict[str, mx.array]:
        """Fuse separate Q, K, V and Gate, Up projections into unified tensors."""
        groups = (
            ("self_attn", "qkv_proj", ("q_proj", "k_proj", "v_proj")),
            ("mlp", "gate_up_proj", ("gate_proj", "up_proj")),
        )
        for i in range(num_layers):
            for module, target, sources in groups:
                prefix = f"model.layers.{i}.{module}"
                for suffix in ("weight", "bias", "scales", "biases"):
                    keys = [f"{prefix}.{name}.{suffix}" for name in sources]
                    if not all(key in weights for key in keys):
                        continue
                    if len({weights[key].dtype for key in keys}) != 1:
                        raise ValueError(f"Cannot fuse different dtypes: {keys}")
                    fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
                    # Release source buffers per tensor instead of retaining a
                    # second model's worth of weights in a lazy concat graph.
                    mx.eval(fused)
                    weights[f"{prefix}.{target}.{suffix}"] = fused

        return weights

    @classmethod
    def sanitize_weights(
        cls, weights: dict[str, mx.array], num_layers: int = 1
    ) -> dict[str, mx.array]:
        """Transform checkpoint weights for testing and offline conversion."""
        return cls._fuse_weights(weights, num_layers=num_layers)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        lazy: bool = False,
        strict: bool = True,
    ) -> "Qwen2Model":
        """Load and initialize a model from a local directory or HF snapshot."""
        from mlx_lm.utils import load_model

        model, _ = load_model(
            Path(model_path),
            lazy=lazy,
            strict=strict,
            get_model_classes=lambda *args, **kwargs: (cls, ModelArgs),
        )
        return model


SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("qwen2", "qwen2.5")
