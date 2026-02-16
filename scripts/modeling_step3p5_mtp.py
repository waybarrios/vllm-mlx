# Copyright 2025 StepFun AI / MLX Community
# MLX-native implementation of Step3p5 with MTP (Multi-Token Prediction) support.
#
# Based on the HuggingFace PyTorch modeling_step3p5.py, rewritten for mlx_lm.
# MTP architecture: 3 prediction layers (originally layers 45-47) with dense MLP,
# per-layer shared_head, eh_proj fusion, and sliding_attention.

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "step3p5"
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    intermediate_size: int = 11264
    num_attention_heads: int = 64
    num_attention_groups: int = 8  # num_key_value_heads for full attention
    head_dim: int = 128
    vocab_size: int = 128896
    rms_norm_eps: float = 1e-5
    rope_theta: Any = 10000.0  # float or list[float]
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 262144
    max_seq_len: int = 262144
    sliding_window: int = 512
    # MoE config
    use_moe: bool = True
    moe_num_experts: int = 288
    moe_top_k: int = 8
    moe_intermediate_size: int = 1280
    moe_every_n_layer: int = 1
    moe_layer_offset: int = 0
    moe_layers_enum: str = ""
    share_expert_dim: int = 1280
    moe_router_activation: str = "sigmoid"
    moe_router_scaling_factor: float = 3.0
    norm_expert_weight: bool = True
    need_fp32_gate: bool = True
    use_moe_router_bias: bool = True
    # Attention config
    att_impl_type: str = "GQA"
    layer_types: list[str] = field(default_factory=list)
    attention_other_setting: dict[str, Any] | None = None
    use_head_wise_attn_gate: bool = True
    use_qk_norm: bool = True
    use_rope_layers: list[bool] = field(default_factory=list)
    partial_rotary_factors: list[float] | None = None
    yarn_only_types: list[str] = field(default_factory=list)
    # SwiGLU clamp limits
    swiglu_limits: list[float | None] = field(default_factory=list)
    swiglu_limits_shared: list[float | None] = field(default_factory=list)
    # MTP config
    num_nextn_predict_layers: int = 0
    # Zero-centered RMSNorm (MLX community already converted to standard format)
    zero_centered: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Parse moe_layers_enum string to list
        if isinstance(self.moe_layers_enum, str) and self.moe_layers_enum.strip():
            self._moe_layer_indices = set(
                int(x) for x in self.moe_layers_enum.strip().split(",")
            )
        else:
            self._moe_layer_indices = set(range(1, self.num_hidden_layers))


class Step3p5RMSNorm(nn.Module):
    """RMSNorm — MLX community weights already have +1 baked in (standard format)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Step3p5MLP(nn.Module):
    """Standard dense SwiGLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        if self.limit is not None and self.limit > 0:
            gate = mx.clip(gate, a_min=None, a_max=self.limit)
            up = mx.clip(up, a_min=-self.limit, a_max=self.limit)
        return self.down_proj(gate * up)


class Step3p5Router(nn.Module):
    """MoE router with sigmoid gating and optional router bias.
    Weight keys: gate.gate.{weight,scales,biases}, gate.router_bias
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.moe_num_experts
        self.top_k = args.moe_top_k
        self.routed_scaling_factor = args.moe_router_scaling_factor
        self.norm_expert_weight = args.norm_expert_weight
        self.use_moe_router_bias = args.use_moe_router_bias
        self.need_fp32_gate = args.need_fp32_gate

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)

        if self.use_moe_router_bias:
            self.router_bias = mx.zeros(self.num_experts)

    def __call__(self, x: mx.array):
        # Always use self.gate(x) — weight may be quantized (packed format)
        router_logits = self.gate(x)
        if self.need_fp32_gate:
            router_logits = router_logits.astype(mx.float32)

        gate_prob = mx.sigmoid(router_logits.astype(mx.float32))

        if self.use_moe_router_bias:
            gate_prob_biased = gate_prob + self.router_bias
            inds = mx.argpartition(gate_prob_biased, kth=-self.top_k, axis=-1)[
                ..., -self.top_k :
            ]
        else:
            inds = mx.argpartition(gate_prob, kth=-self.top_k, axis=-1)[
                ..., -self.top_k :
            ]

        scores = mx.take_along_axis(gate_prob, inds, axis=-1)
        if self.norm_expert_weight:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

        scores = scores * self.routed_scaling_factor
        return inds, scores


class Step3p5MoEBlock(nn.Module):
    """MoE block matching weight key structure: mlp.gate, mlp.switch_mlp, mlp.share_expert."""

    def __init__(
        self,
        args: ModelArgs,
        swiglu_limit: float | None = None,
        swiglu_limit_shared: float | None = None,
    ):
        super().__init__()
        self.gate = Step3p5Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.moe_num_experts
        )
        self.share_expert = Step3p5MLP(
            args.hidden_size, args.share_expert_dim, swiglu_limit=swiglu_limit_shared
        )
        self.limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        if self.limit is not None and self.limit > 0:
            y = mx.clip(y, a_min=-self.limit, a_max=self.limit)
        y = (y * scores[..., None]).sum(axis=-2)
        y = y + self.share_expert(x)
        return y


class Step3p5Attention(nn.Module):
    """Multi-head attention with optional head-wise gating, QK norm, and sliding window."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = args.head_dim

        # Determine attention type
        if args.layer_types and layer_idx < len(args.layer_types):
            self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        else:
            self.is_sliding = layer_idx % 2 == 0

        # Set head counts based on attention type
        if self.is_sliding and args.attention_other_setting:
            self.num_heads = args.attention_other_setting.get(
                "num_attention_heads", args.num_attention_heads
            )
            self.num_kv_heads = args.attention_other_setting.get(
                "num_attention_groups", args.num_attention_groups
            )
        else:
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_attention_groups

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.q_norm = Step3p5RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = Step3p5RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.use_head_wise_attn_gate = args.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)

        # RoPE — determine partial_rotary_factor and theta for this layer
        partial_rotary_factor = 1.0
        if args.partial_rotary_factors and layer_idx < len(args.partial_rotary_factors):
            partial_rotary_factor = args.partial_rotary_factors[layer_idx]

        rope_theta = args.rope_theta
        if isinstance(args.rope_theta, list):
            rope_theta = (
                args.rope_theta[layer_idx]
                if layer_idx < len(args.rope_theta)
                else 10000.0
            )

        # Determine if this layer uses yarn/rope scaling
        rope_scaling = None
        if args.yarn_only_types and args.layer_types:
            if (
                layer_idx < len(args.layer_types)
                and args.layer_types[layer_idx] in args.yarn_only_types
            ):
                rope_scaling = args.rope_scaling
        elif args.rope_scaling:
            rope_scaling = args.rope_scaling

        rotary_dim = int(self.head_dim * partial_rotary_factor)
        self.rope = initialize_rope(
            rotary_dim,
            base=rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape and apply QK norm
        queries = self.q_norm(
            queries.reshape(B, L, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Head-wise gate
        gate = None
        if self.use_head_wise_attn_gate:
            gate = self.g_proj(x)  # [B, L, num_heads]

        # RoPE
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

        # Apply head-wise gating
        if gate is not None:
            output = (
                output.reshape(B, L, self.num_heads, self.head_dim)
                * mx.sigmoid(gate)[..., None]
            ).reshape(B, L, -1)

        return self.o_proj(output)


class Step3p5DecoderLayer(nn.Module):
    """Single transformer decoder layer with attention + MLP/MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Step3p5Attention(args, layer_idx)

        self.input_layernorm = Step3p5RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = Step3p5RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        # Determine swiglu limits for this layer
        swiglu_limit = None
        if args.swiglu_limits and layer_idx < len(args.swiglu_limits):
            val = args.swiglu_limits[layer_idx]
            if val is not None and val != 0:
                swiglu_limit = val

        swiglu_limit_shared = None
        if args.swiglu_limits_shared and layer_idx < len(args.swiglu_limits_shared):
            val = args.swiglu_limits_shared[layer_idx]
            if val is not None and val != 0:
                swiglu_limit_shared = val

        # MoE or dense MLP — both use self.mlp to match weight key naming
        self.is_moe = layer_idx in args._moe_layer_indices
        if self.is_moe:
            self.mlp = Step3p5MoEBlock(
                args, swiglu_limit=swiglu_limit, swiglu_limit_shared=swiglu_limit_shared
            )
        else:
            self.mlp = Step3p5MLP(
                args.hidden_size,
                args.intermediate_size,
                swiglu_limit=swiglu_limit_shared,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        # Self-attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # FFN
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Step3p5Model(nn.Module):
    """Step 3.5 backbone transformer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Step3p5DecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = Step3p5RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
        return_prenorm: bool = False,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        if return_prenorm:
            return self.norm(hidden_states), hidden_states
        return self.norm(hidden_states)


class Step3p5SharedHead(nn.Module):
    """Per-MTP-layer prediction head: norm + linear output projection."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = Step3p5RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output(self.norm(x))


class Step3p5MTPLayer(nn.Module):
    """Single MTP prediction layer.

    Architecture:
      1. Normalize hidden_states (hnorm) and token embedding (enorm) separately
      2. Concatenate and project: [B, L, 2H] -> [B, L, H] via eh_proj
      3. Standard decoder block: attention + dense MLP (NOT MoE)
      4. Per-layer shared_head for logit prediction
    """

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.hnorm = Step3p5RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.enorm = Step3p5RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)

        # MTP uses sliding_attention type — pick a sliding layer_idx for RoPE config
        # Find a sliding_attention layer index for correct RoPE params
        mtp_layer_idx = 1  # default sliding layer
        if args.layer_types:
            for i, lt in enumerate(args.layer_types):
                if lt == "sliding_attention":
                    mtp_layer_idx = i
                    break

        self.self_attn = Step3p5Attention(args, layer_idx=mtp_layer_idx)
        self.mlp = Step3p5MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = Step3p5RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = Step3p5RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.shared_head = Step3p5SharedHead(args)

    def __call__(
        self,
        hidden_states: mx.array,
        input_embeds: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns:
            (mtp_hidden, logits) — hidden for chaining to next MTP layer, logits for this layer
        """
        h = self.hnorm(hidden_states)
        e = self.enorm(input_embeds)
        x = self.eh_proj(mx.concatenate([e, h], axis=-1))

        # Standard decoder: attention + dense MLP
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache) + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x) + residual

        logits = self.shared_head(x)
        return x, logits


class Step3p5MTP(nn.Module):
    """MTP module with multiple prediction layers."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [
            Step3p5MTPLayer(args, layer_idx=i)
            for i in range(args.num_nextn_predict_layers)
        ]


class Model(nn.Module):
    """Step3p5ForCausalLM — MLX-native with MTP support."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Step3p5Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # MTP head
        self._mtp_num_layers = args.num_nextn_predict_layers
        if self._mtp_num_layers > 0:
            self.mtp = Step3p5MTP(args)
        else:
            self.mtp = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
        return_hidden: bool = False,
    ) -> mx.array:
        if return_hidden:
            hidden_states, prenorm_hidden = self.model(
                inputs, cache, return_prenorm=True
            )
        else:
            hidden_states = self.model(inputs, cache)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden_states)
        else:
            out = self.lm_head(hidden_states)

        if return_hidden:
            return out, prenorm_hidden
        return out

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: Any | None = None,
    ) -> mx.array:
        """Run MTP head to predict token n+2 given hidden states and token n+1.

        Uses only the first MTP layer (vllm-mlx compatible single-draft-token mode).

        Args:
            hidden_states: [B, 1, H] prenorm hidden states from main model
            next_token_ids: [B, 1] token IDs for position n+1
            mtp_cache: list of KVCache for MTP layers

        Returns:
            logits: [B, 1, V] logits for token n+2
        """
        if self.mtp is None:
            raise RuntimeError("MTP head not loaded (num_nextn_predict_layers=0)")

        input_embeds = self.model.embed_tokens(next_token_ids)

        layer = self.mtp.layers[0]
        cache_entry = mtp_cache[0] if mtp_cache else None
        mask = create_attention_mask(input_embeds, cache_entry)
        _, logits = layer(hidden_states, input_embeds, mask=mask, cache=cache_entry)
        return logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def make_mtp_cache(self):
        """Create KV cache for MTP layers."""
        if self.mtp is None:
            return None
        return [KVCache() for _ in self.mtp.layers]

    def sanitize(self, weights):
        """Remap weight keys from stored format to module structure."""
        # Check for MTP weights
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)

        # Filter out original model.layers.{45,46,47} if they somehow survived
        weights = {
            k: v
            for k, v in weights.items()
            if not any(k.startswith(f"model.layers.{i}.") for i in [45, 46, 47])
        }

        if not has_mtp_weights or self._mtp_num_layers == 0:
            weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Stack per-expert weights if stored individually (HF PyTorch format)
        per_expert_key = "model.layers.3.mlp.up_proj.0.weight"
        if per_expert_key in weights:
            for layer_idx in range(self.args.num_hidden_layers):
                if layer_idx not in self.args._moe_layer_indices:
                    continue
                prefix = f"model.layers.{layer_idx}.mlp"
                for proj in ["up_proj", "down_proj", "gate_proj"]:
                    expert_keys = [
                        f"{prefix}.{proj}.{e}.weight"
                        for e in range(self.args.moe_num_experts)
                    ]
                    if all(k in weights for k in expert_keys):
                        stacked = mx.stack([weights.pop(k) for k in expert_keys])
                        weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked

        # Handle MoELinear bulk format (mlp.up_proj.weight [E, out, in] → mlp.switch_mlp)
        for layer_idx in range(self.args.num_hidden_layers):
            if layer_idx not in self.args._moe_layer_indices:
                continue
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ["up_proj", "down_proj", "gate_proj"]:
                hf_key = f"{prefix}.{proj}.weight"
                mlx_key = f"{prefix}.switch_mlp.{proj}.weight"
                if hf_key in weights and mlx_key not in weights:
                    weights[mlx_key] = weights.pop(hf_key)

        # RMSNorm: MLX community already added +1 to zero-centered weights (standard format).
        # add_mtp_weights_step3p5.py also added +1 to MTP norms. No adjustment needed.

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # MoE gate routing — use 8-bit
            if "mlp.gate" in path and path.endswith(".gate"):
                return {"group_size": 64, "bits": 8}
            # MTP norms and projections — keep FP
            if "mtp." in path and any(
                x in path
                for x in [
                    ".enorm.",
                    ".hnorm.",
                    ".shared_head.norm.",
                    ".input_layernorm.",
                    ".post_attention_layernorm.",
                    ".q_norm.",
                    ".k_norm.",
                ]
            ):
                return False
            return True

        return predicate
