# SPDX-License-Identifier: Apache-2.0
"""Standard MTP adapter for models with enorm+hnorm+eh_proj+decoder pattern.

This adapter works with any model that follows the DeepSeek V3 MTP architecture:
  1. enorm: RMSNorm on token embedding
  2. hnorm: RMSNorm on hidden state from target model
  3. eh_proj: Linear(hidden_size * 2 -> hidden_size) on concat(hnorm, enorm)
  4. decoder_layer: Full decoder layer (same architecture as main model)
  5. shared lm_head from target model (passed in at call time)

Supported models: DeepSeek V3, DeepSeek V3.2, GLM-5, GLM-4 MoE Lite,
and any future model that follows this pattern.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..mtp_module import MTPModule, MTPModuleConfig


class StandardMTPModule(MTPModule):
    """Generic MTP module for the enorm+hnorm+eh_proj+decoder pattern.

    Architecture:
        1. enorm: RMSNorm on token embedding
        2. hnorm: RMSNorm on hidden state from target model
        3. eh_proj: Linear(hidden_size * 2 -> hidden_size) on concat(hnorm, enorm)
        4. decoder_layer: Full decoder layer (same architecture as main model)
        5. shared_head_norm: RMSNorm before lm_head (optional, may be absent)
        6. shared lm_head from target model (passed in at call time)
    """

    def __init__(
        self,
        config: MTPModuleConfig,
        model_config: Any,
        decoder_layer_cls: type | None = None,
    ):
        super().__init__(config)
        hidden_size = config.hidden_size
        eps = config.rms_norm_eps

        # MTP-specific layers
        self.enorm = nn.RMSNorm(hidden_size, eps=eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=eps)
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Shared head normalization (applied before lm_head)
        self.shared_head_norm = nn.RMSNorm(hidden_size, eps=eps)

        # Full decoder layer (same architecture as main model layers)
        if decoder_layer_cls is None:
            decoder_layer_cls = self._resolve_decoder_cls(model_config)

        # Create decoder layer - try with layer_idx first, fall back without
        try:
            self.decoder_layer = decoder_layer_cls(
                model_config, layer_idx=config.num_hidden_layers
            )
        except TypeError:
            try:
                self.decoder_layer = decoder_layer_cls(model_config)
            except TypeError:
                self.decoder_layer = decoder_layer_cls(args=model_config)

        # Detect cache factory from the model's make_cache pattern.
        # Models like deepseek_v32/glm_moe_dsa use CacheList(KVCache(), KVCache())
        # per layer instead of a plain KVCache().
        self._cache_factory = self._detect_cache_factory(model_config)

    @staticmethod
    def _detect_cache_factory(model_config: Any):
        """Detect the per-layer cache factory from the model's make_cache pattern.

        Imports the model module and inspects its Model.make_cache() output for
        one layer to determine the correct cache type. Returns a callable that
        creates a single layer's cache, or None to fall back to KVCache().
        """
        import importlib

        model_type = getattr(model_config, "model_type", None)
        if model_type is None:
            return None

        # Map model_type to the mlx-lm module name (same map as _resolve_decoder_cls)
        _MODULE_MAP = {
            "deepseek_v3": "deepseek_v3",
            "deepseek_v32": "deepseek_v32",
            "glm_moe_dsa": "deepseek_v32",
            "glm4_moe_lite": "glm4_moe_lite",
        }
        module_name = _MODULE_MAP.get(model_type, model_type)

        try:
            mod = importlib.import_module(f"mlx_lm.models.{module_name}")
            model_cls = getattr(mod, "Model", None)
            if model_cls is None or not hasattr(model_cls, "make_cache"):
                return None

            # Call make_cache on the class to inspect the result.
            # We can't instantiate the full model, so instead probe one layer's
            # cache by creating a minimal dummy instance.
            # Simpler approach: just check if the module imports CacheList and
            # the make_cache source uses it.
            from mlx_lm.models.cache import CacheList, KVCache

            # Create a temporary instance just to call make_cache.
            # This is too expensive. Instead, use a heuristic based on whether
            # the model module references CacheList.
            if hasattr(mod, "CacheList") or "CacheList" in dir(mod):
                # Module uses CacheList - return factory that creates one
                return lambda: CacheList(KVCache(), KVCache())
        except (ImportError, Exception):
            pass

        return None

    @staticmethod
    def _resolve_decoder_cls(model_config: Any) -> type:
        """Auto-resolve decoder layer class from model config's model_type."""
        import importlib

        model_type = getattr(model_config, "model_type", None)
        if model_type is None:
            raise ValueError(
                "Cannot auto-resolve decoder layer class: model_config has no model_type. "
                "Pass decoder_layer_cls explicitly."
            )

        # Map model_type to mlx-lm module name and class
        # Most models use the model_type as the module name
        _MODULE_MAP = {
            "deepseek_v3": ("deepseek_v3", "DecoderLayer"),
            "deepseek_v32": ("deepseek_v32", "DeepseekV32DecoderLayer"),
            "glm_moe_dsa": ("deepseek_v32", "DeepseekV32DecoderLayer"),
            "glm4_moe_lite": ("glm4_moe_lite", "DecoderLayer"),
        }

        if model_type in _MODULE_MAP:
            module_name, class_name = _MODULE_MAP[model_type]
        else:
            # Generic fallback: try model_type as module name, common class names
            module_name = model_type
            class_name = None

        try:
            mod = importlib.import_module(f"mlx_lm.models.{module_name}")
        except ImportError:
            raise ValueError(
                f"Cannot import mlx_lm.models.{module_name} for model_type='{model_type}'. "
                f"Pass decoder_layer_cls explicitly to StandardMTPModule."
            )

        if class_name:
            return getattr(mod, class_name)

        # Try common decoder layer class names
        for name in ["DecoderLayer", "TransformerBlock", f"{model_type.title()}DecoderLayer"]:
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls

        raise ValueError(
            f"Cannot find decoder layer class in mlx_lm.models.{module_name}. "
            f"Pass decoder_layer_cls explicitly."
        )

    def __call__(
        self,
        hidden_states: mx.array,
        token_ids: mx.array,
        embed_fn: Any,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """Single MTP forward step.

        Args:
            hidden_states: (1, seq_len, hidden_size) from target model.
            token_ids: (1, seq_len) token IDs for the last predicted token.
            embed_fn: nn.Embedding from the target model (shared).
            cache: Optional KV cache tuple for the decoder layer.

        Returns:
            new_hidden: (1, seq_len, hidden_size) from the decoder layer.
            The caller (MTPProposer) applies lm_head to produce logits.
        """
        # 1. Embed the token and normalize
        token_embed = embed_fn(token_ids)
        e_norm = self.enorm(token_embed)

        # 2. Normalize hidden state
        h_norm = self.hnorm(hidden_states)

        # 3. Project concatenation
        combined = self.eh_proj(mx.concatenate([h_norm, e_norm], axis=-1))

        # 4. Process through decoder layer
        new_hidden = self.decoder_layer(combined, cache=cache)

        # 5. Apply shared_head normalization
        new_hidden = self.shared_head_norm(new_hidden)

        return new_hidden

    def make_cache(self):
        """Create KV cache for the single decoder layer.

        Returns the correct cache type for the model. Models like
        deepseek_v32/glm_moe_dsa need CacheList(KVCache(), KVCache()),
        while simpler models need a plain KVCache().
        """
        if self._cache_factory is not None:
            return self._cache_factory()
        try:
            from mlx_lm.models.cache import KVCache
            return KVCache()
        except ImportError:
            return None

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        decoder_layer_cls: type | None = None,
    ) -> "StandardMTPModule":
        """Create MTP module from a model's config (ModelArgs).

        Args:
            model_config: The main model's ModelArgs instance.
            decoder_layer_cls: Optional decoder layer class. If None, auto-resolved.

        Returns:
            Configured StandardMTPModule.
        """
        mtp_config = MTPModuleConfig(
            hidden_size=model_config.hidden_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_mtp_layers=getattr(model_config, "num_nextn_predict_layers", 1),
            rms_norm_eps=model_config.rms_norm_eps,
        )
        return cls(mtp_config, model_config, decoder_layer_cls=decoder_layer_cls)


# Keep backward compatibility alias
DeepSeekV3MTPModule = StandardMTPModule
