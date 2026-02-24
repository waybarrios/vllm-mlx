# SPDX-License-Identifier: Apache-2.0
"""Simple MTP adapter for models with decoder-only MTP layers.

This adapter works with models where MTP layers are standard decoder blocks
without the enorm/hnorm/eh_proj preprocessing pattern. Examples include
MiMo and Kimi models.

The forward pass simply runs hidden states through additional decoder layers.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..mtp_module import MTPModule, MTPModuleConfig


class SimpleMTPModule(MTPModule):
    """MTP module for models with simple decoder block MTP layers.

    Architecture:
        1. One or more decoder layers (same type as main model)
        2. Optional norm layer before lm_head
        3. Shared lm_head from target model

    Unlike StandardMTPModule, this does NOT have enorm/hnorm/eh_proj.
    Hidden states pass directly through additional decoder layers.
    """

    def __init__(
        self,
        config: MTPModuleConfig,
        model_config: Any,
        decoder_layer_cls: type | None = None,
    ):
        super().__init__(config)

        if decoder_layer_cls is None:
            raise ValueError("decoder_layer_cls must be provided for SimpleMTPModule")

        # Create MTP decoder layers
        self.layers = []
        base_idx = config.num_hidden_layers
        for i in range(config.num_mtp_layers):
            try:
                layer = decoder_layer_cls(model_config, layer_idx=base_idx + i)
            except TypeError:
                try:
                    layer = decoder_layer_cls(model_config)
                except TypeError:
                    layer = decoder_layer_cls(args=model_config)
            self.layers.append(layer)

        # Optional final norm (some models have it, weights will be loaded if present)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Detect cache factory from the model's make_cache pattern
        self._cache_factory = self._detect_cache_factory(model_config)

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
            token_ids: (1, seq_len) — unused in simple mode (no embed concat).
            embed_fn: nn.Embedding — unused in simple mode.
            cache: Optional KV cache.

        Returns:
            new_hidden: (1, seq_len, hidden_size) for lm_head.
        """
        h = hidden_states
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, cache=layer_cache)
        h = self.norm(h)
        return h

    def make_cache(self):
        """Create per-layer KV caches for all MTP decoder layers.

        Returns the correct cache type per layer. Models like
        deepseek_v32/glm_moe_dsa need CacheList(KVCache(), KVCache()),
        while simpler models need a plain KVCache().
        """
        if self._cache_factory is not None:
            return [self._cache_factory() for _ in self.layers]
        try:
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.layers]
        except ImportError:
            return None

    @staticmethod
    def _detect_cache_factory(model_config):
        """Detect the per-layer cache factory from the model's make_cache pattern.

        Imports the model module and checks if it uses CacheList for its cache
        layers. Returns a callable that creates a single layer's cache, or None
        to fall back to KVCache().
        """
        import importlib

        model_type = getattr(model_config, "model_type", None)
        if model_type is None:
            return None

        try:
            mod = importlib.import_module(f"mlx_lm.models.{model_type}")
            model_cls = getattr(mod, "Model", None)
            if model_cls is None or not hasattr(model_cls, "make_cache"):
                return None

            from mlx_lm.models.cache import CacheList, KVCache

            if hasattr(mod, "CacheList") or "CacheList" in dir(mod):
                return lambda: CacheList(KVCache(), KVCache())
        except (ImportError, Exception):
            pass

        return None

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        decoder_layer_cls: type | None = None,
    ) -> "SimpleMTPModule":
        """Create MTP module from a model's config."""
        mtp_config = MTPModuleConfig(
            hidden_size=model_config.hidden_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_mtp_layers=getattr(model_config, "num_nextn_predict_layers", 1),
            rms_norm_eps=model_config.rms_norm_eps,
        )
        return cls(mtp_config, model_config, decoder_layer_cls=decoder_layer_cls)
