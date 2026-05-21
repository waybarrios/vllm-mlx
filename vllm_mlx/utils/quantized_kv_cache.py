# SPDX-License-Identifier: Apache-2.0
"""
BatchQuantizedKVCache: live inference KV cache quantization with asymmetric bits.

Quantizes KV cache during live inference — each new K/V token is quantized
immediately after computation and stored compressed. Attention operates on
compressed data via mx.quantized_matmul. Supports asymmetric bits (e.g. K=8,
V=4) based on research showing keys need higher precision than values
(AsymKV, KIVI, TurboQuant/ICLR 2026).

VRAM savings (K=8, V=4 vs FP16):
  K: 8/16 = 50%  -> 50% savings
  V: 4/16 = 25%  -> 75% savings
  Total: ~62.5% KV cache memory savings
"""

import importlib
import logging
from typing import List, Optional

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from mlx_lm.models.cache import (
    KVCache,
    _BaseCache,
    create_causal_mask,
    dynamic_roll,
)

logger = logging.getLogger(__name__)


class BatchQuantizedKVCache(_BaseCache):
    """Batch-aware quantized KV cache with asymmetric K/V bit widths.

    Keys and values are stored as quantized 3-tuples (data, scales, biases)
    using mx.quantize(). Attention is performed via mx.quantized_matmul().

    Compatible with BatchKVCache API (update_and_fetch, prepare, finalize,
    filter, extend, extract, merge, make_mask, trim, empty, size, nbytes).
    """

    step = 256

    def __init__(
        self,
        left_padding: List[int],
        k_bits: int = 8,
        v_bits: int = 8,
        group_size: int = 64,
        use_hadamard: bool = False,
    ):
        self.keys = None  # None or (data_uint32, scales, biases) tuple
        self.values = None  # None or (data_uint32, scales, biases) tuple
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.use_hadamard = use_hadamard
        self._right_padding = None

    def _init_quantized_buffer(self, B, n_kv_heads, n_steps, head_dim, bits):
        """Initialize zero-filled quantized buffer for keys or values."""
        el_per_int = 8 * mx.uint32.size // bits
        shape = (B, n_kv_heads, n_steps)
        return (
            mx.zeros((*shape, head_dim // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, head_dim // self.group_size), dtype=mx.float16),
            mx.zeros((*shape, head_dim // self.group_size), dtype=mx.float16),
        )

    def _expand_quantized_buffer(self, buf, n_steps):
        """Expand existing quantized buffer by n_steps along sequence dim."""
        B, n_kv_heads = buf[0].shape[:2]

        def expand(x):
            new_x = mx.zeros((B, n_kv_heads, n_steps, x.shape[-1]), dtype=x.dtype)
            return mx.concatenate([x, new_x], axis=2)

        return tree_map(expand, buf)

    def update_and_fetch(self, keys, values):
        """Quantize new K/V and store in pre-allocated buffers.

        Args:
            keys: (B, n_kv_heads, num_steps, k_head_dim) float array
            values: (B, n_kv_heads, num_steps, v_head_dim) float array

        Returns:
            Tuple of quantized keys and values as 3-tuples, sliced to current offset.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tree_map(lambda x: x[..., :prev, :], self.keys)
                    self.values = tree_map(lambda x: x[..., :prev, :], self.values)
                self.keys = self._expand_quantized_buffer(self.keys, new_steps)
                self.values = self._expand_quantized_buffer(self.values, new_steps)
            else:
                self.keys = self._init_quantized_buffer(
                    B, n_kv_heads, new_steps, k_head_dim, self.k_bits
                )
                self.values = self._init_quantized_buffer(
                    B, n_kv_heads, new_steps, v_head_dim, self.v_bits
                )

        self.offset += num_steps
        self._idx += num_steps

        if self.use_hadamard:
            keys = mx.hadamard_transform(keys)
            values = mx.hadamard_transform(values)

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.k_bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.v_bits)
        for i in range(3):
            self.keys[i][..., prev : self._idx, :] = q_keys[i]
            self.values[i][..., prev : self._idx, :] = q_values[i]

        return (
            tree_map(lambda x: x[..., : self._idx, :], self.keys),
            tree_map(lambda x: x[..., : self._idx, :], self.values),
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty "
                    "BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.keys
            )
            self.values = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2),
                self.values,
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._idx < k[0].shape[2]:
            k = tree_map(lambda x: x[..., : self._idx, :], k)
            v = tree_map(lambda x: x[..., : self._idx, :], v)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys[0].shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """In-place filter to keep just the given indices in the cache."""
        if self.keys is not None:
            self.keys = tree_map(lambda x: x[batch_indices], self.keys)
            self.values = tree_map(lambda x: x[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = tree_map(lambda x: x[..., min_left_pad:, :], self.keys)
                self.values = tree_map(lambda x: x[..., min_left_pad:, :], self.values)
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other: "BatchQuantizedKVCache"):
        """In-place extend this cache with another BatchQuantizedKVCache."""
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)

        def _get_shape(cache):
            if cache.keys is not None:
                return cache.keys[0].shape[2]
            return 0

        L1 = _get_shape(self)
        L2 = _get_shape(other)
        max_size = max(L1, L2)

        def pad(c):
            if c.keys is None:
                # Need to get shapes from the other cache
                ref = self if c is other else other
                B = c.left_padding.shape[0]
                left = max_idx
                right = max_size - left

                def make_zero(x):
                    shape = list(x.shape)
                    shape[0] = B
                    shape[2] = max_size
                    return mx.zeros(shape, dtype=x.dtype)

                k = tree_map(make_zero, ref.keys)
                v = tree_map(make_zero, ref.values)
                left_padding = c.left_padding + left
                return k, v, c.offset, left_padding

            left = max_idx - c._idx
            right = max_size - c.keys[0].shape[2] - left

            def pad_buf(buf):
                def pad_elem(x):
                    p = x
                    if right < 0:
                        p = x[..., :right, :]
                    if left != 0 or max(right, 0) != 0:
                        pads = [
                            (0, 0),
                            (0, 0),
                            (left, max(right, 0)),
                            (0, 0),
                        ]
                        p = mx.pad(p, pads)
                    return p

                return tree_map(pad_elem, buf)

            k = pad_buf(c.keys)
            v = pad_buf(c.values)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self_padded = pad(self)
        other_padded = pad(other)

        self.keys = tree_map(
            lambda a, b: mx.concatenate([a, b], axis=0),
            self_padded[0],
            other_padded[0],
        )
        self.values = tree_map(
            lambda a, b: mx.concatenate([a, b], axis=0),
            self_padded[1],
            other_padded[1],
        )
        self.offset = mx.concatenate([self_padded[2], other_padded[2]])
        self.left_padding = mx.concatenate([self_padded[3], other_padded[3]])
        self._idx = max_idx

    def extract(self, idx: int) -> KVCache:
        """Extract a single sequence, dequantize, and return as KVCache."""
        cache = KVCache()
        padding = self.left_padding[idx].item()
        cache.keys = mx.dequantize(
            *tree_map(
                lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]),
                self.keys,
            ),
            group_size=self.group_size,
            bits=self.k_bits,
        )
        cache.values = mx.dequantize(
            *tree_map(
                lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]),
                self.values,
            ),
            group_size=self.group_size,
            bits=self.v_bits,
        )
        if self.use_hadamard:
            cache.keys = mx.hadamard_transform(cache.keys)
            cache.values = mx.hadamard_transform(cache.values)
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches: List[KVCache], **kwargs) -> "BatchQuantizedKVCache":
        """Merge multiple KVCache objects into a BatchQuantizedKVCache.

        Dequantized KVCaches are quantized during merge.
        """
        k_bits = kwargs.get("k_bits", 8)
        v_bits = kwargs.get("v_bits", 8)
        group_size = kwargs.get("group_size", 64)

        lengths = [
            c.size() if hasattr(c, "size") else (c.offset if c.keys is not None else 0)
            for c in caches
        ]
        max_length = max(lengths) if lengths else 0

        if max_length == 0:
            return cls(
                [0] * len(caches), k_bits=k_bits, v_bits=v_bits, group_size=group_size
            )

        padding = [max_length - length for length in lengths]
        B = len(caches)

        # Get shapes from first non-empty cache
        H = Dk = Dv = 0
        dt = mx.float16
        for c in caches:
            if c.keys is not None:
                H = c.keys.shape[1]
                Dk = c.keys.shape[3]
                Dv = c.values.shape[3]
                dt = c.keys.dtype
                break

        # Assemble FP keys/values with left-padding, then quantize
        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            off = c.offset if hasattr(c, "offset") else c.keys.shape[2]
            keys[i : i + 1, :, p : p + off] = c.keys[..., :off, :]
            values[i : i + 1, :, p : p + off] = c.values[..., :off, :]

        use_hadamard = kwargs.get("use_hadamard", False)
        cache = cls(
            padding,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            use_hadamard=use_hadamard,
        )
        if use_hadamard:
            keys = mx.hadamard_transform(keys)
            values = mx.hadamard_transform(values)
        cache.keys = mx.quantize(keys, group_size=group_size, bits=k_bits)
        cache.values = mx.quantize(values, group_size=group_size, bits=v_bits)
        cache.offset += max_length
        cache._idx = max_length

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


def asymmetric_quantized_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    k_bits: int = 8,
    v_bits: int = 8,
    use_hadamard: bool = False,
) -> mx.array:
    """Scaled dot-product attention with asymmetric quantized K/V.

    Like quantized_scaled_dot_product_attention from mlx_lm base.py,
    but uses separate k_bits/v_bits for K and V matmuls.

    When ``use_hadamard`` is True, queries are Hadamard-rotated before the
    K-matmul (so Q_rot·K_rot^T = Q·K^T) and the output is inverse-rotated
    after the V-matmul (Hadamard is self-inverse with default 1/sqrt(d) scale).
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    if use_hadamard:
        queries = mx.hadamard_transform(queries)

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=k_bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores,
        *q_values,
        transpose=False,
        group_size=group_size,
        bits=v_bits,
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    if use_hadamard:
        out = mx.hadamard_transform(out)

    return out


def install_quantized_kv_cache(
    k_bits: int = 8,
    v_bits: int = 8,
    group_size: int = 64,
    use_hadamard: bool = False,
):
    """Monkey-patch mlx_lm to use BatchQuantizedKVCache for live inference.

    Patches:
      1. _make_cache in mlx_lm.generate: KVCache -> BatchQuantizedKVCache
      2. _merge_caches in mlx_lm.generate: merge with quantization
      3. scaled_dot_product_attention in mlx_lm.models.base: asymmetric SDPA routing

    When ``use_hadamard`` is True, Hadamard rotation (QuaRot) is applied before
    quantization and undone during attention, dramatically improving quantization
    quality by spreading outlier channels uniformly.
    """
    gen_module = importlib.import_module("mlx_lm.generate")
    base_module = importlib.import_module("mlx_lm.models.base")

    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        KVCache as _KVCache,
        RotatingKVCache,
    )
    from mlx_lm.generate import BatchRotatingKVCache

    # --- Patch 1: _make_cache ---
    _original_make_cache = gen_module._make_cache

    def _patched_make_cache(model, left_padding, max_kv_size=None):
        def to_batch_cache(c):
            if type(c) is _KVCache:
                return BatchQuantizedKVCache(
                    left_padding,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    group_size=group_size,
                    use_hadamard=use_hadamard,
                )
            elif isinstance(c, ArraysCache):
                c.left_padding = mx.array(left_padding)
                return c
            elif isinstance(c, RotatingKVCache):
                if c.keep > 0:
                    raise ValueError(
                        "RotatingKVCache with keep tokens is not supported."
                    )
                logger.warning(
                    "RotatingKVCache layer converted to unquantized "
                    "BatchRotatingKVCache (quantization not supported for "
                    "bounded cache). Consider a future BatchQuantized"
                    "RotatingKVCache for full coverage."
                )
                return BatchRotatingKVCache(c.max_size, left_padding)
            elif isinstance(c, CacheList):
                return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
            else:
                raise ValueError(f"{type(c)} does not yet support batching")

        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            return [to_batch_cache(c) for c in cache]
        elif max_kv_size is not None:
            logger.warning(
                "max_kv_size=%d is set but model has no make_cache() method. "
                "All layers will use unquantized BatchRotatingKVCache — "
                "KV cache quantization (K=%d, V=%d) will NOT be applied. "
                "Consider removing --max-kv-size to enable quantization.",
                max_kv_size,
                k_bits,
                v_bits,
            )
            return [
                BatchRotatingKVCache(max_kv_size, left_padding) for _ in model.layers
            ]
        else:
            return [
                BatchQuantizedKVCache(
                    left_padding,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    group_size=group_size,
                    use_hadamard=use_hadamard,
                )
                for _ in model.layers
            ]

    gen_module._make_cache = _patched_make_cache

    # --- Patch 2: _merge_caches ---
    _original_merge_caches = gen_module._merge_caches

    def _patched_merge_caches(caches):
        batch_cache = []
        if not caches:
            return batch_cache
        for i in range(len(caches[0])):
            layer = caches[0][i]
            if isinstance(layer, _KVCache):
                batch_cache.append(
                    BatchQuantizedKVCache.merge(
                        [c[i] for c in caches],
                        k_bits=k_bits,
                        v_bits=v_bits,
                        group_size=group_size,
                        use_hadamard=use_hadamard,
                    )
                )
            elif hasattr(layer, "merge"):
                batch_cache.append(layer.merge([c[i] for c in caches]))
            else:
                raise ValueError(
                    f"{type(layer)} does not yet support batching with history"
                )
        return batch_cache

    gen_module._merge_caches = _patched_merge_caches

    # --- Patch 3: scaled_dot_product_attention ---
    _original_sdpa = base_module.scaled_dot_product_attention
    _original_quantized_sdpa = base_module.quantized_scaled_dot_product_attention

    def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        if hasattr(cache, "k_bits"):
            # Asymmetric quantized KV cache (BatchQuantizedKVCache)
            if sinks is not None:
                raise ValueError("Quantized SDPA does not support attention sinks.")
            return asymmetric_quantized_sdpa(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                group_size=cache.group_size,
                k_bits=cache.k_bits,
                v_bits=cache.v_bits,
                use_hadamard=getattr(cache, "use_hadamard", False),
            )
        elif hasattr(cache, "bits"):
            # Symmetric quantized cache (QuantizedKVCache)
            if sinks is not None:
                raise ValueError("Quantized SDPA does not support attention sinks.")
            return _original_quantized_sdpa(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                sinks=sinks,
            )

    base_module.scaled_dot_product_attention = _patched_sdpa

    hadamard_tag = ", QuaRot=ON" if use_hadamard else ""
    logger.info(
        f"Installed BatchQuantizedKVCache: K={k_bits}-bit, V={v_bits}-bit, "
        f"group_size={group_size}{hadamard_tag}"
    )
