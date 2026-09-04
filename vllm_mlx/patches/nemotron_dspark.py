# SPDX-License-Identifier: Apache-2.0
"""
DSpark block drafter for NemotronH (Nemotron 3.5 Lightning) on MLX.

Port of the DSpark/DFlash semantics from vLLM (``qwen3_dspark.py``,
``qwen3_dflash.py`` and the DSpark speculator), restricted to the
single-sequence greedy regime that vllm-mlx's scheduler spec path serves.

How DSpark works (the parts that matter here):

* The draft NEVER runs its own layers over the context. For every token the
  TARGET processes, the target's hidden states after layers
  ``target_layer_ids`` ([1, 5, 19, 29, 41, 51] for Lightning) are
  concatenated (6 x hidden), fused by ``fc`` to hidden, RMS-normed, and
  projected through each draft layer's K/V heads straight into that layer's
  context KV cache. Context cost per token is a few GEMVs, not a 6-layer
  forward. The aux states come from :func:`_install_aux_taps`, which patches
  the target's forward to stash them (prompt chunks included).

* One draft round is ONE parallel forward of a ``(1 + N)``-token query block
  ``[anchor, mask * N]`` (``mask_token_id`` = 990, ``N <= block_size - 1``).
  Masks sit AT the positions they predict (``sample_from_anchor = false``).
  Attention is causal within the block, sees the context KV through a
  1024-token sliding window, and every head carries a learned attention-sink
  logit (an extra softmax column that absorbs mass and is then dropped).

* Sampling is semi-autoregressive: base logits for all N positions come from
  the single parallel forward (through the TARGET's lm_head — the draft has
  none), then a low-rank Markov head walks left to right adding a transition
  bias conditioned on the previously chosen token,
  ``logits_i = base_i + markov_w2(markov_w1[prev])``, greedy argmax.

* Vocab: the draft speaks the real tokenizer vocab (131072). The target's
  248320 is lm_head padding, so the draft<->target id mapping is the
  identity, base logits are the target lm_head's first 131072 rows, and an
  anchor id >= 131072 (added special tokens) is clamped to the mask id — the
  round simply drafts poorly and verification cleans up.

Two weight artifacts are accepted, checked in this order:

1. ``<model>/dspark-native/model.safetensors`` — an MLX-native NVFP4 repack
   of NVIDIA's checkpoint (bit-exact nibbles, ``mode="nvfp4"``, group 16,
   global scales retained). Loaded at native precision.
2. ``<model>/dspark/weights.safetensors`` — BF16 produced by
   ``scripts/prepare_dspark_weights.py``, re-quantized to 8-bit at load.
   DSpark was trained at NVFP4, so 8-bit is finer than its native format;
   measured agreement is identical to path 1.

``<model>/dspark/config.json`` (NVIDIA's config) is always required.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LAYER_IDS = (1, 5, 19, 29, 41, 51)


class _DSparkAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.scale = self.head_dim**-0.5
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv * self.head_dim
        self.q_proj = nn.Linear(h, q_out, bias=False)
        self.k_proj = nn.Linear(h, kv_out, bias=False)
        self.v_proj = nn.Linear(h, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, h, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.attention_sink_bias = mx.zeros((self.n_heads,))
        self.rope_theta = cfg.get("rope_theta") or 10000


class _DSparkMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DSparkLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(
            cfg["hidden_size"], eps=cfg["rms_norm_eps"]
        )
        self.self_attn = _DSparkAttention(cfg)
        self.mlp = _DSparkMLP(cfg)


class _MarkovHead(nn.Module):
    def __init__(self, vocab: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, vocab, bias=False)


class _ScaledModule(nn.Module):
    """Wrap a bias-free module so its output is multiplied by a constant.

    Used to fold NVFP4 global scales (which MLX's ``nvfp4`` mode does not
    model) into the module output: ``(g * W) @ x == g * (W @ x)``.
    """

    def __init__(self, inner, g: float):
        super().__init__()
        self.inner = inner
        self._g = float(g)

    def __call__(self, x):
        return self.inner(x) * self._g


class DSparkDraft(nn.Module):
    """The drafter plus its single-sequence context KV state."""

    def __init__(self, cfg: dict, margin_tau: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        h = cfg["hidden_size"]
        num_aux = int(cfg.get("num_aux_layers") or len(_target_layer_ids(cfg)))
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], h)
        self.fc = nn.Linear(num_aux * h, h, bias=False)
        self.hidden_norm = nn.RMSNorm(h, eps=cfg["rms_norm_eps"])
        self.layers = [_DSparkLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = nn.RMSNorm(h, eps=cfg["rms_norm_eps"])
        rank = cfg.get("markov_rank") or cfg.get("dspark_markov_rank") or 512
        self.markov_head = _MarkovHead(cfg["vocab_size"], int(rank))

        dflash = cfg.get("dflash_config") or {}
        self.mask_token_id = int(
            cfg.get("mask_token_id", dflash.get("mask_token_id", 990))
        )
        self.vocab_size = int(cfg["vocab_size"])
        self.block_size = int(cfg.get("block_size") or 8)
        self.window = int(
            cfg.get("sliding_window") or dflash.get("swa_window_size") or 1024
        )
        # Adaptive block length: stop drafting at the first position whose
        # Markov-adjusted top1-top2 margin falls below tau (None = always
        # draft the full block). Every wasted draft position costs a full
        # routed-expert read in the verify forward, so this trades a little
        # acceptance for shorter, cheaper verifies.
        self.margin_tau = margin_tau
        self._sliced_lm_head = None
        self.reset_context()

    # -- context KV ---------------------------------------------------------

    def reset_context(self) -> None:
        n_l = len(self.layers)
        self._ctx_k: List[Optional[mx.array]] = [None] * n_l  # [n, n_kv, head_dim]
        self._ctx_v: List[Optional[mx.array]] = [None] * n_l
        self._ctx_len = 0  # absolute positions represented so far

    @property
    def context_length(self) -> int:
        return self._ctx_len

    def append_context(self, aux_cat: mx.array) -> None:
        """Append target aux states for positions ``ctx_len .. ctx_len+n-1``.

        ``aux_cat``: ``[n, num_aux_layers * hidden]`` in sequence order.
        """
        n = aux_cat.shape[0]
        ctx = self.hidden_norm(self.fc(aux_cat))  # [n, h]
        pos0 = self._ctx_len
        for li, layer in enumerate(self.layers):
            a = layer.self_attn
            k = a.k_proj(ctx).reshape(n, a.n_kv, a.head_dim)
            k = a.k_norm(k)
            k = mx.fast.rope(
                k.transpose(1, 0, 2)[None],
                a.head_dim,
                traditional=False,
                base=a.rope_theta,
                scale=1.0,
                offset=pos0,
            )[0].transpose(1, 0, 2)
            v = a.v_proj(ctx).reshape(n, a.n_kv, a.head_dim)
            if self._ctx_k[li] is None:
                self._ctx_k[li], self._ctx_v[li] = k, v
            else:
                self._ctx_k[li] = mx.concatenate([self._ctx_k[li], k], axis=0)
                self._ctx_v[li] = mx.concatenate([self._ctx_v[li], v], axis=0)
            # Sliding window: only the trailing window is ever attended to.
            if self._ctx_k[li].shape[0] > self.window:
                self._ctx_k[li] = self._ctx_k[li][-self.window :]
                self._ctx_v[li] = self._ctx_v[li][-self.window :]
        self._ctx_len += n
        # No eager eval: the next draft/verify sync materializes these.

    # -- draft round --------------------------------------------------------

    def draft(self, anchor_id: int, target_lm_head, n_draft: int) -> List[int]:
        """One block-draft round.

        The anchor sits at position ``ctx_len``; the ``n_draft`` masks at the
        following positions ARE the predictions. Returns up to ``n_draft``
        token ids (greedy, Markov-biased); fewer when ``margin_tau`` cuts the
        block short.
        """
        n_draft = max(1, min(int(n_draft), self.block_size - 1))
        cfg_pos = self._ctx_len
        n_q = 1 + n_draft
        aid = anchor_id if anchor_id < self.vocab_size else self.mask_token_id
        ids = mx.array([aid] + [self.mask_token_id] * n_draft)
        x = self.embed_tokens(ids)  # [n_q, h]

        for li, layer in enumerate(self.layers):
            a = layer.self_attn
            hn = layer.input_layernorm(x)
            q = a.q_proj(hn).reshape(n_q, a.n_heads, a.head_dim)
            k = a.k_proj(hn).reshape(n_q, a.n_kv, a.head_dim)
            v = a.v_proj(hn).reshape(n_q, a.n_kv, a.head_dim)
            q = a.q_norm(q)
            k = a.k_norm(k)
            q = mx.fast.rope(
                q.transpose(1, 0, 2)[None],
                a.head_dim,
                traditional=False,
                base=a.rope_theta,
                scale=1.0,
                offset=cfg_pos,
            )[0].transpose(1, 0, 2)
            k = mx.fast.rope(
                k.transpose(1, 0, 2)[None],
                a.head_dim,
                traditional=False,
                base=a.rope_theta,
                scale=1.0,
                offset=cfg_pos,
            )[0].transpose(1, 0, 2)

            ck, cv = self._ctx_k[li], self._ctx_v[li]
            n_ctx = 0 if ck is None else ck.shape[0]
            keys = k if ck is None else mx.concatenate([ck, k], axis=0)
            vals = v if cv is None else mx.concatenate([cv, v], axis=0)

            # [heads, n_q, n_ctx + n_q] with GQA repeat
            rep = a.n_heads // a.n_kv
            keys_h = mx.repeat(keys.transpose(1, 0, 2), rep, axis=0)
            vals_h = mx.repeat(vals.transpose(1, 0, 2), rep, axis=0)
            q_h = q.transpose(1, 0, 2)
            scores = (q_h @ keys_h.transpose(0, 2, 1)) * a.scale

            # Causal within the block + sliding window over absolute
            # positions. Context keys hold positions
            # [ctx_len - n_ctx .. ctx_len - 1]; query i sits at ctx_len + i.
            q_pos = mx.arange(n_q)[:, None] + self._ctx_len
            k_pos = mx.concatenate(
                [
                    mx.arange(n_ctx) + (self._ctx_len - n_ctx),
                    mx.arange(n_q) + self._ctx_len,
                ]
            )[None, :]
            visible = (k_pos <= q_pos) & (k_pos > q_pos - self.window)
            scores = mx.where(visible[None], scores, mx.array(-mx.inf))

            # Attention sink: an extra per-head logit column, dropped after
            # the softmax (it only steals probability mass).
            sink = mx.broadcast_to(
                a.attention_sink_bias[:, None, None], (a.n_heads, n_q, 1)
            )
            probs = mx.softmax(
                mx.concatenate([scores, sink], axis=-1).astype(mx.float32),
                axis=-1,
            )[..., :-1].astype(x.dtype)
            attn = (probs @ vals_h).transpose(1, 0, 2).reshape(n_q, -1)
            x = x + a.o_proj(attn)
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        hidden = self.norm(x)[1:]  # mask positions only
        head = self._sliced_lm_head
        if head is None:
            base = target_lm_head(hidden)[..., : self.vocab_size]
        else:
            base = head(hidden)  # [N, V_draft]

        # Sequential Markov stage over a gathered top-k (vLLM's
        # _sample_sequential_topk): the transition bias only ever needs to
        # break ties among plausible candidates, so restrict each position to
        # its top-k base logits and gather just those markov_w2 rows
        # (k x rank) instead of the full vocab x rank head. Everything stays
        # lazy — ONE eval for the whole block instead of one per draft.
        k_top = 64
        top_idx = mx.argpartition(-base, k_top - 1, axis=-1)[:, :k_top]  # [N, k]
        top_vals = mx.take_along_axis(base, top_idx, axis=-1)  # [N, k]
        w2 = self.markov_head.markov_w2.weight  # [V, r]

        drafts = []
        margins = []
        prev = mx.array([aid])
        for i in range(n_draft):
            me = self.markov_head.markov_w1(prev)[0]  # [r]
            bias_i = (w2[top_idx[i]] * me).sum(axis=-1)  # [k] gathered rows
            cand = top_vals[i] + bias_i
            top2 = mx.topk(cand, 2)  # ascending
            margins.append(top2[1] - top2[0])
            j = mx.argmax(cand, keepdims=True)
            tok = mx.take(top_idx[i], j)  # [1]
            drafts.append(tok)
            prev = tok
        out = mx.concatenate(drafts)
        mg = mx.stack(margins)
        mx.eval(out, mg)
        toks = [int(t) for t in out]
        tau = self.margin_tau
        if tau is not None:
            mgl = [float(v) for v in mg]
            keep = len(toks)
            for i in range(1, len(toks)):  # always keep at least one draft
                if mgl[i] < tau:
                    keep = i
                    break
            toks = toks[:keep]
        return toks


# -- loading ------------------------------------------------------------------


def _target_layer_ids(cfg: dict) -> List[int]:
    dflash = cfg.get("dflash_config") or {}
    ids = dflash.get("target_layer_ids") or cfg.get("target_layer_ids")
    return [int(i) for i in (ids or DEFAULT_TARGET_LAYER_IDS)]


def _load_native_nvfp4(draft: DSparkDraft, native_file: Path) -> int:
    """Load the MLX-native NVFP4 repack at native precision.

    The quantized modules (fc, the MLPs, markov_w2) run at 4-bit
    (``mode="nvfp4"``, group 16) — no precision confound against NVIDIA's
    own checkpoint. The retained global scale folds into the module output
    (bias-free linears). ``markov_w2`` is the exception: it is consumed via
    row gathers, which packed quantized storage would scramble, so it is
    dequantized exactly (f32 math, one uniform global factor) to BF16.
    """
    raw = mx.load(str(native_file))
    quantized = {k[: -len(".global_scale")] for k in raw if k.endswith(".global_scale")}

    def _is_quantized(path, module):
        return path in quantized and "markov_w2" not in path

    nn.quantize(
        draft, group_size=16, bits=4, mode="nvfp4", class_predicate=_is_quantized
    )

    loadable = []
    for key, value in raw.items():
        if key.endswith(".global_scale"):
            continue
        if key.rsplit(".", 1)[0] == "markov_head.markov_w2":
            continue  # handled below
        loadable.append((key, value))
    w2 = (
        mx.dequantize(
            raw["markov_head.markov_w2.weight"],
            scales=raw["markov_head.markov_w2.scales"],
            group_size=16,
            bits=4,
            mode="nvfp4",
        ).astype(mx.float32)
        * raw["markov_head.markov_w2.global_scale"]
    )
    loadable.append(("markov_head.markov_w2.weight", w2.astype(mx.bfloat16)))
    draft.load_weights(loadable, strict=False)

    def _fold(owner, attr, path):
        setattr(
            owner,
            attr,
            _ScaledModule(getattr(owner, attr), raw[path + ".global_scale"]),
        )

    _fold(draft, "fc", "fc")
    for i, layer in enumerate(draft.layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _fold(layer.mlp, proj, f"layers.{i}.mlp.{proj}")
    mx.eval(draft.parameters())
    return len(quantized)


def _load_bf16_quantize_8bit(draft: DSparkDraft, weights_file: Path) -> None:
    """Load the BF16 dequant and quantize the drafter to 8-bit (group 64).

    BF16 is 1.9 GB — a full draft round then reads as many bytes as a TARGET
    forward (~8 ms), destroying the speedup. Unlike an MTP head (trained in
    BF16, collapses when quantized), DSpark was trained at NVFP4, so 8-bit is
    strictly finer than its native format. Norms, the sink biases and
    ``markov_w2`` (row-gathered) stay in floating point.
    """
    weights = list(mx.load(str(weights_file)).items())
    draft.load_weights(weights, strict=False)

    def _is_quantizable(path, module):
        if "markov_w2" in path:
            return False
        if isinstance(module, (nn.Linear, nn.Embedding)):
            return module.weight.shape[-1] % 64 == 0
        return False

    nn.quantize(draft, group_size=64, bits=8, class_predicate=_is_quantizable)
    mx.eval(draft.parameters())


def _slice_lm_head(draft: DSparkDraft, model) -> None:
    """Slice the target's (quantized) lm_head to the draft vocab once.

    Halves the per-round head cost (248320 -> 131072 rows). Quantized storage
    is row-major, so a row slice stays valid. Falls back to the full head.
    """
    try:
        lh = model.lm_head
        if hasattr(lh, "scales"):
            import copy

            sl = copy.copy(lh)
            sl.weight = lh.weight[: draft.vocab_size]
            sl.scales = lh.scales[: draft.vocab_size]
            if hasattr(lh, "biases"):
                sl.biases = lh.biases[: draft.vocab_size]
            mx.eval(sl.weight, sl.scales)
            draft._sliced_lm_head = sl
            logger.info("[DSpark] sliced lm_head to %d rows", draft.vocab_size)
    except Exception:
        logger.exception("[DSpark] lm_head slice failed — using the full head")


def _install_aux_taps(model, aux_layer_ids: Sequence[int]) -> None:
    """Patch the NemotronH model class so every forward can stash aux states.

    While ``model._dspark_collect`` is true, each forward appends the
    concatenated hidden states after ``aux_layer_ids`` to
    ``model._dspark_pending`` as one ``[n, num_aux * hidden]`` chunk — prompt
    chunks and decode steps alike, in sequence order. Only single-sequence
    forwards are stashed; a multi-sequence forward clears the stash, which
    makes the scheduler's context check fail closed for that request.

    The forward mirrors ``mlx_lm.models.nemotron_h`` exactly (masks, cache
    indexing, block dispatch); with the flag off it defers to the original.
    """
    model._dspark_aux_layers = [int(i) for i in aux_layer_ids]
    if getattr(model, "_dspark_taps_installed", False):
        return

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    base_cls = model.__class__

    class _NemotronHWithDSparkTaps(base_cls):  # type: ignore[misc,valid-type]
        def __call__(self, inputs, cache=None, **kwargs):
            if not getattr(self, "_dspark_collect", False) or kwargs:
                return super().__call__(inputs, cache=cache, **kwargs)
            bb = self.backbone
            hidden_states = bb.embeddings(inputs)
            if cache is None:
                cache = [None] * len(bb.layers)
            attn_mask = create_attention_mask(hidden_states, cache[bb.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[bb.ssm_idx])
            aux_ids = set(self._dspark_aux_layers)
            aux = []
            cache_counter = 0
            for li, layer in enumerate(bb.layers):
                if layer.block_type == "M" or layer.block_type == "*":
                    c = cache[cache_counter]
                    cache_counter += 1
                else:
                    c = None
                mask = attn_mask if layer.block_type == "*" else ssm_mask
                hidden_states = layer(hidden_states, mask=mask, cache=c)
                if li in aux_ids:
                    aux.append(hidden_states)
            if aux and inputs.shape[0] == 1:
                self._dspark_pending.append(mx.concatenate(aux, axis=-1)[0])
            elif aux:
                self._dspark_pending.clear()
            return self.lm_head(bb.norm_f(hidden_states))

    _NemotronHWithDSparkTaps.__name__ = base_cls.__name__ + "WithDSparkTaps"
    _NemotronHWithDSparkTaps.__qualname__ = _NemotronHWithDSparkTaps.__name__
    model.__class__ = _NemotronHWithDSparkTaps
    model._dspark_pending = []
    model._dspark_collect = False
    model._dspark_taps_installed = True


def inject_dspark_support(
    model: Any,
    model_path,
    target_config: dict,
    margin_tau: Optional[float] = None,
) -> bool:
    """Attach a :class:`DSparkDraft` to a NemotronH model as ``model.dspark``.

    Requires ``<model_path>/dspark/config.json`` plus either the native
    NVFP4 repack or the BF16 dequant (see the module docstring). Installs
    the aux-state taps on the target and turns collection on. The
    scheduler's spec step drives everything else. Returns True on success;
    a False return leaves the model untouched and generation unaffected.
    """
    model_path = Path(model_path)
    cfile = model_path / "dspark" / "config.json"
    wfile = model_path / "dspark" / "weights.safetensors"
    native = model_path / "dspark-native" / "model.safetensors"
    if not cfile.exists() or not (wfile.exists() or native.exists()):
        logger.info(
            "[DSpark] no drafter at %s (need dspark/config.json plus "
            "dspark-native/model.safetensors or dspark/weights.safetensors)",
            model_path,
        )
        return False
    if target_config.get("model_type") != "nemotron_h":
        logger.warning(
            "[DSpark] target model_type %r is not nemotron_h; not installing",
            target_config.get("model_type"),
        )
        return False

    cfg = json.loads(cfile.read_text())
    draft = DSparkDraft(cfg, margin_tau=margin_tau)

    if native.exists():
        n_q = _load_native_nvfp4(draft, native)
        logger.info("[DSpark] native NVFP4 weights loaded (%d quantized modules)", n_q)
    else:
        _load_bf16_quantize_8bit(draft, wfile)
        logger.info("[DSpark] BF16 weights loaded and quantized to 8-bit")

    # fc must have landed: all-zeros means the key mapping failed.
    fc_weight = (
        draft.fc.inner.weight
        if isinstance(draft.fc, _ScaledModule)
        else draft.fc.weight
    )
    if float(mx.abs(fc_weight.astype(mx.float32)).sum()) == 0.0:
        logger.error(
            "[DSpark] fc.weight is all zeros — key mapping failed; not installing"
        )
        return False

    _slice_lm_head(draft, model)
    _install_aux_taps(model, _target_layer_ids(cfg))
    model.dspark = draft
    model._dspark_pending = []
    model._dspark_collect = True
    logger.info(
        "[DSpark] drafter ready: %d layers, vocab %d, block %d, aux layers %s, margin_tau=%s",
        len(draft.layers),
        draft.vocab_size,
        draft.block_size,
        model._dspark_aux_layers,
        margin_tau,
    )
    return True
