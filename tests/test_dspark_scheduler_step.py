# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of the DSpark scheduler hook on a tiny hybrid NemotronH.

Runs the real mlx-lm ``BatchGenerator`` over a randomly initialised
Mamba + attention + MLP NemotronH, once plain and once with
``_install_dspark`` driven by scripted drafters:

* ``oracle``  — proposes exactly what the target will emit (all accepted)
* ``garbage`` — always wrong (every round is a full rollback + re-advance)
* ``half``    — right for two positions, then wrong (partial accept)

In every case the speculative path must reproduce the plain greedy token
sequence. That is the greedy-validity bar; the garbage and half cases
exercise the KV trim + recurrent-state restore path on real caches. The
comparison applies the acceptance bar from docs/guides/speculative-decoding.md:
a divergence is tolerated only at a position where the plain run's own
top-1/top-2 logit gap is below a tie threshold (a wider verify forward may
round such a tie the other way). The seed is chosen so no such position
exists, and generation runs on a CPU stream for determinism.
The scripted drafters also assert that the anchor handed to them is the
token they expect at the context position they track, which pins the
position bookkeeping between the target cache, the aux-state stash and the
drafter context.
"""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.generate import BatchGenerator  # noqa: E402
from mlx_lm.models.nemotron_h import Model, ModelArgs  # noqa: E402

from vllm_mlx.patches.nemotron_dspark import _install_aux_taps  # noqa: E402
from vllm_mlx.scheduler import _install_dspark, _SpecDecodeStatsState  # noqa: E402

PATTERN = ["M", "*", "M", "-", "*", "M"]
AUX_LAYERS = [0, 2, 4]
HIDDEN = 32
VOCAB = 97
PROMPT = [3, 17, 5, 88, 41, 2, 9, 60, 33, 12, 7, 45]
N_TOKENS = 40


TIE_TOL = 1e-3  # plain-run top-1/top-2 gap below which a flip is a tie


def _tiny_model(seed: int = 1) -> Model:
    args = ModelArgs(
        model_type="nemotron_h",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=len(PATTERN),
        max_position_embeddings=1024,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern=PATTERN,
    )
    mx.random.seed(seed)
    model = Model(args)
    mx.eval(model.parameters())
    return model


class _ScriptedDraft:
    """Stands in for DSparkDraft: same surface, scripted proposals."""

    block_size = 8

    def __init__(self, oracle, mode: str):
        self.oracle = list(oracle)
        self.mode = mode
        self.calls = 0
        self.diverged_at = None
        self.reset_context()

    def reset_context(self):
        self._ctx_len = 0

    @property
    def context_length(self) -> int:
        return self._ctx_len

    def append_context(self, aux_cat):
        assert aux_cat.ndim == 2 and aux_cat.shape[1] == len(AUX_LAYERS) * HIDDEN
        self._ctx_len += int(aux_cat.shape[0])

    def draft(self, anchor_id: int, target_lm_head, n_draft: int):
        self.calls += 1
        pos = self._ctx_len
        if pos >= len(self.oracle):
            # The hook drafts one more round after the token that hits
            # max_tokens (it cannot know the batch is about to finish);
            # those drafts are never emitted. Any proposal will do.
            return [(anchor_id + 1) % VOCAB]
        # The anchor should be the oracle token at the position we think the
        # cache is at. Record the first mismatch (the test decides whether it
        # was a tolerated tie or a real fault) and draft dummies from then on.
        if self.diverged_at is None and self.oracle[pos] != anchor_id:
            self.diverged_at = pos
        if self.diverged_at is not None:
            return [(anchor_id + 1) % VOCAB]
        truth = self.oracle[pos + 1 : pos + 1 + n_draft]
        if not truth:
            return [(anchor_id + 1) % VOCAB]
        wrong = [(t + 1) % VOCAB for t in truth]
        if self.mode == "oracle":
            return truth
        if self.mode == "garbage":
            return wrong
        if self.mode == "half":
            return truth[:2] + wrong[2:]
        raise ValueError(self.mode)


def _generate(model, install=None):
    """Run one greedy request; returns (tokens, top1-top2 gap per position)."""
    bg = BatchGenerator(
        model,
        max_tokens=N_TOKENS,
        sampler=lambda x: mx.argmax(x, axis=-1),
        prefill_step_size=5,  # several prompt chunks exercise the tap stash
        stream=mx.new_stream(mx.cpu),  # deterministic, and off the GPU
    )
    if install is not None:
        install(bg)
    bg.insert([list(PROMPT)])
    out, gaps = [], []
    while True:
        responses = bg.next_generated()
        if not responses:
            break
        for r in responses:
            out.append(int(r.token))
            top2 = mx.topk(r.logprobs, 2)  # ascending
            gaps.append(float(top2[1] - top2[0]))
            if r.finish_reason is not None:
                return out, gaps
    return out, gaps


@pytest.fixture(scope="module")
def model_and_plain():
    model = _tiny_model()
    plain, gaps = _generate(model)
    assert len(plain) == N_TOKENS
    # A useful oracle must not be degenerate, and must have no near-ties so
    # the comparison below is exact in practice.
    assert len(set(plain)) > 3
    assert min(gaps) > TIE_TOL
    return model, plain, gaps


def _assert_same_greedy_sequence(spec, plain, gaps, label):
    first = next((i for i, (a, b) in enumerate(zip(spec, plain)) if a != b), None)
    if first is None:
        assert len(spec) == len(plain), label
        return
    assert gaps[first] < TIE_TOL, (
        f"{label}: spec decode diverged from plain greedy at position {first} "
        f"where the plain top-1/top-2 gap was {gaps[first]:.3e} (not a tie)"
    )


@pytest.mark.parametrize("mode", ["oracle", "garbage", "half"])
def test_spec_decode_reproduces_plain_greedy(model_and_plain, mode):
    model, plain, gaps = model_and_plain
    drafter = _ScriptedDraft(PROMPT + plain, mode)

    _install_aux_taps(model, AUX_LAYERS)
    model.dspark = drafter
    model._dspark_pending = []
    model._dspark_collect = True
    stats_state = _SpecDecodeStatsState()
    try:

        def _install(bg):
            assert _install_dspark(
                bg, model=model, num_draft_tokens=4, stats_state=stats_state
            )

        spec, _ = _generate(model, install=_install)
    finally:
        model._dspark_collect = False
        model.dspark = None

    _assert_same_greedy_sequence(spec, plain, gaps, mode)
    assert (
        drafter.diverged_at is None
    ), f"{mode}: drafter anchor mismatch at position {drafter.diverged_at}"

    c = stats_state.counters
    assert c["errors"] == 0
    assert c["context_disables"] == 0
    assert c["rounds"] > 0 and drafter.calls == c["rounds"]
    # The hook drafts one wasted round after the token that hits max_tokens;
    # the scripted drafter answers it with a deliberately wrong proposal, so
    # every mode tolerates exactly one rejected tail round.
    if mode == "oracle":
        assert c["drafted"] - c["accepted"] <= 1
        assert c["rounds"] - c["rounds_full"] <= 1
        # a 4-wide block accepted every round: far fewer rounds than tokens
        assert c["rounds"] * 5 >= N_TOKENS
    elif mode == "garbage":
        assert c["accepted"] == 0
        assert c["rounds_full"] == 0
    else:
        assert 0 < c["accepted"] < c["drafted"]
        # only a round whose oracle tail is <= 2 tokens can be fully right
        assert c["rounds_full"] <= 1


def test_stats_snapshot_shape(model_and_plain):
    model, plain, _ = model_and_plain
    drafter = _ScriptedDraft(PROMPT + plain, "oracle")
    _install_aux_taps(model, AUX_LAYERS)
    model.dspark = drafter
    model._dspark_pending = []
    model._dspark_collect = True
    try:
        holder = {}

        def _install(bg):
            _install_dspark(bg, model=model, num_draft_tokens=7)
            holder["bg"] = bg

        _generate(model, install=_install)
    finally:
        model._dspark_collect = False
        model.dspark = None

    stats = holder["bg"].get_spec_decode_stats()
    assert stats["mode"] == "dspark_block"
    assert stats["num_draft_tokens"] == 7  # block_size 8 -> at most 7
    assert stats["enabled"] is True
    assert 0.0 < stats["acceptance_rate"] <= 1.0
    assert stats["mean_accept_len"] > 1.0


# --- cache invariant under batch changes, errors and extraction -----------------


def _generate_schedule(model, schedule, install=None):
    """Run several requests on one generator.

    ``schedule`` is a list of ``(step_index, prompt)``: each prompt is
    inserted just before that many ``next()`` calls have run. Returns
    ``{uid: (tokens, gaps)}``.
    """
    bg = BatchGenerator(
        model,
        max_tokens=N_TOKENS,
        sampler=lambda x: mx.argmax(x, axis=-1),
        prefill_step_size=5,
        stream=mx.new_stream(mx.cpu),
    )
    if install is not None:
        install(bg)
    pending = sorted(schedule)
    out, finished, step = {}, set(), 0
    while True:
        while pending and pending[0][0] <= step:
            _, prompt = pending.pop(0)
            (uid,) = bg.insert([list(prompt)])
            out[uid] = ([], [])
        if not pending and len(finished) == len(out):
            return out
        _, responses = bg.next()
        step += 1
        assert step < 10_000, "generator never finished"
        for r in responses:
            toks, gaps = out[r.uid]
            toks.append(int(r.token))
            top2 = mx.topk(r.logprobs, 2)
            gaps.append(float(top2[1] - top2[0]))
            if r.finish_reason is not None:
                finished.add(r.uid)


def _arm(model, drafter):
    _install_aux_taps(model, AUX_LAYERS)
    model.dspark = drafter
    model._dspark_pending = []
    model._dspark_collect = True


def _disarm(model):
    model._dspark_collect = False
    model.dspark = None


PROMPT_B = [61, 4, 90, 15, 2, 33, 71, 8, 19, 50, 27, 44]


def test_second_request_joining_mid_round_keeps_both_outputs(model_and_plain):
    """A second request merging into the batch while the first holds
    verified-but-unemitted tokens must not corrupt either output. The hook
    rewinds the cache to the emitted prefix before the merge; both requests
    then decode plainly and must match a plain run of the same schedule.

    Oracle mode only: with a 7-wide fully accepted block a round spans 8
    steps, so the merge (which takes ~3 steps of prefill) lands mid-round.
    With rejected drafts a round is 2 steps and the every-cycle context
    check switches speculation off before the merge can land.
    """
    mode = "oracle"
    model, _, _ = model_and_plain
    for n1 in range(2, 14):
        schedule = [(0, PROMPT), (n1, PROMPT_B)]
        plain = _generate_schedule(model, schedule)
        uid_a, uid_b = sorted(plain)
        drafter = _ScriptedDraft(PROMPT + plain[uid_a][0], mode)
        stats_state = _SpecDecodeStatsState()
        _arm(model, drafter)
        try:
            spec = _generate_schedule(
                model,
                schedule,
                install=lambda bg: _install_dspark(
                    bg, model=model, num_draft_tokens=7, stats_state=stats_state
                ),
            )
        finally:
            _disarm(model)
        for uid in (uid_a, uid_b):
            toks, gaps = plain[uid]
            _assert_same_greedy_sequence(spec[uid][0], toks, gaps, f"{mode} n1={n1}")
        assert drafter.diverged_at is None
        c = stats_state.counters
        assert c["errors"] == 0
        if c["despeculations"] >= 1:
            return  # the merge landed mid-round and was handled
    pytest.fail("no insertion step exercised the de-speculation path")


def test_exception_after_verify_rolls_back_and_falls_back(model_and_plain, monkeypatch):
    """An exception after the verify forward must leave the cache at the
    emitted prefix; the request then finishes on plain decode, still
    matching plain greedy."""
    import vllm_mlx.scheduler as sched

    model, plain, gaps = model_and_plain
    real = sched.longest_accepted_prefix
    calls = {"n": 0}

    def flaky(verified, drafted):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected after the verify forward")
        return real(verified, drafted)

    monkeypatch.setattr(sched, "longest_accepted_prefix", flaky)
    drafter = _ScriptedDraft(PROMPT + plain, "oracle")
    stats_state = _SpecDecodeStatsState()
    holder = {}

    def _install(bg):
        _install_dspark(bg, model=model, num_draft_tokens=4, stats_state=stats_state)
        holder["bg"] = bg

    _arm(model, drafter)
    try:
        spec, _ = _generate(model, install=_install)
    finally:
        _disarm(model)

    _assert_same_greedy_sequence(spec, plain, gaps, "injected failure")
    assert drafter.diverged_at is None
    c = stats_state.counters
    assert c["errors"] == 1
    assert c["rounds"] == 2  # the failing round was the last one attempted
    assert holder["bg"].get_spec_decode_stats()["active"] is False


def test_finished_request_cache_holds_only_emitted_tokens(model_and_plain):
    """The cache handed back at finish must not carry verified-but-unemitted
    tokens (it may be stored as a prefix-cache entry)."""
    model, plain, _ = model_and_plain

    def _run(install=None):
        bg = BatchGenerator(
            model,
            max_tokens=N_TOKENS,
            sampler=lambda x: mx.argmax(x, axis=-1),
            prefill_step_size=5,
            stream=mx.new_stream(mx.cpu),
        )
        if install is not None:
            install(bg)
        bg.insert([list(PROMPT)])
        while True:
            for r in bg.next_generated():
                if r.finish_reason is not None:
                    return r

    def _kv_offset(cache):
        return next(int(c.offset) for c in cache if hasattr(c, "keys"))

    plain_final = _run()
    assert _kv_offset(plain_final.prompt_cache) == len(PROMPT) + N_TOKENS

    drafter = _ScriptedDraft(PROMPT + plain, "oracle")
    _arm(model, drafter)
    try:
        spec_final = _run(
            install=lambda bg: _install_dspark(bg, model=model, num_draft_tokens=7)
        )
    finally:
        _disarm(model)
    assert _kv_offset(spec_final.prompt_cache) == len(PROMPT) + N_TOKENS
    assert list(spec_final.all_tokens) == list(plain_final.all_tokens)
