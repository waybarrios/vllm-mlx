# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DSpark block speculative decoding (--spec-draft dspark).

The first group needs no MLX and runs on every CI host: the acceptance
semantics that make verified block decoding greedy-valid, and the NVFP4
decode tables the weight-prep script relies on. The second group exercises
the scheduler plumbing and is skipped where mlx-lm is not importable.
"""

import math

import pytest

from vllm_mlx.spec_utils import (
    FP4_E2M1_VALUES,
    fp8_e4m3_lut,
    fp8_e4m3_to_float,
    longest_accepted_prefix,
)

# --- acceptance semantics ------------------------------------------------------


class TestLongestAcceptedPrefix:
    def test_full_accept_returns_bonus_token(self):
        # target agrees with every draft; verified[k] is the bonus token
        m, correction = longest_accepted_prefix([5, 6, 7, 99], [5, 6, 7])
        assert (m, correction) == (3, 99)

    def test_first_mismatch_returns_targets_own_token(self):
        m, correction = longest_accepted_prefix([5, 6, 42, 99], [5, 6, 7])
        assert (m, correction) == (2, 42)

    def test_zero_accept_still_yields_one_target_token(self):
        # a "failed" round nets the corrected token, never zero progress
        m, correction = longest_accepted_prefix([11, 12, 13, 14], [1, 2, 3])
        assert (m, correction) == (0, 11)

    def test_later_agreement_does_not_rescue_an_earlier_mismatch(self):
        # acceptance is a prefix: once position 0 disagrees, nothing after
        # it can be emitted (it was conditioned on the wrong token)
        m, correction = longest_accepted_prefix([9, 2, 3, 4], [1, 2, 3])
        assert (m, correction) == (0, 9)

    def test_single_draft(self):
        assert longest_accepted_prefix([1, 2], [1]) == (1, 2)
        assert longest_accepted_prefix([3, 2], [1]) == (0, 3)

    def test_length_contract_is_enforced(self):
        with pytest.raises(ValueError, match="len\\(drafted\\) \\+ 1"):
            longest_accepted_prefix([1, 2, 3], [1, 2, 3])


# --- NVFP4 decode tables -------------------------------------------------------


class TestFp8E4m3:
    @pytest.mark.parametrize(
        "byte, value",
        [
            (0x00, 0.0),
            (0x38, 1.0),  # e=7 m=0 -> 1.0
            (0x3C, 1.5),  # e=7 m=4
            (0x40, 2.0),  # e=8 m=0
            (0x08, 2.0**-6),  # smallest normal
            (0x01, 2.0**-9),  # smallest subnormal
            (0x7E, 448.0),  # largest finite e4m3fn value
            (0xB8, -1.0),  # sign bit
            (0xC0, -2.0),
        ],
    )
    def test_known_values(self, byte, value):
        assert fp8_e4m3_to_float(byte) == pytest.approx(value)

    def test_nan_pattern(self):
        assert math.isnan(fp8_e4m3_to_float(0x7F))
        assert math.isnan(fp8_e4m3_to_float(0xFF))

    def test_bit_pattern_is_not_the_integer(self):
        # The bug this table exists to prevent: treating the byte as an int.
        lut = fp8_e4m3_lut()
        assert len(lut) == 256
        assert lut[0x38] == 1.0 and lut[0x38] != 0x38
        finite = [v for v in lut if not math.isnan(v)]
        assert max(finite) == 448.0 and min(finite) == -448.0

    def test_negative_zero_and_symmetry(self):
        lut = fp8_e4m3_lut()
        for b in range(0x7F):
            assert lut[b | 0x80] == pytest.approx(-lut[b])

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            fp8_e4m3_to_float(256)


def test_fp4_e2m1_table():
    assert len(FP4_E2M1_VALUES) == 16
    assert FP4_E2M1_VALUES[:8] == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for i in range(8):
        assert FP4_E2M1_VALUES[i + 8] == -FP4_E2M1_VALUES[i]


# --- scheduler plumbing (needs mlx-lm) ------------------------------------------

mlx_lm = pytest.importorskip("mlx_lm")


def test_scheduler_config_accepts_dspark_and_rejects_bad_values():
    from vllm_mlx.scheduler import SchedulerConfig

    cfg = SchedulerConfig(spec_draft="dspark", spec_num_draft_tokens=4)
    assert cfg.spec_draft == "dspark"
    assert cfg.spec_num_draft_tokens == 4
    assert cfg.spec_draft_margin_tau is None
    with pytest.raises(ValueError, match="spec_draft"):
        SchedulerConfig(spec_draft="eagle")
    with pytest.raises(ValueError, match="spec_num_draft_tokens"):
        SchedulerConfig(spec_draft="dspark", spec_num_draft_tokens=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SchedulerConfig(spec_draft="dspark", enable_mtp=True)


def test_sampling_is_greedy_follows_temperature_only():
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import _sampling_is_greedy

    # make_sampler(temp=0) is argmax regardless of the other knobs
    assert _sampling_is_greedy(SamplingParams(temperature=0.0, top_p=0.9))
    assert _sampling_is_greedy(SamplingParams(temperature=0))
    assert not _sampling_is_greedy(SamplingParams(temperature=0.2, top_p=1.0))
    assert not _sampling_is_greedy(object())


def test_install_dspark_noops_without_drafter(caplog):
    import logging
    from types import SimpleNamespace

    from vllm_mlx.scheduler import _install_dspark

    class _GB:
        def _step(self):
            raise AssertionError("must not be called")

    bg = SimpleNamespace(_generation_batch=_GB(), _next=lambda: [])
    model = SimpleNamespace()  # no .dspark
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        assert _install_dspark(bg, model=model) is False
    assert not hasattr(bg, "get_spec_decode_stats")
    assert bg._next.__name__ == "<lambda>"  # untouched
    assert any("[DSpark] disabled" in rec.message for rec in caplog.records)


def test_install_dspark_noops_without_step_hook(caplog):
    import logging
    from types import SimpleNamespace

    from vllm_mlx.scheduler import _install_dspark

    bg = SimpleNamespace(_generation_batch=object(), _next=lambda: [])
    model = SimpleNamespace(dspark=object())
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        assert _install_dspark(bg, model=model) is False
    assert any("_step" in rec.message for rec in caplog.records)


def test_scheduler_without_drafter_creates_generator_and_warns(caplog):
    """Real construction path: --spec-draft dspark with no drafter loaded must
    warn once and hand back a working generator (plain decode)."""
    import logging
    from types import SimpleNamespace

    from vllm_mlx.scheduler import Request, SamplingParams, Scheduler, SchedulerConfig

    model = SimpleNamespace()  # no .dspark
    tokenizer = SimpleNamespace(
        encode=lambda text: list(range(len(text.split()))),
        decode=lambda ids: " ".join(str(i) for i in ids),
        eos_token_id=0,
        eos_token_ids={0},
    )
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(enable_prefix_cache=False, spec_draft="dspark"),
    )
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
        bg = scheduler._create_batch_generator(SamplingParams(temperature=0.0))
    assert bg is not None
    assert not hasattr(bg, "get_spec_decode_stats")
    assert any("no drafter is loaded" in rec.message for rec in caplog.records)
    assert "spec_decode" not in scheduler.get_stats()

    scheduler.add_request(
        Request(
            request_id="dspark-1",
            prompt="hello there",
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )
    assert scheduler.has_requests()


def test_scheduler_skips_dspark_for_non_greedy_requests(caplog):
    import logging
    from types import SimpleNamespace

    from vllm_mlx.scheduler import SamplingParams, Scheduler, SchedulerConfig

    model = SimpleNamespace(dspark=object())
    tokenizer = SimpleNamespace(
        encode=lambda text: [1, 2, 3],
        decode=lambda ids: "x",
        eos_token_id=0,
        eos_token_ids={0},
    )
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(enable_prefix_cache=False, spec_draft="dspark"),
    )
    with caplog.at_level(logging.INFO, logger="vllm_mlx.scheduler"):
        bg = scheduler._create_batch_generator(SamplingParams(temperature=0.7))
    assert not hasattr(bg, "get_spec_decode_stats")
    assert any("not greedy" in rec.message for rec in caplog.records)
