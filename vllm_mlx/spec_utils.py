# SPDX-License-Identifier: Apache-2.0
"""Pure-Python helpers for block speculative decoding.

Deliberately free of MLX imports so the acceptance semantics and the NVFP4
decode tables can be unit-tested on hosts without Apple Silicon.
"""

from typing import List, Sequence, Tuple

# fp4 e2m1 values indexed by nibble; the high bit is the sign.
FP4_E2M1_VALUES: Tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def fp8_e4m3_to_float(byte: int) -> float:
    """Decode one OCP fp8 e4m3fn bit pattern (bias 7, no infinities).

    NVFP4 block scales are e4m3 *bit patterns* stored in a uint8 tensor.
    Casting that tensor to float gives 0..255 — silently wrong scales that
    still produce a "mostly working" model (teacher-forced draft agreement
    dropped from 29.9% to 15.6% with that bug). Decode the pattern instead.
    """
    if not 0 <= byte <= 0xFF:
        raise ValueError(f"fp8 byte out of range: {byte!r}")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        # subnormal: no implicit leading one, exponent fixed at 1-bias
        return sign * (mantissa / 8.0) * 2.0**-6
    if exponent == 0x0F and mantissa == 0x07:
        return float("nan")
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def fp8_e4m3_lut() -> List[float]:
    """All 256 e4m3fn values, indexed by byte."""
    return [fp8_e4m3_to_float(b) for b in range(256)]


def longest_accepted_prefix(
    verified: Sequence[int], drafted: Sequence[int]
) -> Tuple[int, int]:
    """Greedy block-verify acceptance.

    ``drafted`` is the draft block ``[d1..dk]``. ``verified`` holds the
    target's argmax at every position of the verify forward over
    ``[P, d1..dk]``: ``verified[i]`` is what the target emits after
    ``[.., P, d1..di]``, so it has ``k + 1`` entries and ``verified[k]`` is
    the bonus token that follows a fully accepted block.

    Returns ``(m, correction)``: ``m`` is the number of leading draft tokens
    the target agrees with, and ``correction`` is the target's own token at
    the first disagreement (or the bonus token when ``m == k``). Every
    emitted token is therefore an argmax of the target's logits over the
    emitted prefix — the draft can only change speed, never the output.
    """
    k = len(drafted)
    if len(verified) != k + 1:
        raise ValueError(
            f"verified must have len(drafted) + 1 = {k + 1} entries, "
            f"got {len(verified)}"
        )
    m = 0
    while m < k and verified[m] == drafted[m]:
        m += 1
    return m, verified[m]
