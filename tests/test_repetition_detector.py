# SPDX-License-Identifier: Apache-2.0
"""
Tests for the repetition detector in scheduler.py.
"""

from vllm_mlx.scheduler import _detect_repetition


class TestDetectRepetition:
    """Tests for _detect_repetition function."""

    def test_no_repetition_normal_tokens(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert _detect_repetition(tokens) is False

    def test_single_token_repetition(self):
        """8 identical tokens should trigger detection."""
        tokens = [0, 0, 0, 0, 0, 0, 0, 0]
        assert _detect_repetition(tokens) is True

    def test_single_token_not_enough(self):
        """7 identical tokens should NOT trigger (min_repeat=8)."""
        tokens = [0, 0, 0, 0, 0, 0, 0]
        assert _detect_repetition(tokens) is False

    def test_single_token_at_tail(self):
        """Repetition at the tail of a longer sequence."""
        tokens = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        assert _detect_repetition(tokens) is True

    def test_two_token_pattern(self):
        """Pattern of length 2 repeated 6 times = 12 tokens."""
        tokens = [1, 2] * 6  # [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        assert _detect_repetition(tokens) is True

    def test_two_token_not_enough_repeats(self):
        """Pattern of length 2 repeated only 5 times should NOT trigger."""
        tokens = [1, 2] * 5
        assert _detect_repetition(tokens) is False

    def test_three_token_pattern(self):
        """Pattern of length 3 repeated 6 times = 18 tokens."""
        tokens = [10, 20, 30] * 6
        assert _detect_repetition(tokens) is True

    def test_four_token_pattern(self):
        """Pattern of length 4 repeated 6 times = 24 tokens."""
        tokens = [1, 2, 3, 4] * 6
        assert _detect_repetition(tokens) is True

    def test_pattern_at_tail(self):
        """Pattern repetition at the tail after normal tokens."""
        prefix = [100, 200, 300, 400, 500]
        repeating = [7, 8] * 6
        tokens = prefix + repeating
        assert _detect_repetition(tokens) is True

    def test_empty_tokens(self):
        assert _detect_repetition([]) is False

    def test_short_tokens(self):
        assert _detect_repetition([1, 2, 3]) is False

    def test_almost_repetition(self):
        """Almost repeating but one token is different."""
        tokens = [0, 0, 0, 0, 0, 0, 0, 1]
        assert _detect_repetition(tokens) is False

    def test_custom_min_repeat(self):
        """Custom min_repeat parameter."""
        tokens = [5, 5, 5, 5]
        assert _detect_repetition(tokens, min_repeat=4) is True
        assert _detect_repetition(tokens, min_repeat=5) is False

    def test_mixed_no_false_positive(self):
        """Varied tokens should not trigger."""
        tokens = list(range(32))
        assert _detect_repetition(tokens) is False

    def test_realistic_degenerate_output(self):
        """Simulate realistic degenerate model output (token ID 0 = padding)."""
        # Model starts generating, then degenerates
        normal = [15234, 8821, 3309, 44, 2847]
        degenerate = [0] * 10
        tokens = normal + degenerate
        assert _detect_repetition(tokens) is True
