# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for the Phase 4a speculative decoding package.

Tests cover:
- NgramProposer: pattern matching, edge cases, incremental table build, reset
- RejectionSampler: greedy and stochastic modes, edge cases
- SpecDecodeStats: acceptance rate, mean accepted length, per-position rates, reset
- SpecDecodeMetadata / SpecDecodeConfig: validation, add/get/clear
- SpecDecodeRuntime: propose_drafts, should_disable, accept_and_commit, rollback,
  remove_request, verify_forward
"""

import unittest

import mlx.core as mx

from vllm_mlx.spec_decode import (
    AcceptResult,
    RequestState,
    SpecDecodeConfig,
    SpecDecodeMetadata,
    SpecDecodeRuntime,
    SpecDecodeStats,
    VerifyResult,
)
from vllm_mlx.spec_decode.ngram_proposer import NgramProposer, NgramProposerConfig
from vllm_mlx.spec_decode.rejection_sampler import RejectionResult, RejectionSampler

# ======================================================================
# TestNgramProposer
# ======================================================================


class TestNgramProposer(unittest.TestCase):
    """Tests for NgramProposer pattern matching and state management."""

    def test_pattern_matching_basic(self):
        """Repeated pattern [1,2,3,4,1,2,3] with n=3 should propose [4]."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        token_ids = [1, 2, 3, 4, 1, 2, 3]
        drafts = proposer.propose(token_ids, k=5)

        # The last 2 tokens (n-1=2) are [2, 3] — the search pattern.
        # Earlier occurrence of [2, 3] starts at index 1.
        # Tokens following that match start at index 3: [4].
        # Draft cannot extend past len(token_ids) - prefix_len = 7 - 2 = 5.
        # So draft_start=3, draft_end=min(3+5, 5)=5 => token_ids[3:5] = [4, 1].
        self.assertEqual(drafts, [4, 1])

    def test_no_match_unique_tokens(self):
        """Unique tokens should return empty list — no n-gram matches."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        token_ids = [10, 20, 30, 40, 50]
        drafts = proposer.propose(token_ids, k=5)
        self.assertEqual(drafts, [])

    def test_short_sequence(self):
        """Sequence shorter than n should return empty list."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        # len=2 < n=3
        drafts = proposer.propose([1, 2], k=5)
        self.assertEqual(drafts, [])

    def test_exact_n_length(self):
        """Sequence of exactly n tokens: pattern is last (n-1) but no earlier match."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        drafts = proposer.propose([1, 2, 3], k=5)
        self.assertEqual(drafts, [])

    def test_k_zero(self):
        """k=0 should return empty list."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        token_ids = [1, 2, 3, 4, 1, 2, 3]
        drafts = proposer.propose(token_ids, k=0)
        self.assertEqual(drafts, [])

    def test_multiple_matches_uses_most_recent(self):
        """When multiple n-gram matches exist, the most recent one is used."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        # Pattern [2,3] occurs at indices 1 and 5.
        # Index 1 match: followed by [4, 5]
        # Index 5 match: followed by [99]
        # Most recent match (index 5) should be used -> [99]
        token_ids = [1, 2, 3, 4, 5, 2, 3, 99, 2, 3]
        drafts = proposer.propose(token_ids, k=5)

        # Last 2 tokens are [2,3]. Most recent earlier match is at index 5.
        # draft_start = 5 + 2 = 7, draft_end = min(7+5, 10-2) = 8
        # token_ids[7:8] = [99]
        self.assertEqual(drafts, [99])

    def test_bigram_n2(self):
        """n=2 (bigram): match last 1 token."""
        config = NgramProposerConfig(n=2, max_k=5)
        proposer = NgramProposer(config)

        # Pattern is last (n-1)=1 token: [3].
        # [3] occurs at index 2. Next token at index 3: [4].
        token_ids = [1, 2, 3, 4, 5, 3]
        drafts = proposer.propose(token_ids, k=5)

        # Most recent match of [3] before the last position.
        # Index 2: followed by [4, 5], draft_end = min(3+5, 6-1) = 5
        # But there's also index 5 which IS the last position (query).
        # The table indexes up to max_index_start = seq_len - prefix_len = 6 - 1 = 5.
        # So indices 0..4 are indexed. Gram at index 4 is (5,), at index 2 is (3,).
        # Wait: n-1 = 1 so gram is single token. Positions of (3,): index 2.
        # Actually index 5 is not indexed because the range is start..max_index_start
        # which is range(0, 5) = [0..4]. Token at index 2 is 3, gram is (3,).
        # draft_start = 2 + 1 = 3, draft_end = min(3+5, 5) = 5
        # token_ids[3:5] = [4, 5]
        self.assertEqual(drafts, [4, 5])

    def test_4gram(self):
        """n=4: match last 3 tokens."""
        config = NgramProposerConfig(n=4, max_k=5)
        proposer = NgramProposer(config)

        # Pattern: last 3 tokens = [2, 3, 4]. Earlier occurrence at index 1.
        # draft_start = 1 + 3 = 4, draft_end = min(4+5, 8-3) = 5
        # token_ids[4:5] = [5]
        token_ids = [1, 2, 3, 4, 5, 2, 3, 4]
        drafts = proposer.propose(token_ids, k=5)
        self.assertEqual(drafts, [5])

    def test_incremental_table_build(self):
        """Propose called multiple times on a growing sequence uses incremental indexing."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        # First call: no match
        token_ids = [1, 2, 3]
        drafts = proposer.propose(token_ids, k=5)
        self.assertEqual(drafts, [])

        # Extend sequence and call again — table should be updated incrementally
        token_ids = [1, 2, 3, 4, 1, 2, 3]
        drafts = proposer.propose(token_ids, k=5)
        # Now [2,3] at index 1 matches, draft_start=3, draft_end=min(8,5)=5
        # token_ids[3:5] = [4, 1]
        self.assertEqual(drafts, [4, 1])

        # Verify internal indexed_length was updated (implementation detail)
        self.assertGreater(proposer._indexed_length, 0)

    def test_reset_clears_state(self):
        """reset() clears the n-gram table and indexed length."""
        config = NgramProposerConfig(n=3, max_k=5)
        proposer = NgramProposer(config)

        # Build some state
        token_ids = [1, 2, 3, 4, 1, 2, 3]
        proposer.propose(token_ids, k=5)
        self.assertGreater(len(proposer._ngram_table), 0)
        self.assertGreater(proposer._indexed_length, 0)

        # Reset
        proposer.reset()
        self.assertEqual(len(proposer._ngram_table), 0)
        self.assertEqual(proposer._indexed_length, 0)

    def test_max_k_clamps_k(self):
        """k is clamped to max_k config value."""
        config = NgramProposerConfig(n=3, max_k=1)
        proposer = NgramProposer(config)

        # Even though we ask for k=10, max_k=1 limits to 1 draft token
        token_ids = [1, 2, 3, 4, 5, 6, 2, 3]
        drafts = proposer.propose(token_ids, k=10)
        self.assertLessEqual(len(drafts), 1)

    def test_longer_repeating_pattern(self):
        """Longer repeating sequence produces multiple draft tokens."""
        config = NgramProposerConfig(n=3, max_k=10)
        proposer = NgramProposer(config)

        # [A B C D E A B C] — last 2 tokens [B C], match at index 1
        # draft_start = 1+2 = 3, draft_end = min(3+10, 8-2) = 6
        # token_ids[3:6] = [D, E, A]
        token_ids = [10, 20, 30, 40, 50, 10, 20, 30]
        drafts = proposer.propose(token_ids, k=10)
        self.assertEqual(drafts, [40, 50, 10])


# ======================================================================
# TestRejectionSampler
# ======================================================================


class TestRejectionSampler(unittest.TestCase):
    """Tests for RejectionSampler greedy and stochastic modes."""

    def _make_logits_with_argmax(self, token_id: int, vocab_size: int = 10) -> mx.array:
        """Create a 1D logits vector where argmax is at token_id."""
        logits = mx.zeros((vocab_size,))
        logits = logits.at[token_id].add(mx.array(10.0))
        mx.eval(logits)
        return logits

    def _make_target_logits(
        self, argmax_tokens: list[int], vocab_size: int = 10
    ) -> mx.array:
        """
        Create target logits of shape (1, len(argmax_tokens), vocab_size)
        where position i has argmax at argmax_tokens[i].
        """
        k_plus_1 = len(argmax_tokens)
        logits = mx.zeros((1, k_plus_1, vocab_size))
        for i, tok in enumerate(argmax_tokens):
            # Set a high value at the target token position
            update = mx.zeros((vocab_size,))
            update = update.at[tok].add(mx.array(10.0))
            logits = logits.at[0, i, :].add(update)
        mx.eval(logits)
        return logits

    def test_invalid_method(self):
        """Invalid rejection method raises ValueError."""
        with self.assertRaises(ValueError):
            RejectionSampler(method="invalid")

    def test_greedy_all_accept(self):
        """Greedy: target argmax matches all drafts -> all accepted + bonus."""
        sampler = RejectionSampler(method="greedy")

        # Draft tokens: [3, 5, 7]
        # Target argmax at positions 0,1,2 should be [3, 5, 7], position 3 is bonus
        draft_token_ids = [[3, 5, 7]]
        target_logits = self._make_target_logits([3, 5, 7, 9], vocab_size=10)

        result = sampler(target_logits, draft_token_ids)

        self.assertEqual(result.num_accepted, [3])
        self.assertEqual(result.accepted_token_ids, [[3, 5, 7]])
        self.assertEqual(result.bonus_token_ids, [9])

    def test_greedy_all_reject(self):
        """Greedy: first draft doesn't match -> 0 accepted + correction token."""
        sampler = RejectionSampler(method="greedy")

        # Draft [3, 5, 7] but target argmax at pos 0 is 4 (not 3)
        draft_token_ids = [[3, 5, 7]]
        target_logits = self._make_target_logits([4, 5, 7, 9], vocab_size=10)

        result = sampler(target_logits, draft_token_ids)

        self.assertEqual(result.num_accepted, [0])
        self.assertEqual(result.accepted_token_ids, [[]])
        # Correction token is the target's argmax at position 0
        self.assertEqual(result.bonus_token_ids, [4])

    def test_greedy_partial_accept(self):
        """Greedy: some match, then diverge."""
        sampler = RejectionSampler(method="greedy")

        # Draft [3, 5, 7]: pos 0 matches (3), pos 1 matches (5), pos 2 diverges
        draft_token_ids = [[3, 5, 7]]
        target_logits = self._make_target_logits([3, 5, 8, 9], vocab_size=10)

        result = sampler(target_logits, draft_token_ids)

        self.assertEqual(result.num_accepted, [2])
        self.assertEqual(result.accepted_token_ids, [[3, 5]])
        # Correction token at divergence point (pos 2)
        self.assertEqual(result.bonus_token_ids, [8])

    def test_greedy_empty_drafts(self):
        """Greedy with empty drafts: returns bonus from target logits position 0."""
        sampler = RejectionSampler(method="greedy")

        draft_token_ids = [[]]
        target_logits = self._make_target_logits([6], vocab_size=10)

        result = sampler(target_logits, draft_token_ids)

        self.assertEqual(result.num_accepted, [0])
        self.assertEqual(result.accepted_token_ids, [[]])
        self.assertEqual(result.bonus_token_ids, [6])

    def test_greedy_single_draft(self):
        """Greedy with k=1: single draft token accepted."""
        sampler = RejectionSampler(method="greedy")

        draft_token_ids = [[5]]
        target_logits = self._make_target_logits([5, 2], vocab_size=10)

        result = sampler(target_logits, draft_token_ids)

        self.assertEqual(result.num_accepted, [1])
        self.assertEqual(result.accepted_token_ids, [[5]])
        self.assertEqual(result.bonus_token_ids, [2])

    def test_greedy_batch_size_2(self):
        """Greedy with batch size > 1: independent per-request handling."""
        sampler = RejectionSampler(method="greedy")

        # Request 0: all accept. Request 1: reject at pos 0.
        draft_token_ids = [[3, 5], [2, 4]]

        vocab_size = 10
        logits = mx.zeros((2, 3, vocab_size))
        # Request 0: argmax at [3, 5, 9]
        for i, tok in enumerate([3, 5, 9]):
            logits = logits.at[0, i, tok].add(mx.array(10.0))
        # Request 1: argmax at [7, 4, 8] (7 != 2 -> reject at pos 0)
        for i, tok in enumerate([7, 4, 8]):
            logits = logits.at[1, i, tok].add(mx.array(10.0))
        mx.eval(logits)

        result = sampler(logits, draft_token_ids)

        self.assertEqual(result.num_accepted, [2, 0])
        self.assertEqual(result.accepted_token_ids, [[3, 5], []])
        self.assertEqual(result.bonus_token_ids, [9, 7])

    def test_stochastic_basic(self):
        """Stochastic mode runs without error and returns valid structure."""
        sampler = RejectionSampler(method="stochastic")

        vocab_size = 10
        # Target logits: (1, k+1=3, 10)
        target_logits = mx.random.normal((1, 3, vocab_size))
        # Draft logits: (1, k=2, 10)
        draft_logits = mx.random.normal((1, 2, vocab_size))
        mx.eval(target_logits, draft_logits)

        draft_token_ids = [[1, 2]]

        result = sampler(target_logits, draft_token_ids, draft_logits=draft_logits)

        self.assertIsInstance(result, RejectionResult)
        self.assertEqual(len(result.num_accepted), 1)
        self.assertGreaterEqual(result.num_accepted[0], 0)
        self.assertLessEqual(result.num_accepted[0], 2)
        self.assertIsNotNone(result.bonus_token_ids[0])

    def test_stochastic_requires_draft_logits(self):
        """Stochastic mode raises ValueError when draft_logits is None."""
        sampler = RejectionSampler(method="stochastic")

        target_logits = mx.zeros((1, 3, 10))
        draft_token_ids = [[1, 2]]

        with self.assertRaises(ValueError):
            sampler(target_logits, draft_token_ids, draft_logits=None)

    def test_stochastic_empty_drafts(self):
        """Stochastic with empty drafts returns valid structure."""
        sampler = RejectionSampler(method="stochastic")

        vocab_size = 10
        target_logits = mx.random.normal((1, 1, vocab_size))
        draft_logits = mx.random.normal((1, 0, vocab_size))
        mx.eval(target_logits, draft_logits)

        draft_token_ids = [[]]

        result = sampler(target_logits, draft_token_ids, draft_logits=draft_logits)

        self.assertEqual(result.num_accepted, [0])
        self.assertEqual(result.accepted_token_ids, [[]])
        self.assertIsNotNone(result.bonus_token_ids[0])


# ======================================================================
# TestSpecDecodeStats
# ======================================================================


class TestSpecDecodeStats(unittest.TestCase):
    """Tests for SpecDecodeStats accumulation and computation."""

    def test_acceptance_rate_basic(self):
        """acceptance_rate() computes correctly."""
        stats = SpecDecodeStats()
        stats.update(
            num_drafted=4, num_accepted=3, position_accepted=[True, True, True, False]
        )
        self.assertAlmostEqual(stats.acceptance_rate(), 3 / 4)

    def test_mean_accepted_length_basic(self):
        """mean_accepted_length() computes correctly."""
        stats = SpecDecodeStats()
        stats.update(
            num_drafted=4, num_accepted=3, position_accepted=[True, True, True, False]
        )
        stats.update(
            num_drafted=4, num_accepted=1, position_accepted=[True, False, False, False]
        )
        # Total accepted = 4, total drafts = 2
        self.assertAlmostEqual(stats.mean_accepted_length(), 4 / 2)

    def test_update_accumulates(self):
        """update() accumulates counts correctly over multiple rounds."""
        stats = SpecDecodeStats()

        stats.update(
            num_drafted=3, num_accepted=2, position_accepted=[True, True, False]
        )
        self.assertEqual(stats.num_drafts, 1)
        self.assertEqual(stats.num_draft_tokens, 3)
        self.assertEqual(stats.num_accepted_tokens, 2)

        stats.update(
            num_drafted=5,
            num_accepted=4,
            position_accepted=[True, True, True, True, False],
        )
        self.assertEqual(stats.num_drafts, 2)
        self.assertEqual(stats.num_draft_tokens, 8)
        self.assertEqual(stats.num_accepted_tokens, 6)

    def test_per_position_rates(self):
        """Per-position acceptance rates are tracked correctly."""
        stats = SpecDecodeStats()

        # Round 1: positions [T, F, T]
        stats.update(
            num_drafted=3, num_accepted=2, position_accepted=[True, False, True]
        )
        # Round 2: positions [T, T, F]
        stats.update(
            num_drafted=3, num_accepted=2, position_accepted=[True, True, False]
        )

        rates = stats.acceptance_rate_per_position
        self.assertEqual(len(rates), 3)
        # Position 0: 2/2 = 1.0
        self.assertAlmostEqual(rates[0], 1.0)
        # Position 1: 1/2 = 0.5
        self.assertAlmostEqual(rates[1], 0.5)
        # Position 2: 1/2 = 0.5
        self.assertAlmostEqual(rates[2], 0.5)

    def test_per_position_rates_different_lengths(self):
        """Per-position rates handle rounds with different draft lengths."""
        stats = SpecDecodeStats()

        # Round 1: 2 positions
        stats.update(num_drafted=2, num_accepted=1, position_accepted=[True, False])
        # Round 2: 4 positions (expands the per-position counters)
        stats.update(
            num_drafted=4,
            num_accepted=3,
            position_accepted=[True, True, True, False],
        )

        rates = stats.acceptance_rate_per_position
        self.assertEqual(len(rates), 4)
        # Position 0: 2/2 = 1.0
        self.assertAlmostEqual(rates[0], 1.0)
        # Position 1: 1/2 = 0.5
        self.assertAlmostEqual(rates[1], 0.5)
        # Position 2: 1/1 = 1.0 (only appeared in round 2)
        self.assertAlmostEqual(rates[2], 1.0)
        # Position 3: 0/1 = 0.0
        self.assertAlmostEqual(rates[3], 0.0)

    def test_reset_clears_everything(self):
        """reset() clears all statistics."""
        stats = SpecDecodeStats()
        stats.update(
            num_drafted=5,
            num_accepted=3,
            position_accepted=[True, True, True, False, False],
        )

        stats.reset()

        self.assertEqual(stats.num_drafts, 0)
        self.assertEqual(stats.num_draft_tokens, 0)
        self.assertEqual(stats.num_accepted_tokens, 0)
        self.assertEqual(stats.acceptance_rate_per_position, [])
        self.assertEqual(stats._position_accepted_counts, [])
        self.assertEqual(stats._position_total_counts, [])

    def test_zero_state_no_division_by_zero(self):
        """Zero state (no updates) should not cause division by zero."""
        stats = SpecDecodeStats()

        self.assertEqual(stats.acceptance_rate(), 0.0)
        self.assertEqual(stats.mean_accepted_length(), 0.0)
        self.assertEqual(stats.acceptance_rate_per_position, [])


# ======================================================================
# TestMetadata
# ======================================================================


class TestMetadata(unittest.TestCase):
    """Tests for SpecDecodeConfig validation and SpecDecodeMetadata operations."""

    # --- SpecDecodeConfig ---

    def test_config_valid_methods(self):
        """Valid methods are accepted without error."""
        for method in ("ngram", "eagle", "draft_model"):
            config = SpecDecodeConfig(method=method, num_speculative_tokens=5)
            self.assertEqual(config.method, method)

    def test_config_invalid_method(self):
        """Invalid method raises ValueError."""
        with self.assertRaises(ValueError):
            SpecDecodeConfig(method="invalid_method", num_speculative_tokens=5)

    def test_config_num_speculative_tokens_zero(self):
        """num_speculative_tokens < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            SpecDecodeConfig(method="ngram", num_speculative_tokens=0)

    def test_config_num_speculative_tokens_negative(self):
        """Negative num_speculative_tokens raises ValueError."""
        with self.assertRaises(ValueError):
            SpecDecodeConfig(method="ngram", num_speculative_tokens=-3)

    def test_config_disable_by_batch_size_valid(self):
        """Valid disable_by_batch_size is accepted."""
        config = SpecDecodeConfig(
            method="ngram", num_speculative_tokens=5, disable_by_batch_size=8
        )
        self.assertEqual(config.disable_by_batch_size, 8)

    def test_config_disable_by_batch_size_zero(self):
        """disable_by_batch_size < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            SpecDecodeConfig(
                method="ngram", num_speculative_tokens=5, disable_by_batch_size=0
            )

    def test_config_disable_by_batch_size_none(self):
        """disable_by_batch_size=None is allowed (default)."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=5)
        self.assertIsNone(config.disable_by_batch_size)

    # --- SpecDecodeMetadata ---

    def test_metadata_add_and_get(self):
        """add_request stores and get_draft_tokens retrieves draft tokens."""
        meta = SpecDecodeMetadata()
        meta.add_request("req-1", [10, 20, 30])

        self.assertEqual(meta.get_draft_tokens("req-1"), [10, 20, 30])
        self.assertEqual(meta.num_draft_tokens["req-1"], 3)

    def test_metadata_get_missing_request(self):
        """get_draft_tokens returns empty list for unknown request_id."""
        meta = SpecDecodeMetadata()
        self.assertEqual(meta.get_draft_tokens("nonexistent"), [])

    def test_metadata_multiple_requests(self):
        """Multiple requests are stored independently."""
        meta = SpecDecodeMetadata()
        meta.add_request("req-1", [1, 2])
        meta.add_request("req-2", [3, 4, 5])

        self.assertEqual(meta.get_draft_tokens("req-1"), [1, 2])
        self.assertEqual(meta.get_draft_tokens("req-2"), [3, 4, 5])
        self.assertEqual(meta.num_draft_tokens["req-1"], 2)
        self.assertEqual(meta.num_draft_tokens["req-2"], 3)

    def test_metadata_clear(self):
        """clear() removes all stored data."""
        meta = SpecDecodeMetadata()
        meta.add_request("req-1", [1, 2, 3])
        meta.add_request("req-2", [4, 5])

        meta.clear()

        self.assertEqual(meta.get_draft_tokens("req-1"), [])
        self.assertEqual(meta.get_draft_tokens("req-2"), [])
        self.assertEqual(len(meta.draft_token_ids), 0)
        self.assertEqual(len(meta.num_draft_tokens), 0)

    def test_metadata_add_empty_drafts(self):
        """Adding empty draft list is handled correctly."""
        meta = SpecDecodeMetadata()
        meta.add_request("req-1", [])

        self.assertEqual(meta.get_draft_tokens("req-1"), [])
        self.assertEqual(meta.num_draft_tokens["req-1"], 0)

    def test_metadata_overwrite_request(self):
        """Adding the same request_id again overwrites the previous entry."""
        meta = SpecDecodeMetadata()
        meta.add_request("req-1", [1, 2, 3])
        meta.add_request("req-1", [10, 20])

        self.assertEqual(meta.get_draft_tokens("req-1"), [10, 20])
        self.assertEqual(meta.num_draft_tokens["req-1"], 2)


# ======================================================================
# TestRuntime
# ======================================================================


class _DummyProposer:
    """A simple proposer that returns fixed draft tokens for testing."""

    def __init__(self, drafts: dict[str, list[int]] | None = None):
        self._drafts = drafts or {}
        self._propose_calls: list[tuple[list[int], int]] = []

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        self._propose_calls.append((list(token_ids), k))
        # Return a fixed-length draft of the first k token IDs from the end
        # or from pre-configured drafts matching the sequence
        seq_key = str(token_ids)
        if seq_key in self._drafts:
            return self._drafts[seq_key][:k]
        # Default: return list [100, 101, ... 100+k-1]
        return list(range(100, 100 + k))

    def propose_with_context(self, ctx) -> "Proposal":
        """Delegate to propose() and wrap the result in a Proposal."""
        from vllm_mlx.spec_decode.proposer import Proposal

        tokens = self.propose(ctx.token_ids, ctx.k)
        return Proposal(token_ids=tokens)

    def reset(self):
        # Only clear n-gram state, not the call log (used for test assertions)
        pass


class _DummyModel:
    """Placeholder model for runtime initialization."""

    pass


class TestRuntime(unittest.TestCase):
    """Tests for SpecDecodeRuntime orchestration logic."""

    def _make_runtime(
        self,
        proposer=None,
        method="ngram",
        num_speculative_tokens=3,
        disable_by_batch_size=None,
        rejection_method="greedy",
    ) -> SpecDecodeRuntime:
        """Helper to create a SpecDecodeRuntime with sensible defaults."""
        model = _DummyModel()
        if proposer is None:
            proposer = _DummyProposer()
        config = SpecDecodeConfig(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
            disable_by_batch_size=disable_by_batch_size,
        )
        sampler = RejectionSampler(method=rejection_method)
        return SpecDecodeRuntime(
            model=model,
            proposer=proposer,
            rejection_sampler=sampler,
            config=config,
        )

    def test_propose_drafts_populates_metadata(self):
        """propose_drafts calls proposer and populates metadata for each request."""
        proposer = _DummyProposer()
        runtime = self._make_runtime(proposer=proposer, num_speculative_tokens=3)

        request_states = {
            "req-1": RequestState(request_id="req-1", token_ids=[1, 2, 3], batch_uid=0),
            "req-2": RequestState(
                request_id="req-2", token_ids=[4, 5, 6, 7], batch_uid=1
            ),
        }

        metadata = runtime.propose_drafts(request_states)

        # Verify metadata contains entries for both requests
        self.assertIn("req-1", metadata.draft_token_ids)
        self.assertIn("req-2", metadata.draft_token_ids)

        # Verify proposer was called for each request
        self.assertEqual(len(proposer._propose_calls), 2)

        # Verify sequence lengths are tracked
        self.assertEqual(runtime.get_seq_len("req-1"), 3)
        self.assertEqual(runtime.get_seq_len("req-2"), 4)

    def test_should_disable_no_threshold(self):
        """should_disable returns False when disable_by_batch_size is None."""
        runtime = self._make_runtime(disable_by_batch_size=None)

        self.assertFalse(runtime.should_disable(1))
        self.assertFalse(runtime.should_disable(100))
        self.assertFalse(runtime.should_disable(1000))

    def test_should_disable_below_threshold(self):
        """should_disable returns False when batch_size < threshold."""
        runtime = self._make_runtime(disable_by_batch_size=8)

        self.assertFalse(runtime.should_disable(1))
        self.assertFalse(runtime.should_disable(7))

    def test_should_disable_at_threshold(self):
        """should_disable returns True when batch_size >= threshold."""
        runtime = self._make_runtime(disable_by_batch_size=8)

        self.assertTrue(runtime.should_disable(8))
        self.assertTrue(runtime.should_disable(10))

    def test_accept_and_commit_all_accepted(self):
        """accept_and_commit: all drafts accepted -> committed tokens + bonus."""
        runtime = self._make_runtime(num_speculative_tokens=3)

        # Set up internal seq_len tracking
        runtime._seq_lens["req-1"] = 10

        # Create draft metadata
        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [3, 5, 7])

        # Create verify result with logits where argmax matches drafts
        vocab_size = 10
        logits = mx.zeros((4, vocab_size))  # k+1 = 4 positions
        for i, tok in enumerate([3, 5, 7, 9]):
            logits = logits.at[i, tok].add(mx.array(10.0))
        mx.eval(logits)

        verify_result = VerifyResult(
            target_logits={"req-1": logits},
            request_ids=["req-1"],
        )

        results = runtime.accept_and_commit(verify_result, draft_metadata)

        self.assertIn("req-1", results)
        result = results["req-1"]
        self.assertEqual(result.num_accepted, 3)
        self.assertEqual(result.accepted_tokens, [3, 5, 7, 9])  # 3 accepted + bonus 9
        self.assertEqual(result.bonus_token, 9)
        self.assertEqual(result.rollback_count, 0)

    def test_accept_and_commit_partial_accept(self):
        """accept_and_commit: partial acceptance -> correction token + rollback."""
        runtime = self._make_runtime(num_speculative_tokens=3)
        runtime._seq_lens["req-1"] = 10

        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [3, 5, 7])

        # Logits: pos 0 matches (3), pos 1 diverges (6 != 5)
        vocab_size = 10
        logits = mx.zeros((4, vocab_size))
        for i, tok in enumerate([3, 6, 7, 9]):
            logits = logits.at[i, tok].add(mx.array(10.0))
        mx.eval(logits)

        verify_result = VerifyResult(
            target_logits={"req-1": logits},
            request_ids=["req-1"],
        )

        results = runtime.accept_and_commit(verify_result, draft_metadata)

        result = results["req-1"]
        self.assertEqual(result.num_accepted, 1)
        # Accepted [3] + correction token 6
        self.assertEqual(result.accepted_tokens, [3, 6])
        self.assertEqual(result.bonus_token, 6)
        self.assertEqual(result.rollback_count, 2)  # 3 drafted - 1 accepted

    def test_accept_and_commit_all_rejected(self):
        """accept_and_commit: all drafts rejected -> correction token only."""
        runtime = self._make_runtime(num_speculative_tokens=3)
        runtime._seq_lens["req-1"] = 10

        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [3, 5, 7])

        # Logits: pos 0 diverges (4 != 3)
        vocab_size = 10
        logits = mx.zeros((4, vocab_size))
        for i, tok in enumerate([4, 5, 7, 9]):
            logits = logits.at[i, tok].add(mx.array(10.0))
        mx.eval(logits)

        verify_result = VerifyResult(
            target_logits={"req-1": logits},
            request_ids=["req-1"],
        )

        results = runtime.accept_and_commit(verify_result, draft_metadata)

        result = results["req-1"]
        self.assertEqual(result.num_accepted, 0)
        self.assertEqual(result.accepted_tokens, [4])  # correction token only
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual(result.rollback_count, 3)

    def test_accept_and_commit_no_drafts(self):
        """accept_and_commit: no drafts for request -> bonus token via sampler fallback."""
        runtime = self._make_runtime(num_speculative_tokens=3)
        runtime._seq_lens["req-1"] = 10

        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [])

        # Create logits where argmax is at token 6
        vocab_size = 10
        logits = mx.zeros((1, vocab_size))
        logits = logits.at[0, 6].add(mx.array(10.0))
        mx.eval(logits)

        verify_result = VerifyResult(
            target_logits={"req-1": logits},
            request_ids=["req-1"],
        )

        results = runtime.accept_and_commit(verify_result, draft_metadata)

        result = results["req-1"]
        self.assertEqual(result.num_accepted, 0)
        # Zero-draft still advances by one token via the bonus/fallback
        self.assertEqual(result.bonus_token, 6)
        self.assertEqual(result.accepted_tokens, [6])
        self.assertEqual(result.rollback_count, 0)
        # seq_len should be updated: 10 + 1 = 11
        self.assertEqual(runtime.get_seq_len("req-1"), 11)

    def test_accept_and_commit_updates_stats(self):
        """accept_and_commit updates the runtime's stats tracker."""
        runtime = self._make_runtime(num_speculative_tokens=3)
        runtime._seq_lens["req-1"] = 10

        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [3, 5, 7])

        vocab_size = 10
        logits = mx.zeros((4, vocab_size))
        for i, tok in enumerate([3, 5, 8, 9]):  # 2 accepted, 1 rejected
            logits = logits.at[i, tok].add(mx.array(10.0))
        mx.eval(logits)

        verify_result = VerifyResult(
            target_logits={"req-1": logits},
            request_ids=["req-1"],
        )

        runtime.accept_and_commit(verify_result, draft_metadata)

        stats = runtime.stats
        self.assertEqual(stats.num_drafts, 1)
        self.assertEqual(stats.num_draft_tokens, 3)
        self.assertEqual(stats.num_accepted_tokens, 2)

    def test_accept_and_commit_updates_seq_len(self):
        """accept_and_commit updates internal sequence length tracking."""
        runtime = self._make_runtime(num_speculative_tokens=3)
        runtime._seq_lens["req-1"] = 10

        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [3, 5, 7])

        vocab_size = 10
        logits = mx.zeros((4, vocab_size))
        # All accepted: commit 3 drafts + 1 bonus = 4 tokens
        for i, tok in enumerate([3, 5, 7, 9]):
            logits = logits.at[i, tok].add(mx.array(10.0))
        mx.eval(logits)

        verify_result = VerifyResult(
            target_logits={"req-1": logits},
            request_ids=["req-1"],
        )

        runtime.accept_and_commit(verify_result, draft_metadata)

        # seq_len should be 10 + 4 (3 accepted + 1 bonus) = 14
        self.assertEqual(runtime.get_seq_len("req-1"), 14)

    def test_rollback_does_not_adjust_seq_len(self):
        """rollback no longer adjusts seq_len (deferred to Phase 4b KV cache trim)."""
        runtime = self._make_runtime()
        runtime._seq_lens["req-1"] = 15
        runtime._seq_lens["req-2"] = 20

        runtime.rollback(
            request_ids=["req-1", "req-2"],
            rollback_counts={"req-1": 3, "req-2": 0},
        )

        # seq_lens should remain unchanged — rollback only logs intent now
        self.assertEqual(runtime.get_seq_len("req-1"), 15)
        self.assertEqual(runtime.get_seq_len("req-2"), 20)

    def test_rollback_leaves_seq_len_unchanged(self):
        """rollback no longer modifies seq_len, even with large rollback counts."""
        runtime = self._make_runtime()
        runtime._seq_lens["req-1"] = 2

        runtime.rollback(
            request_ids=["req-1"],
            rollback_counts={"req-1": 100},
        )

        # seq_len should remain unchanged — rollback only logs intent now
        self.assertEqual(runtime.get_seq_len("req-1"), 2)

    def test_rollback_missing_request(self):
        """rollback with unknown request_id does not raise."""
        runtime = self._make_runtime()
        # Should not raise
        runtime.rollback(
            request_ids=["nonexistent"],
            rollback_counts={"nonexistent": 5},
        )

    def test_remove_request_cleans_up(self):
        """remove_request removes the request from internal tracking."""
        runtime = self._make_runtime()
        runtime._seq_lens["req-1"] = 10
        runtime._seq_lens["req-2"] = 20

        runtime.remove_request("req-1")

        self.assertIsNone(runtime.get_seq_len("req-1"))
        self.assertEqual(runtime.get_seq_len("req-2"), 20)

    def test_remove_request_nonexistent(self):
        """remove_request with unknown request_id does not raise."""
        runtime = self._make_runtime()
        # Should not raise
        runtime.remove_request("nonexistent")

    def test_verify_forward_raises_not_implemented(self):
        """verify_forward raises NotImplementedError (Phase 4a stub)."""
        runtime = self._make_runtime()

        with self.assertRaises(NotImplementedError):
            runtime.verify_forward(
                request_ids=["req-1"],
                draft_tokens={"req-1": [1, 2, 3]},
                seq_lens={"req-1": 10},
            )

    def test_stats_property(self):
        """stats property returns the internal SpecDecodeStats instance."""
        runtime = self._make_runtime()
        stats = runtime.stats
        self.assertIsInstance(stats, SpecDecodeStats)

    def test_propose_drafts_passes_correct_k(self):
        """propose_drafts passes config.num_speculative_tokens as k to proposer."""
        proposer = _DummyProposer()
        runtime = self._make_runtime(proposer=proposer, num_speculative_tokens=7)

        request_states = {
            "req-1": RequestState(request_id="req-1", token_ids=[1, 2], batch_uid=0),
        }

        runtime.propose_drafts(request_states)

        # Check the k value passed to proposer
        self.assertEqual(len(proposer._propose_calls), 1)
        _, k_passed = proposer._propose_calls[0]
        self.assertEqual(k_passed, 7)

    def test_accept_and_commit_multiple_requests(self):
        """accept_and_commit handles multiple requests in a single verify result."""
        runtime = self._make_runtime(num_speculative_tokens=2)
        runtime._seq_lens["req-1"] = 10
        runtime._seq_lens["req-2"] = 15

        draft_metadata = SpecDecodeMetadata()
        draft_metadata.add_request("req-1", [3, 5])
        draft_metadata.add_request("req-2", [7])

        vocab_size = 10

        # req-1: all accepted
        logits_1 = mx.zeros((3, vocab_size))
        for i, tok in enumerate([3, 5, 9]):
            logits_1 = logits_1.at[i, tok].add(mx.array(10.0))

        # req-2: rejected at pos 0
        logits_2 = mx.zeros((2, vocab_size))
        for i, tok in enumerate([8, 2]):
            logits_2 = logits_2.at[i, tok].add(mx.array(10.0))

        mx.eval(logits_1, logits_2)

        verify_result = VerifyResult(
            target_logits={"req-1": logits_1, "req-2": logits_2},
            request_ids=["req-1", "req-2"],
        )

        results = runtime.accept_and_commit(verify_result, draft_metadata)

        # req-1: all accepted
        self.assertEqual(results["req-1"].num_accepted, 2)
        self.assertEqual(results["req-1"].accepted_tokens, [3, 5, 9])
        self.assertEqual(results["req-1"].rollback_count, 0)

        # req-2: rejected at pos 0
        self.assertEqual(results["req-2"].num_accepted, 0)
        self.assertEqual(results["req-2"].accepted_tokens, [8])
        self.assertEqual(results["req-2"].rollback_count, 1)


# ======================================================================
# Integration-style tests
# ======================================================================


class TestNgramProposerWithRuntime(unittest.TestCase):
    """Integration tests using NgramProposer with the runtime."""

    def test_full_propose_cycle(self):
        """End-to-end: NgramProposer proposes through the runtime."""
        ngram_config = NgramProposerConfig(n=3, max_k=5, num_speculative_tokens=5)
        proposer = NgramProposer(ngram_config)

        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=5)
        sampler = RejectionSampler(method="greedy")
        runtime = SpecDecodeRuntime(
            model=_DummyModel(),
            proposer=proposer,
            rejection_sampler=sampler,
            config=config,
        )

        # Token sequence with a repeating pattern
        request_states = {
            "req-1": RequestState(
                request_id="req-1",
                token_ids=[1, 2, 3, 4, 5, 1, 2, 3],
                batch_uid=0,
            ),
        }

        metadata = runtime.propose_drafts(request_states)
        drafts = metadata.get_draft_tokens("req-1")

        # With n=3, pattern is [2, 3], match at index 1 -> follow tokens [4, 5, 1]
        self.assertGreater(len(drafts), 0)
        self.assertEqual(drafts, [4, 5, 1])


if __name__ == "__main__":
    unittest.main()
