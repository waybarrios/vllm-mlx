# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for Phase 4b: Speculative decoding scheduler integration.

These tests verify the scheduler's spec decode path using mocked models
and batch generators, without requiring actual LLM inference.
"""

from collections import deque
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_mlx.scheduler import SchedulerConfig
from vllm_mlx.spec_decode import (
    AcceptResult,
    RequestState,
    SpecDecodeConfig,
    SpecDecodeRuntime,
    SpecDecodeStats,
    VerifyResult,
)
from vllm_mlx.spec_decode.metadata import SpecDecodeMetadata
from vllm_mlx.spec_decode.ngram_proposer import NgramProposer, NgramProposerConfig
from vllm_mlx.spec_decode.rejection_sampler import RejectionSampler

# ======================================================================
# TestSchedulerConfigSpecDecode
# ======================================================================


class TestSchedulerConfigSpecDecode:
    """Test SchedulerConfig spec decode fields."""

    def test_default_no_spec_decode(self):
        config = SchedulerConfig()
        assert config.speculative_method is None
        assert config.num_speculative_tokens == 3
        assert config.spec_decode_disable_batch_size is None
        assert config.draft_model_name is None

    def test_ngram_config(self):
        config = SchedulerConfig(
            speculative_method="ngram",
            num_speculative_tokens=5,
            spec_decode_disable_batch_size=8,
        )
        assert config.speculative_method == "ngram"
        assert config.num_speculative_tokens == 5
        assert config.spec_decode_disable_batch_size == 8

    def test_draft_model_config(self):
        config = SchedulerConfig(
            speculative_method="ngram",
            draft_model_name="some-draft-model",
        )
        assert config.draft_model_name == "some-draft-model"

    def test_default_num_speculative_tokens_is_three(self):
        config = SchedulerConfig()
        assert config.num_speculative_tokens == 3

    def test_spec_decode_disable_batch_size_none_by_default(self):
        config = SchedulerConfig()
        assert config.spec_decode_disable_batch_size is None


# ======================================================================
# TestCanSpecDecode
# ======================================================================


class TestCanSpecDecode:
    """Test the _can_spec_decode() guard logic."""

    def _make_scheduler_mock(self, **kwargs):
        """Create a minimal mock scheduler with spec decode fields."""
        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(speculative_method="ngram")
        scheduler._spec_decode_enabled = kwargs.get("enabled", True)
        scheduler._spec_decode_runtime = kwargs.get("runtime", MagicMock())
        scheduler.batch_generator = kwargs.get("batch_generator", MagicMock())
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.unprocessed_prompts = kwargs.get(
                "unprocessed_prompts", []
            )
        if "active_batch" in kwargs:
            scheduler.batch_generator.active_batch = kwargs["active_batch"]
        scheduler.waiting = kwargs.get("waiting", deque())
        scheduler.running = kwargs.get("running", {"req1": MagicMock()})
        scheduler._communicator = kwargs.get("communicator", None)
        scheduler.model = kwargs.get("model", MagicMock(spec=[]))
        # Remove args attr to avoid transformer check by default
        if not kwargs.get("has_model_args", False):
            del scheduler.model.args
        return scheduler

    def test_disabled_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        s = self._make_scheduler_mock(enabled=False)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_no_runtime_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        s = self._make_scheduler_mock(runtime=None)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_no_batch_generator_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        s = self._make_scheduler_mock(batch_generator=None)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_no_active_batch_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        s = self._make_scheduler_mock(active_batch=None)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_waiting_requests_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        s = self._make_scheduler_mock(waiting=deque([MagicMock()]))
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_batch_size_threshold_disables(self):
        from vllm_mlx.scheduler import Scheduler

        runtime = MagicMock()
        runtime.should_disable.return_value = True
        s = self._make_scheduler_mock(runtime=runtime)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_mamba_model_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        model = MagicMock()
        model.args.model_type = "mamba"
        s = self._make_scheduler_mock(model=model, has_model_args=True)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_jamba_model_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        model = MagicMock()
        model.args.model_type = "jamba"
        s = self._make_scheduler_mock(model=model, has_model_args=True)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_nemotron_model_returns_false(self):
        from vllm_mlx.scheduler import Scheduler

        model = MagicMock()
        model.args.model_type = "nemotron"
        s = self._make_scheduler_mock(model=model, has_model_args=True)
        result = Scheduler._can_spec_decode(s)
        assert result is False

    def test_transformer_model_returns_true(self):
        from vllm_mlx.scheduler import Scheduler

        runtime = MagicMock()
        runtime.should_disable.return_value = False
        model = MagicMock()
        model.args.model_type = "llama"
        s = self._make_scheduler_mock(runtime=runtime, model=model, has_model_args=True)
        result = Scheduler._can_spec_decode(s)
        assert result is True

    def test_model_without_args_returns_true(self):
        """A model that lacks an 'args' attribute should pass the check."""
        from vllm_mlx.scheduler import Scheduler

        runtime = MagicMock()
        runtime.should_disable.return_value = False
        s = self._make_scheduler_mock(runtime=runtime)
        # model.args has been deleted (has_model_args=False by default)
        result = Scheduler._can_spec_decode(s)
        assert result is True

    def test_all_conditions_met_returns_true(self):
        from vllm_mlx.scheduler import Scheduler

        runtime = MagicMock()
        runtime.should_disable.return_value = False
        s = self._make_scheduler_mock(runtime=runtime)
        result = Scheduler._can_spec_decode(s)
        assert result is True

    def test_empty_running_dict_uses_zero_batch_size(self):
        """When running dict is empty, batch_size=0 should still pass threshold check."""
        from vllm_mlx.scheduler import Scheduler

        runtime = MagicMock()
        runtime.should_disable.return_value = False
        s = self._make_scheduler_mock(runtime=runtime, running={})
        result = Scheduler._can_spec_decode(s)
        assert result is True


# ======================================================================
# TestInitSpecDecode
# ======================================================================


class TestInitSpecDecode:
    """Test spec decode initialization in scheduler."""

    def test_ngram_proposer_created(self):
        """Test that _init_spec_decode creates NgramProposer for 'ngram' method."""
        from vllm_mlx.scheduler import Scheduler

        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(
            speculative_method="ngram",
            num_speculative_tokens=5,
        )
        scheduler.model = MagicMock()

        Scheduler._init_spec_decode(scheduler)

        assert scheduler._spec_decode_runtime is not None
        runtime = scheduler._spec_decode_runtime
        assert isinstance(runtime, SpecDecodeRuntime)
        assert runtime.config.num_speculative_tokens == 5
        assert runtime.config.method == "ngram"

    def test_ngram_proposer_default_k(self):
        """Test that default num_speculative_tokens (3) is used."""
        from vllm_mlx.scheduler import Scheduler

        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(
            speculative_method="ngram",
            num_speculative_tokens=3,
        )
        scheduler.model = MagicMock()

        Scheduler._init_spec_decode(scheduler)

        runtime = scheduler._spec_decode_runtime
        assert runtime.config.num_speculative_tokens == 3

    def test_disable_by_batch_size_propagated(self):
        """Test that spec_decode_disable_batch_size is propagated to runtime config."""
        from vllm_mlx.scheduler import Scheduler

        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(
            speculative_method="ngram",
            num_speculative_tokens=3,
            spec_decode_disable_batch_size=16,
        )
        scheduler.model = MagicMock()

        Scheduler._init_spec_decode(scheduler)

        runtime = scheduler._spec_decode_runtime
        assert runtime.config.disable_by_batch_size == 16

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        from vllm_mlx.scheduler import Scheduler

        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(
            speculative_method="unknown_method",
            num_speculative_tokens=3,
        )
        scheduler.model = MagicMock()

        # SpecDecodeConfig.__post_init__ validates the method before
        # _init_spec_decode's own check, so we match either message.
        with pytest.raises(
            ValueError, match="(Unknown speculative method|Invalid method)"
        ):
            Scheduler._init_spec_decode(scheduler)

    def test_runtime_uses_greedy_sampler(self):
        """Test that _init_spec_decode uses a greedy RejectionSampler."""
        from vllm_mlx.scheduler import Scheduler

        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(
            speculative_method="ngram",
            num_speculative_tokens=3,
        )
        scheduler.model = MagicMock()

        Scheduler._init_spec_decode(scheduler)

        runtime = scheduler._spec_decode_runtime
        assert runtime.rejection_sampler.method == "greedy"


# ======================================================================
# TestSpecDecodeRuntimeIntegration
# ======================================================================


class TestSpecDecodeRuntimeIntegration:
    """Test the full propose-verify-accept cycle at the runtime level."""

    def test_propose_accept_greedy_full_accept(self):
        """Test that all draft tokens are accepted when model agrees."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        # Build a token sequence with repeating pattern: [1, 2, 3, 1, 2, 3, 1, 2]
        # With n=3 (default), last 2 tokens are [1, 2] (the pattern).
        # Most recent earlier match of [1, 2] is at index 3.
        # draft_start = 3 + 2 = 5, draft_end = min(5+3, 8-2) = 6
        # token_ids[5:6] = [3]
        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }

        # Propose drafts
        metadata = runtime.propose_drafts(request_states)
        draft_tokens = metadata.get_draft_tokens("req1")
        assert len(draft_tokens) > 0

        # Simulate verification where target model agrees with all drafts
        k = len(draft_tokens)
        vocab_size = 10
        target_logits_data = mx.zeros((k + 1, vocab_size))
        for i, token in enumerate(draft_tokens):
            target_logits_data = target_logits_data.at[i, token].add(10.0)
        # Bonus token at position k
        target_logits_data = target_logits_data.at[k, 5].add(10.0)
        mx.eval(target_logits_data)

        verify_result = VerifyResult()
        verify_result.request_ids = ["req1"]
        verify_result.target_logits["req1"] = target_logits_data

        # Accept
        results = runtime.accept_and_commit(verify_result, metadata)
        result = results["req1"]

        assert result.num_accepted == k  # All drafts accepted
        assert result.bonus_token == 5  # Bonus token
        assert len(result.accepted_tokens) == k + 1  # drafts + bonus
        assert result.rollback_count == 0

    def test_propose_accept_greedy_partial_accept(self):
        """Test partial acceptance when model disagrees at some point."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }

        metadata = runtime.propose_drafts(request_states)
        draft_tokens = metadata.get_draft_tokens("req1")

        if len(draft_tokens) >= 2:
            k = len(draft_tokens)
            vocab_size = 10
            target_logits_data = mx.zeros((k + 1, vocab_size))
            # First token matches
            target_logits_data = target_logits_data.at[0, draft_tokens[0]].add(10.0)
            # Second token DISAGREES -- model says token 7 instead
            target_logits_data = target_logits_data.at[1, 7].add(10.0)
            # Rest don't matter
            for i in range(2, k + 1):
                target_logits_data = target_logits_data.at[i, 0].add(10.0)
            mx.eval(target_logits_data)

            verify_result = VerifyResult()
            verify_result.request_ids = ["req1"]
            verify_result.target_logits["req1"] = target_logits_data

            results = runtime.accept_and_commit(verify_result, metadata)
            result = results["req1"]

            assert result.num_accepted == 1  # Only first draft accepted
            assert result.bonus_token == 7  # Correction token
            assert result.rollback_count == k - 1  # Remaining drafts rolled back

    def test_propose_no_drafts(self):
        """Test that short sequences produce no drafts."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3, n=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        # Very short sequence -- no n-gram patterns
        tokens = [1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }

        metadata = runtime.propose_drafts(request_states)
        draft_tokens = metadata.get_draft_tokens("req1")
        assert len(draft_tokens) == 0

    def test_stats_tracking(self):
        """Test that spec decode stats are updated correctly."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        assert runtime.stats.num_drafts == 0
        assert runtime.stats.num_draft_tokens == 0
        assert runtime.stats.num_accepted_tokens == 0

        # Run a cycle
        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }
        metadata = runtime.propose_drafts(request_states)
        draft_tokens = metadata.get_draft_tokens("req1")

        if len(draft_tokens) > 0:
            k = len(draft_tokens)
            vocab_size = 10
            target_logits_data = mx.zeros((k + 1, vocab_size))
            for i, token in enumerate(draft_tokens):
                target_logits_data = target_logits_data.at[i, token].add(10.0)
            target_logits_data = target_logits_data.at[k, 5].add(10.0)
            mx.eval(target_logits_data)

            verify_result = VerifyResult()
            verify_result.request_ids = ["req1"]
            verify_result.target_logits["req1"] = target_logits_data

            runtime.accept_and_commit(verify_result, metadata)

            assert runtime.stats.num_drafts == 1
            assert runtime.stats.num_draft_tokens == k
            assert runtime.stats.num_accepted_tokens == k  # All accepted

    def test_multi_request_propose_and_accept(self):
        """Test propose-verify-accept cycle with multiple concurrent requests."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        # Two requests with different repeating patterns
        request_states = {
            "req1": RequestState(
                request_id="req1",
                token_ids=[1, 2, 3, 1, 2, 3, 1, 2],
                batch_uid=0,
            ),
            "req2": RequestState(
                request_id="req2",
                token_ids=[5, 6, 7, 5, 6, 7, 5, 6],
                batch_uid=1,
            ),
        }

        metadata = runtime.propose_drafts(request_states)

        drafts_req1 = metadata.get_draft_tokens("req1")
        drafts_req2 = metadata.get_draft_tokens("req2")

        # Both should have some drafts due to repeating patterns
        assert len(drafts_req1) > 0
        assert len(drafts_req2) > 0

        # Build verify result for both requests (full accept for both)
        verify_result = VerifyResult()
        verify_result.request_ids = ["req1", "req2"]

        vocab_size = 10
        for rid, drafts in [("req1", drafts_req1), ("req2", drafts_req2)]:
            k = len(drafts)
            logits = mx.zeros((k + 1, vocab_size))
            for i, token in enumerate(drafts):
                logits = logits.at[i, token].add(10.0)
            logits = logits.at[k, 9].add(10.0)  # bonus token
            mx.eval(logits)
            verify_result.target_logits[rid] = logits

        results = runtime.accept_and_commit(verify_result, metadata)

        assert "req1" in results
        assert "req2" in results
        assert results["req1"].num_accepted == len(drafts_req1)
        assert results["req2"].num_accepted == len(drafts_req2)
        assert results["req1"].rollback_count == 0
        assert results["req2"].rollback_count == 0

    def test_seq_len_tracking_across_cycles(self):
        """Test that sequence length is tracked correctly across multiple cycles."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }

        metadata = runtime.propose_drafts(request_states)
        initial_seq_len = runtime.get_seq_len("req1")
        assert initial_seq_len == len(tokens)

        draft_tokens = metadata.get_draft_tokens("req1")
        if len(draft_tokens) > 0:
            k = len(draft_tokens)
            vocab_size = 10
            target_logits_data = mx.zeros((k + 1, vocab_size))
            for i, token in enumerate(draft_tokens):
                target_logits_data = target_logits_data.at[i, token].add(10.0)
            target_logits_data = target_logits_data.at[k, 5].add(10.0)
            mx.eval(target_logits_data)

            verify_result = VerifyResult()
            verify_result.request_ids = ["req1"]
            verify_result.target_logits["req1"] = target_logits_data

            runtime.accept_and_commit(verify_result, metadata)

            # seq_len should advance by num_accepted + 1 (bonus)
            expected_len = initial_seq_len + k + 1
            assert runtime.get_seq_len("req1") == expected_len


# ======================================================================
# TestResponseGeneration
# ======================================================================


class TestResponseGeneration:
    """Test response generation with stop/max-token clipping logic.

    These tests verify the token emission logic used by _step_spec_decode
    without calling the actual scheduler method. The clipping logic is
    extracted and tested in isolation.
    """

    @staticmethod
    def _emit_tokens(accepted_tokens, stop_tokens, tokens_remaining):
        """Replicate the stop/max-token clipping logic from _step_spec_decode."""
        emitted = []
        rollback = 0
        for t_idx, token in enumerate(accepted_tokens):
            if tokens_remaining <= 0:
                unemitted = len(accepted_tokens) - t_idx
                rollback += unemitted
                break
            finish_reason = None
            if token in stop_tokens:
                finish_reason = "stop"
            elif tokens_remaining == 1:
                finish_reason = "length"
            emitted.append((token, finish_reason))
            tokens_remaining -= 1
            if finish_reason is not None:
                unemitted = len(accepted_tokens) - t_idx - 1
                if unemitted > 0:
                    rollback += unemitted
                break
        return emitted, rollback

    def test_stop_token_clips(self):
        """Test that stop tokens correctly terminate response generation."""
        result = AcceptResult(
            accepted_tokens=[1, 2, 3],  # token 2 is a stop token
            num_accepted=2,
            bonus_token=3,
            rollback_count=0,
        )

        stop_tokens = {2}
        tokens_remaining = 10

        emitted, rollback = self._emit_tokens(
            result.accepted_tokens, stop_tokens, tokens_remaining
        )

        assert len(emitted) == 2  # [1, 2(stop)]
        assert emitted[-1] == (2, "stop")
        assert rollback == 1  # token 3 was not emitted

    def test_max_tokens_clips(self):
        """Test that max_tokens limit clips response correctly."""
        result = AcceptResult(
            accepted_tokens=[10, 20, 30, 40],
            num_accepted=3,
            bonus_token=40,
            rollback_count=0,
        )

        tokens_remaining = 2  # Only 2 tokens left before max_tokens
        stop_tokens = set()

        emitted, rollback = self._emit_tokens(
            result.accepted_tokens, stop_tokens, tokens_remaining
        )

        assert len(emitted) == 2  # [10(None), 20("length")]
        assert emitted[0] == (10, None)
        assert emitted[1] == (20, "length")
        assert rollback == 2  # tokens 30, 40 not emitted

    def test_stop_token_at_first_position(self):
        """Test stop token at the very first position."""
        result = AcceptResult(
            accepted_tokens=[99, 10, 20],
            num_accepted=2,
            bonus_token=20,
            rollback_count=0,
        )

        stop_tokens = {99}
        tokens_remaining = 10

        emitted, rollback = self._emit_tokens(
            result.accepted_tokens, stop_tokens, tokens_remaining
        )

        assert len(emitted) == 1
        assert emitted[0] == (99, "stop")
        assert rollback == 2  # tokens 10, 20 not emitted

    def test_zero_tokens_remaining(self):
        """Test that zero tokens_remaining produces no output."""
        result = AcceptResult(
            accepted_tokens=[1, 2, 3],
            num_accepted=2,
            bonus_token=3,
            rollback_count=0,
        )

        emitted, rollback = self._emit_tokens(result.accepted_tokens, set(), 0)

        assert len(emitted) == 0
        assert rollback == 3  # all tokens unemitted

    def test_no_stop_no_limit_emits_all(self):
        """Test that without stop tokens and with plenty of budget, all tokens emit."""
        result = AcceptResult(
            accepted_tokens=[1, 2, 3, 4],
            num_accepted=3,
            bonus_token=4,
            rollback_count=0,
        )

        emitted, rollback = self._emit_tokens(result.accepted_tokens, set(), 100)

        assert len(emitted) == 4
        assert all(fr is None for _, fr in emitted)
        assert rollback == 0

    def test_length_finish_reason_at_exact_boundary(self):
        """Test that 'length' finish reason fires at exactly 1 token remaining."""
        result = AcceptResult(
            accepted_tokens=[10, 20, 30],
            num_accepted=2,
            bonus_token=30,
            rollback_count=0,
        )

        emitted, rollback = self._emit_tokens(result.accepted_tokens, set(), 1)

        assert len(emitted) == 1
        assert emitted[0] == (10, "length")
        assert rollback == 2

    def test_stop_token_takes_priority_over_length(self):
        """If a stop token appears at the last-remaining position, stop takes priority."""
        # With tokens_remaining=1 and the first token being a stop token,
        # stop should take priority
        result = AcceptResult(
            accepted_tokens=[42, 10, 20],
            num_accepted=2,
            bonus_token=20,
            rollback_count=0,
        )

        emitted, rollback = self._emit_tokens(result.accepted_tokens, {42}, 1)

        assert len(emitted) == 1
        assert emitted[0] == (42, "stop")
        assert rollback == 2


# ======================================================================
# TestPhase4aDeferredItems
# ======================================================================


class TestPhase4aDeferredItems:
    """Test deferred items from Phase 4a: stochastic+zero-draft handling."""

    def test_stochastic_zero_draft_sampler(self):
        """Test stochastic rejection with zero draft tokens at the sampler level."""
        sampler = RejectionSampler(method="stochastic")

        # Zero drafts: target_logits shape (1, 1, vocab) for the bonus only
        vocab_size = 10
        target_logits = mx.zeros((1, 1, vocab_size))
        target_logits = target_logits.at[0, 0, 5].add(10.0)
        mx.eval(target_logits)

        # Empty draft logits: shape (1, 0, vocab)
        draft_logits = mx.zeros((1, 0, vocab_size))

        result = sampler(
            target_logits=target_logits,
            draft_token_ids=[[]],
            draft_logits=draft_logits,
        )

        assert result.num_accepted[0] == 0
        assert result.bonus_token_ids[0] is not None
        assert len(result.accepted_token_ids[0]) == 0

    def test_runtime_stochastic_zero_draft(self):
        """Test runtime handles stochastic + zero-draft via empty draft_logits."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = MagicMock()
        sampler = RejectionSampler(method="stochastic")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        # Simulate zero drafts for a request
        metadata = SpecDecodeMetadata()
        metadata.add_request("req1", [])  # No draft tokens

        vocab_size = 10
        # Shape (1, vocab_size): the runtime will expand_dims to (1, 1, vocab)
        # which is what the sampler expects for zero-draft: (batch=1, k+1=1, vocab)
        target_logits = mx.zeros((1, vocab_size))
        target_logits = target_logits.at[0, 5].add(10.0)
        mx.eval(target_logits)

        verify_result = VerifyResult()
        verify_result.request_ids = ["req1"]
        verify_result.target_logits["req1"] = target_logits

        # This should NOT crash
        results = runtime.accept_and_commit(verify_result, metadata)
        result = results["req1"]
        assert result.num_accepted == 0
        assert result.bonus_token is not None

    def test_runtime_greedy_zero_draft(self):
        """Test runtime handles greedy + zero-draft correctly."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = MagicMock()
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        metadata = SpecDecodeMetadata()
        metadata.add_request("req1", [])

        vocab_size = 10
        # Shape (1, vocab_size): runtime will expand_dims to (1, 1, vocab)
        # which is (batch=1, k+1=1, vocab) as the sampler expects
        target_logits = mx.zeros((1, vocab_size))
        target_logits = target_logits.at[0, 8].add(10.0)
        mx.eval(target_logits)

        verify_result = VerifyResult()
        verify_result.request_ids = ["req1"]
        verify_result.target_logits["req1"] = target_logits

        results = runtime.accept_and_commit(verify_result, metadata)
        result = results["req1"]
        assert result.num_accepted == 0
        assert result.bonus_token == 8
        assert result.accepted_tokens == [8]
        assert result.rollback_count == 0

    def test_stochastic_zero_draft_bonus_token_in_valid_range(self):
        """Test that stochastic zero-draft produces a token within vocab range."""
        sampler = RejectionSampler(method="stochastic")

        vocab_size = 100
        target_logits = mx.zeros((1, 1, vocab_size))
        # Make token 42 very likely
        target_logits = target_logits.at[0, 0, 42].add(100.0)
        mx.eval(target_logits)

        draft_logits = mx.zeros((1, 0, vocab_size))

        result = sampler(
            target_logits=target_logits,
            draft_token_ids=[[]],
            draft_logits=draft_logits,
        )

        bonus = result.bonus_token_ids[0]
        assert bonus is not None
        assert 0 <= bonus < vocab_size


# ======================================================================
# TestSpecDecodeConfigValidation
# ======================================================================


class TestSpecDecodeConfigValidation:
    """Test SpecDecodeConfig validation for integration correctness."""

    def test_valid_ngram_config(self):
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=5)
        assert config.method == "ngram"
        assert config.num_speculative_tokens == 5

    def test_disable_by_batch_size_propagation(self):
        config = SpecDecodeConfig(
            method="ngram",
            num_speculative_tokens=3,
            disable_by_batch_size=16,
        )
        assert config.disable_by_batch_size == 16

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid method"):
            SpecDecodeConfig(method="invalid", num_speculative_tokens=3)

    def test_zero_speculative_tokens_raises(self):
        with pytest.raises(ValueError):
            SpecDecodeConfig(method="ngram", num_speculative_tokens=0)


# ======================================================================
# TestRollbackIntegration
# ======================================================================


class TestRollbackIntegration:
    """Test rollback behavior in integration scenarios."""

    def test_rollback_after_partial_accept(self):
        """Full cycle: propose -> verify -> partial accept -> rollback."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }

        metadata = runtime.propose_drafts(request_states)
        draft_tokens = metadata.get_draft_tokens("req1")

        if len(draft_tokens) > 0:
            k = len(draft_tokens)
            vocab_size = 10

            # All rejected at position 0
            target_logits_data = mx.zeros((k + 1, vocab_size))
            # Position 0 disagrees
            target_logits_data = target_logits_data.at[0, 9].add(10.0)
            for i in range(1, k + 1):
                target_logits_data = target_logits_data.at[i, 0].add(10.0)
            mx.eval(target_logits_data)

            verify_result = VerifyResult()
            verify_result.request_ids = ["req1"]
            verify_result.target_logits["req1"] = target_logits_data

            results = runtime.accept_and_commit(verify_result, metadata)
            result = results["req1"]

            assert result.num_accepted == 0
            assert result.rollback_count == k

            # Rollback should not crash
            runtime.rollback(
                request_ids=["req1"],
                rollback_counts={"req1": result.rollback_count},
            )

    def test_rollback_with_zero_count(self):
        """Rollback with zero count is a no-op."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = MagicMock()
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)
        runtime._seq_lens["req1"] = 10

        # Should not crash or modify anything
        runtime.rollback(
            request_ids=["req1"],
            rollback_counts={"req1": 0},
        )

        assert runtime.get_seq_len("req1") == 10

    def test_remove_request_after_completion(self):
        """Test that removing a request cleans up internal state."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = MagicMock()
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)
        runtime._seq_lens["req1"] = 50
        runtime._seq_lens["req2"] = 100

        runtime.remove_request("req1")

        assert runtime.get_seq_len("req1") is None
        assert runtime.get_seq_len("req2") == 100


# ======================================================================
# TestShouldDisableIntegration
# ======================================================================


class TestShouldDisableIntegration:
    """Test batch size threshold logic."""

    def test_no_threshold_never_disables(self):
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = MagicMock()
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        assert runtime.should_disable(0) is False
        assert runtime.should_disable(1) is False
        assert runtime.should_disable(100) is False
        assert runtime.should_disable(10000) is False

    def test_threshold_disables_at_limit(self):
        config = SpecDecodeConfig(
            method="ngram", num_speculative_tokens=3, disable_by_batch_size=8
        )
        proposer = MagicMock()
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        assert runtime.should_disable(7) is False
        assert runtime.should_disable(8) is True
        assert runtime.should_disable(9) is True

    def test_threshold_boundary(self):
        config = SpecDecodeConfig(
            method="ngram", num_speculative_tokens=3, disable_by_batch_size=1
        )
        proposer = MagicMock()
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        assert runtime.should_disable(0) is False
        assert runtime.should_disable(1) is True


# ======================================================================
# TestStatsIntegration
# ======================================================================


class TestStatsIntegration:
    """Test statistics tracking through the full pipeline."""

    def test_stats_acceptance_rate_after_cycle(self):
        """Test acceptance rate computation after a full cycle."""
        config = SpecDecodeConfig(method="ngram", num_speculative_tokens=3)
        proposer = NgramProposer(NgramProposerConfig(num_speculative_tokens=3))
        sampler = RejectionSampler(method="greedy")
        model = MagicMock()

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        tokens = [1, 2, 3, 1, 2, 3, 1, 2]
        request_states = {
            "req1": RequestState(request_id="req1", token_ids=tokens, batch_uid=0),
        }

        metadata = runtime.propose_drafts(request_states)
        draft_tokens = metadata.get_draft_tokens("req1")

        if len(draft_tokens) > 0:
            k = len(draft_tokens)
            vocab_size = 10
            target_logits_data = mx.zeros((k + 1, vocab_size))

            # Accept all drafts
            for i, token in enumerate(draft_tokens):
                target_logits_data = target_logits_data.at[i, token].add(10.0)
            target_logits_data = target_logits_data.at[k, 5].add(10.0)
            mx.eval(target_logits_data)

            verify_result = VerifyResult()
            verify_result.request_ids = ["req1"]
            verify_result.target_logits["req1"] = target_logits_data

            runtime.accept_and_commit(verify_result, metadata)

            assert runtime.stats.acceptance_rate() == 1.0
            assert runtime.stats.mean_accepted_length() == k

    def test_stats_reset(self):
        """Test that stats can be reset."""
        stats = SpecDecodeStats()
        stats.update(
            num_drafted=3,
            num_accepted=2,
            position_accepted=[True, True, False],
        )

        assert stats.num_drafts == 1
        stats.reset()
        assert stats.num_drafts == 0
        assert stats.num_draft_tokens == 0
        assert stats.num_accepted_tokens == 0

    def test_stats_per_position_tracking(self):
        """Test per-position acceptance rate tracking."""
        stats = SpecDecodeStats()

        # Round 1: position 0 accepted, positions 1 and 2 rejected
        stats.update(
            num_drafted=3,
            num_accepted=1,
            position_accepted=[True, False, False],
        )

        # Round 2: all accepted
        stats.update(
            num_drafted=3,
            num_accepted=3,
            position_accepted=[True, True, True],
        )

        rates = stats.acceptance_rate_per_position
        assert len(rates) == 3
        assert rates[0] == 1.0  # 2/2
        assert rates[1] == 0.5  # 1/2
        assert rates[2] == 0.5  # 1/2


# ======================================================================
# TestCanSpecDecodeTP
# ======================================================================


class TestCanSpecDecodeTP:
    """Test that _can_spec_decode allows spec decode under TP (Phase 5)."""

    def _make_scheduler_mock(self, **kwargs):
        """Create a minimal mock scheduler with spec decode fields."""
        scheduler = MagicMock()
        scheduler.config = SchedulerConfig(speculative_method="ngram")
        scheduler._spec_decode_enabled = kwargs.get("enabled", True)
        scheduler._spec_decode_runtime = kwargs.get("runtime", MagicMock())
        scheduler.batch_generator = kwargs.get("batch_generator", MagicMock())
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.unprocessed_prompts = kwargs.get(
                "unprocessed_prompts", []
            )
        if "active_batch" in kwargs:
            scheduler.batch_generator.active_batch = kwargs["active_batch"]
        scheduler.waiting = kwargs.get("waiting", deque())
        scheduler.running = kwargs.get("running", {"req1": MagicMock()})
        scheduler._communicator = kwargs.get("communicator", None)
        scheduler.model = kwargs.get("model", MagicMock(spec=[]))
        if not kwargs.get("has_model_args", False):
            del scheduler.model.args
        return scheduler

    def test_tp_distributed_allows_spec_decode(self):
        """Phase 5: TP mode should NOT disable spec decode anymore."""
        from vllm_mlx.scheduler import Scheduler

        comm = MagicMock()
        comm.is_distributed = True
        runtime = MagicMock()
        runtime.should_disable.return_value = False

        s = self._make_scheduler_mock(communicator=comm, runtime=runtime)
        result = Scheduler._can_spec_decode(s)
        assert result is True

    def test_tp_distributed_with_threshold_still_disables(self):
        """TP mode with batch size over threshold should still disable."""
        from vllm_mlx.scheduler import Scheduler

        comm = MagicMock()
        comm.is_distributed = True
        runtime = MagicMock()
        runtime.should_disable.return_value = True

        s = self._make_scheduler_mock(communicator=comm, runtime=runtime)
        result = Scheduler._can_spec_decode(s)
        assert result is False


# NOTE: TestWorkerSpecDecodeStep tests moved to distributed TP PR
# (requires vllm_mlx.distributed and vllm_mlx.distributed_launcher)
