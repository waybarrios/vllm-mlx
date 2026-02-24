# SPDX-License-Identifier: Apache-2.0
"""
Speculative decoding runtime: orchestrates draft-verify-accept cycles.

This module provides the central runtime abstraction for speculative decoding.
It coordinates the proposer (draft token generation), target model verification,
rejection sampling (accept/reject decisions), and KV cache rollback for
rejected positions.

The runtime operates on a per-step basis:
    1. propose_drafts()  — generate k draft tokens per request via the proposer
    2. verify_forward()  — run target model on draft tokens to obtain logits
    3. accept_and_commit() — rejection-sample to decide which drafts to keep
    4. rollback()         — adjust KV cache state for rejected positions

Phase 4a scope: defines the interface and core logic. Actual model forward
pass integration and scheduler wiring happen in Phase 4b.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from .metadata import SpecDecodeConfig, SpecDecodeMetadata
from .metrics import SpecDecodeStats
from .proposer import BaseProposer, ProposalContext

if TYPE_CHECKING:
    from .rejection_sampler import RejectionResult, RejectionSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supporting data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestState:
    """
    Minimal state needed by the runtime per request.

    Attributes:
        request_id: Unique identifier for the request.
        token_ids: Full token sequence (prompt + generated tokens so far).
        batch_uid: The UID assigned by the BatchGenerator for this request.
    """

    request_id: str
    token_ids: list[int]
    batch_uid: int
    hidden_states: Any | None = None


@dataclass
class VerifyResult:
    """
    Result of target model verification forward pass.

    Contains the logits produced by the target model for each request's
    draft token positions (k verification positions + 1 bonus position).

    Attributes:
        target_logits: Mapping from request_id to the target model's logits.
            Each value is an mx.array of shape (k+1, vocab_size).
        request_ids: Ordered list of request IDs that were verified.
    """

    target_logits: dict[str, Any] = field(default_factory=dict)
    draft_logits: dict[str, Any] = field(default_factory=dict)
    request_ids: list[str] = field(default_factory=list)


@dataclass
class AcceptResult:
    """
    Result of rejection sampling for a single request.

    Attributes:
        accepted_tokens: Tokens to commit (accepted drafts + correction/bonus).
        num_accepted: Number of accepted DRAFT tokens (not counting bonus).
        bonus_token: Bonus token appended when all drafts are accepted, or
            None if at least one draft was rejected.
        rollback_count: Number of KV cache positions to rollback (rejected
            draft positions that were speculatively computed).
    """

    accepted_tokens: list[int]
    num_accepted: int
    bonus_token: int | None
    rollback_count: int


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class SpecDecodeRuntime:
    """
    Orchestrates the speculative decoding draft-verify-accept cycle.

    This class holds references to the proposer, rejection sampler, and
    configuration, and exposes a step-by-step API that the scheduler can
    drive. Per-request state (sequence lengths, active draft info) is
    tracked internally.

    Typical usage from the scheduler (Phase 4b)::

        runtime = SpecDecodeRuntime(model, proposer, sampler, config)

        # Each decoding step:
        if not runtime.should_disable(batch_size):
            metadata = runtime.propose_drafts(request_states)
            verify   = runtime.verify_forward(req_ids, metadata.draft_token_ids, seq_lens)
            results  = runtime.accept_and_commit(verify, metadata)
            runtime.rollback(req_ids, {r: v.rollback_count for r, v in results.items()})
    """

    def __init__(
        self,
        model: Any,
        proposer: BaseProposer,
        rejection_sampler: RejectionSampler,
        config: SpecDecodeConfig,
    ) -> None:
        """
        Initialize the speculative decoding runtime.

        Args:
            model: The target model (nn.Module-like). Used in verify_forward
                to obtain target logits. Actual wiring happens in Phase 4b.
            proposer: The draft token proposer instance.
            rejection_sampler: The rejection sampler used to accept/reject
                draft tokens based on target model logits.
            config: Speculative decoding configuration.
        """
        self.model = model
        self.proposer = proposer
        self.rejection_sampler = rejection_sampler
        self.config = config

        # Per-request tracked sequence lengths (request_id -> seq_len).
        # Updated after accept_and_commit and rollback operations.
        self._seq_lens: dict[str, int] = {}

        # Statistics tracker
        self._stats = SpecDecodeStats()
        # Set the rolling window size from config
        self._stats.set_window_size(config.auto_disable_window)

        # Auto-disable state
        self._auto_disabled: bool = False
        self._steps_since_disable: int = 0
        self._auto_disable_logged: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_drafts(
        self,
        request_states: dict[str, RequestState],
    ) -> SpecDecodeMetadata:
        """
        Generate draft tokens for all active requests using the proposer.

        Iterates over each request and calls the proposer to generate up to
        ``config.num_speculative_tokens`` draft tokens based on the request's
        current token sequence.

        If the recent acceptance rate has dropped below the configured
        threshold, speculation is temporarily auto-disabled.  Every
        ``auto_disable_window`` steps after disabling, a single probe round
        is executed to check whether conditions have improved.

        Args:
            request_states: Mapping of request_id -> RequestState containing
                the full token sequence and batch UID per request.

        Returns:
            SpecDecodeMetadata populated with draft tokens per request.
            Returns empty metadata when auto-disabled (no drafts proposed).
        """
        # --- Auto-disable based on acceptance rate feedback ---
        threshold = self.config.auto_disable_threshold
        window = self.config.auto_disable_window

        if self._auto_disabled:
            self._steps_since_disable += 1
            # Periodic re-evaluation: try one probe round
            if self._steps_since_disable >= window:
                self._steps_since_disable = 0
                # Allow this round to proceed as a probe
                logger.info(
                    "Spec decode auto-disable: running probe round to re-evaluate "
                    "(recent acceptance rate=%.3f, threshold=%.3f)",
                    self._stats.recent_acceptance_rate(),
                    threshold,
                )
            else:
                # Still disabled — return empty metadata
                return SpecDecodeMetadata()
        elif threshold > 0.0 and self._stats.should_auto_disable(threshold):
            # Transition to disabled state
            self._auto_disabled = True
            self._steps_since_disable = 0
            if not self._auto_disable_logged:
                logger.info(
                    "Spec decode auto-disabled: recent acceptance rate %.3f "
                    "below threshold %.3f (window=%d). Will probe every %d steps.",
                    self._stats.recent_acceptance_rate(),
                    threshold,
                    window,
                    window,
                )
                self._auto_disable_logged = True
            return SpecDecodeMetadata()

        metadata = SpecDecodeMetadata()
        k = self.config.num_speculative_tokens

        for request_id, state in request_states.items():
            self.proposer.reset()
            ctx = ProposalContext(
                request_id=request_id,
                token_ids=state.token_ids,
                k=k,
                hidden_states=state.hidden_states,
            )
            proposal = self.proposer.propose_with_context(ctx)
            draft_tokens = proposal.token_ids
            metadata.add_request(request_id, draft_tokens)

            # Track current sequence length for later rollback calculations
            self._seq_lens[request_id] = len(state.token_ids)

        return metadata

    def verify_forward(
        self,
        request_ids: list[str],
        draft_tokens: dict[str, list[int]],
        seq_lens: dict[str, int],
    ) -> VerifyResult:
        """
        Run target model forward pass on draft tokens to get verification logits.

        For each request, the draft tokens are fed to the target model. The
        model produces logits for all (k+1) positions: k verification logits
        (one per draft token) plus 1 bonus/correction logit for the position
        after the last draft token.

        Args:
            request_ids: List of request IDs to verify.
            draft_tokens: Mapping of request_id -> draft token list.
            seq_lens: Mapping of request_id -> current sequence length
                (before drafts were appended).

        Returns:
            VerifyResult containing target logits per request.

        Raises:
            NotImplementedError: Always, in Phase 4a. The actual model
                forward pass will be wired in Phase 4b when scheduler
                integration is complete.

        NOTE:
            Phase 4b will replace this with a real batched model forward
            that feeds all draft tokens through the target model and
            extracts per-position logits.
        """
        raise NotImplementedError(
            "verify_forward requires target model integration (Phase 4b). "
            "This method defines the interface; the actual implementation "
            "will be wired when the scheduler integration is complete."
        )

    def accept_and_commit(
        self,
        verify_result: VerifyResult,
        draft_metadata: SpecDecodeMetadata,
    ) -> dict[str, AcceptResult]:
        """
        Run rejection sampling and commit accepted tokens.

        For each request in the verify result, calls the rejection sampler
        with the target logits and draft tokens to decide which drafts to
        accept. Computes rollback counts and optional bonus tokens.

        Args:
            verify_result: The result from verify_forward containing
                target model logits per request.
            draft_metadata: The draft token metadata from propose_drafts.

        Returns:
            Mapping of request_id -> AcceptResult with accepted tokens,
            acceptance count, optional bonus token, and rollback count.
        """
        results: dict[str, AcceptResult] = {}

        for request_id in verify_result.request_ids:
            draft_token_ids = draft_metadata.get_draft_tokens(request_id)
            num_drafted = len(draft_token_ids)

            if num_drafted == 0:
                # Still need to advance by one token — use target logits position 0
                target_logits_req = verify_result.target_logits.get(request_id)
                if target_logits_req is not None:
                    batched_logits = mx.expand_dims(target_logits_req, axis=0)
                    # For stochastic mode, pass empty draft logits (1, 0, vocab)
                    # to avoid ValueError from missing draft_logits.
                    empty_draft_logits = None
                    if self.rejection_sampler.method == "stochastic":
                        vocab_size = target_logits_req.shape[-1]
                        empty_draft_logits = mx.zeros((1, 0, vocab_size))
                    rejection_result = self.rejection_sampler(
                        target_logits=batched_logits,
                        draft_token_ids=[[]],
                        draft_logits=empty_draft_logits,
                    )
                    bonus = rejection_result.bonus_token_ids[0]
                    committed = [bonus] if bonus is not None else []
                    results[request_id] = AcceptResult(
                        accepted_tokens=committed,
                        num_accepted=0,
                        bonus_token=bonus,
                        rollback_count=0,
                    )
                    if request_id in self._seq_lens and committed:
                        self._seq_lens[request_id] += len(committed)
                else:
                    results[request_id] = AcceptResult(
                        accepted_tokens=[],
                        num_accepted=0,
                        bonus_token=None,
                        rollback_count=0,
                    )
                continue

            target_logits = verify_result.target_logits.get(request_id)

            # Run rejection sampling — the sampler expects batch inputs,
            # so wrap the single-request data in a batch dimension.
            batched_logits = mx.expand_dims(target_logits, axis=0)

            # Get draft logits if available (needed for stochastic rejection)
            draft_logits_for_req = verify_result.draft_logits.get(request_id)
            batched_draft_logits = None
            if draft_logits_for_req is not None:
                batched_draft_logits = mx.expand_dims(draft_logits_for_req, axis=0)

            rejection_result: RejectionResult = self.rejection_sampler(
                target_logits=batched_logits,
                draft_token_ids=[draft_token_ids],
                draft_logits=batched_draft_logits,
            )

            # Unwrap batch dimension (single request at index 0)
            num_accepted = rejection_result.num_accepted[0]
            accepted_draft_tokens = draft_token_ids[:num_accepted]

            # The correction/bonus token from the rejection sampler
            bonus_token: int | None = rejection_result.bonus_token_ids[0]

            # Build the final committed token list
            committed_tokens = list(accepted_draft_tokens)
            if bonus_token is not None:
                committed_tokens.append(bonus_token)

            # Rollback count: draft positions that were NOT accepted.
            # These KV cache entries need to be invalidated.
            rollback_count = num_drafted - num_accepted

            results[request_id] = AcceptResult(
                accepted_tokens=committed_tokens,
                num_accepted=num_accepted,
                bonus_token=bonus_token,
                rollback_count=rollback_count,
            )

            # Update per-request sequence length tracking
            if request_id in self._seq_lens:
                self._seq_lens[request_id] += len(committed_tokens)

            # Update statistics
            position_accepted = [
                i < num_accepted for i in range(num_drafted)
            ]
            self._stats.update(
                num_drafted=num_drafted,
                num_accepted=num_accepted,
                position_accepted=position_accepted,
            )

        # Check if we should re-enable after a probe round
        self._check_auto_reenable()

        return results

    def rollback(
        self,
        request_ids: list[str],
        rollback_counts: dict[str, int],
    ) -> None:
        """
        Rollback KV cache for rejected draft tokens.

        Uses lazy rollback: adjusts tracked sequence lengths rather than
        physically removing KV cache entries. The overwritten positions will
        be recomputed during the next forward pass when new tokens are
        inserted at those positions.

        Args:
            request_ids: Requests needing rollback.
            rollback_counts: Mapping of request_id -> number of positions
                to rollback in the KV cache.

        NOTE:
            Phase 4a tracks state changes only. Actual KV cache manipulation
            (e.g., trimming cache tensors) will be implemented in Phase 4b.
        """
        for request_id in request_ids:
            count = rollback_counts.get(request_id, 0)
            if count <= 0:
                continue
            # Phase 4a: only log rollback intent.
            # Phase 4b will implement actual KV cache trimming.
            logger.debug(
                "Rollback %d positions for request %s (KV cache trim deferred to Phase 4b)",
                count,
                request_id,
            )

    def should_disable(self, batch_size: int) -> bool:
        """
        Check if speculative decoding should be disabled for the current batch.

        When the batch size exceeds the configured threshold, speculation
        adds more overhead than benefit because the GPU is already saturated
        with real requests.

        Args:
            batch_size: Current number of active requests in the batch.

        Returns:
            True if speculative decoding should be disabled, False otherwise.
        """
        if self.config.disable_by_batch_size is None:
            return False
        return batch_size >= self.config.disable_by_batch_size

    @property
    def stats(self) -> SpecDecodeStats:
        """Return the accumulated speculative decoding statistics."""
        return self._stats

    @property
    def auto_disabled(self) -> bool:
        """Whether speculative decoding is currently auto-disabled."""
        return self._auto_disabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_auto_reenable(self) -> None:
        """Re-enable speculation if a probe round shows improved acceptance.

        Called after ``accept_and_commit`` updates statistics.  If the
        runtime was auto-disabled and the most recent acceptance rate now
        meets the threshold, speculation is re-enabled.
        """
        if not self._auto_disabled:
            return
        threshold = self.config.auto_disable_threshold
        rate = self._stats.recent_acceptance_rate()
        if rate >= threshold:
            self._auto_disabled = False
            self._steps_since_disable = 0
            self._auto_disable_logged = False
            logger.info(
                "Spec decode re-enabled: recent acceptance rate %.3f "
                ">= threshold %.3f",
                rate,
                threshold,
            )

    def get_seq_len(self, request_id: str) -> int | None:
        """
        Get the tracked sequence length for a request.

        Args:
            request_id: The request ID to look up.

        Returns:
            The current sequence length, or None if the request is not tracked.
        """
        return self._seq_lens.get(request_id)

    def remove_request(self, request_id: str) -> None:
        """
        Remove a completed or cancelled request from internal tracking.

        Args:
            request_id: The request ID to remove.
        """
        self._seq_lens.pop(request_id, None)
        if hasattr(self.proposer, 'remove_request'):
            self.proposer.remove_request(request_id)
