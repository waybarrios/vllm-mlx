# SPDX-License-Identifier: Apache-2.0
"""
Rejection sampling for speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class RejectionResult:
    """
    Result of rejection sampling for a batch.

    Attributes:
        accepted_token_ids: Per-request list of accepted tokens.
        num_accepted: Per-request count of accepted tokens.
        bonus_token_ids: Per-request bonus token (or None).
    """

    accepted_token_ids: list[list[int]]  # per-request list of accepted tokens
    num_accepted: list[int]  # per-request count of accepted tokens
    bonus_token_ids: list[int | None]  # per-request bonus token (or None)


class RejectionSampler:
    """
    Rejection sampler for speculative decoding.

    Supports greedy and stochastic rejection methods.
    """

    def __init__(self, method: str = "greedy") -> None:
        """
        Initialize the rejection sampler.

        Args:
            method: Rejection method. Must be "greedy" or "stochastic".

        Raises:
            ValueError: If method is not supported.
        """
        if method not in {"greedy", "stochastic"}:
            raise ValueError(
                f"Invalid rejection method '{method}'. "
                "Must be 'greedy' or 'stochastic'."
            )
        self.method = method

    def __call__(
        self,
        target_logits: mx.array,
        draft_token_ids: list[list[int]],
        draft_logits: mx.array | None = None,
    ) -> RejectionResult:
        """
        Run rejection sampling for a batch of draft tokens.

        Args:
            target_logits: Target-model logits with shape (batch, k + 1, vocab).
            draft_token_ids: Per-request draft token IDs.
            draft_logits: Draft-model logits with shape (batch, k, vocab).
                Required for stochastic rejection.

        Returns:
            A RejectionResult containing accepted tokens and bonus tokens.
        """
        if self.method == "greedy":
            return self._greedy_rejection(target_logits, draft_token_ids)
        return self._stochastic_rejection(target_logits, draft_token_ids, draft_logits)

    def _greedy_rejection(
        self,
        target_logits: mx.array,
        draft_token_ids: list[list[int]],
    ) -> RejectionResult:
        """Greedy rejection using vectorized argmax matching."""
        batch_size = len(draft_token_ids)

        # Compute all argmax target tokens at once: shape (batch, k+1)
        target_tokens = mx.argmax(target_logits, axis=-1)

        # Find max draft length for padding
        draft_lengths = [len(d) for d in draft_token_ids]
        max_k = max(draft_lengths) if draft_lengths else 0

        if max_k == 0:
            # All requests have empty drafts — bonus is target_tokens[:, 0]
            bonus_arr = target_tokens[:, 0]
            mx.eval(bonus_arr)
            bonus_list = bonus_arr.tolist()
            return RejectionResult(
                accepted_token_ids=[[] for _ in range(batch_size)],
                num_accepted=[0] * batch_size,
                bonus_token_ids=bonus_list,
            )

        # Build padded draft array: shape (batch, max_k)
        # Use -1 as pad value (will never match any argmax token)
        draft_padded = [d + [-1] * (max_k - len(d)) for d in draft_token_ids]
        draft_arr = mx.array(draft_padded)  # (batch, max_k)

        # Compare: match[b, i] = (target_tokens[b, i] == draft_arr[b, i])
        match_mask = target_tokens[:, :max_k] == draft_arr  # (batch, max_k)

        # Mask out positions beyond each request's actual draft length
        # Create a length mask: valid[b, i] = (i < draft_lengths[b])
        positions = mx.arange(max_k)  # (max_k,)
        length_arr = mx.array(draft_lengths)  # (batch,)
        valid_mask = positions[None, :] < length_arr[:, None]  # (batch, max_k)

        # Positions beyond draft length should not match (force False)
        match_mask = match_mask & valid_mask

        # Cumulative product to find first rejection point
        # cumprod_mask[b, i] = 1 if all tokens 0..i matched, 0 otherwise
        cumprod_mask = mx.cumprod(match_mask.astype(mx.int32), axis=1)

        # Number accepted per request = sum of cumprod mask
        num_accepted_arr = mx.sum(cumprod_mask, axis=1)  # (batch,)

        # Bonus token: target_tokens[b, num_accepted[b]]
        # Use gather indexing
        bonus_indices = num_accepted_arr  # (batch,)
        # target_tokens shape is (batch, k+1), index along axis=1
        bonus_arr = mx.take_along_axis(
            target_tokens, bonus_indices[:, None], axis=1
        ).squeeze(
            1
        )  # (batch,)

        # Single eval call for all results
        mx.eval(target_tokens, num_accepted_arr, bonus_arr, cumprod_mask)

        # Convert to Python lists
        num_accepted_list = num_accepted_arr.tolist()
        bonus_list = bonus_arr.tolist()

        # Build accepted_token_ids from the draft tokens (they are known to match)
        accepted_token_ids: list[list[int]] = []
        for b in range(batch_size):
            n = num_accepted_list[b]
            accepted_token_ids.append(draft_token_ids[b][:n])

        return RejectionResult(
            accepted_token_ids=accepted_token_ids,
            num_accepted=num_accepted_list,
            bonus_token_ids=bonus_list,
        )

    def _stochastic_rejection(
        self,
        target_logits: mx.array,
        draft_token_ids: list[list[int]],
        draft_logits: mx.array | None,
    ) -> RejectionResult:
        """Stochastic rejection sampling with vectorized acceptance checks."""
        if draft_logits is None:
            raise ValueError("draft_logits must be provided for stochastic rejection.")

        batch_size = len(draft_token_ids)
        draft_lengths = [len(d) for d in draft_token_ids]
        max_k = max(draft_lengths) if draft_lengths else 0

        # p_target covers k+1 positions, p_draft covers k positions
        p_target = mx.softmax(target_logits, axis=-1)  # (batch, k+1, vocab)
        p_draft = mx.softmax(draft_logits, axis=-1)  # (batch, k, vocab)

        if max_k == 0:
            # All requests have empty drafts — sample bonus from target
            bonus_sampled = mx.random.categorical(target_logits[:, 0, :])  # (batch,)
            mx.eval(bonus_sampled)
            bonus_list = bonus_sampled.tolist()
            return RejectionResult(
                accepted_token_ids=[[] for _ in range(batch_size)],
                num_accepted=[0] * batch_size,
                bonus_token_ids=bonus_list,
            )

        eps = 1e-10

        # Build padded draft token index array: shape (batch, max_k)
        # Pad with 0 (a valid but irrelevant token index for masked positions)
        draft_padded = [d + [0] * (max_k - len(d)) for d in draft_token_ids]
        draft_arr = mx.array(draft_padded)  # (batch, max_k)

        # Gather target and draft probabilities at draft token positions
        # p_target[:, :max_k, :] has shape (batch, max_k, vocab)
        # We need p_target[b, i, draft_arr[b, i]] for each b, i
        draft_idx_expanded = draft_arr[:, :, None]  # (batch, max_k, 1)
        p_t_at_draft = mx.take_along_axis(
            p_target[:, :max_k, :], draft_idx_expanded, axis=2
        ).squeeze(
            2
        )  # (batch, max_k)
        p_d_at_draft = mx.take_along_axis(
            p_draft[:, :max_k, :], draft_idx_expanded, axis=2
        ).squeeze(
            2
        )  # (batch, max_k)

        # Safe division for acceptance ratio
        p_d_safe = mx.maximum(p_d_at_draft, eps)
        ratios = p_t_at_draft / p_d_safe  # (batch, max_k)

        # Generate all random numbers at once
        r = mx.random.uniform(shape=(batch_size, max_k))  # (batch, max_k)

        # Accept where r < min(1, ratio)
        accept_mask = r < mx.minimum(mx.array(1.0), ratios)  # (batch, max_k)

        # Mask out positions beyond each request's draft length
        positions = mx.arange(max_k)  # (max_k,)
        length_arr = mx.array(draft_lengths)  # (batch,)
        valid_mask = positions[None, :] < length_arr[:, None]  # (batch, max_k)
        accept_mask = accept_mask & valid_mask

        # Cumulative product to find first rejection
        cumprod_mask = mx.cumprod(
            accept_mask.astype(mx.int32), axis=1
        )  # (batch, max_k)
        num_accepted_arr = mx.sum(cumprod_mask, axis=1)  # (batch,)

        # Evaluate everything needed before Python-side branching
        mx.eval(num_accepted_arr, p_target, p_draft)
        num_accepted_list = num_accepted_arr.tolist()

        # Now compute bonus tokens per request
        # For rejected requests: sample from adjusted distribution at rejection position
        # For all-accepted requests: sample from target at position k
        bonus_tokens: list[int] = []
        # Collect all bonus sampling operations, then eval once
        bonus_arrays: list[mx.array] = []

        for b in range(batch_size):
            n = num_accepted_list[b]
            k_b = draft_lengths[b]

            if n == k_b:
                # All draft tokens accepted — sample bonus from target at pos k
                bonus_arrays.append(mx.random.categorical(target_logits[b, k_b, :]))
            else:
                # Rejected at position n — sample from adjusted distribution
                adjusted = mx.maximum(p_target[b, n, :] - p_draft[b, n, :], 0.0)
                adjusted_sum = mx.sum(adjusted)
                # Use adjusted if sum > 0, else fall back to target probs
                adjusted_normalized = mx.where(
                    adjusted_sum > 0.0,
                    adjusted / mx.maximum(adjusted_sum, eps),
                    p_target[b, n, :],
                )
                bonus_arrays.append(mx.random.categorical(mx.log(adjusted_normalized)))

        # Single eval for all bonus token samples
        mx.eval(*bonus_arrays)
        bonus_list = [int(a.item()) for a in bonus_arrays]

        # Build accepted_token_ids
        accepted_token_ids: list[list[int]] = []
        for b in range(batch_size):
            n = num_accepted_list[b]
            accepted_token_ids.append(draft_token_ids[b][:n])

        return RejectionResult(
            accepted_token_ids=accepted_token_ids,
            num_accepted=num_accepted_list,
            bonus_token_ids=bonus_list,
        )
