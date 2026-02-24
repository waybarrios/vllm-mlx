# SPDX-License-Identifier: Apache-2.0
"""MTP (Multi-Token Prediction) proposer for speculative decoding.

Uses the target model's hidden states and a lightweight MTP module to
propose draft tokens without a separate draft model.
"""

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .mtp_module import MTPModule
from .proposer import BaseProposer, Proposal, ProposalContext, ProposerConfig

logger = logging.getLogger(__name__)


@dataclass
class MTPProposerConfig(ProposerConfig):
    """Configuration for the MTP proposer.

    Attributes:
        num_speculative_tokens: Number of draft tokens to propose per step.
    """

    pass


class MTPProposer(BaseProposer):
    """Proposer that uses MTP module for draft token generation.

    Unlike n-gram or draft model proposers, MTP requires hidden states
    from the target model's last layer. When hidden states are not available
    (e.g., first step), it returns no proposals.

    Args:
        config: MTPProposerConfig with proposal parameters.
        mtp_module: The MTP module for forward passes.
        embed_fn: Shared embedding function (nn.Embedding) from target model.
        lm_head: Shared language model head (nn.Linear) from target model.
    """

    def __init__(
        self,
        config: MTPProposerConfig,
        mtp_module: MTPModule,
        embed_fn: Any,
        lm_head: Any,
    ):
        super().__init__(config)
        self.mtp_module = mtp_module
        self.embed_fn = embed_fn
        self.lm_head = lm_head

    def propose_with_context(self, ctx: ProposalContext) -> Proposal:
        """Generate draft tokens using MTP module.

        Args:
            ctx: ProposalContext with hidden_states from target model.

        Returns:
            Proposal with draft token IDs. Empty if no hidden states available.
        """
        if ctx.hidden_states is None:
            return Proposal(token_ids=[])

        drafts: list[int] = []
        h = ctx.hidden_states  # (1, 1, hidden_size)
        last_token = ctx.token_ids[-1]
        cache = self.mtp_module.make_cache()

        for _ in range(ctx.k):
            # MTP forward: hidden + token -> new_hidden
            new_hidden = self.mtp_module(
                h, mx.array([[last_token]]), self.embed_fn, cache=cache,
            )

            # Apply shared lm_head to get logits
            logits = self.lm_head(new_hidden)
            mx.eval(logits)

            # Greedy sampling
            token = mx.argmax(logits[:, -1, :], axis=-1).item()
            drafts.append(token)

            # Update state for next step
            last_token = token
            h = new_hidden

        return Proposal(token_ids=drafts)

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """Cannot propose without hidden states context."""
        return []

    def reset(self) -> None:
        """No-op: MTP creates fresh caches each step, no persistent state."""
        pass

    def remove_request(self, request_id: str) -> None:
        """No-op: MTP caches are created fresh each step."""
        pass
