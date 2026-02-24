# SPDX-License-Identifier: Apache-2.0
"""
Abstract base class for speculative decoding proposers.

A proposer generates draft tokens given a context sequence. Different
implementations (n-gram, EAGLE, draft model) provide different tradeoffs
between draft quality (acceptance rate) and drafting cost.

The propose() method works per-request (single sequence). The caller
(scheduler/runtime) is responsible for batching across requests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProposerConfig:
    """
    Base configuration for all proposers.

    Subclasses should extend this with method-specific fields
    (e.g., n-gram window size, draft model path).

    Attributes:
        num_speculative_tokens: Default number of draft tokens to propose (k).
    """

    num_speculative_tokens: int = 5


@dataclass
class ProposalContext:
    """Context for propose_with_context(), carrying hidden states for MTP.

    Attributes:
        request_id: Unique request identifier.
        token_ids: Full token sequence (prompt + generated so far).
        k: Number of draft tokens to propose.
        hidden_states: Last-layer hidden states from target model (for MTP).
    """

    request_id: str
    token_ids: list[int]
    k: int
    hidden_states: Any | None = None


@dataclass
class Proposal:
    """Result of a proposal, wrapping draft token IDs.

    Attributes:
        token_ids: Proposed draft token IDs.
    """

    token_ids: list[int]


class BaseProposer(ABC):
    """
    Abstract base class for draft token proposers.

    Each proposer implements a strategy for generating candidate tokens
    that will be verified by the target model. Proposers operate on a
    single request at a time; the runtime handles batching.

    Subclasses must implement:
        - propose(): Generate k draft tokens given a context.
        - reset(): Clear any internal state.
    """

    def __init__(self, config: ProposerConfig) -> None:
        """
        Initialize the proposer with configuration.

        Args:
            config: Proposer configuration parameters.
        """
        self.config = config

    @abstractmethod
    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """
        Propose k draft tokens given context token_ids.

        Args:
            token_ids: The context token IDs (prompt + generated so far).
            k: Number of draft tokens to propose.

        Returns:
            A list of k proposed token IDs. May return fewer than k if the
            proposer cannot generate enough candidates (e.g., n-gram miss).
        """
        ...

    def propose_with_context(self, ctx: ProposalContext) -> Proposal:
        """Propose draft tokens using rich context (default: delegates to propose()).

        Subclasses that need hidden states (e.g., MTP) should override this.
        The default implementation ignores extra context fields and calls propose().

        Args:
            ctx: ProposalContext with token_ids, k, and optional hidden_states.

        Returns:
            Proposal containing the draft token IDs.
        """
        tokens = self.propose(ctx.token_ids, ctx.k)
        return Proposal(token_ids=tokens)

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state.

        Called when starting a new request or when the proposer needs
        to clear cached state (e.g., n-gram tables, draft model KV cache).
        """
        ...
