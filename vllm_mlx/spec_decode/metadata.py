# SPDX-License-Identifier: Apache-2.0
"""
Data classes for speculative decoding configuration and metadata.

These provide the core data structures used across the speculative decoding
pipeline to configure behavior and pass draft token information between
the proposer, verifier, and runtime components.
"""

from dataclasses import dataclass, field


@dataclass
class SpecDecodeConfig:
    """
    Configuration for speculative decoding.

    Controls which speculation method to use, how many tokens to draft,
    and when to automatically disable speculation for large batches or
    low acceptance rates.

    Attributes:
        method: Speculation method. One of "ngram", "eagle", "draft_model", or "mtp".
        num_speculative_tokens: Number of draft tokens to propose per step (k).
        disable_by_batch_size: If set, automatically disable speculative decoding
            when the batch size exceeds this threshold. This avoids wasting compute
            on drafting when the GPU is already saturated with real requests.
        auto_disable_threshold: Acceptance rate threshold below which speculative
            decoding is automatically disabled. Set to 0.0 to never auto-disable
            based on acceptance rate. Default: 0.4.
        auto_disable_window: Number of recent speculation rounds over which to
            evaluate the acceptance rate for auto-disable decisions. Default: 50.
    """

    method: str
    num_speculative_tokens: int
    disable_by_batch_size: int | None = None
    auto_disable_threshold: float = 0.4
    auto_disable_window: int = 50

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_methods = {"ngram", "eagle", "draft_model", "mtp"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. Must be one of {valid_methods}"
            )
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )
        if self.disable_by_batch_size is not None and self.disable_by_batch_size < 1:
            raise ValueError(
                f"disable_by_batch_size must be >= 1, got {self.disable_by_batch_size}"
            )
        if not (0.0 <= self.auto_disable_threshold <= 1.0):
            raise ValueError(
                f"auto_disable_threshold must be in [0.0, 1.0], got {self.auto_disable_threshold}"
            )
        if self.auto_disable_window < 1:
            raise ValueError(
                f"auto_disable_window must be >= 1, got {self.auto_disable_window}"
            )


@dataclass
class SpecDecodeMetadata:
    """
    Per-step metadata carrying draft tokens for verification.

    Created by the proposer and consumed by the verifier. Maps request IDs
    to their proposed draft token sequences.

    Attributes:
        draft_token_ids: Mapping from request_id to the list of draft token IDs
            proposed for that request.
        num_draft_tokens: Mapping from request_id to the number of draft tokens
            proposed. This is redundant with len(draft_token_ids[req_id]) but
            provided for convenience and to avoid repeated len() calls in hot paths.
    """

    draft_token_ids: dict[str, list[int]] = field(default_factory=dict)
    num_draft_tokens: dict[str, int] = field(default_factory=dict)

    def add_request(self, request_id: str, token_ids: list[int]) -> None:
        """
        Add draft tokens for a request.

        Args:
            request_id: The unique identifier for the request.
            token_ids: The list of draft token IDs proposed for this request.
        """
        self.draft_token_ids[request_id] = token_ids
        self.num_draft_tokens[request_id] = len(token_ids)

    def get_draft_tokens(self, request_id: str) -> list[int]:
        """
        Get draft tokens for a request.

        Args:
            request_id: The unique identifier for the request.

        Returns:
            The list of draft token IDs, or an empty list if no drafts exist.
        """
        return self.draft_token_ids.get(request_id, [])

    def clear(self) -> None:
        """Remove all draft token metadata."""
        self.draft_token_ids.clear()
        self.num_draft_tokens.clear()
