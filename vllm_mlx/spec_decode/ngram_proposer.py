# SPDX-License-Identifier: Apache-2.0
"""
N-gram based draft token proposer for speculative decoding.

Uses n-gram pattern matching on the existing token sequence to predict
likely continuations. This is a CPU-only, pure-Python implementation
that requires no model weights — it simply exploits repetitive patterns
in the generated text (e.g., repeated phrases, boilerplate, structured
output) to propose draft tokens at near-zero cost.

Algorithm:
    Given context token_ids and n-gram size n, take the last (n-1) tokens
    as a search pattern. Scan earlier positions in the sequence for the
    same (n-1)-gram. When a match is found, the tokens that follow the
    match become the draft proposal. If multiple matches exist, the most
    recent match (closest to the end) is preferred, as it is most likely
    to reflect the current generation context.
"""

from dataclasses import dataclass, field

from .proposer import BaseProposer, ProposerConfig


@dataclass
class NgramProposerConfig(ProposerConfig):
    """
    Configuration for the n-gram proposer.

    Attributes:
        n: The n-gram size. The proposer searches for the last (n-1) tokens
            as a pattern in earlier context. Higher values require longer
            exact matches, reducing false positives but also reducing hit
            rate. Default is 3 (trigram: match last 2 tokens).
        max_k: Maximum number of draft tokens to propose per call. The
            actual number returned may be fewer if the matched context
            does not have enough following tokens. Default is 5.
    """

    n: int = 3
    max_k: int = 5


class NgramProposer(BaseProposer):
    """
    Draft token proposer based on n-gram pattern matching.

    Maintains an incrementally-built lookup table mapping (n-1)-grams to
    the positions where they occur in the token sequence. On each call
    to propose(), the table is extended with any new tokens, then queried
    for the most recent match of the trailing (n-1)-gram pattern.

    This proposer is effective for:
    - Repetitive or structured output (JSON, XML, code boilerplate)
    - Long documents with recurring phrases
    - Any context where the model tends to repeat earlier patterns

    The proposer is CPU-only and adds negligible latency.
    """

    def __init__(self, config: NgramProposerConfig) -> None:
        """
        Initialize the n-gram proposer.

        Args:
            config: N-gram proposer configuration.
        """
        super().__init__(config)
        self._config: NgramProposerConfig = config

        # Lookup table: maps (n-1)-gram tuples to a list of positions
        # where the gram starts. Positions are appended in order, so
        # the last element is always the most recent occurrence.
        self._ngram_table: dict[tuple[int, ...], list[int]] = {}

        # How many tokens from the input sequence have already been
        # indexed into the lookup table. This allows incremental
        # updates without rescanning the entire sequence.
        self._indexed_length: int = 0

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """
        Propose up to k draft tokens by n-gram pattern matching.

        Takes the last (n-1) tokens of token_ids as a search pattern,
        finds the most recent earlier occurrence of this pattern, and
        returns the tokens that follow it.

        Args:
            token_ids: The full context token IDs (prompt + generated).
            k: Number of draft tokens to propose. Clamped to max_k.

        Returns:
            A list of up to k proposed token IDs. Returns an empty list
            if no n-gram match is found, if the input is too short, or
            if k is 0.
        """
        n = self._config.n
        max_k = self._config.max_k

        # Clamp k to configured maximum
        k = min(k, max_k)

        # Early exits
        if k <= 0 or len(token_ids) < n:
            return []

        prefix_len = n - 1

        # Build / extend the n-gram lookup table incrementally.
        # We index all (n-1)-grams up to (but not including) the last
        # (n-1) tokens, because the last (n-1) tokens form the query
        # pattern and should not match themselves.
        self._update_table(token_ids)

        # The pattern to search for: last (n-1) tokens
        pattern = tuple(token_ids[-prefix_len:])

        # Look up positions where this pattern occurs
        positions = self._ngram_table.get(pattern)
        if not positions:
            return []

        # Use the most recent match (last in the list).
        # The match position points to where the (n-1)-gram starts.
        # Draft tokens begin at (match_pos + prefix_len).
        match_pos = positions[-1]
        draft_start = match_pos + prefix_len

        # Collect up to k tokens following the matched pattern.
        # Cannot go past the start of the query pattern itself
        # (i.e., len(token_ids) - prefix_len).
        draft_end = min(draft_start + k, len(token_ids) - prefix_len)

        if draft_start >= draft_end:
            return []

        return token_ids[draft_start:draft_end]

    def reset(self) -> None:
        """
        Reset internal state.

        Clears the n-gram lookup table and indexed length counter.
        Must be called when switching to a new request to avoid
        cross-contamination of n-gram statistics.
        """
        self._ngram_table.clear()
        self._indexed_length = 0

    def _update_table(self, token_ids: list[int]) -> None:
        """
        Incrementally update the n-gram lookup table with new tokens.

        Only indexes (n-1)-grams that have not been seen before (based
        on self._indexed_length). The indexable range excludes the last
        (n-1) tokens, which form the current query pattern.

        Args:
            token_ids: The full context token IDs.
        """
        prefix_len = self._config.n - 1
        seq_len = len(token_ids)

        # The last (n-1) tokens are the query pattern; do not index them.
        # So the last indexable start position is (seq_len - prefix_len - 1).
        max_index_start = seq_len - prefix_len

        # Start from where we left off last time
        start = self._indexed_length

        # If the sequence has been truncated or reset, rebuild from scratch
        if start > seq_len:
            self._ngram_table.clear()
            start = 0

        for i in range(start, max_index_start):
            gram = tuple(token_ids[i : i + prefix_len])
            if gram not in self._ngram_table:
                self._ngram_table[gram] = []
            self._ngram_table[gram].append(i)

        self._indexed_length = max_index_start
