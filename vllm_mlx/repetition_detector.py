"""Detect degenerate repeating token patterns during generation."""


class RepetitionDetector:
    """Sliding-window detector for repeating token sequences.

    Checks periodically (every ``check_interval`` tokens) whether the
    last ``window`` tokens contain a pattern of length 2-``max_pattern``
    repeated at least ``min_repeats`` times consecutively.

    Usage::

        det = RepetitionDetector()
        for token_id in generate():
            if det.check(token_id):
                break  # degenerate loop detected
    """

    def __init__(
        self,
        window: int = 200,
        max_pattern: int = 50,
        min_repeats: int = 3,
        check_interval: int = 20,
    ):
        self.window = window
        self.max_pattern = max_pattern
        self.min_repeats = min_repeats
        self.check_interval = check_interval
        self._tokens: list[int] = []
        self._count = 0

    def check(self, token_id: int) -> bool:
        """Record a token and return True if a repetition loop is detected."""
        self._tokens.append(token_id)
        self._count += 1

        # Only keep the sliding window
        if len(self._tokens) > self.window:
            self._tokens = self._tokens[-self.window :]

        # Check periodically to stay lightweight
        if self._count % self.check_interval != 0:
            return False

        return self._is_repeating()

    def _is_repeating(self) -> bool:
        tokens = self._tokens
        n = len(tokens)
        # Need at least min_repeats * 2 tokens for shortest pattern (len 2)
        if n < self.min_repeats * 2:
            return False

        for pat_len in range(2, min(self.max_pattern + 1, n // self.min_repeats + 1)):
            pattern = tokens[-pat_len:]
            repeats = 1
            pos = n - 2 * pat_len
            while pos >= 0:
                if tokens[pos : pos + pat_len] == pattern:
                    repeats += 1
                    if repeats >= self.min_repeats:
                        return True
                    pos -= pat_len
                else:
                    break

        return False

    def reset(self):
        """Clear state for a new generation."""
        self._tokens.clear()
        self._count = 0
