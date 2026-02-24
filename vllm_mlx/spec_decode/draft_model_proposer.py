# SPDX-License-Identifier: Apache-2.0
"""
Draft model based proposer for speculative decoding.

Uses a smaller autoregressive language model (loaded via mlx-lm) to generate
draft tokens. The draft model runs entirely on the local node and proposes
tokens that are then verified in parallel by the (potentially distributed)
target model.

This approach produces higher-quality drafts than n-gram matching because
the draft model has learned language structure, but at the cost of running
a small model forward pass for each draft token.

The draft model is loaded once via load() and reused across all propose()
calls. Each propose() call creates a fresh KV cache (since different
requests have different contexts), processes the prompt to build the cache,
then generates k tokens autoregressively.

For best acceptance rate, greedy decoding (argmax) is used by default.
"""

import logging
from dataclasses import dataclass

from .proposer import BaseProposer, ProposerConfig

logger = logging.getLogger(__name__)


@dataclass
class DraftModelProposerConfig(ProposerConfig):
    """
    Configuration for the draft model proposer.

    Attributes:
        draft_model_name: Path or HuggingFace name of the draft model.
            This should be a small model (~3-8B parameters) that fits on
            a single node. Example: "~/models/Moonlight-16B-A3B-Instruct-4-bit".
        temperature: Sampling temperature. 0.0 means greedy (argmax),
            which typically gives the best acceptance rate.
        top_p: Top-p (nucleus) sampling parameter. Only used when
            temperature > 0.
    """

    draft_model_name: str = ""
    temperature: float = 0.0
    top_p: float = 1.0


class DraftModelProposer(BaseProposer):
    """
    Draft token proposer using a smaller autoregressive model.

    Loads a small language model via mlx-lm and uses it to generate
    candidate tokens autoregressively. The model is loaded once and
    reused across all requests. Each propose() call creates a fresh
    KV cache for the given context.

    This proposer is effective for:
    - General-purpose text where n-gram patterns are insufficient
    - High-quality drafts that increase acceptance rate
    - Scenarios where the draft model and target model share vocabulary

    The draft model adds latency proportional to k forward passes through
    the small model, but this is typically much cheaper than k passes
    through the target model (especially when the target is distributed).
    """

    def __init__(self, config: DraftModelProposerConfig) -> None:
        """
        Initialize the draft model proposer.

        The model is NOT loaded here; call load() before the first propose().

        Args:
            config: Draft model proposer configuration.
        """
        super().__init__(config)
        self._config: DraftModelProposerConfig = config
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """
        Load the draft model and tokenizer.

        Must be called before the first propose() call. Uses mlx_lm.load()
        to load the model weights and tokenizer from the configured path.

        Raises:
            ImportError: If mlx-lm is not installed.
            Exception: If the model cannot be loaded.
        """
        if self._model is not None:
            return  # Already loaded

        try:
            import mlx_lm

            logger.info(
                f"Loading draft model: {self._config.draft_model_name}"
            )
            self._model, self._tokenizer = mlx_lm.load(
                self._config.draft_model_name,
                tokenizer_config={"trust_remote_code": True},
            )
            logger.info(
                f"Draft model loaded successfully: {self._config.draft_model_name}"
            )
        except ImportError:
            raise ImportError(
                "mlx-lm is required for the draft model proposer. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load draft model: {e}")
            raise

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        """
        Propose k draft tokens using the draft model.

        Performs k autoregressive forward passes through the draft model:
        1. Process the full prompt through the model to build a KV cache.
        2. Sample a token from the last position's logits.
        3. Feed the sampled token back and repeat k times.

        Uses greedy decoding (argmax) when temperature is 0.0 for best
        acceptance rate. A fresh KV cache is created per call since each
        request has different context.

        Args:
            token_ids: The context token IDs (prompt + generated so far).
            k: Number of draft tokens to propose.

        Returns:
            A list of k proposed token IDs.

        Raises:
            RuntimeError: If the model has not been loaded via load().
        """
        if self._model is None:
            raise RuntimeError(
                "Draft model not loaded. Call load() before propose()."
            )

        if k <= 0 or len(token_ids) == 0:
            return []

        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        # Create a fresh KV cache for this request
        cache = make_prompt_cache(self._model)

        # Process prompt through model to build KV cache
        # Model expects shape (batch=1, seq_len)
        prompt = mx.array(token_ids)[None]
        logits = self._model(prompt, cache=cache)
        mx.eval(logits)

        # Generate k tokens autoregressively
        drafted: list[int] = []
        for _ in range(k):
            # Sample from last position logits
            if self._config.temperature == 0.0:
                # Greedy: take argmax
                token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                # Temperature sampling
                scaled_logits = logits[:, -1, :] / self._config.temperature
                probs = mx.softmax(scaled_logits, axis=-1)
                token = mx.random.categorical(mx.log(probs))

            mx.eval(token)
            token_id = token.item()
            drafted.append(token_id)

            # Feed back for next step
            next_input = mx.array([[token_id]])
            logits = self._model(next_input, cache=cache)
            mx.eval(logits)

        return drafted

    def reset(self) -> None:
        """
        Reset internal state.

        No persistent state is maintained between propose() calls since
        a fresh KV cache is created each time. This method is provided
        for interface compliance.
        """
        pass
