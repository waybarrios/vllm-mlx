# SPDX-License-Identifier: Apache-2.0
"""
Resumable model download with retry/timeout support.

Pre-downloads models via huggingface_hub.snapshot_download() with
configurable timeout and retry logic before passing to mlx-lm/mlx-vlm.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Mirrors mlx_lm.utils._download() default allow_patterns
LLM_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]

# Mirrors mlx_vlm.utils.get_model_path() allow_patterns
MLLM_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
]


@dataclass
class DownloadConfig:
    """Configuration for model download behavior."""

    download_timeout: int = 300
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    offline: bool = False


def ensure_model_downloaded(
    model_name: str,
    config: DownloadConfig | None = None,
    is_mllm: bool = False,
) -> Path:
    """
    Ensure a model is available locally, downloading with retry if needed.

    Args:
        model_name: HuggingFace model name or local path.
        config: Download configuration. Uses defaults if None.
        is_mllm: If True, use MLLM download patterns (broader file set).

    Returns:
        Path to the local model directory.

    Raises:
        RuntimeError: If download fails after all retries.
        KeyboardInterrupt: Propagated immediately without retry.
    """
    if config is None:
        config = DownloadConfig()

    model_path = Path(model_name)
    if model_path.exists():
        logger.info(f"Model found at local path: {model_path}")
        return model_path

    if config.offline:
        logger.info(f"Offline mode: looking for cached {model_name}")
        try:
            result = Path(snapshot_download(model_name, local_files_only=True))
            logger.info(f"Found cached model at {result}")
            return result
        except Exception as e:
            raise RuntimeError(
                f"Model '{model_name}' not found in local cache. "
                f"Download it first without --offline flag."
            ) from e

    allow_patterns = MLLM_ALLOW_PATTERNS if is_mllm else LLM_ALLOW_PATTERNS

    # Set HF download timeout via environment variable
    old_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(config.download_timeout)

    last_error = None
    try:
        for attempt in range(1, config.max_retries + 1):
            try:
                logger.info(
                    f"Downloading model {model_name} "
                    f"(attempt {attempt}/{config.max_retries}, "
                    f"timeout={config.download_timeout}s)"
                )
                result = Path(
                    snapshot_download(
                        model_name,
                        allow_patterns=allow_patterns,
                    )
                )
                logger.info(f"Model downloaded successfully to {result}")
                return result
            except KeyboardInterrupt:
                logger.warning("Download interrupted by user.")
                raise
            except Exception as e:
                last_error = e
                if attempt < config.max_retries:
                    wait = config.retry_backoff_base**attempt
                    logger.warning(
                        f"Download attempt {attempt} failed: {e}. "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Download failed after {config.max_retries} attempts."
                    )

        raise RuntimeError(
            f"Failed to download '{model_name}' after {config.max_retries} "
            f"attempts. Last error: {last_error}\n"
            f"Run the same command again to resume the download."
        )
    finally:
        # Restore original env var
        if old_timeout is None:
            os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
        else:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = old_timeout
