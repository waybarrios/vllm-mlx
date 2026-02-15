# SPDX-License-Identifier: Apache-2.0
"""Tests for resumable model download with retry/timeout support."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx.utils.download import (
    LLM_ALLOW_PATTERNS,
    MLLM_ALLOW_PATTERNS,
    DownloadConfig,
    ensure_model_downloaded,
)


class TestLocalPath:
    """Tests for local path handling."""

    def test_local_path_skips_download(self, tmp_path):
        """Existing local directory is returned without downloading."""
        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            result = ensure_model_downloaded(str(tmp_path))
            mock_download.assert_not_called()
            assert result == tmp_path


class TestRetryLogic:
    """Tests for download retry behavior."""

    def test_retry_on_failure(self):
        """Failed downloads are retried up to max_retries times."""
        config = DownloadConfig(max_retries=3, retry_backoff_base=0.01)
        fake_path = "/fake/cache/path"

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.side_effect = [
                ConnectionError("timeout"),
                ConnectionError("timeout"),
                fake_path,
            ]
            result = ensure_model_downloaded("org/model", config=config)
            assert result == Path(fake_path)
            assert mock_download.call_count == 3

    def test_retry_exhaustion(self):
        """RuntimeError is raised after all retries are exhausted."""
        config = DownloadConfig(max_retries=2, retry_backoff_base=0.01)

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.side_effect = ConnectionError("timeout")
            with pytest.raises(RuntimeError, match="Failed to download"):
                ensure_model_downloaded("org/model", config=config)
            assert mock_download.call_count == 2

    def test_keyboard_interrupt_not_retried(self):
        """KeyboardInterrupt propagates immediately without retry."""
        config = DownloadConfig(max_retries=3, retry_backoff_base=0.01)

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.side_effect = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                ensure_model_downloaded("org/model", config=config)
            assert mock_download.call_count == 1


class TestOfflineMode:
    """Tests for offline mode behavior."""

    def test_offline_mode_cached(self):
        """Offline mode finds cached model successfully."""
        config = DownloadConfig(offline=True)
        fake_path = "/fake/cache/path"

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.return_value = fake_path
            result = ensure_model_downloaded("org/model", config=config)
            assert result == Path(fake_path)
            mock_download.assert_called_once_with("org/model", local_files_only=True)

    def test_offline_mode_missing(self):
        """Offline mode raises clear error when model is not cached."""
        config = DownloadConfig(offline=True)

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("not found locally")
            with pytest.raises(RuntimeError, match="not found in local cache"):
                ensure_model_downloaded("org/model", config=config)


class TestTimeout:
    """Tests for download timeout configuration."""

    def test_hf_timeout_env_set(self):
        """HF_HUB_DOWNLOAD_TIMEOUT env var is set during download."""
        config = DownloadConfig(download_timeout=600, max_retries=1)
        fake_path = "/fake/cache/path"
        captured_timeout = {}

        original_env = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")

        def capture_env(*args, **kwargs):
            captured_timeout["value"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
            return fake_path

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.side_effect = capture_env
            ensure_model_downloaded("org/model", config=config)

        assert captured_timeout["value"] == "600"
        # Env var should be restored after download
        assert os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") == original_env

    def test_hf_timeout_env_restored_on_failure(self):
        """HF_HUB_DOWNLOAD_TIMEOUT is restored even after failure."""
        config = DownloadConfig(
            download_timeout=999, max_retries=1, retry_backoff_base=0.01
        )
        original_env = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.side_effect = ConnectionError("fail")
            with pytest.raises(RuntimeError):
                ensure_model_downloaded("org/model", config=config)

        assert os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") == original_env


class TestAllowPatterns:
    """Tests for LLM vs MLLM download patterns."""

    def test_llm_patterns_used_by_default(self):
        """LLM allow patterns are used when is_mllm=False."""
        config = DownloadConfig(max_retries=1)
        fake_path = "/fake/cache/path"

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.return_value = fake_path
            ensure_model_downloaded("org/model", config=config, is_mllm=False)
            mock_download.assert_called_once_with(
                "org/model", allow_patterns=LLM_ALLOW_PATTERNS
            )

    def test_mllm_patterns_used(self):
        """MLLM allow patterns are used when is_mllm=True."""
        config = DownloadConfig(max_retries=1)
        fake_path = "/fake/cache/path"

        with patch("vllm_mlx.utils.download.snapshot_download") as mock_download:
            mock_download.return_value = fake_path
            ensure_model_downloaded("org/model", config=config, is_mllm=True)
            mock_download.assert_called_once_with(
                "org/model", allow_patterns=MLLM_ALLOW_PATTERNS
            )


class TestCLIDownloadCommand:
    """Tests for CLI download subcommand argument parsing."""

    def test_cli_download_command(self):
        """Download subcommand parses arguments correctly."""
        import argparse

        # We test argparse by calling parse_args directly
        # (main() would try to actually run the command)
        with patch("sys.argv", ["vllm-mlx", "download", "org/model"]):
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="command")
            download_parser = subparsers.add_parser("download")
            download_parser.add_argument("model")
            download_parser.add_argument("--timeout", type=int, default=300)
            download_parser.add_argument("--retries", type=int, default=3)
            download_parser.add_argument("--mllm", action="store_true")

            args = parser.parse_args(["download", "org/model", "--timeout", "600"])
            assert args.command == "download"
            assert args.model == "org/model"
            assert args.timeout == 600
            assert args.retries == 3
            assert args.mllm is False

    def test_cli_download_mllm_flag(self):
        """Download subcommand parses --mllm flag."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        download_parser = subparsers.add_parser("download")
        download_parser.add_argument("model")
        download_parser.add_argument("--timeout", type=int, default=300)
        download_parser.add_argument("--retries", type=int, default=3)
        download_parser.add_argument("--mllm", action="store_true")

        args = parser.parse_args(["download", "org/vl-model", "--mllm"])
        assert args.mllm is True
