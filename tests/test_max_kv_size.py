# SPDX-License-Identifier: Apache-2.0
"""Tests for --max-kv-size CLI flag and SchedulerConfig plumbing."""

import pytest


class TestMaxKvSizeCLI:
    """Verify --max-kv-size argparse integration."""

    def _parse(self, args_list):
        """Build the serve parser and parse args."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        return parser.parse_args(["serve", "test-model"] + args_list)

    def test_default_is_none(self):
        args = self._parse([])
        assert args.max_kv_size is None

    def test_explicit_value(self):
        args = self._parse(["--max-kv-size", "65536"])
        assert args.max_kv_size == 65536

    def test_with_continuous_batching(self):
        args = self._parse(["--continuous-batching", "--max-kv-size", "32768"])
        assert args.max_kv_size == 32768
        assert args.continuous_batching is True


class TestSchedulerConfigMaxKvSize:
    """Verify SchedulerConfig accepts and stores max_kv_size."""

    def test_default_zero(self):
        pytest.importorskip("mlx.core")
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig()
        assert config.max_kv_size == 0

    def test_set_value(self):
        pytest.importorskip("mlx.core")
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig(max_kv_size=65536)
        assert config.max_kv_size == 65536


class TestMLLMSchedulerConfigMaxKvSize:
    """Verify MLLMSchedulerConfig accepts and stores max_kv_size."""

    def test_default_zero(self):
        pytest.importorskip("mlx.core")
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig()
        assert config.max_kv_size == 0

    def test_set_value(self):
        pytest.importorskip("mlx.core")
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(max_kv_size=32768)
        assert config.max_kv_size == 32768
