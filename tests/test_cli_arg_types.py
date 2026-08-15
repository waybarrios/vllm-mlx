# SPDX-License-Identifier: Apache-2.0
"""Tests for the argparse type helpers in vllm_mlx/cli_arg_types.py.

Focused on parse_auto_or_positive_int_arg / make_auto_or_positive_int_arg_parser,
which back --embedding-max-length: invalid values must be rejected, not
just validated in passing but never exercised by a test.
"""

import argparse

import pytest

from vllm_mlx.cli_arg_types import (
    make_auto_or_positive_int_arg_parser,
    parse_auto_or_positive_int_arg,
)


class TestParseAutoOrPositiveIntArg:
    def test_auto_returns_none(self):
        assert parse_auto_or_positive_int_arg("auto", "--embedding-max-length") is None

    def test_auto_is_case_and_whitespace_insensitive(self):
        assert (
            parse_auto_or_positive_int_arg(" Auto ", "--embedding-max-length") is None
        )

    def test_positive_integer_is_accepted(self):
        assert parse_auto_or_positive_int_arg("4096", "--embedding-max-length") == 4096

    @pytest.mark.parametrize("bad_value", ["0", "-1", "-4096"])
    def test_non_positive_integer_is_rejected(self, bad_value):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            parse_auto_or_positive_int_arg(bad_value, "--embedding-max-length")

    @pytest.mark.parametrize("bad_value", ["abc", "4096.0", "", "1024tokens"])
    def test_non_integer_is_rejected(self, bad_value):
        with pytest.raises(argparse.ArgumentTypeError, match="integer"):
            parse_auto_or_positive_int_arg(bad_value, "--embedding-max-length")

    def test_error_message_includes_option_name(self):
        """The option name is threaded through so a CLI user sees which flag
        was invalid, not a generic message."""
        with pytest.raises(argparse.ArgumentTypeError, match="--embedding-max-length"):
            parse_auto_or_positive_int_arg("-1", "--embedding-max-length")


class TestMakeAutoOrPositiveIntArgParser:
    """The factory cli.py actually passes as `type=` for --embedding-max-length."""

    def test_accepts_auto_and_positive_integers(self):
        parser = make_auto_or_positive_int_arg_parser("--embedding-max-length")
        assert parser("auto") is None
        assert parser("1024") == 1024

    def test_rejects_invalid_values(self):
        parser = make_auto_or_positive_int_arg_parser("--embedding-max-length")
        with pytest.raises(argparse.ArgumentTypeError):
            parser("not-a-number")
        with pytest.raises(argparse.ArgumentTypeError):
            parser("0")


class TestEmbeddingMaxLengthCliWiring:
    def test_invalid_value_rejected_at_argument_parsing(self, capsys):
        """Regression: --embedding-max-length must reject an invalid value
        (e.g. negative) at CLI parsing time via argparse, not silently
        accept it and fail later inside the server."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "--embedding-max-length", "-1"])
        assert "--embedding-max-length" in capsys.readouterr().err

    def test_auto_and_positive_values_parse_successfully(self):
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        assert (
            parser.parse_args(
                ["serve", "--embedding-max-length", "auto"]
            ).embedding_max_length
            is None
        )
        assert (
            parser.parse_args(
                ["serve", "--embedding-max-length", "2048"]
            ).embedding_max_length
            == 2048
        )
