# SPDX-License-Identifier: Apache-2.0
"""Argparse type helpers shared by CLI entrypoints."""

import argparse
import json
import math
from collections.abc import Callable
from typing import Any


def parse_json_object_arg(value: str, option_name: str) -> dict[str, Any]:
    """Parse and validate that an option value is a JSON object."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a valid JSON object: {exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{option_name} must be a JSON object")

    return parsed


def make_json_object_arg_parser(option_name: str) -> Callable[[str], dict[str, Any]]:
    """Create an argparse type parser for JSON object options."""

    def _parser(value: str) -> dict[str, Any]:
        return parse_json_object_arg(value, option_name)

    return _parser


def positive_int_arg(value: str) -> int:
    """Parse an argparse integer that must be greater than zero."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_positive_int_arg(value: str, option_name: str) -> int:
    """Parse and validate that an option value is a positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{option_name} must be a positive integer")
    return parsed


def make_positive_int_arg_parser(option_name: str) -> Callable[[str], int]:
    """Create an argparse type parser for positive integer options."""

    def _parser(value: str) -> int:
        return parse_positive_int_arg(value, option_name)

    return _parser


def parse_positive_finite_float(value: Any, value_name: str) -> float:
    """Parse a numeric value that must be positive and finite."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{value_name} must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{value_name} must be a positive finite number")
    return parsed


def memory_budget_gb_arg(value: str) -> float:
    """Parse a registry memory budget that can be represented in bytes."""
    option_name = "--memory-budget-gb"
    try:
        parsed = parse_positive_finite_float(value, option_name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    bytes_value = parsed * (1024**3)
    if not math.isfinite(bytes_value):
        raise argparse.ArgumentTypeError(f"{option_name} is too large")
    if int(bytes_value) == 0:
        raise argparse.ArgumentTypeError(f"{option_name} must be at least 1 byte")
    return parsed


def parse_auto_or_positive_int_arg(value: str, option_name: str) -> int | None:
    """Parse an option that is either the literal 'auto' or a positive integer."""
    if value.strip().lower() == "auto":
        return None
    return parse_positive_int_arg(value, option_name)


def make_auto_or_positive_int_arg_parser(
    option_name: str,
) -> Callable[[str], int | None]:
    """Create an argparse type parser for 'auto'-or-positive-integer options."""

    def _parser(value: str) -> int | None:
        return parse_auto_or_positive_int_arg(value, option_name)

    return _parser
