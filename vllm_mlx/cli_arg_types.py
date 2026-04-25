# SPDX-License-Identifier: Apache-2.0
"""Argparse type helpers shared by CLI entrypoints."""

import argparse
import json
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
