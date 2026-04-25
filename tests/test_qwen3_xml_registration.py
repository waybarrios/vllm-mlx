# SPDX-License-Identifier: Apache-2.0
"""Verify qwen3_coder resolves to Qwen3XMLToolParser, not HermesToolParser."""

import pytest


def test_qwen3_coder_resolves_to_qwen3_xml():
    pytest.importorskip("transformers")
    from vllm_mlx.tool_parsers import ToolParserManager
    from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import Qwen3XMLToolParser

    cls = ToolParserManager.get_tool_parser("qwen3_coder")
    assert cls is Qwen3XMLToolParser


def test_qwen3_xml_aliases_all_resolve():
    pytest.importorskip("transformers")
    from vllm_mlx.tool_parsers import ToolParserManager
    from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import Qwen3XMLToolParser

    for name in ["qwen3_xml", "qwen3.5", "qwen3_coder"]:
        cls = ToolParserManager.get_tool_parser(name)
        assert cls is Qwen3XMLToolParser, f"{name} -> {cls.__name__}"


def test_hermes_does_not_claim_qwen3_coder():
    pytest.importorskip("transformers")
    from vllm_mlx.tool_parsers import ToolParserManager
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    cls = ToolParserManager.get_tool_parser("qwen3_coder")
    assert cls is not HermesToolParser


def test_hermes_still_registered():
    pytest.importorskip("transformers")
    from vllm_mlx.tool_parsers import ToolParserManager
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    for name in ["hermes", "nous"]:
        cls = ToolParserManager.get_tool_parser(name)
        assert cls is HermesToolParser, f"{name} -> {cls.__name__}"
