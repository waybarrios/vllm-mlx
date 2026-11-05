# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS reasoning parser for Harmony channel-based model output."""

from .harmony_parser import HarmonyReasoningParser


class GptOssReasoningParser(HarmonyReasoningParser):
    """Share Harmony body boundaries while retaining the plain-text fallback.

    Routing each accumulated response by only its last channel loses earlier
    bodies in a coalesced delta and exposes partial controls in steady phases.
    The Harmony parser tracks emitted body offsets across those boundaries.
    """

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if "<|channel|>" not in model_output:
            return None, model_output or None
        return super().extract_reasoning(model_output)
