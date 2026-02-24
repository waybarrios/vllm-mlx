# SPDX-License-Identifier: Apache-2.0
"""
Speculative decoding package for vllm-mlx.

Provides the core abstractions for speculative decoding, where a fast
proposer (n-gram, EAGLE, draft model, or MTP) generates candidate tokens
that are then verified in parallel by the target model.

Key components:
- SpecDecodeConfig: Configuration for speculation method and parameters.
- SpecDecodeMetadata: Per-step draft token data passed between components.
- BaseProposer / ProposerConfig: Abstract interface for draft token proposers.
- ProposalContext / Proposal: Context object pattern for MTP hidden states.
- MTPModule / MTPProposer: Multi-Token Prediction proposer infrastructure.
- SpecDecodeStats: Performance tracking and acceptance rate statistics.
- SpecDecodeRuntime: Orchestrates draft-verify-accept cycles.
- RequestState / VerifyResult / AcceptResult: Supporting data classes.

Usage:
    from vllm_mlx.spec_decode import SpecDecodeConfig, BaseProposer

    config = SpecDecodeConfig(method="ngram", num_speculative_tokens=5)
"""

from .draft_model_proposer import DraftModelProposer, DraftModelProposerConfig
from .metadata import SpecDecodeConfig, SpecDecodeMetadata
from .metrics import SpecDecodeStats
from .mtp_module import MTPModule, MTPModuleConfig, load_mtp_weights, detect_mtp_prefix, detect_mtp_style
from .mtp_proposer import MTPProposer, MTPProposerConfig
from .proposer import BaseProposer, Proposal, ProposalContext, ProposerConfig
from .runtime import AcceptResult, RequestState, SpecDecodeRuntime, VerifyResult

__all__ = [
    # Configuration and metadata
    "SpecDecodeConfig",
    "SpecDecodeMetadata",
    # Proposer abstractions
    "BaseProposer",
    "ProposerConfig",
    "ProposalContext",
    "Proposal",
    # Draft model proposer
    "DraftModelProposer",
    "DraftModelProposerConfig",
    # MTP proposer
    "MTPModule",
    "MTPModuleConfig",
    "MTPProposer",
    "MTPProposerConfig",
    "load_mtp_weights",
    "detect_mtp_prefix",
    "detect_mtp_style",
    # Metrics
    "SpecDecodeStats",
    # Runtime
    "SpecDecodeRuntime",
    "RequestState",
    "VerifyResult",
    "AcceptResult",
]
