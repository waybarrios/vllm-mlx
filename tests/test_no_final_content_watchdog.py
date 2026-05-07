"""No-final-content watchdog regressions for thinking routes."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


def _load_thinking_processor():
    class _FakeArray:
        pass

    fake_mx = types.SimpleNamespace(array=_FakeArray)
    sys.modules.setdefault("mlx", types.ModuleType("mlx"))
    sys.modules["mlx.core"] = fake_mx

    path = (
        Path(__file__).resolve().parents[1]
        / "vllm_mlx/constrained/thinking_processor.py"
    )
    loader = importlib.machinery.SourceFileLoader(
        "thinking_processor_under_test", str(path)
    )
    spec = importlib.util.spec_from_loader("thinking_processor_under_test", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_no_final_content_watchdog_forces_transition_before_budget():
    module = _load_thinking_processor()
    proc = module.ThinkingAwareLogitsProcessor(
        start_token_ids=[1],
        end_token_ids=[2],
        thinking_token_budget=100,
        prompt_has_think_tag=True,
        no_final_content_token_limit=3,
    )

    for token_id in [10, 11, 12]:
        proc._advance_with_token(token_id)

    assert proc.state is module.Phase.TRANSITIONING
    assert proc.watchdog_was_enforced is True
    assert proc.thinking_tokens == 3


def test_budget_transition_does_not_mark_watchdog_enforced():
    module = _load_thinking_processor()
    proc = module.ThinkingAwareLogitsProcessor(
        start_token_ids=[1],
        end_token_ids=[2],
        thinking_token_budget=2,
        prompt_has_think_tag=True,
        no_final_content_token_limit=10,
    )

    for token_id in [10, 11]:
        proc._advance_with_token(token_id)

    assert proc.state is module.Phase.TRANSITIONING
    assert proc.watchdog_was_enforced is False
