# SPDX-License-Identifier: Apache-2.0
"""Regression tests for load_assistant_drafter's config-driven architecture
dispatch (ex load_gemma4_assistant_drafter — see fix/mllm-draft-loader-
architecture-dispatch).

These test the dispatch logic itself (config.json -> mlx_vlm.utils.load_model),
not real weight loading — mlx_vlm.utils.load_model is monkeypatched so no
actual model architecture or safetensors content is needed. The existing
tests in test_mllm_assistant_drafter.py cover chat-time drafter wiring and
inject `_draft_model` directly, bypassing this loader entirely; these tests
cover the loader itself, which nothing previously exercised.
"""

import json
import sys
from types import SimpleNamespace

import pytest


def _write_drafter_checkpoint(tmp_path, model_type: str):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize("model_type", ["gemma4_assistant", "qwen3_5_mtp", "eagle3"])
def test_load_assistant_drafter_dispatches_any_model_type_through_load_model(
    tmp_path, monkeypatch, model_type
):
    """No architecture is hardcoded: every model_type goes through the same
    mlx_vlm.utils.load_model call, just with a different config.json."""
    from vllm_mlx.models.mllm import load_assistant_drafter

    drafter_path = _write_drafter_checkpoint(tmp_path, model_type)
    captured = {}

    class FakeDrafterModel:
        def eval(self):
            captured["eval_called"] = True
            return self

    def fake_load_model(path):
        captured["path"] = path
        return FakeDrafterModel()

    monkeypatch.setitem(
        sys.modules, "mlx_vlm.utils", SimpleNamespace(load_model=fake_load_model)
    )

    result = load_assistant_drafter(str(drafter_path))

    assert captured["path"] == drafter_path
    assert captured["eval_called"] is True
    assert isinstance(result, FakeDrafterModel)


def test_load_assistant_drafter_missing_config_raises_file_not_found(tmp_path):
    from vllm_mlx.models.mllm import load_assistant_drafter

    (tmp_path / "model.safetensors").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="config"):
        load_assistant_drafter(str(tmp_path))


def test_load_assistant_drafter_missing_weights_raises_file_not_found(tmp_path):
    from vllm_mlx.models.mllm import load_assistant_drafter

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp"}))

    with pytest.raises(FileNotFoundError, match="weights"):
        load_assistant_drafter(str(tmp_path))


def test_load_assistant_drafter_missing_mlx_vlm_raises_import_error(
    tmp_path, monkeypatch
):
    from vllm_mlx.models.mllm import load_assistant_drafter

    drafter_path = _write_drafter_checkpoint(tmp_path, "qwen3_5_mtp")
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", None)

    with pytest.raises(ImportError, match="mlx-vlm"):
        load_assistant_drafter(str(drafter_path))
