# SPDX-License-Identifier: Apache-2.0
"""Source-level regression for derived text-model quantization plumbing."""

import importlib
import json
from pathlib import Path
import sys
import types

import pytest


def _stub_module(name, *, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    return module


@pytest.mark.parametrize(
    "global_mode, expected_mode", [("mxfp8", "mxfp8"), (None, "affine")]
)
def test_build_text_model_forwards_global_mode_without_overriding_layer_dicts(
    tmp_path, monkeypatch, global_mode, expected_mode
):
    """Global mode reaches quantize while explicit layer dicts stay authoritative."""

    mlx = _stub_module("mlx", package=True)
    mlx_core = _stub_module("mlx.core")
    mlx_nn = _stub_module("mlx.nn")
    mlx_utils = _stub_module("mlx.utils")
    mlx.core = mlx_core
    mlx.nn = mlx_nn
    mlx.utils = mlx_utils
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", mlx_nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)

    mlx_utils.tree_flatten = lambda _parameters: [
        ("global.scales", object()),
        ("language_model.override.scales", object()),
    ]

    class FakeTextModelArgs:
        @classmethod
        def from_dict(cls, config):
            return config

    class FakeTextModel:
        mtp = None

        def __init__(self, _args):
            self.loaded = []

        def load_weights(self, weights, strict=False):
            self.loaded.append((weights, strict))

        def train(self, _mode):
            return self

    quantize_calls = []

    def record_quantize(model, **kwargs):
        quantize_calls.append(kwargs)
        predicate = kwargs["class_predicate"]

        class Quantizable:
            def to_quantized(self):
                return None

        assert predicate("global", Quantizable()) is True
        assert predicate("language_model.override", Quantizable()) == {
            "group_size": 64,
            "bits": 8,
        }

    mlx_nn.quantize = record_quantize

    mlx_lm = _stub_module("mlx_lm", package=True)
    mlx_lm_models = _stub_module("mlx_lm.models", package=True)
    mlx_lm_qwen = _stub_module("mlx_lm.models.qwen3_5")
    mlx_lm_qwen.TextModel = FakeTextModel
    mlx_lm_qwen.TextModelArgs = FakeTextModelArgs
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5", mlx_lm_qwen)

    model_path = tmp_path / "derived"
    model_path.mkdir()
    quantization = {
        "group_size": 32,
        "bits": 8,
        "language_model.override": {"group_size": 64, "bits": 8},
    }
    if global_mode is not None:
        quantization["mode"] = global_mode
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "quantization": quantization})
    )

    class FakeLanguageModel:
        def parameters(self):
            return []

    class FakeVLM:
        language_model = FakeLanguageModel()

    source_path = Path(__file__).parents[1] / "vllm_mlx/text_model_from_vlm.py"
    spec = importlib.util.spec_from_file_location(
        "text_model_from_vlm_under_test", source_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.build_text_model(FakeVLM(), model_path, enable_mtp=False)

    assert len(quantize_calls) == 1
    assert quantize_calls[0]["group_size"] == 32
    assert quantize_calls[0]["bits"] == 8
    assert quantize_calls[0]["mode"] == expected_mode
