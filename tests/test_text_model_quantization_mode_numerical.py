# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight numerical check for VLM-to-TextModel mode transfer."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_mxfp8_global_and_affine_override_round_trip(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class TextModelArgs:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class TextModel(nn.Module):
        mtp = None

        def __init__(self, _args):
            super().__init__()
            self.embedding = nn.Embedding(4, 32)
            self.mxfp8_linear = nn.Linear(64, 64)
            self.affine_linear = nn.Linear(64, 64)

    mx.random.seed(0)
    source = TextModel(TextModelArgs())
    nn.quantize(
        source,
        group_size=32,
        bits=8,
        mode="mxfp8",
        class_predicate=lambda path, _module: (
            {"group_size": 64, "bits": 8} if path == "affine_linear" else True
        ),
    )
    mx.eval(source.parameters())

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "quantization": {
                    "group_size": 32,
                    "bits": 8,
                    "mode": "mxfp8",
                    "language_model.affine_linear": {"group_size": 64, "bits": 8},
                },
            }
        )
    )

    source_path = Path(__file__).parents[1] / "vllm_mlx/text_model_from_vlm.py"
    spec = importlib.util.spec_from_file_location(
        "text_model_quantization_numerical", source_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "_import_text_model_classes", lambda _kind: (TextModel, TextModelArgs)
    )

    class VLM:
        language_model = source

    derived = module.build_text_model(VLM(), model_path, enable_mtp=False)
    assert derived is not None
    assert derived.embedding.mode == "mxfp8"
    assert derived.mxfp8_linear.mode == "mxfp8"
    assert derived.affine_linear.mode == "affine"

    indices = mx.array([0])
    values = mx.random.normal((1, 64))
    for name, inputs in (
        ("embedding", indices),
        ("mxfp8_linear", values),
        ("affine_linear", values),
    ):
        expected = getattr(source, name)(inputs)
        actual = getattr(derived, name)(inputs)
        mx.eval(expected, actual)
        assert bool(mx.allclose(actual, expected, atol=0, rtol=0)), name
