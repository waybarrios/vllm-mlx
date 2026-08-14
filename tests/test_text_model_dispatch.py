# SPDX-License-Identifier: Apache-2.0
"""Text-model class selection for VLM-derived TextModels.

The dispatch used to match one exact ``model_type`` string and send everything
else to the Qwen3.5 text model. That is a guess, and a wrong guess does not
fail where it is made: it fails deep inside the chosen constructor with an
error naming neither the model nor the class, ``build_text_model`` returns
None, and the engine carries on with ``_text_model=None`` — a route quietly
losing its backend.

Concretely, Gemma 4 reports ``gemma4_text`` on some checkpoints and
``gemma4_unified_text`` on others. The latter reached ``qwen3_5.TextModelArgs``,
which leaves ``num_experts`` as None, and died on ``args.num_experts > 0``.
"""

import logging

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from vllm_mlx.text_model_from_vlm import (  # noqa: E402
    _import_text_model_classes,
    build_text_model,
)


@pytest.mark.parametrize(
    "model_type",
    ["gemma4_text", "gemma4_unified_text", "gemma4_unified", "gemma4"],
)
def test_every_gemma4_variant_gets_the_gemma4_text_model(model_type):
    """A family must cover its variants, not one spelling of it."""
    Model, ModelArgs = _import_text_model_classes(model_type)
    assert Model.__module__ == "mlx_lm.models.gemma4_text", (
        f"{model_type!r} selected {Model.__module__}.{Model.__qualname__}; "
        "a Gemma 4 config passed to another family dies on a field it has no "
        "opinion about"
    )
    assert ModelArgs.__module__ == "mlx_lm.models.gemma4_text"


def test_gemma4_unified_text_config_actually_constructs():
    """The regression, end to end: this config used to raise.

    ``qwen3_5.TextModelArgs.from_dict`` leaves ``num_experts`` as None for a
    Gemma 4 config, and ``qwen3_5.DecoderLayer.__init__`` compares it to 0:
    ``TypeError: '>' not supported between instances of 'NoneType' and 'int'``.
    """
    text_config = {
        "model_type": "gemma4_unified_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
    }
    _, ModelArgs = _import_text_model_classes(text_config["model_type"])
    args = ModelArgs.from_dict(text_config)
    assert getattr(args, "num_hidden_layers", None) == 2


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_6_text", "", "unknown"])
def test_unmatched_types_keep_the_generic_fallback(model_type):
    """Unknown families must not start raising — that would be a regression.

    qwen3_5.TextModel handles dense and MoE natively, so it stays the default.
    """
    Model, ModelArgs = _import_text_model_classes(model_type)
    assert Model.__module__ == "mlx_lm.models.qwen3_5"
    assert Model.__qualname__ == "TextModel"
    assert ModelArgs.__qualname__ == "TextModelArgs"


def test_failure_names_the_model_type_and_the_chosen_class(tmp_path, caplog):
    """The old log line was a bare TypeError from someone else's constructor.

    Without the model_type and the class that was picked, the only way to find
    out which family was guessed is to bisect the dispatch by hand.
    """
    (tmp_path / "config.json").write_text(
        '{"text_config": {"model_type": "gemma4_unified_text", '
        '"num_hidden_layers": "not-an-int"}}'
    )

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.ERROR, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None

    assert caplog.records, "a failed build must be logged"
    message = caplog.records[-1].getMessage()
    assert "gemma4_unified_text" in message, message
    assert "mlx_lm.models.gemma4_text" in message, message
    assert caplog.records[-1].exc_info is not None, "traceback must be preserved"


def test_missing_config_is_not_reported_as_a_build_failure(tmp_path, caplog):
    """No config.json is a "not applicable", not an error worth a traceback."""

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.ERROR, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None
    assert not caplog.records
