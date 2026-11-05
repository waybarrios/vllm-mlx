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


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_qwen4_exp_stays_on_mlx_vlm_text_path(tmp_path, caplog, model_type):
    """Qwen4-Exp is not compatible with the generic Qwen3.5 TextModel."""
    (tmp_path / "config.json").write_text(
        '{"text_config": {"model_type": "' + model_type + '"}}'
    )

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.INFO, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None

    assert "mlx-vlm text path" in caplog.records[-1].getMessage()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def _install_recording_qwen3_5(monkeypatch):
    """Swap in a qwen3_5 module that records which configs reach it."""
    import sys
    import types

    from vllm_mlx import text_model_from_vlm

    seen: list[dict] = []

    class TextModelArgs:
        @classmethod
        def from_dict(cls, config):
            seen.append(config)
            return config

    class TextModel:
        def __init__(self, args):
            self.args = args
            self.mtp = None

        def load_weights(self, weights, strict=False):
            pass

        def train(self, mode=True):
            return self

    module = types.ModuleType("mlx_lm.models.qwen3_5")
    module.TextModel = TextModel
    module.TextModelArgs = TextModelArgs
    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5", module)
    monkeypatch.setattr(text_model_from_vlm.mlx.utils, "tree_flatten", lambda _p: [])
    return seen, TextModel


# Text-relevant fields of mlx-community/Qwen2-VL-2B-Instruct-4bit's real
# config.json (flat, no text_config): dense mrope attention, 12 heads over 1536,
# no hybrid linear-attention layout -- not something the Qwen3.5 skeleton fits.
_QWEN2_VL_CONFIG = {
    "architectures": ["Qwen2VLForConditionalGeneration"],
    "model_type": "qwen2_vl",
    "hidden_size": 1536,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "num_hidden_layers": 28,
    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    "vision_config": {"depth": 32, "embed_dim": 1280},
}


@pytest.mark.parametrize(
    "config",
    [
        _QWEN2_VL_CONFIG,
        {**_QWEN2_VL_CONFIG, "model_type": "qwen2_5_vl"},
        {"model_type": "qwen3_vl", "text_config": {"model_type": "qwen3_vl_text"}},
    ],
    ids=["qwen2_vl", "qwen2_5_vl", "qwen3_vl"],
)
def test_vlm_families_without_an_extractor_stay_on_mlx_vlm(
    tmp_path, monkeypatch, caplog, config
):
    """A resolved snapshot must not make an incompatible VLM extractable.

    Regression for #780 review: once the CLI passed the resolved snapshot into
    SimpleEngine, build_text_model could read Qwen2-VL's config.json and fed it
    to the generic Qwen3.5 TextModel. Text-only chat has to stay on the
    loaded mlx-vlm model instead.
    """
    import json

    seen, _ = _install_recording_qwen3_5(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps(config))

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.INFO, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None

    assert seen == [], "the Qwen3.5 skeleton must not be built for this family"
    assert "mlx-vlm text path" in caplog.records[-1].getMessage()


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}},
        {
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text", "num_experts": 256},
        },
    ],
    ids=["qwen3_5", "qwen3_5_moe"],
)
def test_qwen3_5_family_still_uses_the_extractor(tmp_path, monkeypatch, config):
    """Qwen3.5 / 3.6 / 3.8 snapshots keep the mlx-lm text route."""
    import json

    seen, TextModel = _install_recording_qwen3_5(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps(config))

    class _Vlm:
        class language_model:
            @staticmethod
            def parameters():
                return {}

    text_model = build_text_model(_Vlm(), tmp_path, enable_mtp=False)

    assert isinstance(text_model, TextModel)
    assert seen == [config["text_config"]]


class TestBuildTextModelPathContract:
    """Path normalization and MTP dispatch, without model/native execution."""

    def test_none_model_does_not_touch_the_path(self):
        class HostilePath:
            def __bool__(self):
                raise AssertionError("path inspected before the None-model return")

        assert build_text_model(None, HostilePath()) is None

    @pytest.mark.parametrize("failure", ["constructor", "exists"])
    def test_normalization_and_exists_errors_escape_the_build_try(
        self, monkeypatch, tmp_path, caplog, failure
    ):
        from pathlib import Path

        from vllm_mlx import text_model_from_vlm as module

        events = []

        def normalize(value):
            events.append(("normalize", value))
            if failure == "constructor":
                return Path(123)
            return tmp_path

        def exists(path):
            events.append(("exists", path))
            raise OSError("config existence failed")

        monkeypatch.setattr(module, "Path", normalize)
        monkeypatch.setattr(Path, "exists", exists)
        with pytest.raises(TypeError if failure == "constructor" else OSError):
            build_text_model(object(), "relative/model")
        assert events == [("normalize", "relative/model")] + (
            [("exists", tmp_path / "config.json")] if failure == "exists" else []
        )
        assert not caplog.records

    @pytest.mark.parametrize("path_kind", ["empty", "dot", "relative", "absolute"])
    @pytest.mark.parametrize("mtp_mode", ["disabled", "default", "loaded"])
    def test_normalized_path_identity_and_mtp_order(
        self, tmp_path, monkeypatch, caplog, path_kind, mtp_mode
    ):
        import json
        import sys
        import types
        from pathlib import Path

        from vllm_mlx import text_model_from_vlm as module

        monkeypatch.chdir(tmp_path)
        (tmp_path / "relative").mkdir()
        raw = {"empty": "", "dot": Path("."), "relative": "relative"}.get(
            path_kind, str(tmp_path)
        )
        normalized = Path(raw)
        config = {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1}
        (normalized / "config.json").write_text(json.dumps(config))
        seen, TextModel = _install_recording_qwen3_5(monkeypatch)
        events = []

        def normalize(value):
            events.append(("normalize", value))
            return normalized

        def exists(path):
            events.append(("exists", path))
            return True

        def load_mtp(path):
            assert path is normalized
            events.append(("mtp", path))
            return [("mtp.weight", "opaque")] if mtp_mode == "loaded" else []

        def inject(model, path, loaded_config):
            assert path is normalized
            assert isinstance(model, TextModel)
            assert loaded_config == config
            events.append(("inject", path))

        injector = types.ModuleType("vllm_mlx.patches.qwen3_5_mtp")
        injector.inject_mtp_support = inject
        monkeypatch.setitem(sys.modules, injector.__name__, injector)
        monkeypatch.setattr(module, "Path", normalize)
        monkeypatch.setattr(Path, "exists", exists)
        monkeypatch.setattr(module, "_load_mtp_weights", load_mtp)
        monkeypatch.setattr(
            TextModel,
            "load_weights",
            lambda self, weights, strict: events.append(("weights", weights, strict)),
        )
        monkeypatch.setattr(
            TextModel, "train", lambda self, mode: events.append(("train", mode))
        )
        vlm = types.SimpleNamespace(
            language_model=types.SimpleNamespace(parameters=lambda: {})
        )
        kwargs = {"enable_mtp": False} if mtp_mode == "disabled" else {}
        result = build_text_model(vlm, raw, **kwargs)
        if path_kind == "empty":
            assert result is None
            assert events == seen == []
            assert not caplog.records
            return

        assert isinstance(result, TextModel)
        assert seen == [config]
        expected = [("normalize", raw), ("exists", normalized / "config.json")]
        if mtp_mode != "disabled":
            expected.append(("mtp", normalized))
        expected.append(("weights", [], False))
        if mtp_mode == "loaded":
            expected.append(("weights", [("mtp.weight", "opaque")], False))
        if mtp_mode != "disabled":
            expected.append(("inject", normalized))
        expected.append(("train", False))
        assert events == expected
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert [(r.msg, r.args) for r in warnings] == (
            [("No MTP weights found in %s", (normalized.name,))]
            if mtp_mode == "default"
            else []
        )

    @pytest.mark.parametrize("config", ["{", '{"model_type": "qwen4_exp"}'])
    def test_malformed_and_unsupported_configs_do_not_construct(
        self, tmp_path, monkeypatch, caplog, config
    ):
        seen, _ = _install_recording_qwen3_5(monkeypatch)
        (tmp_path / "config.json").write_text(config)
        with caplog.at_level(logging.INFO, logger="vllm_mlx.text_model_from_vlm"):
            assert build_text_model(object(), str(tmp_path)) is None
        assert seen == []
        assert len(caplog.records) == 1
        record = caplog.records[0]
        if config == "{":
            assert record.levelno == logging.ERROR
            assert record.exc_info is not None
            assert "Failed to build TextModel" in record.getMessage()
        else:
            assert record.levelno == logging.INFO
            assert "mlx-vlm text path" in record.getMessage()

    def test_public_signature_and_type_hints(self):
        import inspect
        from pathlib import Path
        from typing import Any, get_type_hints

        signature = inspect.signature(build_text_model)
        assert list(signature.parameters) == ["vlm_model", "model_path", "enable_mtp"]
        assert signature.parameters["model_path"].default is inspect.Parameter.empty
        assert signature.parameters["enable_mtp"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["enable_mtp"].default is True
        assert get_type_hints(build_text_model) == {
            "vlm_model": Any,
            "model_path": str | Path,
            "enable_mtp": bool,
            "return": Any | None,
        }


class TestMtpWeightProducerContracts:
    """Real helper/builder contracts with opaque leaves, not native arrays."""

    @pytest.fixture
    def named_index(self, tmp_path):
        import json

        def write(weight_map):
            (tmp_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
            for shard in weight_map.values():
                (tmp_path / shard).touch()
            return tmp_path

        return write

    @pytest.fixture
    def recording_builder(self, tmp_path, monkeypatch):
        import json
        import sys
        import types

        from vllm_mlx import text_model_from_vlm as module

        events = []

        class Args:
            @classmethod
            def from_dict(cls, config):
                events.append("args")
                return config

        class Model:
            def __init__(self, args):
                events.append("construct")
                self.mtp = None

            def load_weights(self, weights, strict):
                events.append(("weights", weights, strict))

            def train(self, mode):
                events.append(("train", mode))

            def modules(self):
                events.append("modules")
                return []

        config = {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "quantization": {"bits": 4, "group_size": 32},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        monkeypatch.setattr(
            module, "_import_text_model_classes", lambda _: (Model, Args)
        )
        injector = types.ModuleType("vllm_mlx.patches.qwen3_5_mtp")
        injector.inject_mtp_support = lambda *args: events.append("inject")
        monkeypatch.setitem(sys.modules, injector.__name__, injector)
        monkeypatch.setitem(module.mx.__dict__, "eval", lambda _: events.append("eval"))
        monkeypatch.setitem(
            module.nn.__dict__,
            "quantize",
            lambda *args, **kwargs: events.append("quantize"),
        )
        vlm = types.SimpleNamespace(
            language_model=types.SimpleNamespace(parameters=lambda: {})
        )
        return module, vlm, events, Model

    @pytest.mark.parametrize(
        "kind", ["hostile", "metadata", "tuple", "mapping", "none", "list", "scalar"]
    )
    def test_rejects_non_dict_before_result_operations(
        self, tmp_path, monkeypatch, named_index, kind
    ):
        from collections import UserDict

        from vllm_mlx import text_model_from_vlm as module

        operations = []

        class Hostile:
            def touched(self, *args):
                operations.append("touched")
                raise AssertionError("non-dict result was inspected")

            __contains__ = __iter__ = __getitem__ = __repr__ = __str__ = touched

        key = "language_model.model.mtp.weight"
        results = {
            "hostile": Hostile(),
            "metadata": ({key: object()}, {"secret": "metadata"}),
            "tuple": (1, 2),
            "mapping": UserDict({key: object()}),
            "none": None,
            "list": [key],
            "scalar": 3,
        }
        named_index({key: "draft.npy"})
        calls = []

        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return results[kind]

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        with pytest.raises(TypeError) as caught:
            module._load_mtp_weights(tmp_path)
        assert str(caught.value) == (
            "MTP shard must contain a dictionary of named weight arrays"
        )
        assert operations == []
        assert calls == [((str(tmp_path / "draft.npy"),), {})]

    @pytest.mark.parametrize("suffix", ["safetensors", "npz", "gguf", "subclass"])
    def test_valid_named_results_preserve_grouping_names_and_leaf_identity(
        self, tmp_path, monkeypatch, named_index, suffix
    ):
        from vllm_mlx import text_model_from_vlm as module

        class NamedDict(dict):
            pass

        first, second, third = object(), object(), object()
        a = "language_model.model.mtp.a"
        b = "other.mtp.b"
        c = "model.mtp.a"
        shard = "first." + suffix
        named_index({a: shard, b: "second.npz", c: shard, "unrelated": shard})
        values = {
            shard: (NamedDict if suffix == "subclass" else dict)(
                {a: first, c: third, "extra": object()}
            ),
            "second.npz": {b: second},
        }
        calls = []

        def load(path, /, format=None, return_metadata=False, *, stream=None):
            calls.append((path, format, return_metadata, stream))
            return values[path.rsplit("/", 1)[-1]]

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        loaded = module._load_mtp_weights(tmp_path)
        assert [name for name, _ in loaded] == ["model.mtp.a", "model.mtp.a", b]
        assert all(
            value is expected
            for (_, value), expected in zip(loaded, [first, third, second])
        )
        assert calls == [
            (str(tmp_path / shard), None, False, None),
            (str(tmp_path / "second.npz"), None, False, None),
        ]

    @pytest.mark.parametrize(
        "case", ["no-index", "default-map", "unrelated", "empty", "absent"]
    )
    def test_permissive_empty_cases_do_not_become_completeness_validation(
        self, tmp_path, monkeypatch, named_index, case
    ):
        from vllm_mlx import text_model_from_vlm as module

        calls = []
        if case == "default-map":
            (tmp_path / "model.safetensors.index.json").write_text("{}")
        elif case != "no-index":
            named_index(
                {"unrelated" if case == "unrelated" else "model.mtp.a": "draft.npz"}
            )

        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return {} if case == "empty" else {"different": object()}

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        assert module._load_mtp_weights(tmp_path) == []
        assert calls == (
            [((str(tmp_path / "draft.npz"),), {})]
            if case in {"empty", "absent"}
            else []
        )

    def test_missing_shard_warns_and_repeated_json_key_keeps_last_value(
        self, tmp_path, monkeypatch, caplog
    ):
        from vllm_mlx import text_model_from_vlm as module

        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.mtp.a":"old.npz","model.mtp.a":"new.npz",'
            '"model.mtp.b":"missing.npz"}}'
        )
        (tmp_path / "new.npz").touch()
        leaf, calls = object(), []

        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return {"model.mtp.a": leaf}

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        result = module._load_mtp_weights(tmp_path)
        assert len(result) == 1 and result[0][0] == "model.mtp.a"
        assert result[0][1] is leaf
        assert calls == [((str(tmp_path / "new.npz"),), {})]
        assert [(r.levelno, r.msg, r.args) for r in caplog.records] == [
            (logging.WARNING, "MTP shard not found: %s", ("missing.npz",))
        ]

    def test_invalid_second_shard_aborts_without_partial_return_or_third_load(
        self, tmp_path, monkeypatch, named_index
    ):
        from vllm_mlx import text_model_from_vlm as module

        named_index({f"model.mtp.{i}": f"{i}.npz" for i in range(3)})
        calls = []

        def load(path):
            calls.append(path)
            return {"model.mtp.0": object()} if len(calls) == 1 else ({}, {})

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        with pytest.raises(
            TypeError,
            match="^MTP shard must contain a dictionary of named weight arrays$",
        ):
            module._load_mtp_weights(tmp_path)
        assert calls == [str(tmp_path / "0.npz"), str(tmp_path / "1.npz")]

    @pytest.mark.parametrize("failure", ["shape", "value-error", "os-error"])
    def test_builder_uses_real_helper_and_existing_logged_none_fallback(
        self, tmp_path, monkeypatch, named_index, recording_builder, caplog, failure
    ):
        module, vlm, events, _ = recording_builder
        secret = "private-result-metadata-payload"
        named_index({"model.mtp.a": secret + ".npz"})
        original = (
            ValueError("loader failure")
            if failure == "value-error"
            else OSError("loader failure")
        )

        def load(*args, **kwargs):
            events.append(("load", args, kwargs))
            if failure != "shape":
                raise original
            return ({"model.mtp.a": object()}, {secret: secret})

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        if failure != "shape":
            with pytest.raises(type(original)) as caught:
                module._load_mtp_weights(tmp_path)
            assert caught.value is original
            events.clear()
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            assert build_text_model(vlm, tmp_path) is None
        assert events == [
            "args",
            "construct",
            ("load", (str(tmp_path / (secret + ".npz")),), {}),
        ]
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.ERROR and record.exc_info is not None
        if failure == "shape":
            assert type(record.exc_info[1]) is TypeError
            assert (
                str(record.exc_info[1])
                == "MTP shard must contain a dictionary of named weight arrays"
            )
            formatted = logging.Formatter().format(record)
            assert secret not in formatted and str(tmp_path) not in formatted
        else:
            assert record.exc_info[1] is original

    @pytest.mark.parametrize("enable_mtp", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    def test_actual_flatten_names_drive_quantization_and_backbone_list_identity(
        self, tmp_path, monkeypatch, named_index, recording_builder, enable_mtp, nested
    ):
        import types

        module, vlm, events, Model = recording_builder
        weight, scales, other, draft = object(), object(), object(), object()
        tree = (
            {"layers": [{"weight": weight, "scales": scales}], "tail": (other,)}
            if nested
            else {}
        )
        vlm.language_model.parameters = lambda: tree
        actual_flatten = module.mlx.utils.tree_flatten
        flattened = []

        def flatten(*args, **kwargs):
            assert len(args) == 1 and args[0] is tree and kwargs == {}
            result = actual_flatten(*args, **kwargs)
            flattened.append(result)
            return result

        # Observe only the builder call; real flatten recursion uses its module global.
        monkeypatch.setattr(
            module,
            "mlx",
            types.SimpleNamespace(utils=types.SimpleNamespace(tree_flatten=flatten)),
        )
        named_index({"language_model.model.mtp.scales": "draft.npz"})

        def load(*args, **kwargs):
            events.append(("load", args, kwargs))
            return {"language_model.model.mtp.scales": draft}

        def quantize(model, *, bits, group_size, class_predicate):
            assert isinstance(model, Model) and (bits, group_size) == (4, 32)
            capable = types.SimpleNamespace(to_quantized=lambda: None)
            assert class_predicate("layers.0", capable) is nested
            assert class_predicate("model.mtp", capable) is enable_mtp
            assert class_predicate("absent", capable) is False
            assert class_predicate("layers.0", object()) is False
            events.append("quantize")

        monkeypatch.setitem(module.mx.__dict__, "load", load)
        monkeypatch.setitem(module.nn.__dict__, "quantize", quantize)
        assert isinstance(build_text_model(vlm, tmp_path, enable_mtp=enable_mtp), Model)
        assert len(flattened) == 1 and isinstance(flattened[0], list)
        assert [name for name, _ in flattened[0]] == (
            ["layers.0.weight", "layers.0.scales", "tail.0"] if nested else []
        )
        assert all(
            value is expected
            for (_, value), expected in zip(flattened[0], [weight, scales, other])
        )
        transfers = [
            event
            for event in events
            if isinstance(event, tuple) and event[0] == "weights"
        ]
        assert transfers[0][1] is flattened[0] and transfers[0][2] is False
        assert len(transfers) == (2 if enable_mtp else 1)
        if enable_mtp:
            assert transfers[1][1][0][0] == "model.mtp.scales"
            assert transfers[1][1][0][1] is draft and transfers[1][2] is False
        assert sum(isinstance(e, tuple) and e[0] == "load" for e in events) == int(
            enable_mtp
        )
        assert events[-3:] == [("train", False), "modules", "eval"]
