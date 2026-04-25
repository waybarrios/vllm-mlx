# SPDX-License-Identifier: Apache-2.0
"""Tests for model artifact inspection, acquisition, and conversion helpers."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_mlx.model_workflow import (
    CONVERSION_MANIFEST_NAME,
    MODEL_MANIFEST_NAME,
    QUALIFICATION_REQUEST_NAME,
    REGISTRATION_MANIFEST_NAME,
    AcquisitionOptions,
    ConversionOptions,
    QualificationOptions,
    RegistrationOptions,
    _drop_none,
    acquire_model,
    convert_model,
    inspect_model,
    qualify_model,
    register_model,
)


def test_inspect_local_model_reports_size_and_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "quantization": {"bits": 4, "group_size": 64},
                "max_position_embeddings": 32768,
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)

    payload = inspect_model(str(tmp_path))

    assert payload["source"] == "local"
    assert payload["file_count"] == 2
    assert payload["model_family"]["model_type"] == "qwen3"
    assert payload["mlx"]["looks_like_mlx_artifact"] is True
    assert payload["mlx"]["needs_conversion"] is False


def test_inspect_hf_model_uses_metadata_without_weight_download(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "qwen3_5_moe",
                    "mtp_num_hidden_layers": 1,
                    "max_position_embeddings": 1_000_000,
                }
            }
        )
    )
    siblings = [
        SimpleNamespace(rfilename="config.json", size=100),
        SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=1000),
        SimpleNamespace(rfilename="tokenizer.json", size=200),
    ]
    info = SimpleNamespace(sha="abc123", siblings=siblings)

    with (
        patch("vllm_mlx.model_workflow.HfApi") as mock_api,
        patch("vllm_mlx.model_workflow.hf_hub_download") as mock_download,
    ):
        mock_api.return_value.model_info.return_value = info
        mock_download.return_value = str(config_path)
        payload = inspect_model("org/model", revision="main")

    assert payload["revision"] == "abc123"
    assert payload["total_size_bytes"] == 1300
    assert payload["model_files_size_gb"] == 0.0
    assert payload["model_family"]["model_type"] == "qwen3_5_moe"
    assert payload["mlx"]["needs_conversion"] is True
    assert payload["warnings"] == [
        "very large advertised context; choose an explicit serving context before loading"
    ]


def test_inspect_gptq_model_not_detected_as_mlx(tmp_path):
    """GPTQ/AWQ models with quantization_config must not be treated as MLX artifacts."""
    model_dir = tmp_path / "gptq-source"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        )
    )
    (model_dir / "model.safetensors").write_bytes(b"x" * 1024)

    payload = inspect_model(str(model_dir))

    assert payload["mlx"]["looks_like_mlx_artifact"] is False
    assert payload["mlx"]["needs_conversion"] is True


def test_inspect_awq_model_not_detected_as_mlx(tmp_path):
    """AWQ models with quantization_config must not be treated as MLX artifacts."""
    model_dir = tmp_path / "awq-source"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        )
    )

    payload = inspect_model(str(model_dir))

    assert payload["mlx"]["looks_like_mlx_artifact"] is False
    assert payload["mlx"]["needs_conversion"] is True


def test_acquire_model_finalizes_target_and_writes_manifest(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"
    seen_env = {}

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization": {"bits": 4}})
        )
        (staging / "model.safetensors").write_bytes(b"weights")
        seen_env["fast_transfer"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        return str(staging)

    with (
        patch("vllm_mlx.model_workflow.find_spec", return_value=object()),
        patch("vllm_mlx.model_workflow.snapshot_download") as mock_download,
    ):
        mock_download.side_effect = fake_snapshot_download
        manifest = acquire_model(
            "org/model",
            options=AcquisitionOptions(
                revision="rev1",
                target_dir=str(target),
                staging_dir=str(staging_root),
                fast_transfer=True,
            ),
        )

    assert seen_env["fast_transfer"] == "1"
    assert target.exists()
    assert manifest["model_id"] == "org/model"
    assert manifest["path"] == str(target)
    assert manifest["fast_transfer"]["enabled"] is True
    manifest_path = target / MODEL_MANIFEST_NAME
    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text())
    assert saved["inspection"]["file_count"] == 2


def test_acquire_model_disables_fast_transfer_when_package_missing(tmp_path):
    target = tmp_path / "model-final"

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        return str(staging)

    with (
        patch("vllm_mlx.model_workflow.find_spec", return_value=None),
        patch("vllm_mlx.model_workflow.snapshot_download") as mock_download,
    ):
        mock_download.side_effect = fake_snapshot_download
        manifest = acquire_model(
            "org/model",
            options=AcquisitionOptions(target_dir=str(target), fast_transfer=True),
        )

    assert manifest["fast_transfer"]["requested"] is True
    assert manifest["fast_transfer"]["enabled"] is False
    assert "not installed" in manifest["fast_transfer"]["reason"]


def test_acquire_model_refuses_existing_target(tmp_path):
    target = tmp_path / "model-final"
    target.mkdir()

    with patch("vllm_mlx.model_workflow.snapshot_download") as mock_download:
        try:
            acquire_model(
                "org/model", options=AcquisitionOptions(target_dir=str(target))
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("expected FileExistsError")

    mock_download.assert_not_called()


def test_convert_model_dry_run_records_mlx_lm_command(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    payload = convert_model(
        ConversionOptions(
            model=str(source),
            output_path=str(output),
            quantize=True,
            q_bits=3,
            q_group_size=64,
            q_mode="affine",
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["backend"] == "mlx-lm"
    assert payload["recipe"]["q_bits"] == 3
    assert "--quantize" in payload["command"]
    assert "--q-bits" in payload["command"]
    assert str(output) in payload["command"]


def test_convert_model_success_writes_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    def fake_run(*args, **kwargs):
        output.mkdir()
        (output / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization": {"bits": 4}})
        )
        (output / "model.safetensors").write_bytes(b"weights")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = convert_model(
            ConversionOptions(model=str(source), output_path=str(output))
        )

    assert payload["status"] == "succeeded"
    assert (output / CONVERSION_MANIFEST_NAME).exists()


def test_convert_model_failure_returns_status_and_stderr(tmp_path):
    """Conversion failure must return status=failed with stderr captured."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="conversion error")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = convert_model(
            ConversionOptions(model=str(source), output_path=str(output))
        )

    assert payload["status"] == "failed"
    assert payload["returncode"] == 1
    assert payload["stderr"] == "conversion error"
    assert not (output / CONVERSION_MANIFEST_NAME).exists()


def test_read_json_handles_malformed_json(tmp_path):
    """_read_json must not raise on malformed JSON."""
    from vllm_mlx.model_workflow import _read_json

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert _read_json(bad) == {}


# --- Registration tests ---


def test_register_model_writes_manifest_from_artifact(tmp_path):
    artifact = tmp_path / "artifact-model"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "quantization": {"bits": 4}})
    )
    (artifact / MODEL_MANIFEST_NAME).write_text(
        json.dumps({"kind": "vllm-mlx-model-artifact", "model_id": "org/model"})
    )

    payload = register_model(
        RegistrationOptions(
            artifact_path=str(artifact),
            model_id="qwen-test",
            served_model_name="qwen-test-served",
            preset_alias="fast-qwen",
            mllm=True,
            tool_call_parser="qwen3_coder",
            reasoning_parser="qwen3",
            default_temperature=0.6,
            default_top_p=0.95,
            default_top_k=20,
            default_min_p=0.0,
            default_presence_penalty=0.0,
            default_repetition_penalty=1.0,
            chat_template_kwargs={"enable_thinking": True},
            feature_flags=["prefix_cache"],
        )
    )

    assert payload["kind"] == "vllm-mlx-model-registration"
    assert payload["model_id"] == "qwen-test"
    assert payload["served_model_name"] == "qwen-test-served"
    assert payload["preset_alias"] == "fast-qwen"
    assert payload["mllm"] is True
    assert payload["production_ready"] is False
    assert payload["qualification_required"] is True
    assert payload["serving_defaults"]["top_k"] == 20
    assert payload["serving_defaults"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert payload["parser_policy"]["reasoning_parser"] == "qwen3"
    assert payload["source_manifests"]["acquisition"]["payload"]["model_id"] == (
        "org/model"
    )
    assert (artifact / REGISTRATION_MANIFEST_NAME).exists()


def test_register_model_with_minimal_defaults(tmp_path):
    """register_model with only artifact_path must produce a valid manifest."""
    artifact = tmp_path / "minimal-model"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps({"model_type": "llama"}))

    payload = register_model(RegistrationOptions(artifact_path=str(artifact)))

    assert payload["kind"] == "vllm-mlx-model-registration"
    assert payload["model_id"] == "minimal-model"
    assert payload["served_model_name"] == "minimal-model"
    assert payload["mllm"] is None
    assert payload["preset_alias"] is None
    assert payload["serving_defaults"] == {}
    assert payload["parser_policy"] == {}
    assert payload["feature_flags"] == []
    assert payload["production_ready"] is False
    assert (artifact / REGISTRATION_MANIFEST_NAME).exists()


def test_register_model_requires_local_directory(tmp_path):
    missing = tmp_path / "missing"

    try:
        register_model(RegistrationOptions(artifact_path=str(missing)))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_register_model_rejects_file_path(tmp_path):
    """artifact_path must be a directory, not a file."""
    a_file = tmp_path / "not-a-dir.txt"
    a_file.write_text("hello")

    try:
        register_model(RegistrationOptions(artifact_path=str(a_file)))
    except NotADirectoryError:
        pass
    else:
        raise AssertionError("expected NotADirectoryError")


# --- Qualification tests ---


def test_qualify_model_dry_run_records_bench_command(tmp_path):
    output = tmp_path / QUALIFICATION_REQUEST_NAME

    payload = qualify_model(
        QualificationOptions(
            model_id="qwen-test",
            server_url="http://127.0.0.1:8090",
            workload_path="/tmp/workload.json",
            output_path=str(output),
            result_path="/tmp/results.json",
            repetitions=3,
            dry_run=True,
            extra_args=["--tag", "nightly"],
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["production_ready"] is False
    assert "--workload" in payload["command"]
    assert "/tmp/workload.json" in payload["command"]
    assert "--tag" in payload["command"]
    assert "environment" in payload
    assert "python" in payload["environment"]
    assert output.exists()


def test_qualify_model_runs_command_and_records_failure(tmp_path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=7, stdout="", stderr="bad workload")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = qualify_model(
            QualificationOptions(
                model_id="qwen-test",
                workload_path="/tmp/workload.json",
            )
        )

    assert payload["status"] == "failed"
    assert payload["returncode"] == 7
    assert payload["stderr"] == "bad workload"


def test_qualify_model_success_path(tmp_path):
    """qualification with returncode=0 should report succeeded."""
    output = tmp_path / "qual-result.json"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = qualify_model(
            QualificationOptions(
                model_id="qwen-test",
                output_path=str(output),
            )
        )

    assert payload["status"] == "succeeded"
    assert payload["returncode"] == 0
    assert payload["production_ready"] is False
    assert output.exists()


# --- _drop_none tests ---


def test_drop_none_preserves_zero_values():
    """Zero-valued defaults (0, 0.0, False, empty string) must survive _drop_none."""
    result = _drop_none(
        {
            "temperature": 0.0,
            "top_k": 0,
            "presence_penalty": 0.0,
            "flag": False,
            "name": "",
            "removed": None,
        }
    )
    assert result == {
        "temperature": 0.0,
        "top_k": 0,
        "presence_penalty": 0.0,
        "flag": False,
        "name": "",
    }
