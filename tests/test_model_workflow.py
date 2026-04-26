# SPDX-License-Identifier: Apache-2.0
"""Tests for model artifact inspection, acquisition, and conversion helpers."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx.model_workflow import (
    CONVERSION_MANIFEST_NAME,
    MODEL_MANIFEST_NAME,
    AcquisitionOptions,
    ConversionOptions,
    acquire_model,
    convert_model,
    inspect_model,
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
        with pytest.raises(FileExistsError):
            acquire_model(
                "org/model", options=AcquisitionOptions(target_dir=str(target))
            )

    mock_download.assert_not_called()


def test_convert_model_dry_run_records_mlx_lm_command(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    payload = convert_model(
        ConversionOptions(
            source_path=str(source),
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
            ConversionOptions(source_path=str(source), output_path=str(output))
        )

    assert payload["status"] == "succeeded"
    assert (output / CONVERSION_MANIFEST_NAME).exists()


def test_convert_model_failure_reports_status_and_stderr(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="conversion error")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = convert_model(
            ConversionOptions(source_path=str(source), output_path=str(output))
        )

    assert payload["status"] == "failed"
    assert payload["returncode"] == 1
    assert payload["stderr"] == "conversion error"
    assert "output_inspection" not in payload
    assert "manifest_path" not in payload


def test_inspect_gptq_model_is_not_detected_as_mlx(tmp_path):
    """GPTQ/AWQ quantization_config must not trigger has_mlx_signals."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)

    payload = inspect_model(str(tmp_path))

    assert payload["mlx"]["looks_like_mlx_artifact"] is False
    assert payload["mlx"]["needs_conversion"] is True
