# SPDX-License-Identifier: Apache-2.0
"""Model acquisition, inspection, and conversion workflow helpers.

The functions in this module intentionally avoid loading model weights. They
collect repository/file metadata, download artifacts, and record manifests so a
model can be qualified before it is served.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .utils.download import LLM_ALLOW_PATTERNS, MLLM_ALLOW_PATTERNS

MODEL_MANIFEST_NAME = "vllm_mlx_model_manifest.json"
CONVERSION_MANIFEST_NAME = "vllm_mlx_conversion_manifest.json"
REGISTRATION_MANIFEST_NAME = "vllm_mlx_registration_manifest.json"
QUALIFICATION_REQUEST_NAME = "vllm_mlx_qualification_request.json"

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class AcquisitionOptions:
    """Options for Hugging Face model acquisition."""

    revision: str | None = None
    target_dir: str | None = None
    staging_dir: str | None = None
    is_mllm: bool = False
    fast_transfer: bool = True
    local_files_only: bool = False


@dataclass(frozen=True)
class ConversionOptions:
    """Options for the mlx-lm conversion backend."""

    source_path: str
    output_path: str
    quantize: bool = False
    q_bits: int | None = None
    q_group_size: int | None = None
    q_mode: str | None = None
    quant_predicate: str | None = None
    dtype: str | None = None
    trust_remote_code: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RegistrationOptions:
    """Options for generating a portable model registration manifest."""

    artifact_path: str
    model_id: str | None = None
    served_model_name: str | None = None
    preset_alias: str | None = None
    output_path: str | None = None
    mllm: bool | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    feature_flags: list[str] | None = None


@dataclass(frozen=True)
class QualificationOptions:
    """Options for creating or running a bench-serve qualification handoff."""

    model_id: str
    server_url: str = "http://127.0.0.1:8080"
    workload_path: str | None = None
    output_path: str | None = None
    result_path: str | None = None
    repetitions: int | None = None
    dry_run: bool = False
    extra_args: list[str] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bytes_to_gb(size: int | float | None) -> float | None:
    if size is None:
        return None
    return round(float(size) / (1024**3), 3)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _local_file_inventory(path: Path) -> tuple[list[dict[str, Any]], int]:
    files = []
    total = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
        total += size
        files.append({"path": str(item.relative_to(path)), "size": size})
    return files, total


def _hf_file_inventory(
    model_id: str, *, revision: str | None, local_files_only: bool
) -> tuple[list[dict[str, Any]], int | None, str | None]:
    if local_files_only:
        return [], None, revision

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    files = []
    total = 0
    total_known = True
    for sibling in getattr(info, "siblings", []) or []:
        filename = getattr(sibling, "rfilename", None)
        if not filename:
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            total_known = False
        else:
            total += int(size)
        files.append({"path": filename, "size": size})
    return files, total if total_known else None, getattr(info, "sha", revision)


def _hf_config(
    model_id: str, *, revision: str | None, local_files_only: bool
) -> dict[str, Any]:
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    return _read_json(Path(config_path))


def _config_value(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get(key)
    return None


def _model_family(config: dict[str, Any]) -> dict[str, Any]:
    architectures = _config_value(config, "architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    max_context = (
        _config_value(config, "max_position_embeddings")
        or _config_value(config, "max_sequence_length")
        or _config_value(config, "seq_length")
        or _config_value(config, "model_max_length")
    )
    return {
        "model_type": _config_value(config, "model_type"),
        "architectures": architectures,
        "torch_dtype": _config_value(config, "torch_dtype"),
        "max_context": max_context,
        "quantization": config.get("quantization") or config.get("quantization_config"),
        "has_text_config": isinstance(config.get("text_config"), dict),
        "has_vision_config": isinstance(config.get("vision_config"), dict),
        "mtp_num_hidden_layers": _config_value(config, "mtp_num_hidden_layers"),
    }


def _estimate_fit(
    *,
    total_bytes: int | None,
    model_files_bytes: int | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_context = _model_family(config).get("max_context")
    warnings = []
    if isinstance(max_context, int) and max_context >= 262_144:
        warnings.append(
            "very large advertised context; choose an explicit serving context before loading"
        )

    # Conversion normally needs source weights, output weights, and temporary
    # shards. Keep this conservative without pretending to know architecture
    # residency exactly.
    disk_floor = None
    if total_bytes is not None:
        disk_floor = int(total_bytes * 2.2)

    memory_floor = model_files_bytes or total_bytes
    return {
        "download_size_gb": _bytes_to_gb(total_bytes),
        "model_file_size_gb": _bytes_to_gb(model_files_bytes),
        "estimated_conversion_disk_gb": _bytes_to_gb(disk_floor),
        "rough_load_memory_gb": _bytes_to_gb(memory_floor),
        "warnings": warnings,
    }


def _model_file_bytes(files: list[dict[str, Any]]) -> int | None:
    total = 0
    known = False
    for entry in files:
        path = str(entry.get("path", ""))
        if not path.endswith((".safetensors", ".bin", ".gguf")):
            continue
        size = entry.get("size")
        if size is None:
            return None
        known = True
        total += int(size)
    return total if known else None


def _looks_like_mlx_name(model: str, *, source: str) -> bool:
    name = model.lower() if source == "huggingface" else Path(model).name.lower()
    return (
        name.startswith("mlx-community/")
        or "-mlx" in name
        or "_mlx" in name
        or name.endswith("mlx")
    )


def _is_model_id(value: str) -> bool:
    return bool(_MODEL_ID_RE.fullmatch(value))


def _fast_transfer_env(requested: bool) -> tuple[dict[str, str], dict[str, Any]]:
    if not requested:
        return {}, {"requested": False, "enabled": False, "reason": "disabled"}
    if find_spec("hf_transfer") is None:
        return (
            {},
            {
                "requested": True,
                "enabled": False,
                "reason": "hf_transfer package is not installed",
            },
        )
    return (
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        {"requested": True, "enabled": True, "reason": "enabled"},
    )


def inspect_model(
    model: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Inspect a local model path or Hugging Face model id without loading weights."""
    model_path = Path(model).expanduser()
    warnings = []

    if model_path.exists():
        files, total_bytes = _local_file_inventory(model_path)
        config = _read_json(model_path / "config.json")
        resolved_revision = None
        source = "local"
        location = str(model_path)
    else:
        if not _is_model_id(model):
            raise ValueError(
                f"{model!r} is not an existing path or a Hugging Face model id"
            )
        source = "huggingface"
        location = model
        files, total_bytes, resolved_revision = _hf_file_inventory(
            model, revision=revision, local_files_only=local_files_only
        )
        try:
            config = _hf_config(
                model, revision=revision, local_files_only=local_files_only
            )
        except Exception as exc:
            config = {}
            warnings.append(f"could not read config.json: {exc}")

    model_files_bytes = _model_file_bytes(files)
    family = _model_family(config)
    estimate = _estimate_fit(
        total_bytes=total_bytes,
        model_files_bytes=model_files_bytes,
        config=config,
    )
    warnings.extend(estimate.pop("warnings"))
    has_name_signal = _looks_like_mlx_name(model, source=source) and (
        source == "huggingface" or bool(config)
    )
    has_mlx_signals = bool(family.get("quantization")) or has_name_signal

    return {
        "model": model,
        "source": source,
        "location": location,
        "revision": resolved_revision or revision,
        "inspected_at": _now_iso(),
        "file_count": len(files),
        "total_size_bytes": total_bytes,
        "total_size_gb": _bytes_to_gb(total_bytes),
        "model_files_size_gb": _bytes_to_gb(model_files_bytes),
        "model_family": family,
        "mlx": {
            "looks_like_mlx_artifact": has_mlx_signals,
            "needs_conversion": not has_mlx_signals,
        },
        "fit_estimate": estimate,
        "warnings": warnings,
    }


def acquire_model(
    model_id: str,
    *,
    options: AcquisitionOptions | None = None,
) -> dict[str, Any]:
    """Download a model repository and write a finalized artifact manifest."""
    if options is None:
        options = AcquisitionOptions()
    if not _is_model_id(model_id):
        raise ValueError(f"{model_id!r} is not a Hugging Face model id")

    allow_patterns = MLLM_ALLOW_PATTERNS if options.is_mllm else LLM_ALLOW_PATTERNS
    env_updates, fast_transfer = _fast_transfer_env(options.fast_transfer)

    old_env = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    try:
        if options.target_dir:
            target = Path(options.target_dir).expanduser()
            if target.exists():
                raise FileExistsError(f"target path already exists: {target}")
            staging_root = (
                Path(options.staging_dir).expanduser()
                if options.staging_dir
                else target.parent
            )
            staging_root.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=staging_root)
            )
            try:
                downloaded = Path(
                    snapshot_download(
                        model_id,
                        revision=options.revision,
                        allow_patterns=allow_patterns,
                        local_dir=str(staging),
                        local_files_only=options.local_files_only,
                    )
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(target))
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            final_path = target
        else:
            final_path = Path(
                snapshot_download(
                    model_id,
                    revision=options.revision,
                    allow_patterns=allow_patterns,
                    local_files_only=options.local_files_only,
                )
            )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    inspection = inspect_model(str(final_path), revision=options.revision)
    manifest = {
        "kind": "vllm-mlx-model-artifact",
        "model_id": model_id,
        "revision": options.revision,
        "path": str(final_path),
        "created_at": _now_iso(),
        "allow_patterns": allow_patterns,
        "fast_transfer": fast_transfer,
        "local_files_only": options.local_files_only,
        "inspection": inspection,
    }
    manifest_path = final_path / MODEL_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _conversion_command(options: ConversionOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        options.source_path,
        "--mlx-path",
        options.output_path,
    ]
    if options.quantize:
        command.append("--quantize")
    if options.q_bits is not None:
        command.extend(["--q-bits", str(options.q_bits)])
    if options.q_group_size is not None:
        command.extend(["--q-group-size", str(options.q_group_size)])
    if options.q_mode:
        command.extend(["--q-mode", options.q_mode])
    if options.quant_predicate:
        command.extend(["--quant-predicate", options.quant_predicate])
    if options.dtype:
        command.extend(["--dtype", options.dtype])
    if options.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def convert_model(options: ConversionOptions) -> dict[str, Any]:
    """Run mlx-lm conversion and record the exact recipe."""
    command = _conversion_command(options)
    started = _now_iso()
    source_inspection = inspect_model(options.source_path)
    result = {
        "kind": "vllm-mlx-conversion",
        "backend": "mlx-lm",
        "command": command,
        "source_path": options.source_path,
        "output_path": options.output_path,
        "started_at": started,
        "dry_run": options.dry_run,
        "recipe": {
            "quantize": options.quantize,
            "q_bits": options.q_bits,
            "q_group_size": options.q_group_size,
            "q_mode": options.q_mode,
            "quant_predicate": options.quant_predicate,
            "dtype": options.dtype,
            "trust_remote_code": options.trust_remote_code,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "source_inspection": source_inspection,
    }

    if options.dry_run:
        result["status"] = "dry_run"
        return result

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result["returncode"] = completed.returncode
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    result["completed_at"] = _now_iso()
    result["status"] = "succeeded" if completed.returncode == 0 else "failed"
    if completed.returncode != 0:
        return result

    output_path = Path(options.output_path).expanduser()
    output_inspection = inspect_model(str(output_path))
    result["output_inspection"] = output_inspection
    manifest_path = output_path / CONVERSION_MANIFEST_NAME
    _write_json(manifest_path, result)
    result["manifest_path"] = str(manifest_path)
    return result


def _existing_manifests(path: Path) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for name, key in (
        (MODEL_MANIFEST_NAME, "acquisition"),
        (CONVERSION_MANIFEST_NAME, "conversion"),
    ):
        manifest_path = path / name
        if manifest_path.exists():
            manifests[key] = {
                "path": str(manifest_path),
                "payload": _read_json(manifest_path),
            }
    return manifests


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def register_model(options: RegistrationOptions) -> dict[str, Any]:
    """Write a portable registration manifest for a finalized local artifact.

    This deliberately does not mutate a production registry. The manifest is a
    handoff artifact that Ops or a deployment tool can apply after qualification.
    """
    artifact = Path(options.artifact_path).expanduser()
    if not artifact.exists():
        raise FileNotFoundError(f"artifact path does not exist: {artifact}")
    if not artifact.is_dir():
        raise NotADirectoryError(f"artifact path must be a directory: {artifact}")

    inspection = inspect_model(str(artifact))
    model_id = options.model_id or artifact.name
    serving_defaults = _drop_none(
        {
            "temperature": options.default_temperature,
            "top_p": options.default_top_p,
            "top_k": options.default_top_k,
            "min_p": options.default_min_p,
            "presence_penalty": options.default_presence_penalty,
            "repetition_penalty": options.default_repetition_penalty,
            "chat_template_kwargs": options.chat_template_kwargs,
        }
    )
    parser_policy = _drop_none(
        {
            "tool_call_parser": options.tool_call_parser,
            "reasoning_parser": options.reasoning_parser,
        }
    )
    payload = {
        "kind": "vllm-mlx-model-registration",
        "schema_version": 1,
        "created_at": _now_iso(),
        "model_id": model_id,
        "served_model_name": options.served_model_name or model_id,
        "preset_alias": options.preset_alias,
        "artifact_path": str(artifact),
        "mllm": options.mllm,
        "feature_flags": options.feature_flags or [],
        "serving_defaults": serving_defaults,
        "parser_policy": parser_policy,
        "inspection": inspection,
        "source_manifests": _existing_manifests(artifact),
        "qualification_required": True,
        "production_ready": False,
    }

    output = (
        Path(options.output_path).expanduser()
        if options.output_path
        else artifact / REGISTRATION_MANIFEST_NAME
    )
    _write_json(output, payload)
    payload["manifest_path"] = str(output)
    return payload


def _qualification_command(options: QualificationOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "bench-serve",
        "--url",
        options.server_url,
        "--model",
        options.model_id,
        "--format",
        "json",
    ]
    if options.workload_path:
        command.extend(["--workload", options.workload_path])
    if options.repetitions is not None:
        command.extend(["--repetitions", str(options.repetitions)])
    if options.result_path:
        command.extend(["--output", options.result_path])
    if options.extra_args:
        command.extend(options.extra_args)
    return command


def qualify_model(options: QualificationOptions) -> dict[str, Any]:
    """Create or run a bench-serve qualification handoff."""
    command = _qualification_command(options)
    payload = {
        "kind": "vllm-mlx-model-qualification",
        "schema_version": 1,
        "created_at": _now_iso(),
        "model_id": options.model_id,
        "server_url": options.server_url,
        "workload_path": options.workload_path,
        "result_path": options.result_path,
        "repetitions": options.repetitions,
        "dry_run": options.dry_run,
        "command": command,
        "production_ready": False,
    }

    if not options.dry_run:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        payload["returncode"] = completed.returncode
        payload["stdout"] = completed.stdout
        payload["stderr"] = completed.stderr
        payload["completed_at"] = _now_iso()
        payload["status"] = "succeeded" if completed.returncode == 0 else "failed"
    else:
        payload["status"] = "dry_run"

    if options.output_path:
        output = Path(options.output_path).expanduser()
        _write_json(output, payload)
        payload["manifest_path"] = str(output)
    return payload
