# SPDX-License-Identifier: Apache-2.0
"""Check the local configuration passed across each real client boundary."""

import json
import stat
from datetime import datetime
from pathlib import Path

import pytest

from scripts.client_acceptance.clients import prepare_client

CLIENTS = ("opencode", "pi", "codex", "claude", "copilot", "cline", "openclaw")
BASE_URL = "http://127.0.0.1:8123/v1"
MODEL = 'org/model-"quoted"-Ω'
PROMPT = "Read input.txt, copy its token to output.txt, and reply with it."


def prepare(tmp_path, name, **overrides):
    (tmp_path / "workspace").mkdir(exist_ok=True)
    (tmp_path / "home").mkdir(exist_ok=True)
    kwargs = dict(
        name=name,
        executable="/opt/client bin/agent",
        root=tmp_path,
        base_url=BASE_URL,
        model=MODEL,
        prompt=PROMPT,
        timeout=90,
    )
    kwargs.update(overrides)
    return prepare_client(**kwargs)


def read_json(path):
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize("name", CLIENTS)
def test_preparation_preserves_arguments_without_reading_ambient_credentials(
    tmp_path, monkeypatch, name
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-be-copied")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-secret-must-not-be-copied")
    plan = prepare(tmp_path, name)
    assert plan.name == name
    assert plan.argv[0] == "/opt/client bin/agent"
    assert PROMPT in plan.argv
    serialized = json.dumps([plan.argv, plan.env])
    serialized += "".join(
        path.read_text() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert "ambient-secret-must-not-be-copied" not in serialized
    assert "local-test" in serialized
    assert not (tmp_path / "workspace" / ".git").exists()


@pytest.mark.parametrize("name", CLIENTS)
def test_preparing_again_replaces_the_proxy_endpoint(tmp_path, name):
    prepare(tmp_path, name)
    plan = prepare(tmp_path, name, base_url="http://127.0.0.1:9123/v1")
    serialized = json.dumps([plan.argv, plan.env])
    serialized += "".join(
        path.read_text() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert "http://127.0.0.1:8123" not in serialized
    assert "http://127.0.0.1:9123" in serialized


@pytest.mark.parametrize("name", CLIENTS)
def test_per_run_credential_reaches_private_config_without_process_arguments(
    tmp_path, name
):
    credential = "inert-per-run-proxy-credential"
    plan = prepare(tmp_path, name, api_key=credential)
    assert credential not in json.dumps(plan.argv)
    env_keys = {
        "codex": "VLLM_MLX_CLIENT_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "copilot": "COPILOT_PROVIDER_API_KEY",
    }
    config_keys = {
        "opencode": ("opencode.json", ("provider", "vllm-mlx", "options", "apiKey")),
        "pi": ("models.json", ("providers", "vllm-mlx", "apiKey")),
        "cline": (
            "settings/providers.json",
            ("providers", "openai-compatible", "settings", "apiKey"),
        ),
        "openclaw": ("openclaw.json", ("models", "providers", "vllm", "apiKey")),
    }
    if name in env_keys:
        assert plan.env[env_keys[name]] == credential
    else:
        filename, keys = config_keys[name]
        path = tmp_path / name / filename
        value = read_json(path)
        for key in keys:
            value = value[key]
        assert value == credential
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_opencode_pins_chat_endpoint_and_denies_non_file_tools(tmp_path):
    plan = prepare(tmp_path, "opencode")
    config = read_json(plan.env["OPENCODE_CONFIG"])
    provider = config["provider"]["vllm-mlx"]
    assert plan.protocol == "chat"
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == BASE_URL
    assert MODEL in provider["models"]
    assert config["model"] == config["small_model"] == f"vllm-mlx/{MODEL}"
    assert config["enabled_providers"] == ["vllm-mlx"]
    assert config["permission"]["*"] == "deny"
    assert config["permission"]["read"] == "allow"
    assert config["permission"]["edit"] == "allow"
    assert config["share"] == "disabled"
    assert plan.env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert plan.env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    assert "--pure" in plan.argv


def test_opencode_disables_title_requests_and_automatic_compaction(tmp_path):
    plan = prepare(tmp_path, "opencode")
    config = read_json(plan.env["OPENCODE_CONFIG"])
    assert config["agent"]["title"]["disable"] is True
    assert config["compaction"]["auto"] is False
    assert config["compaction"]["prune"] is False
    assert config["small_model"] == f"vllm-mlx/{MODEL}"
    assert config["snapshot"] is False


def test_pi_uses_custom_provider_with_discovery_disabled(tmp_path):
    plan = prepare(tmp_path, "pi")
    config = read_json(Path(plan.env["PI_CODING_AGENT_DIR"]) / "models.json")
    provider = config["providers"]["vllm-mlx"]
    assert plan.protocol == "chat"
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == BASE_URL
    assert provider["models"][0]["id"] == MODEL
    assert "--api-key" not in plan.argv
    assert plan.argv[plan.argv.index("--tools") + 1] == "read,edit,write"
    assert plan.env["PI_OFFLINE"] == "1"
    for flag in (
        "--no-extensions",
        "--no-skills",
        "--no-context-files",
        "--no-prompt-templates",
        "--no-session",
    ):
        assert flag in plan.argv


def test_codex_selects_responses_and_preserves_sandbox(tmp_path):
    plan = prepare(tmp_path, "codex")
    overrides = {
        plan.argv[index + 1].split("=", 1)[0]: plan.argv[index + 1].split("=", 1)[1]
        for index, value in enumerate(plan.argv)
        if value == "-c"
    }
    assert plan.protocol == "responses"
    assert json.loads(overrides["model_providers.vllm-mlx.base_url"]) == BASE_URL
    assert json.loads(overrides["model_providers.vllm-mlx.wire_api"]) == "responses"
    assert overrides["model_providers.vllm-mlx.requires_openai_auth"] == "false"
    assert overrides["sandbox_workspace_write.network_access"] == "false"
    assert overrides["project_doc_max_bytes"] == "0"
    assert plan.argv[plan.argv.index("--model") + 1] == MODEL
    assert plan.argv[plan.argv.index("--sandbox") + 1] == "workspace-write"
    assert "--ignore-user-config" in plan.argv
    assert "--ignore-rules" in plan.argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in plan.argv
    assert Path(plan.env["CODEX_HOME"]).is_relative_to(tmp_path)


def test_claude_uses_messages_origin_and_api_key_only_bare_mode(tmp_path):
    plan = prepare(tmp_path, "claude")
    assert plan.protocol == "anthropic"
    assert plan.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8123"
    assert plan.env["ANTHROPIC_API_KEY"] == "local-test"
    assert "--bare" in plan.argv
    assert "--restricted" in plan.argv
    assert "--strict-mcp-config" in plan.argv
    assert plan.argv[plan.argv.index("--setting-sources") + 1] == ""
    assert plan.argv[plan.argv.index("--tools") + 1] == "Read,Edit,Write"
    assert plan.argv[plan.argv.index("--model") + 1] == MODEL


def test_copilot_uses_offline_byok_and_file_tool_allowlist(tmp_path):
    plan = prepare(tmp_path, "copilot")
    assert plan.protocol == "chat"
    assert plan.env["COPILOT_PROVIDER_BASE_URL"] == BASE_URL
    assert plan.env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert plan.env["COPILOT_MODEL"] == MODEL
    assert plan.env["COPILOT_OFFLINE"] == "true"
    assert "--available-tools=view,edit,create,apply_patch" in plan.argv
    assert "--deny-tool=shell" in plan.argv
    assert "--disable-builtin-mcps" in plan.argv
    assert "--no-custom-instructions" in plan.argv


def test_cline_run_consumes_private_provider_configuration(tmp_path):
    plan = prepare(tmp_path, "cline")
    assert plan.protocol == "chat"
    state = Path(plan.argv[plan.argv.index("--data-dir") + 1])
    config = read_json(state / "settings" / "providers.json")
    assert config["version"] == 1
    assert config["lastUsedProvider"] == "openai-compatible"
    assert config["modes"] == {}
    provider = config["providers"]["openai-compatible"]
    assert provider["settings"] == {
        "provider": "openai-compatible",
        "apiKey": "local-test",
        "model": MODEL,
        "baseUrl": BASE_URL,
    }
    assert provider["tokenSource"] == "manual"
    assert datetime.fromisoformat(provider["updatedAt"].replace("Z", "+00:00")).tzinfo
    assert plan.argv[plan.argv.index("--provider") + 1] == "openai-compatible"
    assert plan.argv[plan.argv.index("--model") + 1] == MODEL
    assert plan.argv[plan.argv.index("--timeout") + 1] == "90"
    assert json.loads(plan.env["CLINE_COMMAND_PERMISSIONS"])["deny"] == ["*"]


def test_openclaw_pins_config_in_embedded_exec_without_channel_delivery(tmp_path):
    plan = prepare(tmp_path, "openclaw")
    config = read_json(plan.argv[plan.argv.index("--config") + 1])
    provider = config["models"]["providers"]["vllm"]
    assert plan.protocol == "responses"
    assert plan.argv[1:3] == ["agent", "exec"]
    assert provider["baseUrl"] == BASE_URL
    assert provider["api"] == "openai-responses"
    assert provider["models"][0]["id"] == MODEL
    assert config["agents"]["defaults"]["model"]["primary"] == f"vllm/{MODEL}"
    assert config["tools"]["allow"] == ["read", "edit", "write"]
    assert config["tools"]["fs"]["workspaceOnly"] is True
    assert config["plugins"]["allow"] == ["vllm"]
    assert "--deliver" not in plan.argv
    assert "--auth-env-only" not in plan.argv  # Incompatible with --config upstream.
    assert Path(plan.env["OPENCLAW_STATE_DIR"]).is_relative_to(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "unknown"},
        {"executable": "relative-client"},
        {"model": ""},
        {"timeout": 0},
        {"base_url": "https://example.com/v1"},
        {"base_url": "http://user:secret@127.0.0.1:8000/v1"},
        {"base_url": "http://127.0.0.1:8000/v1?secret=value"},
    ],
)
def test_invalid_launch_inputs_fail_before_writing_configuration(tmp_path, overrides):
    overrides = dict(overrides)
    with pytest.raises(ValueError):
        prepare(tmp_path, overrides.pop("name", "pi"), **overrides)
    assert not any(path.is_file() for path in tmp_path.rglob("*"))
