# SPDX-License-Identifier: Apache-2.0
"""Prepare isolated, noninteractive clients for a loopback acceptance proxy.

These adapters target the documented CLI contracts checked in September 2026.
Unsupported flags must fail visibly on older clients. The caller owns process
timeouts, an allowlisted environment, isolated HOME/XDG directories, and cleanup.
Preparing a plan never launches a client or reads an existing client profile.
"""

import ipaddress
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

CLIENT_NAMES = ("opencode", "pi", "codex", "claude", "copilot", "cline", "openclaw")
LOCAL_KEY = "local-test"


@dataclass(frozen=True)
class ClientPlan:
    name: str
    protocol: str
    argv: list[str]
    env: dict[str, str]


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)
    return str(path)


def prepare_client(
    name: str,
    executable: str,
    root: Path,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    *,
    api_key: str = LOCAL_KEY,
) -> ClientPlan:
    """Write private configuration and return argv without executing commands."""
    if name not in CLIENT_NAMES:
        raise ValueError("Unknown acceptance client")
    if not Path(executable).is_absolute():
        raise ValueError("Client executable must be an absolute path")
    if not model.strip() or timeout <= 0:
        raise ValueError("A model and positive timeout are required")
    parsed = urlsplit(base_url)
    try:
        loopback = (
            parsed.hostname == "localhost"
            or ipaddress.ip_address(parsed.hostname or "").is_loopback
        )
        port = parsed.port
    except ValueError:
        loopback, port = False, None
    if (
        parsed.scheme != "http"
        or not loopback
        or port is None
        or parsed.path != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Client endpoint must be a loopback HTTP /v1 proxy URL")

    root = root.resolve()
    workspace = str(root / "workspace")
    state = root / name
    state.mkdir(mode=0o700, exist_ok=True)
    env: dict[str, str] = {}
    protocol = "chat"

    if name == "opencode":
        config = _write_json(
            state / "opencode.json",
            {
                "model": f"vllm-mlx/{model}",
                "small_model": f"vllm-mlx/{model}",
                "enabled_providers": ["vllm-mlx"],
                "provider": {
                    "vllm-mlx": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": base_url, "apiKey": api_key},
                        "models": {
                            model: {
                                "name": model,
                                "limit": {"context": 32768, "output": 4096},
                            }
                        },
                    }
                },
                "permission": {"*": "deny", "read": "allow", "edit": "allow"},
                # Title generation otherwise makes a separate tools-free model
                # request before the fixture task. Keep this run task-focused.
                "agent": {"title": {"disable": True}},
                "compaction": {"auto": False, "prune": False},
                "share": "disabled",
                "autoupdate": False,
                "snapshot": False,
                "formatter": False,
                "lsp": False,
                "plugin": [],
                "mcp": {},
                "instructions": [],
            },
        )
        env = {
            "OPENCODE_CONFIG": config,
            "OPENCODE_CONFIG_DIR": str(state),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
        }
        argv = [
            executable,
            "run",
            "--pure",
            "--model",
            f"vllm-mlx/{model}",
            "--format",
            "json",
            "--dir",
            workspace,
            prompt,
        ]
    elif name == "pi":
        _write_json(
            state / "models.json",
            {
                "providers": {
                    "vllm-mlx": {
                        "baseUrl": base_url,
                        "api": "openai-completions",
                        "apiKey": api_key,
                        "models": [
                            {
                                "id": model,
                                "reasoning": False,
                                "contextWindow": 32768,
                                "maxTokens": 4096,
                            }
                        ],
                    }
                }
            },
        )
        env = {"PI_CODING_AGENT_DIR": str(state), "PI_OFFLINE": "1"}
        argv = [
            executable,
            "--print",
            "--mode",
            "json",
            "--provider",
            "vllm-mlx",
            "--model",
            model,
            "--tools",
            "read,edit,write",
            "--thinking",
            "off",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--no-prompt-templates",
            "--no-themes",
            "--no-approve",
            prompt,
        ]
    elif name == "codex":
        protocol = "responses"
        env = {"CODEX_HOME": str(state), "VLLM_MLX_CLIENT_KEY": api_key}
        argv = [
            executable,
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--cd",
            workspace,
            "--model",
            model,
        ]
        # Codex's normal file-read workflow uses its command tool. Preserve the
        # OS sandbox rather than disabling that tool or bypassing permissions.
        config = {
            "model_provider": "vllm-mlx",
            "model_providers.vllm-mlx.name": "Local acceptance proxy",
            "model_providers.vllm-mlx.base_url": base_url,
            "model_providers.vllm-mlx.wire_api": "responses",
            "model_providers.vllm-mlx.env_key": "VLLM_MLX_CLIENT_KEY",
            "model_providers.vllm-mlx.requires_openai_auth": False,
            "model_providers.vllm-mlx.request_max_retries": 0,
            "model_providers.vllm-mlx.stream_max_retries": 0,
            "model_providers.vllm-mlx.stream_idle_timeout_ms": timeout * 1000,
            "approval_policy": "never",
            "sandbox_workspace_write.network_access": False,
            "allow_login_shell": False,
            "shell_environment_policy.inherit": "none",
            "shell_environment_policy.set.PATH": os.defpath,
            "project_doc_max_bytes": 0,
            "web_search": "disabled",
            "features.apps": False,
            "features.plugins": False,
            "features.hooks": False,
            "features.multi_agent": False,
            "features.remote_models": False,
            "features.skip_host_skill_discovery": True,
        }
        for key, value in config.items():
            argv.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        argv.append(prompt)
    elif name == "claude":
        protocol = "anthropic"
        env = {
            "CLAUDE_CONFIG_DIR": str(state),
            # Anthropic's SDK appends /v1/messages to an origin base URL.
            "ANTHROPIC_BASE_URL": base_url.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "API_TIMEOUT_MS": str(timeout * 1000),
        }
        argv = [
            executable,
            "--print",
            "--bare",
            "--restricted",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--tools",
            "Read,Edit,Write",
            "--allowedTools",
            "Read,Edit,Write",
            "--permission-mode",
            "acceptEdits",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--",
            prompt,
        ]
    elif name == "copilot":
        env = {
            "COPILOT_HOME": str(state),
            "COPILOT_PROVIDER_BASE_URL": base_url,
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_API_KEY": api_key,
            "COPILOT_MODEL": model,
            "COPILOT_OFFLINE": "true",
            "COPILOT_AUTO_UPDATE": "false",
        }
        argv = [
            executable,
            "--output-format=json",
            "--available-tools=view,edit,create,apply_patch",
            "--allow-tool=read",
            "--allow-tool=write",
            "--deny-tool=shell",
            "--disable-builtin-mcps",
            "--no-custom-instructions",
            "--no-auto-update",
            "--no-bash-env",
            "--no-remote",
            "--no-remote-export",
            "--no-ask-user",
            "--prompt",
            prompt,
        ]
    elif name == "cline":
        env = {
            "CLINE_COMMAND_PERMISSIONS": json.dumps({"deny": ["*"]}),
            "CLINE_SESSION_BACKEND_MODE": "local",
        }
        # Cline 3.0.61 reads this store beneath the root command's --data-dir.
        # Writing it directly keeps the proxy credential out of auth's argv.
        _write_json(
            state / "settings" / "providers.json",
            {
                "version": 1,
                "lastUsedProvider": "openai-compatible",
                "modes": {},
                "providers": {
                    "openai-compatible": {
                        "settings": {
                            "provider": "openai-compatible",
                            "apiKey": api_key,
                            "model": model,
                            "baseUrl": base_url,
                        },
                        "updatedAt": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "tokenSource": "manual",
                    }
                },
            },
        )
        argv = [
            executable,
            "--data-dir",
            str(state),
            "--cwd",
            workspace,
            "--provider",
            "openai-compatible",
            "--model",
            model,
            "--json",
            "--timeout",
            str(timeout),
            "--auto-approve",
            "true",
            "--",
            prompt,
        ]
    else:  # openclaw
        # OpenClaw's Chat replay policy rewrites call IDs. Its Responses
        # transport preserves standard call_ IDs for exact round-trip checks.
        protocol = "responses"
        config = _write_json(
            state / "openclaw.json",
            {
                "models": {
                    "providers": {
                        "vllm": {
                            "baseUrl": base_url,
                            "apiKey": api_key,
                            "api": "openai-responses",
                            "timeoutSeconds": timeout,
                            "models": [
                                {
                                    "id": model,
                                    "name": model,
                                    "reasoning": False,
                                    "input": ["text"],
                                    "contextWindow": 32768,
                                    "maxTokens": 4096,
                                    "cost": {
                                        "input": 0,
                                        "output": 0,
                                        "cacheRead": 0,
                                        "cacheWrite": 0,
                                    },
                                }
                            ],
                        }
                    }
                },
                "agents": {
                    "defaults": {
                        "workspace": workspace,
                        "model": {"primary": f"vllm/{model}", "fallbacks": []},
                    }
                },
                "tools": {
                    "allow": ["read", "edit", "write"],
                    "fs": {"workspaceOnly": True},
                },
                "plugins": {"allow": ["vllm"], "slots": {"memory": "none"}},
            },
        )
        env = {
            "OPENCLAW_STATE_DIR": str(state),
            # Keep startup in the runner's process group so its timeout can
            # reap the CLI before the temporary profile is removed.
            "OPENCLAW_NO_RESPAWN": "1",
            "NODE_DISABLE_COMPILE_CACHE": "1",
        }
        # --auth-env-only rejects --config. Empty HOME and explicit state keep
        # this pinned provider independent of existing CLI credential stores.
        argv = [
            executable,
            "agent",
            "exec",
            "--config",
            config,
            "--cwd",
            workspace,
            "--model",
            f"vllm/{model}",
            "--code-mode",
            "direct",
            "--json",
            "--timeout",
            str(timeout),
            "--",
            prompt,
        ]

    return ClientPlan(name, protocol, argv, env)
