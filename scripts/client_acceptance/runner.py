# SPDX-License-Identifier: Apache-2.0
"""Run installed coding clients against an explicitly selected local model."""

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from urllib.parse import urlsplit

PROMPT = (
    "Read input.txt in the current workspace using a tool. "
    "Copy its token to output.txt as one line. Leave input.txt unchanged. "
    "Reply with that token after writing the file. Work only with those two files. "
    "Do not use the network."
)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    timed_out: bool
    version: str | None = None


def isolated_environment(root: Path) -> dict[str, str]:
    """Build a child environment without operator credentials or client profiles."""
    directories = {
        "HOME": root / "home",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_STATE_HOME": root / "state",
        "TMPDIR": root / "tmp",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {
        **{name: str(path) for name, path in directories.items()},
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "en_US.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "NO_COLOR": "1",
        "TERM": "dumb",
    }


def run_process(
    argv: list[str],
    env: dict[str, str],
    cwd: Path,
    timeout: float,
    *,
    capture_version: bool = False,
) -> ProcessResult:
    """Bound the process lifetime and terminate descendants on every exit path.

    Client transcripts are discarded. Only a numeric version is extracted when
    explicitly probing --version; arbitrary subprocess output never enters reports.
    """
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output if capture_version else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=max(0.01, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also stop helpers left behind by a successfully exited CLI.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        version = None
        if capture_version and not timed_out and process.returncode == 0:
            output.seek(0)
            match = re.search(
                rb"\b\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?\b", output.read(8192)
            )
            if match:
                version = match.group().decode("ascii")
        return ProcessResult(process.returncode, timed_out, version)


def _file_matches(path: Path, expected: bytes) -> bool:
    """Inspect a bounded regular file without following a final symlink or FIFO."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as file:
            if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                return False
            return file.read(len(expected) + 1) == expected
    except OSError:
        return False


def evaluate_run(
    returncode: int,
    timed_out: bool,
    workspace: Path,
    token: str,
    proof: dict,
) -> list[str]:
    reasons = []
    if timed_out:
        reasons.append("client_timeout")
    if returncode != 0:
        reasons.append("client_exit_nonzero")
    if not (
        proof.get("requests", 0) >= 2
        and proof.get("streamed")
        and proof.get("tool_calls", 0) > 0
        and proof.get("tool_results", 0) > 0
        and proof.get("completed")
        and proof.get("answer_matches")
        and not proof.get("errors")
    ):
        reasons.append("incomplete_protocol_evidence")
    expected = (token + "\n").encode("ascii")
    if not _file_matches(workspace / "input.txt", expected):
        reasons.append("input_modified")
    if not _file_matches(workspace / "output.txt", expected):
        reasons.append("incorrect_file_edit")
    return reasons


def _model_available(
    base_url: str, model: str, timeout: float, client_key: str
) -> bool:
    """Probe through the same proxy used by the client, without redirect handling."""
    url = urlsplit(base_url)
    connection = HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        headers = {"Authorization": "Bearer " + client_key} if client_key else {}
        connection.request("GET", url.path + "/models", headers=headers)
        response = connection.getresponse()
        data = response.read(1_048_577)
        if response.status != 200 or len(data) > 1_048_576:
            return False
        body = json.loads(data)
        return isinstance(body, dict) and any(
            isinstance(item, dict) and item.get("id") == model
            for item in body.get("data", [])
        )
    except (OSError, ValueError, TypeError, RecursionError, HTTPException):
        return False
    finally:
        connection.close()


def run_client(
    name: str,
    base_url: str,
    model: str,
    timeout: int,
    api_key: str = "",
    expected_version: str | None = None,
) -> dict:
    from .clients import LOCAL_KEY, prepare_client
    from .observe import RecordingProxy

    result = {
        "client": name,
        "status": "unavailable",
        "version": None,
        "expected_version": expected_version,
        "reasons": [],
        "evidence": {},
    }
    executable = shutil.which(name)
    if not executable:
        result["reasons"] = ["executable_missing"]
        return result
    executable = str(Path(executable).absolute())
    url = urlsplit(base_url)
    host = f"[{url.hostname}]" if url.hostname == "::1" else url.hostname
    base_url = f"http://{host}:{url.port or 80}/v1"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vllm-mlx-acceptance-") as directory:
        root = Path(directory)
        env = isolated_environment(root)
        workspace = root / "workspace"
        workspace.mkdir()
        token = secrets.token_hex(16)
        (workspace / "input.txt").write_text(token + "\n")
        (workspace / "output.txt").write_text("pending\n")
        try:
            # Preparing before the probe also isolates client-specific version startup.
            plan = prepare_client(
                name, executable, root, base_url, model, PROMPT, timeout
            )
            env.update(plan.env)
            version = run_process(
                [executable, "--version"],
                env,
                workspace,
                min(timeout, 10),
                capture_version=True,
            )
            result["version"] = version.version
            result["protocol"] = plan.protocol
            if version.version is None:
                result["reasons"] = ["version_probe_failed"]
                return result
            if expected_version and version.version != expected_version:
                result["reasons"] = ["client_version_mismatch"]
                return result
            with RecordingProxy(
                base_url,
                model,
                plan.protocol,
                api_key=api_key,
                timeout=max(0.01, timeout - (time.monotonic() - started)),
            ) as proxy:
                if not _model_available(
                    proxy.base_url, model, min(timeout, 5), proxy.client_key
                ):
                    result["reasons"] = ["server_or_model_unavailable"]
                    return result
                plan = prepare_client(
                    name,
                    executable,
                    root,
                    proxy.base_url,
                    model,
                    PROMPT,
                    timeout,
                    api_key=proxy.client_key or LOCAL_KEY,
                )
                env.update(plan.env)
                result["status"] = "failed"
                process = run_process(
                    plan.argv,
                    env,
                    workspace,
                    timeout - (time.monotonic() - started),
                )
                drained = proxy.wait_for_idle(
                    max(0.0, timeout - (time.monotonic() - started))
                )
                proof = proxy.evidence.summary(token)
                result["evidence"] = proof
                result["returncode"] = process.returncode
                result["reasons"] = evaluate_run(
                    process.returncode, process.timed_out, workspace, token, proof
                )
                if not drained:
                    result["reasons"].append("proxy_drain_timeout")
                if not result["reasons"]:
                    result["status"] = "passed"
        except (OSError, ValueError) as exc:
            # Exception messages can contain subprocess output or credentials.
            result["status"] = "failed"
            result["reasons"] = ["runner_error"]
            result["error_type"] = type(exc).__name__
        finally:
            result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def _revision(value: str) -> str:
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})", value):
        raise argparse.ArgumentTypeError(
            "use an immutable 40/64-digit revision or sha256 digest"
        )
    return value


def _positive_timeout(value: str) -> int:
    timeout = int(value)
    if not 1 <= timeout <= 3600:
        raise argparse.ArgumentTypeError("timeout must be between 1 and 3600 seconds")
    return timeout


def main(argv: list[str] | None = None) -> int:
    from .clients import CLIENT_NAMES
    from .observe import RecordingProxy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clients", nargs="+", choices=CLIENT_NAMES, default=["opencode", "pi"]
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--model", required=True, help="Exact ID advertised by /v1/models"
    )
    parser.add_argument("--model-revision", required=True, type=_revision)
    parser.add_argument("--server-revision", required=True, type=_revision)
    parser.add_argument(
        "--expect-version", action="append", default=[], metavar="CLIENT=VERSION"
    )
    parser.add_argument(
        "--timeout", type=_positive_timeout, default=180, help="Seconds per client"
    )
    parser.add_argument(
        "--api-key-env",
        default="VLLM_MLX_API_KEY",
        help="Environment variable holding the upstream server key",
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="Destination JSON report"
    )
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("the client runner requires macOS or Linux process groups")
    if not args.model.strip() or any(ord(char) < 32 for char in args.model):
        parser.error("model must be a nonempty printable ID")
    versions = {}
    for item in args.expect_version:
        name, separator, version = item.partition("=")
        if (
            not separator
            or name not in args.clients
            or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?", version)
        ):
            parser.error("--expect-version requires a selected CLIENT=VERSION")
        versions[name] = version
    try:
        # Validate even when every requested executable is missing.
        RecordingProxy(args.base_url, args.model, "chat")
    except ValueError:
        parser.error("base URL must be an HTTP loopback endpoint ending in /v1")
    key = os.environ.get(args.api_key_env, "")
    results = [
        run_client(
            name, args.base_url, args.model, args.timeout, key, versions.get(name)
        )
        for name in dict.fromkeys(args.clients)
    ]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": args.model_revision,
        "server_revision": args.server_revision,
        "revision_source": "operator_provided",
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for item in results:
        reasons = ", ".join(item["reasons"]) or "tool loop and file edit verified"
        print(f"{item['client']}: {item['status']} ({reasons})")
    if any(item["status"] == "failed" for item in results):
        return 1
    return 0 if all(item["status"] == "passed" for item in results) else 2


if __name__ == "__main__":
    sys.exit(main())
