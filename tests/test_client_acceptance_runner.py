# SPDX-License-Identifier: Apache-2.0
"""Portable runner checks. These do not establish real-model acceptance."""

import json
import os
import subprocess
import sys
import time

import pytest

from scripts.client_acceptance.runner import (
    evaluate_run,
    isolated_environment,
    main,
    run_client,
    run_process,
)


@pytest.fixture
def workspace(tmp_path):
    work = tmp_path / "workspace"
    work.mkdir()
    (work / "input.txt").write_text("fixture-token\n")
    (work / "output.txt").write_text("fixture-token\n")
    return work


@pytest.fixture
def proof():
    return {
        "requests": 3,
        "streamed": True,
        "tool_calls": 2,
        "tool_results": 2,
        "completed": True,
        "answer_matches": True,
        "errors": [],
    }


def test_pass_requires_protocol_evidence_and_actual_edit(workspace, proof):
    assert evaluate_run(0, False, workspace, "fixture-token", proof) == []
    (workspace / "output.txt").write_text("pending\n")
    assert "incorrect_file_edit" in evaluate_run(
        0, False, workspace, "fixture-token", proof
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("requests", 1),
        ("streamed", False),
        ("tool_calls", 0),
        ("tool_results", 0),
        ("completed", False),
        ("answer_matches", False),
        ("errors", ["upstream_error"]),
    ],
)
def test_successful_exit_cannot_replace_protocol_proof(workspace, proof, field, value):
    proof[field] = value
    assert "incomplete_protocol_evidence" in evaluate_run(
        0, False, workspace, "fixture-token", proof
    )


def test_timeout_and_nonzero_exit_fail_even_after_correct_edit(workspace, proof):
    reasons = evaluate_run(7, True, workspace, "fixture-token", proof)
    assert "client_timeout" in reasons
    assert "client_exit_nonzero" in reasons


def test_changed_input_and_symlink_output_are_rejected(workspace, proof, tmp_path):
    (workspace / "input.txt").write_text("changed\n")
    target = tmp_path / "outside.txt"
    target.write_text("fixture-token\n")
    (workspace / "output.txt").unlink()
    (workspace / "output.txt").symlink_to(target)
    reasons = evaluate_run(0, False, workspace, "fixture-token", proof)
    assert "input_modified" in reasons
    assert "incorrect_file_edit" in reasons


def test_non_regular_output_is_rejected_without_blocking(workspace, proof):
    (workspace / "output.txt").unlink()
    os.mkfifo(workspace / "output.txt")
    assert "incorrect_file_edit" in evaluate_run(
        0, False, workspace, "fixture-token", proof
    )


def test_child_environment_excludes_ambient_secrets_and_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inert-parent-sentinel")
    monkeypatch.setenv("GITHUB_TOKEN", "inert-parent-sentinel")
    monkeypatch.setenv("HTTP_PROXY", "http://unwanted.invalid")
    monkeypatch.setenv("NODE_OPTIONS", "--require=unwanted.js")
    env = isolated_environment(tmp_path)
    output = subprocess.check_output(
        [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
        env=env,
        text=True,
    )
    child = json.loads(output)
    for name in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "HTTP_PROXY", "NODE_OPTIONS"):
        assert name not in child
    assert child["HOME"] == str(tmp_path / "home")
    assert child["XDG_CONFIG_HOME"].startswith(str(tmp_path))
    assert child["TMPDIR"] == str(tmp_path / "tmp")


def test_process_timeout_kills_descendant_before_delayed_write(tmp_path):
    marker = tmp_path / "escaped.txt"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    )
    result = run_process(
        [sys.executable, "-c", parent],
        isolated_environment(tmp_path),
        tmp_path,
        timeout=0.15,
    )
    assert result.timed_out
    time.sleep(1.1)
    assert not marker.exists()


def test_process_output_is_not_retained_in_result(tmp_path):
    result = run_process(
        [sys.executable, "-c", "print('inert-output-sentinel')"],
        isolated_environment(tmp_path),
        tmp_path,
        timeout=5,
    )
    assert result.returncode == 0
    assert "inert-output-sentinel" not in repr(result)


def test_missing_clients_produce_report_and_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    report_path = tmp_path / "report.json"
    result = main(
        [
            "--clients",
            "pi",
            "opencode",
            "--model",
            "served-model",
            "--model-revision",
            "a" * 40,
            "--server-revision",
            "b" * 40,
            "--report",
            str(report_path),
        ]
    )
    assert result == 2
    report = json.loads(report_path.read_text())
    assert [item["status"] for item in report["results"]] == [
        "unavailable",
        "unavailable",
    ]
    assert all(item["reasons"] == ["executable_missing"] for item in report["results"])
    assert report["model_revision"] == "a" * 40
    assert report["revision_source"] == "operator_provided"


def test_mutable_model_revision_is_rejected_before_launch(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--model",
                "served-model",
                "--model-revision",
                "main",
                "--server-revision",
                "b" * 40,
                "--report",
                str(tmp_path / "r.json"),
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", ["version_mismatch", "nested_discovery"])
def test_unavailable_prerequisites_preserve_report_without_running_task(
    tmp_path, monkeypatch, failure
):
    from tests.test_client_acceptance_observe import upstream

    marker = tmp_path / "task-started"
    executable = tmp_path / "pi"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "if '--version' in sys.argv:\n    print('pi 1.2.3')\n"
        f"else:\n    Path({str(marker)!r}).touch()\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    nesting = 10_000
    body = b'{"data":[],"extra":' + b"[" * nesting + b"0" + b"]" * nesting + b"}"
    report_path = tmp_path / "report.json"
    with upstream([(200, {"Content-Type": "application/json"}, body)]) as (
        url,
        received,
    ):
        status = main(
            [
                "--clients",
                "pi",
                "--base-url",
                url,
                "--model",
                "served-model",
                "--model-revision",
                "a" * 40,
                "--server-revision",
                "b" * 40,
                "--expect-version",
                "pi=9.9.9" if failure == "version_mismatch" else "pi=1.2.3",
                "--report",
                str(report_path),
            ]
        )
    assert status == 2
    result = json.loads(report_path.read_text())["results"][0]
    assert result["status"] == "unavailable"
    assert result["reasons"] == [
        (
            "client_version_mismatch"
            if failure == "version_mismatch"
            else "server_or_model_unavailable"
        )
    ]
    assert not marker.exists()
    assert len(received) == (0 if failure == "version_mismatch" else 1)


def _scripted_tool_loop(
    tmp_path,
    monkeypatch,
    *,
    timeout=10,
    final_delay=0,
    inference_elapsed=0,
    client_returncode=0,
    api_key="",
):
    """Exercise the real runner using a scripted client and HTTP model peer."""
    from tests.test_client_acceptance_observe import exchange, sse, upstream

    token = "fixture-7d1849-secret"
    monkeypatch.setattr(
        "scripts.client_acceptance.runner.secrets.token_hex", lambda _: token
    )
    executable = tmp_path / "pi"
    executable.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
from pathlib import Path
from urllib.request import Request, urlopen
if '--version' in sys.argv:
    print('pi 1.2.3')
    raise SystemExit(0)
provider = json.loads((Path(os.environ['PI_CODING_AGENT_DIR']) / 'models.json').read_text())['providers']['vllm-mlx']
messages = [{'role': 'user', 'content': sys.argv[-1]}]
for turn in range(3):
    body = {'model': provider['models'][0]['id'], 'stream': True, 'messages': messages,
            'metadata': {'workspace': os.getcwd()}}
    request = Request(provider['baseUrl'] + '/chat/completions', data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json',
                               'Authorization': 'Bearer ' + provider['apiKey']})
    with urlopen(request, timeout=5) as response:
        events = []
        for raw in response:
            line = raw.decode().strip()
            if line.startswith('data: {'):
                events.append(json.loads(line[6:]))
            if turn == 2 and line == 'data: [DONE]':
                break
    calls = [call for event in events for choice in event.get('choices', [])
             for call in choice.get('delta', {}).get('tool_calls', [])]
    if not calls:
        raise SystemExit(CLIENT_RETURN_CODE)
    call = calls[0]
    args = json.loads(call['function']['arguments'])
    if call['function']['name'] == 'read':
        output = Path(args['path']).read_text()
    else:
        Path(args['path']).write_text(args['content'])
        output = 'file written'
    messages.extend([{'role': 'assistant', 'tool_calls': [call]},
                     {'role': 'tool', 'tool_call_id': call['id'], 'content': output}])
raise SystemExit(1)
""".replace("CLIENT_RETURN_CODE", str(client_returncode)))
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    read_call, _, final = exchange("chat")
    read_call = read_call.replace(b"fixture.txt", b"input.txt")
    write_call = sse(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_write",
                                "type": "function",
                                "function": {
                                    "name": "write",
                                    "arguments": json.dumps(
                                        {"path": "output.txt", "content": token + "\n"}
                                    ),
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        b"[DONE]",
    )
    replies = [
        (200, {"Content-Type": "application/json"}, b'{"data":[{"id":"served-model"}]}')
    ]
    if inference_elapsed:
        from types import SimpleNamespace

        from scripts.client_acceptance import observe

        # Advance only the observer's clock to exercise a slow model without a
        # minute-long test. Real processes, sockets, and parsing still execute.
        clock = [time.monotonic()]
        monkeypatch.setattr(
            observe, "time", SimpleNamespace(monotonic=lambda: clock[0])
        )

        def first_response(wfile):
            clock[0] += inference_elapsed
            wfile.write(read_call)
            wfile.flush()

    else:
        first_response = read_call

    def final_response(wfile):
        wfile.write(final)
        wfile.flush()
        time.sleep(final_delay)

    replies += [
        (200, {"Content-Type": "text/event-stream"}, body)
        for body in (first_response, write_call, final_response)
    ]
    with upstream(replies) as (base_url, received):
        result = run_client(
            "pi",
            base_url,
            "served-model",
            timeout,
            api_key=api_key,
            expected_version="1.2.3",
        )
    return result, received, token


def test_runner_uses_private_proxy_credential_for_authenticated_server(
    tmp_path, monkeypatch
):
    key = "inert-upstream-sentinel"
    result, received, token = _scripted_tool_loop(tmp_path, monkeypatch, api_key=key)
    assert result["status"] == "passed", result
    assert len(received) == 4
    assert all(
        headers["Authorization"] == "Bearer " + key for _, headers, _ in received
    )
    assert key not in json.dumps(result)
    assert token not in json.dumps(result)


def test_runner_executes_tool_loop_through_proxy_and_removes_workspace(
    tmp_path, monkeypatch
):
    """A scripted executable verifies runner integration, not any installed client."""
    from pathlib import Path

    result, received, token = _scripted_tool_loop(tmp_path, monkeypatch)
    assert result["status"] == "passed", result
    assert result["evidence"]["tool_results"] == 1
    assert result["evidence"]["requests"] == 3
    assert result["version"] == "1.2.3"
    workspace = Path(json.loads(received[1][2])["metadata"]["workspace"])
    assert not workspace.parent.exists()
    assert token not in json.dumps(result)


def test_inference_can_exceed_sixty_seconds_within_requested_client_budget(
    tmp_path, monkeypatch
):
    result, _, _ = _scripted_tool_loop(
        tmp_path, monkeypatch, timeout=180, inference_elapsed=65
    )
    assert result["status"] == "passed", result


def test_runner_waits_for_final_stream_evidence_after_client_exits(
    tmp_path, monkeypatch
):
    result, _, _ = _scripted_tool_loop(tmp_path, monkeypatch, final_delay=0.2)
    assert result["status"] == "passed", result
    assert result["evidence"]["completed"] is True


@pytest.mark.parametrize("returncode", [0, 7])
def test_final_stream_drain_stops_at_client_budget_and_keeps_exit_failure(
    tmp_path, monkeypatch, returncode
):
    started = time.monotonic()
    result, _, _ = _scripted_tool_loop(
        tmp_path,
        monkeypatch,
        timeout=1,
        final_delay=2,
        client_returncode=returncode,
    )
    assert result["status"] == "failed", result
    assert "proxy_drain_timeout" in result["reasons"]
    assert ("client_exit_nonzero" in result["reasons"]) == bool(returncode)
    assert time.monotonic() - started < 1.5
