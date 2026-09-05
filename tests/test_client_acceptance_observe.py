# SPDX-License-Identifier: Apache-2.0
"""Harness verification with scripted HTTP peers, not MLX acceptance evidence."""

import importlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest


def observer():
    # Keep the initial missing implementation a behavioral assertion failure.
    try:
        return importlib.import_module("scripts.client_acceptance.observe")
    except ModuleNotFoundError:
        pytest.fail("The client acceptance recording proxy is not implemented")


def sse(*events):
    return b"".join(
        b"data: "
        + (event if isinstance(event, bytes) else json.dumps(event).encode())
        + b"\n\n"
        for event in events
    )


TOKEN = "fixture-7d1849-secret"
PATHS = {
    "chat": "/chat/completions",
    "responses": "/responses",
    "anthropic": "/messages",
}


def exchange(protocol, *, result_id="call_1", answer=TOKEN):
    """Literal wire fixtures for one model read call and its client result."""
    if protocol == "chat":
        call = sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"path":"fixture.txt"}',
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
        result = {
            "messages": [{"role": "tool", "tool_call_id": result_id, "content": TOKEN}]
        }
        final = sse(
            {
                "choices": [
                    {"index": 0, "delta": {"content": answer}, "finish_reason": None}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            b"[DONE]",
        )
    elif protocol == "responses":
        item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path":"fixture.txt"}',
            "status": "completed",
        }
        call = sse(
            {
                "type": "response.output_item.added",
                "item": {**item, "status": "in_progress"},
            },
            {"type": "response.output_item.done", "item": item},
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [item]},
            },
        )
        result = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": result_id,
                    "output": [{"type": "input_text", "text": TOKEN}],
                }
            ]
        }
        final = sse(
            {"type": "response.output_text.delta", "delta": answer},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": answer}],
                        }
                    ],
                },
            },
        )
    else:
        call = sse(
            {
                "type": "message_start",
                "message": {"id": "msg_1", "role": "assistant", "content": []},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "read",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"path":"fixture.txt"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        )
        result = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result_id,
                            "content": [{"type": "text", "text": TOKEN}],
                        }
                    ],
                }
            ]
        }
        final = sse(
            {
                "type": "message_start",
                "message": {"id": "msg_2", "role": "assistant", "content": []},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": answer},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        )
    return call, result, final


@contextmanager
def upstream(replies):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append((self.path, dict(self.headers), body))
            status, headers, payload = replies.pop(0)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Connection", "close")
            if not callable(payload):
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if callable(payload):
                payload(self.wfile)
            else:
                self.wfile.write(payload)

        do_GET = do_POST

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post(base, path, payload, headers=None):
    url = urlsplit(base)
    conn = HTTPConnection(url.hostname, url.port, timeout=3)
    conn.request(
        "POST",
        url.path + path,
        json.dumps(payload),
        {"Content-Type": "application/json", **(headers or {})},
    )
    response = conn.getresponse()
    status, body = response.status, response.read()
    conn.close()
    return status, body


def reply(payload, status=200):
    return status, {"Content-Type": "text/event-stream"}, payload


@pytest.mark.parametrize(
    "protocol,method,path",
    [
        ("chat", "GET", "/models"),
        ("chat", "POST", "/chat/completions"),
        ("responses", "POST", "/responses"),
        ("anthropic", "POST", "/messages?beta=true"),
        ("anthropic", "POST", "/messages/count_tokens?beta=true"),
    ],
)
@pytest.mark.parametrize("credential", [None, "incorrect", "upstream-secret"])
def test_protected_proxy_requires_its_own_key_without_recording_rejected_requests(
    protocol, method, path, credential
):
    with upstream([]) as (url, received):
        with observer().RecordingProxy(
            url, "test-model", protocol, api_key="upstream-secret"
        ) as proxy:
            address = urlsplit(proxy.base_url)
            connection = HTTPConnection(address.hostname, address.port, timeout=3)
            headers = {"Authorization": "Bearer " + credential} if credential else {}
            connection.request(method, address.path + path, headers=headers)
            response = connection.getresponse()
            assert response.status == 401
            response.read()
            connection.close()
            assert received == []
            assert proxy.evidence.summary(TOKEN)["errors"] == []
            assert proxy.evidence.summary(TOKEN)["requests"] == 0


def test_proxy_keys_are_private_to_each_authenticated_run():
    first = observer().RecordingProxy(
        "http://127.0.0.1:8000/v1", "test-model", "chat", api_key="upstream-secret"
    )
    second = observer().RecordingProxy(
        "http://127.0.0.1:8000/v1", "test-model", "chat", api_key="upstream-secret"
    )
    assert first.client_key != second.client_key
    assert len(first.client_key) >= 32
    assert first.client_key != "upstream-secret"
    assert first.client_key not in json.dumps(first.evidence.summary(TOKEN))


@pytest.mark.parametrize(
    "path,method", [("/models", "GET"), ("/messages/count_tokens", "POST")]
)
def test_authenticated_discovery_and_token_count_use_proxy_key(path, method):
    with upstream([(200, {"Content-Type": "application/json"}, b"{}")]) as (
        url,
        received,
    ):
        with observer().RecordingProxy(
            url, "test-model", "anthropic", api_key="upstream-secret"
        ) as proxy:
            address = urlsplit(proxy.base_url)
            connection = HTTPConnection(address.hostname, address.port, timeout=3)
            connection.request(
                method,
                address.path + path,
                json.dumps({"model": "test-model"}) if method == "POST" else None,
                {"x-api-key": proxy.client_key},
            )
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.close()
        assert received[0][1]["Authorization"] == "Bearer upstream-secret"


@pytest.mark.parametrize("protocol", PATHS)
def test_observes_real_streamed_tool_round_trip_without_leaking_payload(protocol):
    call, result, final = exchange(protocol)
    with upstream([reply(call), reply(final)]) as (url, received):
        with observer().RecordingProxy(
            url, "test-model", protocol, api_key="upstream-secret"
        ) as proxy:
            first = post(
                proxy.base_url,
                PATHS[protocol],
                {"model": "test-model", "stream": True},
                {
                    "Authorization": "Bearer " + proxy.client_key,
                    "Cookie": "private=cookie",
                    "x-api-key": "client-key",
                },
            )
            second = post(
                proxy.base_url,
                PATHS[protocol],
                {"model": "test-model", "stream": True, **result},
                (
                    {"x-api-key": proxy.client_key}
                    if protocol == "anthropic"
                    else {"Authorization": "Bearer " + proxy.client_key}
                ),
            )
            assert first == (200, call)
            assert second == (200, final)
            assert proxy.evidence.passed(TOKEN)
            summary = proxy.evidence.summary(TOKEN)
            assert summary == {
                "requests": 2,
                "streamed": True,
                "tool_calls": 1,
                "tool_results": 1,
                "completed": True,
                "answer_matches": True,
                "errors": [],
            }
            serialized = json.dumps(summary)
            assert TOKEN not in serialized and "secret" not in serialized
        headers = {k.lower(): v for k, v in received[0][1].items()}
        assert headers["authorization"] == "Bearer upstream-secret"
        assert "cookie" not in headers
        assert headers.get("x-api-key") == (
            "upstream-secret" if protocol == "anthropic" else None
        )


@pytest.mark.parametrize("protocol", PATHS)
@pytest.mark.parametrize(
    "failure",
    [
        "unmatched_result",
        "replayed_call",
        "truncated",
        "later_response",
        "error_event",
        "token_outside_result",
    ],
)
def test_does_not_accept_replayed_or_incomplete_evidence(protocol, failure):
    call, result, final = exchange(
        protocol, result_id="other" if failure == "unmatched_result" else "call_1"
    )
    if failure == "replayed_call":
        call = exchange(protocol, answer="not the fixture")[2]
        # Historical assistant messages are client data, not issuance evidence.
        result["history"] = {"tool_calls": [{"id": "call_1"}]}
    if failure == "truncated":
        final = final[: final.rfind(b"data:")]
    if failure == "error_event":
        final += sse({"type": "error", "error": {"message": "secret backend error"}})
    if failure == "token_outside_result":
        result = json.loads(json.dumps(result).replace(TOKEN, "other text"))
        result["metadata"] = {"token": TOKEN}
    replies = [reply(call), reply(final)]
    if failure == "later_response":
        replies.append(reply(exchange(protocol, answer="wrong final answer")[2]))
    with upstream(replies) as (url, _):
        with observer().RecordingProxy(url, "test-model", protocol) as proxy:
            post(
                proxy.base_url, PATHS[protocol], {"model": "test-model", "stream": True}
            )
            post(
                proxy.base_url,
                PATHS[protocol],
                {"model": "test-model", "stream": True, **result},
            )
            if failure == "later_response":
                post(
                    proxy.base_url,
                    PATHS[protocol],
                    {"model": "test-model", "stream": True},
                )
            assert not proxy.evidence.passed(TOKEN)
            assert "secret backend error" not in json.dumps(
                proxy.evidence.summary(TOKEN)
            )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/v1",
        "http://example.org/v1",
        "http://127.0.0.2/v1",
        "http://127.0.0.1/v1?key=secret",
        "http://user:secret@localhost/v1",
        "http://localhost/other",
        "http://localhost/v1#fragment",
    ],
)
def test_rejects_non_loopback_or_ambiguous_upstream(url):
    with pytest.raises(ValueError):
        observer().RecordingProxy(url, "test-model", "chat")


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/responses", {"model": "test-model"}),
        ("/chat/completions?x=1", {"model": "test-model"}),
        ("/chat/completions", {"model": "wrong-model"}),
    ],
)
def test_rejects_wrong_path_or_model_without_forwarding(path, payload):
    with upstream([]) as (url, received):
        with observer().RecordingProxy(url, "test-model", "chat") as proxy:
            status, _ = post(proxy.base_url, path, payload)
            assert status in (400, 404)
            assert received == []
            assert not proxy.evidence.passed(TOKEN)


@pytest.mark.parametrize("status", [302, 401, 500])
def test_upstream_failures_are_not_followed_and_are_sanitized(status):
    with upstream(
        [
            (
                status,
                {"Location": "http://example.org/private"},
                b"private backend secret",
            )
        ]
    ) as (url, received):
        with observer().RecordingProxy(url, "test-model", "chat") as proxy:
            _, body = post(
                proxy.base_url, PATHS["chat"], {"model": "test-model", "stream": True}
            )
            assert b"private backend secret" not in body
            assert not proxy.evidence.passed(TOKEN)
            assert proxy.evidence.summary(TOKEN)["errors"]
        assert len(received) == 1


def test_request_and_response_capture_are_bounded(monkeypatch):
    module = observer()
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 256)
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 256)
    with upstream([reply(b"x" * 1024)]) as (url, received):
        with module.RecordingProxy(url, "test-model", "chat") as proxy:
            status, _ = post(
                proxy.base_url,
                PATHS["chat"],
                {"model": "test-model", "padding": "x" * 300},
            )
            assert status == 413
            _, body = post(
                proxy.base_url, PATHS["chat"], {"model": "test-model", "stream": True}
            )
            assert len(body) <= 256
            assert not proxy.evidence.passed(TOKEN)
            assert proxy.evidence.summary(TOKEN)["errors"]
        assert len(received) == 1


def test_summary_only_counts_matching_results_containing_fixture_token():
    call, result, final = exchange("chat")
    result["messages"][0]["content"] = "a successful result without the fixture"
    with upstream([reply(call), reply(final)]) as (url, _):
        with observer().RecordingProxy(url, "test-model", "chat") as proxy:
            post(proxy.base_url, PATHS["chat"], {"model": "test-model", "stream": True})
            post(
                proxy.base_url,
                PATHS["chat"],
                {"model": "test-model", "stream": True, **result},
            )
            assert proxy.evidence.summary(TOKEN)["tool_results"] == 0


def test_discovery_failure_does_not_poison_later_inference_evidence():
    call, result, final = exchange("chat")
    with upstream([(404, {}, b"discovery unavailable"), reply(call), reply(final)]) as (
        url,
        _,
    ):
        with observer().RecordingProxy(url, "test-model", "chat") as proxy:
            parsed = urlsplit(proxy.base_url)
            connection = HTTPConnection(parsed.hostname, parsed.port, timeout=3)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            connection.close()
            post(proxy.base_url, PATHS["chat"], {"model": "test-model", "stream": True})
            post(
                proxy.base_url,
                PATHS["chat"],
                {"model": "test-model", "stream": True, **result},
            )
            assert proxy.evidence.passed(TOKEN)
            assert proxy.evidence.summary(TOKEN)["requests"] == 2


def test_unterminated_event_after_completion_does_not_pass():
    call, result, final = exchange("chat")
    with upstream([reply(call), reply(final + b'data: {"error":')]) as (url, _):
        with observer().RecordingProxy(url, "test-model", "chat") as proxy:
            post(proxy.base_url, PATHS["chat"], {"model": "test-model", "stream": True})
            post(
                proxy.base_url,
                PATHS["chat"],
                {"model": "test-model", "stream": True, **result},
            )
            assert not proxy.evidence.passed(TOKEN)


def test_proxy_streams_before_upstream_finishes_and_times_out_stalled_stream():
    release = threading.Event()
    prefix = b'data: {"choices":[{"index":0,"delta":{"content":"ready"}}]}\n\n'

    def stalled(wfile):
        wfile.write(prefix)
        wfile.flush()
        release.wait(timeout=2)

    started = time.monotonic()
    with upstream([reply(stalled)]) as (url, _):
        with observer().RecordingProxy(url, "test-model", "chat", timeout=0.2) as proxy:
            parsed = urlsplit(proxy.base_url)
            connection = HTTPConnection(parsed.hostname, parsed.port, timeout=3)
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps({"model": "test-model", "stream": True}),
            )
            response = connection.getresponse()
            assert response.read(len(prefix)) == prefix
            assert not release.is_set()
            response.read()
            connection.close()
            assert "upstream_timeout" in proxy.evidence.summary(TOKEN)["errors"]
            assert not proxy.evidence.passed(TOKEN)
        release.set()
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("query", ["", "?beta=true"])
def test_anthropic_sdk_beta_query_preserves_tool_round_trip(query):
    call, result, final = exchange("anthropic")
    path = "/messages" + query
    with upstream([reply(call), reply(final)]) as (url, received):
        with observer().RecordingProxy(url, "test-model", "anthropic") as proxy:
            assert post(
                proxy.base_url, path, {"model": "test-model", "stream": True}
            ) == (200, call)
            assert post(
                proxy.base_url, path, {"model": "test-model", "stream": True, **result}
            ) == (200, final)
            assert proxy.evidence.passed(TOKEN)
        assert [entry[0] for entry in received] == ["/v1/messages" + query] * 2


@pytest.mark.parametrize("query", ["", "?beta=true"])
def test_anthropic_sdk_beta_token_count_is_not_an_inference(query):
    payload = b'{"input_tokens": 12}'
    with upstream([(200, {"Content-Type": "application/json"}, payload)]) as (
        url,
        received,
    ):
        with observer().RecordingProxy(url, "test-model", "anthropic") as proxy:
            assert post(
                proxy.base_url,
                "/messages/count_tokens" + query,
                {"model": "test-model", "messages": []},
            ) == (200, payload)
            assert proxy.evidence.summary(TOKEN)["requests"] == 0
        assert received[0][0] == "/v1/messages/count_tokens" + query


@pytest.mark.parametrize(
    "path",
    [
        "/messages?beta=false",
        "/messages?beta=true&api_key=secret",
        "/messages?api_key=secret",
        "/messages?beta=true&beta=true",
        "/messages/count_tokens?beta=false",
        "/models?beta=true",
        "/messages/batches?beta=true",
    ],
)
def test_anthropic_query_allowlist_rejects_unrelated_parameters_and_paths(path):
    with upstream([]) as (url, received):
        with observer().RecordingProxy(url, "test-model", "anthropic") as proxy:
            status, body = post(
                proxy.base_url, path, {"model": "test-model", "stream": True}
            )
            assert status == 404
            assert b"secret" not in body
            assert received == []


def test_anthropic_beta_query_keeps_model_validation():
    with upstream([]) as (url, received):
        with observer().RecordingProxy(url, "test-model", "anthropic") as proxy:
            status, _ = post(
                proxy.base_url,
                "/messages?beta=true",
                {"model": "other-model", "stream": True},
            )
            assert status == 400
            assert received == []


@pytest.mark.parametrize("protocol", PATHS)
def test_continuation_waits_for_terminal_stream_eof(protocol):
    call, result, final = exchange(protocol)
    release = threading.Event()

    def delayed_eof(wfile):
        wfile.write(call)
        wfile.flush()
        release.wait(timeout=2)

    with upstream([reply(delayed_eof), reply(final)]) as (url, received):
        with observer().RecordingProxy(url, "test-model", protocol, timeout=2) as proxy:
            parsed = urlsplit(proxy.base_url)
            connection = HTTPConnection(parsed.hostname, parsed.port, timeout=3)
            connection.request(
                "POST",
                "/v1" + PATHS[protocol],
                json.dumps({"model": "test-model", "stream": True}),
            )
            response = connection.getresponse()
            assert response.read(len(call)) == call
            with ThreadPoolExecutor(max_workers=1) as executor:
                continuation = executor.submit(
                    post,
                    proxy.base_url,
                    PATHS[protocol],
                    {"model": "test-model", "stream": True, **result},
                )
                try:
                    # The next request must wait until issuance evidence is final.
                    with pytest.raises(FutureTimeout):
                        continuation.result(timeout=0.1)
                finally:
                    release.set()
                    response.read()
                    connection.close()
                assert continuation.result(timeout=2) == (200, final)
            assert proxy.evidence.passed(TOKEN)
            assert len(received) == 2


def test_request_queue_wait_is_bounded():
    with upstream([]) as (url, received):
        with observer().RecordingProxy(url, "test-model", "chat", timeout=0.1) as proxy:
            # Hold the same lock used by an unfinished upstream exchange.
            proxy._serial.acquire()
            try:
                started = time.monotonic()
                status, body = post(
                    proxy.base_url,
                    PATHS["chat"],
                    {"model": "test-model", "stream": True},
                )
                elapsed = time.monotonic() - started
                assert status == 504
                assert json.loads(body)["error"]["message"] == "request_queue_timeout"
                assert 0.08 <= elapsed < 1
                assert received == []
                assert not proxy.evidence.passed(TOKEN)
            finally:
                proxy._serial.release()


def parallel_calls(protocol):
    """Two complete server-issued calls sharing one model response."""
    call, _, _ = exchange(protocol)
    events = []
    second_blocks = []
    for line in call.splitlines():
        if not line.startswith(b"data: "):
            continue
        if line == b"data: [DONE]":
            events.append(b"[DONE]")
            continue
        event = json.loads(line[6:])
        second = json.loads(json.dumps(event).replace("call_1", "call_2"))
        if protocol == "chat":
            delta = event["choices"][0]["delta"]
            if "tool_calls" in delta:
                other = second["choices"][0]["delta"]["tool_calls"][0]
                other["index"] = 1
                delta["tool_calls"].append(other)
        elif protocol == "responses":
            if event["type"] in (
                "response.output_item.added",
                "response.output_item.done",
            ):
                second["item"]["id"] = "fc_2"
                events.extend([event, second])
                continue
            if event["type"] == "response.completed":
                other = second["response"]["output"][0]
                other["id"] = "fc_2"
                event["response"]["output"].append(other)
        elif event["type"].startswith("content_block_"):
            second["index"] = 1
            events.append(event)
            second_blocks.append(second)
            if event["type"] == "content_block_stop":
                events.extend(second_blocks)
                second_blocks = []
            continue
        events.append(event)
    return sse(*events)


@pytest.mark.parametrize("protocol", PATHS)
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
@pytest.mark.parametrize("missing_result", [False, True], ids=["complete", "missing"])
def test_final_completion_requires_results_for_every_call(
    protocol, parallel, missing_result
):
    call, first_result, final = exchange(protocol)
    second_call = call.replace(b"call_1", b"call_2")
    _, second_result, _ = exchange(protocol, result_id="call_2")
    second_result = json.loads(json.dumps(second_result).replace(TOKEN, "file written"))
    key = "input" if protocol == "responses" else "messages"
    final_results = {
        key: first_result[key] + ([] if missing_result else second_result[key])
    }
    replies = (
        [reply(parallel_calls(protocol)), reply(final)]
        if parallel
        else [reply(call), reply(second_call), reply(final)]
    )
    with upstream(replies) as (url, _):
        with observer().RecordingProxy(url, "test-model", protocol) as proxy:
            post(
                proxy.base_url, PATHS[protocol], {"model": "test-model", "stream": True}
            )
            if not parallel:
                post(
                    proxy.base_url,
                    PATHS[protocol],
                    {"model": "test-model", "stream": True, **first_result},
                )
            post(
                proxy.base_url,
                PATHS[protocol],
                {"model": "test-model", "stream": True, **final_results},
            )
            assert proxy.evidence.passed(TOKEN) is (not missing_result)
            summary = proxy.evidence.summary(TOKEN)
            assert summary["completed"] is (not missing_result)
            assert summary["tool_calls"] == 2
            assert summary["tool_results"] == 1
            if missing_result:
                assert "missing_tool_results" in summary["errors"]
            else:
                assert summary["errors"] == []


@pytest.mark.parametrize("protocol", PATHS)
@pytest.mark.parametrize(
    "release_before_timeout", [True, False], ids=["drained", "timeout"]
)
def test_idle_wait_finalizes_terminal_response_evidence(
    protocol, release_before_timeout
):
    call, result, final = exchange(protocol)
    release = threading.Event()

    def delayed_eof(wfile):
        wfile.write(final)
        wfile.flush()
        release.wait(timeout=2)

    with upstream([reply(call), reply(delayed_eof)]) as (url, _):
        with observer().RecordingProxy(url, "test-model", protocol, timeout=2) as proxy:
            post(
                proxy.base_url, PATHS[protocol], {"model": "test-model", "stream": True}
            )
            parsed = urlsplit(proxy.base_url)
            connection = HTTPConnection(parsed.hostname, parsed.port, timeout=3)
            connection.request(
                "POST",
                "/v1" + PATHS[protocol],
                json.dumps({"model": "test-model", "stream": True, **result}),
            )
            response = connection.getresponse()
            assert response.read(len(final)) == final
            try:
                wait_for_idle = getattr(proxy, "wait_for_idle", None)
                assert callable(wait_for_idle)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    idle = executor.submit(wait_for_idle, 0.3)
                    if release_before_timeout:
                        with pytest.raises(FutureTimeout):
                            idle.result(timeout=0.05)
                        release.set()
                        assert idle.result(timeout=1) is True
                        assert proxy.evidence.passed(TOKEN)
                    else:
                        assert idle.result(timeout=1) is False
                        assert (
                            "stream_finalize_timeout"
                            in proxy.evidence.summary(TOKEN)["errors"]
                        )
                        assert not proxy.evidence.passed(TOKEN)
            finally:
                release.set()
                response.read()
                connection.close()


def test_idle_wait_supports_an_exhausted_client_budget():
    proxy = observer().RecordingProxy("http://127.0.0.1:8000/v1", "test-model", "chat")
    wait_for_idle = getattr(proxy, "wait_for_idle", None)
    assert callable(wait_for_idle)
    assert wait_for_idle(0) is True
    with proxy._serial:
        assert wait_for_idle(0) is False
    assert proxy.evidence.summary(TOKEN)["errors"] == ["stream_finalize_timeout"]
