# SPDX-License-Identifier: Apache-2.0
"""Bounded loopback proxy collecting protocol evidence without payload reports."""

import http.client
import json
import math
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_REQUESTS = 128
_PATHS = {
    "chat": "/v1/chat/completions",
    "responses": "/v1/responses",
    "anthropic": "/v1/messages",
}


def _objects(value):
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, dict):
        return _text(value.get("text", value.get("content", "")))
    return ""


class Evidence:
    """Keep bounded in-memory evidence; summaries contain only counts and flags."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.requests = 0
        self.streamed = False
        self.completed = False
        self._answer = ""
        self._issued = set()
        self._results = {}
        self._errors = []
        self._lock = threading.RLock()

    def error(self, code):
        with self._lock:
            if code not in self._errors:
                self._errors.append(code)

    def begin(self, body):
        with self._lock:
            self.requests += 1
            self.completed = False
            self._answer = ""
            if self.requests > MAX_REQUESTS:
                self.error("request_limit")
                return False
            if body.get("stream") is not True:
                self.error("stream_required")
            results = []
            if self.protocol == "responses":
                results = [
                    (item.get("call_id"), item.get("output"))
                    for item in _objects(body.get("input"))
                    if item.get("type") == "function_call_output"
                ]
            else:
                for message in _objects(body.get("messages")):
                    if self.protocol == "chat" and message.get("role") == "tool":
                        results.append(
                            (message.get("tool_call_id"), message.get("content"))
                        )
                    elif self.protocol == "anthropic" and message.get("role") == "user":
                        for item in _objects(message.get("content")):
                            if item.get("type") == "tool_result":
                                if item.get("is_error"):
                                    self.error("tool_result_error")
                                results.append(
                                    (item.get("tool_use_id"), item.get("content"))
                                )
            for call_id, result in results:
                if isinstance(call_id, str) and call_id in self._issued:
                    # Retain one bounded payload per observed call, not repeated history.
                    self._results.setdefault(call_id, _text(result))
            return True

    def finish(self, stream):
        with self._lock:
            stream.end()
            if not stream.completed:
                self.error("incomplete_stream")
            for code in stream.errors:
                self.error(code)
            if stream.completed and not stream.errors:
                self.streamed = True
                for call_id in stream.calls:
                    if call_id in self._issued:
                        self.error("duplicate_tool_call_id")
                    self._issued.add(call_id)
                self.completed = not stream.calls and stream.final
                if self.completed and self._issued.difference(self._results):
                    self.error("missing_tool_results")
                    self.completed = False
                self._answer = stream.answer if self.completed else ""

    def summary(self, token):
        with self._lock:
            return {
                "requests": self.requests,
                "streamed": self.streamed,
                "tool_calls": len(self._issued),
                "tool_results": sum(
                    bool(token) and token in result for result in self._results.values()
                ),
                "completed": self.completed,
                "answer_matches": bool(token) and token in self._answer,
                "errors": list(self._errors),
            }

    def passed(self, token):
        with self._lock:
            report = self.summary(token)
            return bool(
                report["requests"] >= 2
                and report["streamed"]
                and report["completed"]
                and report["answer_matches"]
                and not report["errors"]
                and any(token in result for result in self._results.values())
            )


class _Stream:
    """Parse SSE incrementally, accepting only server-emitted calls and text."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.completed = False
        self.final = False
        self.answer = ""
        self.calls = set()
        self.errors = set()
        self._buffer = b""
        self._data = []
        self._stop = None
        self._chat_calls = {}

    def feed(self, chunk):
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                if self._data:
                    self._event(b"\n".join(self._data))
                self._data = []
            elif line.startswith(b"data:"):
                self._data.append(line[5:].lstrip(b" "))

    def end(self):
        if self._buffer.strip() or self._data:
            self.errors.add("incomplete_sse_event")

    def _event(self, raw):
        if raw == b"[DONE]":
            if self.protocol == "chat":
                self.completed = self._stop in ("stop", "tool_calls")
                self.final = self._stop == "stop"
                for call in self._chat_calls.values():
                    try:
                        valid = (
                            call.get("id")
                            and call.get("name")
                            and isinstance(json.loads(call.get("arguments", "")), dict)
                        )
                    except (ValueError, TypeError):
                        valid = False
                    if valid:
                        self.calls.add(call["id"])
                    else:
                        self.errors.add("invalid_tool_call")
            return
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError):
            self.errors.add("invalid_sse_json")
            return
        if not isinstance(event, dict):
            self.errors.add("invalid_sse_event")
            return
        if event.get("error") or event.get("type") in (
            "error",
            "response.failed",
            "response.incomplete",
        ):
            self.errors.add("upstream_error_event")
            return
        if self.completed:
            self.errors.add("event_after_completion")
            return
        if self.protocol == "chat":
            self._chat(event)
        elif self.protocol == "responses":
            self._responses(event)
        else:
            self._anthropic(event)

    def _chat(self, event):
        for choice in _objects(event.get("choices")):
            if choice.get("index", 0) != 0:
                self.errors.add("multiple_choices")
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                self.errors.add("invalid_sse_event")
                continue
            self.answer += _text(delta.get("content"))
            for part in _objects(delta.get("tool_calls")):
                index = part.get("index")
                if not isinstance(index, int):
                    self.errors.add("invalid_tool_call")
                    continue
                call = self._chat_calls.setdefault(index, {})
                function = part.get("function") or {}
                if not isinstance(function, dict):
                    self.errors.add("invalid_tool_call")
                    continue
                for key, value in (
                    ("id", part.get("id")),
                    ("name", function.get("name")),
                    ("arguments", function.get("arguments")),
                ):
                    if isinstance(value, str):
                        call[key] = call.get(key, "") + value
            if choice.get("finish_reason"):
                self._stop = choice["finish_reason"]

    def _responses(self, event):
        kind = event.get("type")
        if kind == "response.output_text.delta":
            self.answer += _text(event.get("delta"))
        elif kind in ("response.output_item.added", "response.output_item.done"):
            self._response_call(event.get("item"))
        elif kind == "response.completed":
            response = event.get("response")
            if (
                not isinstance(response, dict)
                or response.get("status") != "completed"
                or response.get("error")
            ):
                self.errors.add("response_not_completed")
                return
            output = _objects(response.get("output"))
            # The completed response is authoritative, even when earlier deltas differed.
            self.answer = "".join(
                _text(item.get("content"))
                for item in output
                if item.get("type") == "message" and item.get("role") == "assistant"
            )
            for item in output:
                self._response_call(item)
            self.completed = True
            self.final = not self.calls

    def _response_call(self, item):
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id and item.get("name"):
                self.calls.add(call_id)
            else:
                self.errors.add("invalid_tool_call")

    def _anthropic(self, event):
        kind = event.get("type")
        if kind == "content_block_start":
            block = event.get("content_block") or {}
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    call_id = block.get("id")
                    if isinstance(call_id, str) and call_id and block.get("name"):
                        self.calls.add(call_id)
                    else:
                        self.errors.add("invalid_tool_call")
                elif block.get("type") == "text":
                    self.answer += _text(block.get("text"))
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                self.answer += _text(delta.get("text"))
        elif kind == "message_delta":
            delta = event.get("delta") or {}
            if isinstance(delta, dict):
                self._stop = delta.get("stop_reason")
        elif kind == "message_stop":
            self.completed = self._stop in ("end_turn", "stop_sequence", "tool_use")
            self.final = self._stop in ("end_turn", "stop_sequence")


class RecordingProxy:
    """Forward only selected inference and model discovery to literal loopback."""

    def __init__(self, upstream_url, model, protocol, api_key="", timeout=60):
        url = urlsplit(upstream_url)
        try:
            port = url.port or 80
        except ValueError:
            raise ValueError("Invalid upstream port") from None
        if (
            url.scheme != "http"
            or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username is not None
            or url.password is not None
            or url.path.rstrip("/") != "/v1"
            or url.query
            or url.fragment
        ):
            raise ValueError("Upstream must be an HTTP loopback /v1 base URL")
        if protocol not in _PATHS or not isinstance(model, str) or not model:
            raise ValueError("Invalid protocol or model")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.model = model
        self.protocol = protocol
        self.timeout = timeout
        # Resolve the permitted localhost spelling directly, without trusting DNS.
        self._host = "127.0.0.1" if url.hostname == "localhost" else url.hostname
        self._port = port
        self._api_key = api_key
        # Preserve the upstream authentication boundary without sharing its key
        # with client processes or exposing a key in command-line arguments.
        self.client_key = secrets.token_urlsafe(32) if api_key else ""
        self.evidence = Evidence(protocol)
        self._serial = threading.Lock()
        self._server = None
        self._thread = None

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.settimeout(owner.timeout)

            def log_message(self, *args):
                pass

            def do_GET(self):
                self._dispatch()

            def do_POST(self):
                self._dispatch()

            def _reject(self, status, code, *, record=True):
                if record:
                    self._error(code)
                self.close_connection = True
                payload = json.dumps(
                    {"error": {"message": code, "type": "acceptance_proxy_error"}}
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def _error(self, code):
                if not (self.command == "GET" and self.path == "/v1/models"):
                    owner.evidence.error(code)

            def _dispatch(self):
                if owner.client_key:
                    credentials = [("Authorization", "Bearer " + owner.client_key)]
                    if owner.protocol == "anthropic":
                        credentials.append(("x-api-key", owner.client_key))
                    authorized = any(
                        len(self.headers.get_all(name, [])) == 1
                        and secrets.compare_digest(
                            self.headers[name].encode("utf-8"), expected.encode("ascii")
                        )
                        for name, expected in credentials
                    )
                    if not authorized:
                        self._reject(401, "authentication_required", record=False)
                        return
                allowed = self.command == "GET" and self.path == "/v1/models"
                # The Anthropic SDK's beta Messages resource adds this exact
                # query for create/stream and countTokens. No other query is
                # needed by the acceptance clients.
                inference_paths = {_PATHS[owner.protocol]}
                if owner.protocol == "anthropic":
                    inference_paths.add("/v1/messages?beta=true")
                inference = self.command == "POST" and self.path in inference_paths
                count_tokens = (
                    owner.protocol == "anthropic"
                    and self.command == "POST"
                    and self.path
                    in {
                        "/v1/messages/count_tokens",
                        "/v1/messages/count_tokens?beta=true",
                    }
                )
                if not (allowed or inference or count_tokens):
                    self._reject(404, "path_rejected")
                    return
                if (
                    self.headers.get("Transfer-Encoding")
                    or len(self.headers.get_all("Content-Length", [])) > 1
                ):
                    self._reject(400, "invalid_request_framing")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._reject(400, "invalid_content_length")
                    return
                if length < 0 or length > MAX_REQUEST_BYTES:
                    self._reject(413, "request_too_large")
                    return
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        self._reject(400, "incomplete_request")
                        return
                    body = json.loads(raw) if self.command == "POST" else {}
                except (ValueError, UnicodeError, RecursionError, OSError):
                    self._reject(400, "invalid_request_body")
                    return
                if self.command == "POST" and (
                    not isinstance(body, dict) or body.get("model") != owner.model
                ):
                    self._reject(400, "model_rejected")
                    return
                # SDKs can send the next tool result as soon as they receive a
                # terminal SSE event, before the preceding HTTP stream closes.
                # Wait until its issuance evidence is finalized before begin().
                if not owner._serial.acquire(timeout=owner.timeout):
                    self._reject(504, "request_queue_timeout")
                    return
                try:
                    if inference and not owner.evidence.begin(body):
                        self._reject(429, "request_limit")
                        return
                    self._forward(raw, inference)
                finally:
                    owner._serial.release()

            def _forward(self, raw, inference):
                deadline = time.monotonic() + owner.timeout
                connection = http.client.HTTPConnection(
                    owner._host, owner._port, timeout=owner.timeout
                )
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream" if inference else "application/json",
                    "Accept-Encoding": "identity",
                }
                if owner._api_key:
                    headers["Authorization"] = "Bearer " + owner._api_key
                    if owner.protocol == "anthropic":
                        headers["x-api-key"] = owner._api_key
                if owner.protocol == "anthropic":
                    headers["anthropic-version"] = "2023-06-01"
                stream = _Stream(owner.protocol) if inference else None
                started = False
                self.close_connection = True
                try:
                    connection.request(
                        self.command, self.path, body=raw or None, headers=headers
                    )
                    response = connection.getresponse()
                    if response.status != 200:
                        self._reject(502, "upstream_http_error")
                        return
                    content_type = response.getheader("Content-Type", "")
                    if (
                        inference
                        and content_type.split(";", 1)[0].strip().lower()
                        != "text/event-stream"
                    ):
                        self._reject(502, "stream_required")
                        return
                    if (
                        response.getheader("Content-Encoding", "identity").lower()
                        != "identity"
                    ):
                        self._reject(502, "unsupported_content_encoding")
                        return
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "text/event-stream" if inference else "application/json",
                    )
                    self.send_header("Connection", "close")
                    self.end_headers()
                    started = True
                    size = 0
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        # HTTPConnection may relinquish a closing response socket.
                        if connection.sock is not None:
                            connection.sock.settimeout(remaining)
                        elif response.fp is not None:
                            response.fp.raw._sock.settimeout(remaining)
                        chunk = response.read1(min(4096, MAX_RESPONSE_BYTES - size + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            self._error("response_too_large")
                            break
                        if stream:
                            stream.feed(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (TimeoutError, socket.timeout):
                    self._error("upstream_timeout")
                    if not started:
                        self._reject(504, "upstream_timeout")
                except (OSError, http.client.HTTPException, ValueError, RecursionError):
                    self._error("upstream_io_error")
                    if not started:
                        self._reject(502, "upstream_io_error")
                finally:
                    connection.close()
                    if stream:
                        owner.evidence.finish(stream)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        return self

    def wait_for_idle(self, timeout: float) -> bool:
        """Wait for active forwarding and evidence collection within the budget."""
        if not self._serial.acquire(timeout=max(0.0, timeout)):
            self.evidence.error("stream_finalize_timeout")
            return False
        self._serial.release()
        return True

    def __exit__(self, exc_type, exc_value, traceback):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)
