# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit Outlines backend admission."""

import pytest
from fastapi import HTTPException

from vllm_mlx.api.models import ChatCompletionRequest, Message, ResponseFormat
from vllm_mlx.api.responses_models import ResponsesRequest
import vllm_mlx.server as server


class TestOutlinesBackendAdmission:
    """Explicit Outlines selection must fail before model acquisition."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"response_format": {"type": "outlines"}},
            {"response_format": {"type": "json_schema", "backend": "outlines"}},
            {"guided_decoding_backend": "outlines"},
            {"guided_decoding": {"backend": " Outlines "}},
            {"guided_decoding": {"guided_decoding_backend": "OUTLINES"}},
            {"structured_outputs": {"backend": "outlines"}},
            {"structured_outputs": {"guided_decoding_backend": "outlines"}},
        ],
    )
    @pytest.mark.anyio
    async def test_explicit_selector_is_rejected(self, payload):
        class RawRequest:
            async def json(self):
                return payload

        with pytest.raises(HTTPException) as excinfo:
            await server._preflight_response_format_backend(RawRequest())

        assert excinfo.value.status_code == 422
        assert (
            excinfo.value.detail == "response_format backend 'outlines' is unavailable"
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"response_format": {"type": "text"}},
            {"response_format": {"type": "json_object"}},
            {"response_format": {"type": "future_vendor_mode"}},
            {"guided_decoding_backend": None},
            {"backend": "outlines"},
            {"chat_template_kwargs": {"backend": "outlines"}},
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "backend_value",
                        "schema": {
                            "type": "object",
                            "properties": {"backend": {"const": "outlines"}},
                        },
                    },
                }
            },
        ],
    )
    def test_unrelated_or_supported_values_are_not_rejected(self, payload):
        assert server._request_uses_outlines_backend(payload) is False

    @pytest.mark.parametrize(
        "response_format",
        [
            {"type": "outlines"},
            {"type": "json_schema", "backend": " OUTLINES "},
            ResponseFormat(type="outlines"),
        ],
    )
    def test_internal_preparation_defensively_rejects_outlines(self, response_format):
        with pytest.raises(HTTPException) as excinfo:
            server._prepare_json_logits_processor(
                object(),
                [{"role": "user", "content": "return JSON"}],
                response_format,
                tools=None,
                tool_choice=None,
            )

        assert excinfo.value.status_code == 422
        assert (
            excinfo.value.detail == "response_format backend 'outlines' is unavailable"
        )

    @pytest.mark.anyio
    async def test_chat_route_rejects_before_engine_acquisition(self, monkeypatch):
        acquired = False

        async def fail_if_acquired(*_args, **_kwargs):
            nonlocal acquired
            acquired = True
            raise AssertionError("engine acquisition must not run")

        class RawRequest:
            async def json(self):
                return {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "guided_decoding_backend": "outlines",
                }

        monkeypatch.setattr(server, "_validate_model_name", lambda _model: None)
        monkeypatch.setattr(
            server, "_acquire_default_engine_for_request", fail_if_acquired
        )
        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="return JSON")],
        )

        with pytest.raises(HTTPException) as excinfo:
            await server.create_chat_completion(request, RawRequest())

        assert excinfo.value.status_code == 422
        assert acquired is False

    @pytest.mark.anyio
    async def test_chat_route_without_raw_request_rejects_before_acquisition(
        self, monkeypatch
    ):
        acquired = False

        async def fail_if_acquired(*_args, **_kwargs):
            nonlocal acquired
            acquired = True
            raise AssertionError("engine acquisition must not run")

        monkeypatch.setattr(server, "_validate_model_name", lambda _model: None)
        monkeypatch.setattr(
            server, "_acquire_default_engine_for_request", fail_if_acquired
        )
        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="return JSON")],
            response_format=ResponseFormat(type="outlines"),
        )

        with pytest.raises(HTTPException) as excinfo:
            await server.create_chat_completion(request, None)

        assert excinfo.value.status_code == 422
        assert acquired is False

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.anyio
    async def test_responses_route_rejects_raw_selector_before_engine(
        self, monkeypatch, stream
    ):
        engine_requested = False

        def fail_if_requested():
            nonlocal engine_requested
            engine_requested = True
            raise AssertionError("engine lookup must not run")

        class RawRequest:
            async def json(self):
                return {
                    "model": "served-model",
                    "input": "return JSON",
                    "stream": stream,
                    "structured_outputs": {"backend": "outlines"},
                }

        monkeypatch.setattr(server, "get_engine", fail_if_requested)
        request = ResponsesRequest(
            model="served-model",
            input="return JSON",
            stream=stream,
        )

        with pytest.raises(HTTPException) as excinfo:
            await server.create_response(request, RawRequest())

        assert excinfo.value.status_code == 422
        assert engine_requested is False

    @pytest.mark.anyio
    async def test_responses_internal_object_rejects_before_engine(self, monkeypatch):
        engine_requested = False

        def fail_if_requested():
            nonlocal engine_requested
            engine_requested = True
            raise AssertionError("engine lookup must not run")

        monkeypatch.setattr(server, "get_engine", fail_if_requested)
        request = ResponsesRequest(model="served-model", input="return JSON")
        request.text.format.type = "outlines"

        with pytest.raises(HTTPException) as excinfo:
            await server.create_response(request, None)

        assert excinfo.value.status_code == 422
        assert engine_requested is False
