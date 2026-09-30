# SPDX-License-Identifier: Apache-2.0
"""Focused regressions for external server integration requests."""

from types import SimpleNamespace
from unittest.mock import Mock

from tests.server_integration_helpers import get_chat_completion_payload


def test_chat_completion_uses_model_advertised_by_models_endpoint():
    """The chat integration request must use the server's advertised model ID."""
    models_response = SimpleNamespace(
        status_code=200,
        json=lambda: {"data": [{"id": "served-model"}]},
    )
    requests = SimpleNamespace(
        get=Mock(return_value=models_response),
    )

    request_payload = get_chat_completion_payload(requests, "http://localhost:8000")

    requests.get.assert_called_once_with("http://localhost:8000/v1/models", timeout=5)
    assert request_payload["model"] == "served-model"
