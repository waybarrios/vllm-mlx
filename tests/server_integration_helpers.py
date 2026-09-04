# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by external server integration tests."""


def get_chat_completion_payload(requests, server_url):
    """Build a chat payload using the model ID advertised by the server."""
    response = requests.get(f"{server_url}/v1/models", timeout=5)
    assert response.status_code == 200

    models = response.json().get("data", [])
    assert models
    model_name = models[0].get("id")
    assert model_name

    return {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
    }
