# SPDX-License-Identifier: Apache-2.0
"""Cross-platform tests for optional endpoint model resolution policies."""

import pytest
from fastapi import HTTPException

from vllm_mlx.endpoint_model_policies import (
    resolve_embedding_model_name,
    resolve_stt_model_name,
    resolve_tts_model_name,
)


class TestEmbeddingModelPolicy:
    def test_allowlisted_embedding_model_passes(self):
        assert (
            resolve_embedding_model_name("mlx-community/multilingual-e5-small-mlx")
            == "mlx-community/multilingual-e5-small-mlx"
        )

    def test_unknown_embedding_model_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_embedding_model_name("attacker/unknown-embedding")

        assert exc_info.value.status_code == 400
        assert "attacker/unknown-embedding" in exc_info.value.detail
        assert "--embedding-model" in exc_info.value.detail

    def test_locked_embedding_model_can_be_custom(self):
        assert (
            resolve_embedding_model_name(
                "custom/private-embedding",
                locked_model="custom/private-embedding",
            )
            == "custom/private-embedding"
        )

    def test_locked_embedding_model_rejects_other_request(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_embedding_model_name(
                "mlx-community/all-MiniLM-L6-v2-4bit",
                locked_model="custom/private-embedding",
            )

        assert exc_info.value.status_code == 400
        assert "custom/private-embedding" in exc_info.value.detail


class TestAudioModelPolicy:
    def test_stt_alias_resolves_to_configured_model(self):
        assert (
            resolve_stt_model_name("whisper-large-v3")
            == "mlx-community/whisper-large-v3-mlx"
        )

    def test_stt_full_model_id_is_accepted(self):
        model_name = "mlx-community/parakeet-tdt-0.6b-v2"
        assert resolve_stt_model_name(model_name) == model_name

    def test_stt_unknown_model_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_stt_model_name("attacker/unknown-stt")

        assert exc_info.value.status_code == 400
        assert "attacker/unknown-stt" in exc_info.value.detail
        assert "whisper-large-v3" in exc_info.value.detail

    def test_tts_alias_resolves_to_configured_model(self):
        assert resolve_tts_model_name("kokoro") == "mlx-community/Kokoro-82M-bf16"

    def test_tts_full_model_id_is_accepted(self):
        model_name = "mlx-community/chatterbox-turbo-fp16"
        assert resolve_tts_model_name(model_name) == model_name

    def test_tts_unknown_model_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_tts_model_name("attacker/unknown-tts")

        assert exc_info.value.status_code == 400
        assert "attacker/unknown-tts" in exc_info.value.detail
        assert "kokoro" in exc_info.value.detail
