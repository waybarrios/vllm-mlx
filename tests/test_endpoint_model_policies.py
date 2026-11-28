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
    @pytest.mark.parametrize(
        "model_name",
        [
            "mlx-community/ModernBERT-base-mlx",
            "mlx-community/all-MiniLM-L6-v2-4bit",
            "mlx-community/bert-base-uncased-mlx",
            "mlx-community/bge-large-en-v1.5-4bit",
            "mlx-community/embeddinggemma-300m-6bit",
            "mlx-community/multilingual-e5-large-mlx",
            "mlx-community/multilingual-e5-small-mlx",
        ],
    )
    def test_each_allowlisted_model_preserves_exact_name(self, model_name):
        assert resolve_embedding_model_name(model_name) == model_name

    def test_unknown_model_rejection_preserves_exact_response_detail(self):
        with pytest.raises(HTTPException) as exc:
            resolve_embedding_model_name("unknown")
        assert exc.value.status_code == 400
        assert exc.value.detail == (
            "Embedding model 'unknown' is not available. "
            "Request-time embedding model loading is limited to the supported "
            "allowlist: mlx-community/ModernBERT-base-mlx, "
            "mlx-community/all-MiniLM-L6-v2-4bit, "
            "mlx-community/bert-base-uncased-mlx, "
            "mlx-community/bge-large-en-v1.5-4bit, "
            "mlx-community/embeddinggemma-300m-6bit, "
            "mlx-community/multilingual-e5-large-mlx, "
            "mlx-community/multilingual-e5-small-mlx. "
            "To use a different embedding model, start the server with "
            "--embedding-model <model>."
        )

    def test_locked_model_rejection_preserves_exact_response_detail(self):
        with pytest.raises(HTTPException) as exc:
            resolve_embedding_model_name("unknown", locked_model="custom/model")
        assert exc.value.status_code == 400
        assert exc.value.detail == (
            "Embedding model 'unknown' is not available. "
            "This server was started with --embedding-model custom/model. "
            "Only 'custom/model' can be used for embeddings. Restart the "
            "server with a different --embedding-model to use 'unknown'."
        )

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
    @pytest.mark.parametrize(
        "resolver, alias, model_name",
        [
            (
                resolve_stt_model_name,
                "whisper-large-v3",
                "mlx-community/whisper-large-v3-mlx",
            ),
            (
                resolve_stt_model_name,
                "whisper-large-v3-turbo",
                "mlx-community/whisper-large-v3-turbo",
            ),
            (
                resolve_stt_model_name,
                "whisper-medium",
                "mlx-community/whisper-medium-mlx",
            ),
            (
                resolve_stt_model_name,
                "whisper-small",
                "mlx-community/whisper-small-mlx",
            ),
            (resolve_stt_model_name, "parakeet", "mlx-community/parakeet-tdt-0.6b-v2"),
            (
                resolve_stt_model_name,
                "parakeet-v3",
                "mlx-community/parakeet-tdt-0.6b-v3",
            ),
            (resolve_tts_model_name, "kokoro", "mlx-community/Kokoro-82M-bf16"),
            (resolve_tts_model_name, "kokoro-4bit", "mlx-community/Kokoro-82M-4bit"),
            (
                resolve_tts_model_name,
                "chatterbox",
                "mlx-community/chatterbox-turbo-fp16",
            ),
            (
                resolve_tts_model_name,
                "chatterbox-4bit",
                "mlx-community/chatterbox-turbo-4bit",
            ),
            (
                resolve_tts_model_name,
                "vibevoice",
                "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            ),
            (resolve_tts_model_name, "voxcpm", "mlx-community/VoxCPM1.5"),
        ],
    )
    def test_each_alias_and_exact_configured_id_resolves(
        self, resolver, alias, model_name
    ):
        assert resolver(alias) == model_name
        assert resolver(model_name) == model_name

    @pytest.mark.parametrize(
        "resolver, expected_detail",
        [
            (
                resolve_stt_model_name,
                "Transcription model 'unknown' is not available. "
                "Supported request models are: parakeet, parakeet-v3, "
                "whisper-large-v3, whisper-large-v3-turbo, whisper-medium, "
                "whisper-small. Exact configured model IDs for those aliases "
                "are also accepted.",
            ),
            (
                resolve_tts_model_name,
                "Speech model 'unknown' is not available. "
                "Supported request models are: chatterbox, chatterbox-4bit, "
                "kokoro, kokoro-4bit, vibevoice, voxcpm. Exact configured model "
                "IDs for those aliases are also accepted.",
            ),
        ],
    )
    def test_unknown_model_rejection_preserves_exact_response_detail(
        self, resolver, expected_detail
    ):
        with pytest.raises(HTTPException) as exc:
            resolver("unknown")
        assert exc.value.status_code == 400
        assert exc.value.detail == expected_detail

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
