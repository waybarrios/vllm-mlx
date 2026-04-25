# SPDX-License-Identifier: Apache-2.0
"""Request-time model resolution policies for optional endpoints.

These endpoints intentionally do not expose arbitrary Hugging Face loading from
user-controlled request bodies. Unknown model names must be rejected before any
engine instantiation or download path is reached.
"""

from fastapi import HTTPException

_EMBEDDING_MODELS = frozenset(
    {
        "mlx-community/ModernBERT-base-mlx",
        "mlx-community/all-MiniLM-L6-v2-4bit",
        "mlx-community/bert-base-uncased-mlx",
        "mlx-community/bge-large-en-v1.5-4bit",
        "mlx-community/embeddinggemma-300m-6bit",
        "mlx-community/multilingual-e5-large-mlx",
        "mlx-community/multilingual-e5-small-mlx",
    }
)

_STT_MODEL_ALIASES = {
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}

_TTS_MODEL_ALIASES = {
    "kokoro": "mlx-community/Kokoro-82M-bf16",
    "kokoro-4bit": "mlx-community/Kokoro-82M-4bit",
    "chatterbox": "mlx-community/chatterbox-turbo-fp16",
    "chatterbox-4bit": "mlx-community/chatterbox-turbo-4bit",
    "vibevoice": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
    "voxcpm": "mlx-community/VoxCPM1.5",
}


def _with_identity_aliases(model_map: dict[str, str]) -> dict[str, str]:
    expanded = dict(model_map)
    for model_name in model_map.values():
        expanded[model_name] = model_name
    return expanded


_STT_MODEL_MAP = _with_identity_aliases(_STT_MODEL_ALIASES)
_TTS_MODEL_MAP = _with_identity_aliases(_TTS_MODEL_ALIASES)


def _reject_unknown_embedding_model(requested_model: str) -> None:
    supported = ", ".join(sorted(_EMBEDDING_MODELS))
    raise HTTPException(
        status_code=400,
        detail=(
            f"Embedding model '{requested_model}' is not available. "
            "Request-time embedding model loading is limited to the supported "
            f"allowlist: {supported}. To use a different embedding model, start "
            "the server with --embedding-model <model>."
        ),
    )


def _reject_unknown_audio_model(
    endpoint: str,
    requested_model: str,
    supported_aliases: dict[str, str],
) -> None:
    aliases = ", ".join(sorted(supported_aliases))
    raise HTTPException(
        status_code=400,
        detail=(
            f"{endpoint} model '{requested_model}' is not available. "
            f"Supported request models are: {aliases}. Exact configured model IDs "
            "for those aliases are also accepted."
        ),
    )


def resolve_embedding_model_name(
    requested_model: str,
    *,
    locked_model: str | None = None,
) -> str:
    """Resolve the embedding model for a request or raise HTTP 400."""
    if locked_model is not None:
        if requested_model == locked_model:
            return locked_model
        raise HTTPException(
            status_code=400,
            detail=(
                f"Embedding model '{requested_model}' is not available. "
                f"This server was started with --embedding-model {locked_model}. "
                f"Only '{locked_model}' can be used for embeddings. Restart the "
                f"server with a different --embedding-model to use '{requested_model}'."
            ),
        )

    if requested_model in _EMBEDDING_MODELS:
        return requested_model

    _reject_unknown_embedding_model(requested_model)


def resolve_stt_model_name(requested_model: str) -> str:
    """Resolve an STT request model alias or configured model ID."""
    if requested_model in _STT_MODEL_MAP:
        return _STT_MODEL_MAP[requested_model]
    _reject_unknown_audio_model("Transcription", requested_model, _STT_MODEL_ALIASES)


def resolve_tts_model_name(requested_model: str) -> str:
    """Resolve a TTS request model alias or configured model ID."""
    if requested_model in _TTS_MODEL_MAP:
        return _TTS_MODEL_MAP[requested_model]
    _reject_unknown_audio_model("Speech", requested_model, _TTS_MODEL_ALIASES)
