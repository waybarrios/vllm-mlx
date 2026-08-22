# SPDX-License-Identifier: Apache-2.0
"""
Tests for audio support (STT, TTS, audio processing).

Note: Some tests require mlx-audio to be installed.
"""

import pytest
import numpy as np


def _install_fake_stt_loader(monkeypatch, model=None, load_model=None):
    """Install a minimal mlx-audio module tree around a fake STT model."""
    import sys
    from types import ModuleType

    utils_module = ModuleType("mlx_audio.stt.utils")
    utils_module.load_model = load_model or (lambda _model_name: model)
    stt_module = ModuleType("mlx_audio.stt")
    stt_module.utils = utils_module
    mlx_audio_module = ModuleType("mlx_audio")
    mlx_audio_module.stt = stt_module
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils_module)


class TestSTTEngine:
    """Tests for Speech-to-Text engine."""

    def test_init_whisper(self):
        """Test STT engine initialization with Whisper."""
        from vllm_mlx.audio.stt import STTEngine

        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        assert engine.model_name == "mlx-community/whisper-large-v3-mlx"
        assert engine._is_parakeet is False
        assert engine._loaded is False

    def test_init_parakeet(self):
        """Test STT engine initialization with Parakeet."""
        from vllm_mlx.audio.stt import STTEngine

        engine = STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
        assert engine._is_parakeet is True

    def test_default_models(self):
        """Test default model constants."""
        from vllm_mlx.audio.stt import DEFAULT_WHISPER_MODEL, DEFAULT_PARAKEET_MODEL

        assert "whisper" in DEFAULT_WHISPER_MODEL.lower()
        assert "parakeet" in DEFAULT_PARAKEET_MODEL.lower()

    def test_transcription_result(self):
        """Test TranscriptionResult dataclass."""
        from vllm_mlx.audio.stt import TranscriptionResult

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.5,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.5

    @pytest.mark.parametrize(
        ("model_name", "processor_repo"),
        [
            ("mlx-community/whisper-tiny-mlx", "openai/whisper-tiny"),
            ("mlx-community/whisper-small-mlx", "openai/whisper-small"),
            ("mlx-community/whisper-medium-mlx", "openai/whisper-medium"),
            ("mlx-community/whisper-large-v3-mlx", "openai/whisper-large-v3"),
            (
                "mlx-community/whisper-large-v3-turbo",
                "openai/whisper-large-v3-turbo",
            ),
        ],
    )
    def test_load_recovers_missing_whisper_processor(
        self, monkeypatch, model_name, processor_repo
    ):
        """Documented MLX checkpoints should use their canonical processor."""
        from types import SimpleNamespace

        from vllm_mlx.audio.stt import STTEngine

        model = SimpleNamespace(_processor=None)
        _install_fake_stt_loader(monkeypatch, model)

        processor = object()
        calls = []

        def fake_from_pretrained(requested_repo):
            calls.append(requested_repo)
            return processor

        monkeypatch.setattr(
            "transformers.WhisperProcessor.from_pretrained",
            fake_from_pretrained,
        )

        engine = STTEngine(model_name)
        engine.load()

        assert model._processor is processor
        assert calls == [processor_repo]
        assert engine._loaded is True

    def test_load_preserves_existing_whisper_processor(self, monkeypatch):
        """A processor bundled with the MLX checkpoint remains authoritative."""
        from types import SimpleNamespace

        from vllm_mlx.audio.stt import STTEngine

        processor = object()
        model = SimpleNamespace(_processor=processor)
        _install_fake_stt_loader(monkeypatch, model)

        def unexpected_processor_load(_model_name):
            raise AssertionError("processor should not be replaced")

        monkeypatch.setattr(
            "transformers.WhisperProcessor.from_pretrained",
            unexpected_processor_load,
        )

        engine = STTEngine("mlx-community/whisper-small-mlx")
        engine.load()

        assert model._processor is processor

    def test_processor_load_failure_does_not_retain_model(self, monkeypatch):
        """A failed processor download should leave the engine retry-safe."""
        from types import SimpleNamespace

        from vllm_mlx.audio.stt import STTEngine

        model = SimpleNamespace(_processor=None)
        load_observations = []
        engine = STTEngine("mlx-community/whisper-small-mlx")

        def fake_load_model(_model_name):
            load_observations.append(engine.model)
            return model

        _install_fake_stt_loader(monkeypatch, load_model=fake_load_model)

        def fail_processor_load(_model_name):
            raise OSError("offline")

        monkeypatch.setattr(
            "transformers.WhisperProcessor.from_pretrained",
            fail_processor_load,
        )

        with pytest.raises(OSError, match="offline"):
            engine.load()
        with pytest.raises(OSError, match="offline"):
            engine.load()

        assert load_observations == [None, None]
        assert engine.model is None
        assert engine._loaded is False

    def test_load_does_not_attach_processor_to_parakeet(self, monkeypatch):
        """Parakeet keeps its native mlx-audio loading path."""
        from types import SimpleNamespace

        from vllm_mlx.audio.stt import STTEngine

        model = SimpleNamespace(_processor=None)
        _install_fake_stt_loader(monkeypatch, model)

        def unexpected_processor_load(_model_name):
            raise AssertionError("Whisper processor should not load for Parakeet")

        monkeypatch.setattr(
            "transformers.WhisperProcessor.from_pretrained",
            unexpected_processor_load,
        )

        engine = STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
        engine.load()

        assert model._processor is None

    def test_transcription_duration_supports_dictionary_segments(self):
        """mlx-audio Whisper returns segment dictionaries."""
        from types import SimpleNamespace

        from vllm_mlx.audio.stt import STTEngine

        engine = STTEngine("mlx-community/whisper-small-mlx")
        engine._loaded = True
        engine.model = SimpleNamespace(
            generate=lambda *_args, **_kwargs: SimpleNamespace(
                text="hello",
                language="en",
                segments=[{"start": 0.0, "end": 1.25, "text": "hello"}],
            )
        )

        result = engine.transcribe("speech.wav")

        assert result.duration == 1.25


class TestTTSEngine:
    """Tests for Text-to-Speech engine."""

    def test_init_kokoro(self):
        """Test TTS engine initialization with Kokoro."""
        from vllm_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        assert engine.model_name == "mlx-community/Kokoro-82M-bf16"
        assert engine._model_family == "kokoro"
        assert engine._loaded is False

    def test_init_chatterbox(self):
        """Test TTS engine initialization with Chatterbox."""
        from vllm_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/chatterbox-turbo-fp16")
        assert engine._model_family == "chatterbox"

    def test_init_vibevoice(self):
        """Test TTS engine initialization with VibeVoice."""
        from vllm_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VibeVoice-Realtime-0.5B-4bit")
        assert engine._model_family == "vibevoice"

    def test_init_voxcpm(self):
        """Test TTS engine initialization with VoxCPM."""
        from vllm_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VoxCPM1.5")
        assert engine._model_family == "voxcpm"

    def test_available_voices(self):
        """Test voice lists."""
        from vllm_mlx.audio.tts import KOKORO_VOICES, CHATTERBOX_VOICES

        assert "af_heart" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 5
        assert "default" in CHATTERBOX_VOICES

    def test_get_voices(self):
        """Test get_voices method."""
        from vllm_mlx.audio.tts import TTSEngine

        kokoro = TTSEngine("mlx-community/Kokoro-82M-bf16")
        voices = kokoro.get_voices()
        assert "af_heart" in voices

    def test_audio_output(self):
        """Test AudioOutput dataclass."""
        from vllm_mlx.audio.tts import AudioOutput

        audio = np.zeros(24000, dtype=np.float32)
        output = AudioOutput(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
        )
        assert output.sample_rate == 24000
        assert output.duration == 1.0
        assert len(output.audio) == 24000


class TestAudioProcessor:
    """Tests for audio processor (SAM-Audio)."""

    def test_init(self):
        """Test audio processor initialization."""
        from vllm_mlx.audio.processor import AudioProcessor

        processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
        assert processor.model_name == "mlx-community/sam-audio-large-fp16"
        assert processor._loaded is False

    def test_default_model(self):
        """Test default SAM-Audio model."""
        from vllm_mlx.audio.processor import DEFAULT_SAM_MODEL

        assert "sam-audio" in DEFAULT_SAM_MODEL.lower()

    def test_separation_result(self):
        """Test SeparationResult dataclass."""
        from vllm_mlx.audio.processor import SeparationResult

        target = np.zeros(44100, dtype=np.float32)
        residual = np.zeros(44100, dtype=np.float32)

        result = SeparationResult(
            target=target,
            residual=residual,
            sample_rate=44100,
            peak_memory=1.5,
        )
        assert result.sample_rate == 44100
        assert result.peak_memory == 1.5
        assert len(result.target) == 44100


class TestAPIModels:
    """Tests for audio API models."""

    def test_audio_url(self):
        """Test AudioUrl model."""
        from vllm_mlx.api.models import AudioUrl

        url = AudioUrl(url="file://test.mp3")
        assert url.url == "file://test.mp3"

    def test_content_part_audio(self):
        """Test ContentPart with audio."""
        from vllm_mlx.api.models import ContentPart

        part = ContentPart(type="audio_url", audio_url={"url": "test.mp3"})
        assert part.type == "audio_url"
        # Pydantic converts dict to AudioUrl model
        assert part.audio_url.url == "test.mp3"

    def test_transcription_request(self):
        """Test AudioTranscriptionRequest model."""
        from vllm_mlx.api.models import AudioTranscriptionRequest

        req = AudioTranscriptionRequest(
            model="whisper-large-v3",
            language="en",
        )
        assert req.model == "whisper-large-v3"
        assert req.language == "en"
        assert req.response_format == "json"

    def test_speech_request(self):
        """Test AudioSpeechRequest model."""
        from vllm_mlx.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(
            model="kokoro",
            input="Hello world",
            voice="af_heart",
            speed=1.2,
        )
        assert req.model == "kokoro"
        assert req.input == "Hello world"
        assert req.voice == "af_heart"
        assert req.speed == 1.2

    def test_transcription_response(self):
        """Test AudioTranscriptionResponse model."""
        from vllm_mlx.api.models import AudioTranscriptionResponse

        resp = AudioTranscriptionResponse(
            text="Hello world",
            language="en",
            duration=2.5,
        )
        assert resp.text == "Hello world"


class TestAudioImports:
    """Test that all audio modules can be imported."""

    def test_import_audio_module(self):
        """Test importing main audio module."""
        from vllm_mlx.audio import (
            STTEngine,
            TTSEngine,
            AudioProcessor,
        )

        assert STTEngine is not None
        assert TTSEngine is not None
        assert AudioProcessor is not None

    def test_import_api_models(self):
        """Test importing audio API models."""
        from vllm_mlx.api import (
            AudioUrl,
            AudioTranscriptionRequest,
        )

        assert AudioUrl is not None
        assert AudioTranscriptionRequest is not None


# Integration tests (require mlx-audio installed)
@pytest.mark.skip(reason="Requires mlx-audio and models downloaded")
class TestAudioIntegration:
    """Integration tests for audio (require models)."""

    def test_whisper_transcription(self):
        """Test Whisper transcription."""
        from vllm_mlx.audio import transcribe_audio

        result = transcribe_audio(
            "test_audio.wav",
            model_name="mlx-community/whisper-small-mlx",
        )
        assert result.text is not None

    def test_kokoro_tts(self):
        """Test Kokoro TTS generation."""
        from vllm_mlx.audio import generate_speech

        audio = generate_speech(
            "Hello world",
            model_name="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
        )
        assert audio.audio is not None
        assert audio.sample_rate > 0

    def test_sam_audio_separation(self):
        """Test SAM-Audio voice separation."""
        from vllm_mlx.audio import separate_voice

        target, residual = separate_voice(
            "test_audio.wav",
            model_name="mlx-community/sam-audio-small",
        )
        assert target is not None
        assert residual is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
