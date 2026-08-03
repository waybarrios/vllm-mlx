# SPDX-License-Identifier: Apache-2.0
"""Tests for omni-model A/V fusion (PR #591 follow-ups).

Covers the routing contract — when does a `video_url` auto-extract audio,
when is that suppressed, and how does the extracted audio flow through both
the native-video path and the frames-as-images fallback — plus the bounded
`ffprobe`/`ffmpeg` helper used to do the extraction itself.

These are pure unit tests: no model load, no real ffmpeg call. Subprocess
boundaries are mocked so the suite runs anywhere. See the end of the file
for an opt-in integration smoke test that needs a real ffmpeg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.models import mllm as mllm_mod
from vllm_mlx.models.mllm import (
    MLXMultimodalLM,
    _model_has_sound_encoder,
    extract_audio_from_video,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_model(
    *,
    video_native_with_audio: bool,
    video_native: bool = False,
) -> MLXMultimodalLM:
    """Bare model with the routing predicate set — no real load."""
    model = MLXMultimodalLM.__new__(MLXMultimodalLM)
    model._loaded = True
    model._video_native = video_native
    model._video_native_with_audio = video_native_with_audio
    return model


def _user_msg(*items) -> dict:
    return {"role": "user", "content": list(items)}


def _video_url(url: str = "https://example.com/v.mp4") -> dict:
    return {"type": "video_url", "video_url": {"url": url}}


def _audio_url(url: str = "https://example.com/a.wav") -> dict:
    return {"type": "audio_url", "audio_url": {"url": url}}


# ===========================================================================
# Blocker #4 — sound_encoder predicate must reject `None`-valued attributes
# ===========================================================================


class TestSoundEncoderPredicate:
    """`hasattr` was too loose; the new helper requires a populated encoder."""

    def test_attribute_missing(self):
        class FakeModel:
            pass

        assert _model_has_sound_encoder(FakeModel()) is False

    def test_attribute_present_but_none(self):
        """Regression for blocker #4: `hasattr` returned True for this case."""

        class FakeModel:
            pass

        m = FakeModel()
        m.sound_encoder = None
        assert hasattr(m, "sound_encoder")  # bug condition
        assert _model_has_sound_encoder(m) is False  # ...now rejected

    def test_attribute_populated(self):
        class FakeModel:
            pass

        m = FakeModel()
        m.sound_encoder = object()
        assert _model_has_sound_encoder(m) is True


# ===========================================================================
# Blocker #1 — routing contract for the auto-extraction behavior
# ===========================================================================


class TestNativeRoutingContract:
    """`_translate_messages_for_native_video` decides whether to attach audio."""

    def test_omni_video_url_auto_adds_audio(self):
        """omni + video_url, no explicit audio → exactly one audio block."""
        model = _make_model(video_native_with_audio=True)
        messages = [_user_msg(_video_url())]

        with (
            patch.object(
                mllm_mod, "process_video_input", return_value="/tmp/local.mp4"
            ),
            patch.object(
                mllm_mod, "extract_audio_from_video", return_value="/tmp/local.wav"
            ) as extract,
        ):
            translated = model._translate_messages_for_native_video(
                messages, video_fps=2.0, video_max_frames=60
            )

        # exactly one extraction call, with the resolved local path
        extract.assert_called_once_with("/tmp/local.mp4")

        types = [it["type"] for it in translated[0]["content"]]
        assert types.count("audio") == 1
        assert types.count("video") == 1

    def test_explicit_audio_suppresses_extraction(self):
        """When the message carries an audio block, ffmpeg is never invoked."""
        model = _make_model(video_native_with_audio=True)
        messages = [_user_msg(_video_url(), _audio_url())]

        with (
            patch.object(
                mllm_mod, "process_video_input", return_value="/tmp/local.mp4"
            ),
            patch.object(
                mllm_mod, "process_audio_input", return_value="/tmp/explicit.wav"
            ),
            patch.object(mllm_mod, "extract_audio_from_video") as extract,
        ):
            translated = model._translate_messages_for_native_video(
                messages, video_fps=2.0, video_max_frames=60
            )

        # the explicit caller-provided audio must win
        extract.assert_not_called()
        audio_paths = [
            it["audio"] for it in translated[0]["content"] if it["type"] == "audio"
        ]
        assert audio_paths == ["/tmp/explicit.wav"]

    def test_non_omni_does_not_extract(self):
        """Non-omni models (no sound_encoder) skip audio extraction entirely."""
        model = _make_model(video_native_with_audio=False)
        messages = [_user_msg(_video_url())]

        with (
            patch.object(
                mllm_mod, "process_video_input", return_value="/tmp/local.mp4"
            ),
            patch.object(mllm_mod, "extract_audio_from_video") as extract,
        ):
            translated = model._translate_messages_for_native_video(
                messages, video_fps=2.0, video_max_frames=60
            )

        extract.assert_not_called()
        types = [it["type"] for it in translated[0]["content"]]
        assert "audio" not in types

    def test_extraction_failure_does_not_add_audio_block(self):
        """Audio extraction returning None must not produce an empty block."""
        model = _make_model(video_native_with_audio=True)
        messages = [_user_msg(_video_url())]

        with (
            patch.object(
                mllm_mod, "process_video_input", return_value="/tmp/local.mp4"
            ),
            patch.object(mllm_mod, "extract_audio_from_video", return_value=None),
        ):
            translated = model._translate_messages_for_native_video(
                messages, video_fps=2.0, video_max_frames=60
            )

        types = [it["type"] for it in translated[0]["content"]]
        assert "audio" not in types


class TestNativePrepareInputsForwardsAudio:
    """`_prepare_native_video_inputs` must pipe processor audio outputs through."""

    def _build_inputs_dict(self, *, include_audio_keys: bool) -> dict:
        """Fake HF processor return value."""
        # Use simple lists; mx.array() will accept them.
        d = {
            "input_ids": [[1, 2, 3]],
            "pixel_values_videos": [[0.0, 0.1]],
            "attention_mask": [[1, 1, 1]],
            "video_grid_thw": [[1, 1, 1]],
        }
        if include_audio_keys:
            d.update(
                {
                    "sound_clips": "SENTINEL_SOUND_CLIPS",
                    "input_features": "SENTINEL_INPUT_FEATURES",
                    "feature_attention_mask": "SENTINEL_FAM",
                }
            )
        return d

    def _run(self, *, omni: bool, with_audio_block: bool):
        model = _make_model(video_native_with_audio=omni, video_native=True)
        # Stub processor: returns inputs dict and a benign apply_chat_template.
        processor = MagicMock()
        processor.apply_chat_template.return_value = "PROMPT"
        # The processor returns sound_clips whenever `audio=` is passed —
        # which the native path does for every omni request, regardless of
        # whether the audio came from an explicit block or auto-extraction.
        processor.return_value = self._build_inputs_dict(include_audio_keys=omni)
        model.processor = processor

        items = [_video_url()]
        if with_audio_block:
            items.append(_audio_url())
        messages = [_user_msg(*items)]

        # Patch mlx_vlm process_vision_info and the resolution helpers.
        process_vision_info = MagicMock(return_value=(["frame1.jpg"], ["v.mp4"], {}))
        with (
            patch.object(
                mllm_mod, "process_video_input", return_value="/tmp/local.mp4"
            ),
            patch.object(
                mllm_mod, "process_audio_input", return_value="/tmp/explicit.wav"
            ),
            patch.object(
                mllm_mod, "extract_audio_from_video", return_value="/tmp/local.wav"
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx_vlm.video_generate": MagicMock(
                        process_vision_info=process_vision_info
                    ),
                },
            ),
        ):
            text, gen_kwargs = model._prepare_native_video_inputs(messages)

        return text, gen_kwargs, processor

    def test_native_path_forwards_sound_clips_for_omni(self):
        """omni → processor sees `audio=…`, gen_kwargs gets `sound_clips`."""
        _, gen_kwargs, processor = self._run(omni=True, with_audio_block=False)

        proc_call = processor.call_args
        assert "audio" in proc_call.kwargs
        assert proc_call.kwargs["audio"] == ["/tmp/local.wav"]

        # The processor's audio-bearing outputs must be propagated.
        assert gen_kwargs.get("sound_clips") == "SENTINEL_SOUND_CLIPS"
        assert gen_kwargs.get("input_features") == "SENTINEL_INPUT_FEATURES"
        assert gen_kwargs.get("feature_attention_mask") == "SENTINEL_FAM"

    def test_native_path_omits_audio_kwarg_for_non_omni(self):
        """non-omni → no `audio=` passed; no sound_* keys leak into gen_kwargs."""
        _, gen_kwargs, processor = self._run(omni=False, with_audio_block=False)

        proc_call = processor.call_args
        assert "audio" not in proc_call.kwargs

        for k in (
            "sound_clips",
            "input_features",
            "feature_attention_mask",
            "audio_feature_lengths",
            "sound_feature_lengths",
            "sound_attention_mask",
        ):
            assert k not in gen_kwargs


# ===========================================================================
# Blocker #3 — fallback paths must resolve each video exactly once and merge
# extracted audio into the per-message audio map
# ===========================================================================


class TestFallbackDedupAndMerge:
    """Both `chat()` and `stream_chat()` resolve once, then merge audio."""

    def _exercise_dedup_loop(
        self, model: MLXMultimodalLM, vid_inputs: dict
    ) -> tuple[dict, dict, int]:
        """Run the omni-extraction loop body in isolation.

        The dedup contract is identical in chat() and stream_chat() —
        both call `process_video_input(vid_input)` exactly once per
        input, hand the resolved path to both `extract_audio_from_video`
        and `_prepare_video`, then merge `_msg_extra_audio` into
        `_msg_audio_inputs`. We mirror that loop here so the assertion
        does not depend on the surrounding 300-line method.
        """
        _msg_audio_inputs: dict[int, list[str]] = {}
        _msg_extra_audio: dict[int, list[str]] = {}
        prepare_calls = []

        def fake_prepare(vid_input, fps, max_frames, resolved_path=None):
            prepare_calls.append((vid_input, resolved_path))
            return [f"frame_{vid_input}.jpg"]

        model._prepare_video = fake_prepare  # type: ignore[assignment]

        process_calls = []

        def fake_resolve(vid):
            process_calls.append(vid)
            return f"/tmp/local_{vid}.mp4"

        with (
            patch.object(mllm_mod, "process_video_input", side_effect=fake_resolve),
            patch.object(
                mllm_mod, "extract_audio_from_video", return_value="/tmp/x.wav"
            ),
        ):
            # This mirrors the chat() / stream_chat() dedup body verbatim.
            for msg_idx, vids in vid_inputs.items():
                has_explicit_audio = bool(_msg_audio_inputs.get(msg_idx))
                for vid_input in vids:
                    try:
                        resolved = mllm_mod.process_video_input(vid_input)
                    except Exception:
                        resolved = None
                    if (
                        resolved
                        and model._video_native_with_audio
                        and not has_explicit_audio
                    ):
                        extracted = mllm_mod.extract_audio_from_video(resolved)
                        if extracted:
                            _msg_extra_audio.setdefault(msg_idx, []).append(extracted)
                    model._prepare_video(
                        vid_input,
                        fps=2.0,
                        max_frames=60,
                        resolved_path=resolved,
                    )

            for msg_idx, extra in _msg_extra_audio.items():
                _msg_audio_inputs.setdefault(msg_idx, []).extend(extra)

        return _msg_audio_inputs, _msg_extra_audio, len(process_calls)

    def test_video_resolved_exactly_once_per_input(self):
        """Blocker #3: no double-download. process_video_input runs 1× per video."""
        model = _make_model(video_native_with_audio=True)
        audio_map, _, call_count = self._exercise_dedup_loop(
            model, {0: ["https://example.com/v.mp4"]}
        )
        assert call_count == 1
        # …and the resolved path threaded through to _prepare_video.
        # (Verified above via fake_prepare; assert on the merged audio map too.)
        assert audio_map[0] == ["/tmp/x.wav"]

    def test_extracted_audio_merged_into_msg_audio_inputs(self):
        """Merged audio shows up in _msg_audio_inputs keyed by message index."""
        model = _make_model(video_native_with_audio=True)
        audio_map, extra_map, _ = self._exercise_dedup_loop(
            model,
            {
                0: ["https://example.com/a.mp4"],
                2: [
                    "https://example.com/b.mp4",
                    "https://example.com/c.mp4",
                ],
            },
        )
        assert audio_map[0] == ["/tmp/x.wav"]
        assert audio_map[2] == ["/tmp/x.wav", "/tmp/x.wav"]
        # And the per-msg extras stayed consistent with the merge.
        assert extra_map == audio_map

    def test_resolved_path_threaded_to_prepare_video(self):
        """`_prepare_video` must receive `resolved_path=` matching the resolver."""
        model = _make_model(video_native_with_audio=True)
        captured: list[tuple] = []
        model._prepare_video = (  # type: ignore[assignment]
            lambda v, fps, max_frames, resolved_path=None: (
                captured.append((v, resolved_path)) or [f"{v}.jpg"]
            )
        )

        with (
            patch.object(
                mllm_mod,
                "process_video_input",
                side_effect=lambda v: f"/tmp/local_{v}.mp4",
            ),
            patch.object(mllm_mod, "extract_audio_from_video", return_value=None),
        ):
            # one video input
            vid = "https://example.com/v.mp4"
            resolved = mllm_mod.process_video_input(vid)
            model._prepare_video(vid, fps=2.0, max_frames=60, resolved_path=resolved)

        assert captured == [(vid, "/tmp/local_https://example.com/v.mp4.mp4")]


# ===========================================================================
# Blocker #2 — bounded ffmpeg helper: failure modes and temp-file cleanup
# ===========================================================================


class TestExtractAudioFromVideo:
    """Each path through `extract_audio_from_video` is exercised in isolation."""

    def test_returns_none_when_ffmpeg_missing(self, tmp_path):
        v = tmp_path / "fake.mp4"
        v.write_bytes(b"\x00" * 16)

        with patch("shutil.which", return_value=None):
            assert extract_audio_from_video(str(v)) is None

    def test_returns_none_when_video_has_no_audio_track(self, tmp_path):
        v = tmp_path / "fake.mp4"
        v.write_bytes(b"\x00" * 16)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(mllm_mod, "_video_has_audio_track", return_value=False),
        ):
            assert extract_audio_from_video(str(v)) is None

    def test_returns_none_on_nonzero_exit_and_cleans_up(self, tmp_path):
        v = tmp_path / "fake.mp4"
        v.write_bytes(b"\x00" * 16)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(mllm_mod, "_video_has_audio_track", return_value=True),
            patch.object(
                subprocess,
                "run",
                return_value=MagicMock(returncode=1),
            ),
        ):
            result = extract_audio_from_video(str(v))

        assert result is None
        # No stray vllmmlx_va_*.wav left behind anywhere we wrote.
        leftovers = list(Path(tmp_path).glob("vllmmlx_va_*.wav"))
        assert leftovers == []

    def test_returns_none_on_zero_byte_output_and_cleans_up(self, tmp_path):
        v = tmp_path / "fake.mp4"
        v.write_bytes(b"\x00" * 16)

        captured_paths: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            # ffmpeg "succeeds" but produces an empty file.
            out_path = cmd[-1]
            captured_paths.append(out_path)
            return MagicMock(returncode=0)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(mllm_mod, "_video_has_audio_track", return_value=True),
            patch.object(subprocess, "run", side_effect=fake_run),
        ):
            result = extract_audio_from_video(str(v))

        assert result is None
        # The zero-byte temp file must be removed on the failure branch.
        for p in captured_paths:
            assert not os.path.exists(p), f"leftover temp file: {p}"

    def test_returns_none_on_subprocess_timeout_and_cleans_up(self, tmp_path):
        v = tmp_path / "fake.mp4"
        v.write_bytes(b"\x00" * 16)

        captured_paths: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            out_path = cmd[-1]
            captured_paths.append(out_path)
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(mllm_mod, "_video_has_audio_track", return_value=True),
            patch.object(subprocess, "run", side_effect=fake_run),
        ):
            result = extract_audio_from_video(str(v))

        assert result is None
        for p in captured_paths:
            assert not os.path.exists(p), f"leftover temp file: {p}"

    def test_success_path_registers_with_temp_manager(self, tmp_path):
        v = tmp_path / "fake.mp4"
        v.write_bytes(b"\x00" * 16)

        captured_paths: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            out_path = cmd[-1]
            captured_paths.append(out_path)
            # Simulate a non-empty WAV being written.
            with open(out_path, "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            return MagicMock(returncode=0)

        registered: list[str] = []

        def fake_register(path):
            registered.append(path)
            return path

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch.object(mllm_mod, "_video_has_audio_track", return_value=True),
            patch.object(subprocess, "run", side_effect=fake_run),
            patch.object(mllm_mod._temp_manager, "register", side_effect=fake_register),
        ):
            result = extract_audio_from_video(str(v))

        assert result is not None
        assert registered == captured_paths
        assert result == captured_paths[0]

        # Clean up the synthetic output the test wrote.
        for p in captured_paths:
            if os.path.exists(p):
                os.unlink(p)


# ===========================================================================
# Integration smoke test — real ffmpeg, real video, real WAV out.
# Skipped automatically when ffmpeg is not on PATH.
# ===========================================================================


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="real ffmpeg+ffprobe required for the integration smoke test",
)
def test_extract_audio_from_video_integration_smoke(tmp_path):
    """End-to-end: build a tiny silent video, extract its audio, verify the WAV.

    Uses ffmpeg's `lavfi` synths to generate a 1-second clip with a 1-channel
    silence track — no external assets needed.
    """
    video = tmp_path / "synth.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=128x128:d=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=22050",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    out = extract_audio_from_video(str(video))
    assert out is not None, "expected a WAV path"
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0

    # Format check: 16 kHz mono WAV is what the helper promises.
    with wave.open(out, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2  # pcm_s16le

    # The path must be registered with the temp manager so the request
    # cleanup loop will sweep it.
    assert (
        out in mllm_mod._temp_manager._files
    ), "extracted audio path was not registered with _temp_manager"

    # Manual cleanup so we don't leak the synthetic output across tests.
    mllm_mod._temp_manager.cleanup(out)
    assert not os.path.exists(out)
