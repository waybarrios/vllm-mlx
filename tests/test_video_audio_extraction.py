# SPDX-License-Identifier: Apache-2.0
"""The Omni audio-extraction subprocesses must never touch the terminal.

ffmpeg polls stdin for interactive keys. When the server runs in a
background process group - which every job-control launcher creates - a
background read of the controlling terminal raises SIGTTIN against the
whole group, server included. The symptom is a request frozen at 0% CPU
with no error, until subprocess.run's timeout fires and the audio is
silently dropped. Both the ffmpeg flag and the stdin redirect are asserted
so a refactor cannot quietly reintroduce either half.

Also pinned here: an auto-extracted soundtrack gets its own template audio
placeholder, and across messages the audio list follows message order, so
each audio feature binds to its own message's slot (chat and streaming).

Runs without MLX: nothing here needs a model; mlx-vlm is stubbed.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.models import mllm


@pytest.fixture
def captured_runs(monkeypatch, tmp_path):
    """Route ffmpeg/ffprobe through a recorder instead of the real binaries."""
    calls: list[dict] = []

    def fake_run(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        # extract_audio_from_video checks the WAV is non-empty afterwards.
        if argv[0] == "ffmpeg":
            open(argv[-1], "wb").write(b"\0" * 64)
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 0, stdout="audio\n", stderr="")

    # mllm imports subprocess/shutil inside the functions, so patch the
    # modules themselves rather than attributes on mllm.
    import shutil

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mllm._temp_manager, "register", lambda p: p, raising=False)
    return calls


def test_ffmpeg_never_reads_the_terminal(captured_runs, tmp_path):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"\0" * 16)

    out = mllm.extract_audio_from_video(str(video))

    assert out is not None and out.endswith(".wav")
    ffmpeg = [c for c in captured_runs if c["argv"][0] == "ffmpeg"]
    assert len(ffmpeg) == 1
    assert "-nostdin" in ffmpeg[0]["argv"], "ffmpeg must not poll stdin"
    assert (
        ffmpeg[0]["kwargs"].get("stdin") is subprocess.DEVNULL
    ), "ffmpeg must inherit a closed stdin, not the launcher's tty"


def test_ffprobe_never_reads_the_terminal(captured_runs, tmp_path):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"\0" * 16)

    assert mllm._video_has_audio_track(str(video)) is True
    ffprobe = [c for c in captured_runs if c["argv"][0] == "ffprobe"]
    assert len(ffprobe) == 1
    assert ffprobe[0]["kwargs"].get("stdin") is subprocess.DEVNULL


# ---------------------------------------------------------------------------
# The extracted audio must also reach the chat template as a placeholder.
#
# Extraction alone is not enough: the file goes to generate(audio=...) and the
# encoder produces N features, but the model splices those into N placeholder
# tokens that only the chat template emits - and it emits one per audio content
# PART. The video branch used to insert frames and nothing else, so an Omni
# model got N features and zero slots:
#   ValueError: Sound token count (0) does not match feature count (36)
# ---------------------------------------------------------------------------


def _parts(messages, *, frames, audio):
    built = mllm._build_mllm_chat_messages(
        messages,
        all_image_urls=[],
        video_frame_counts=frames,
        video_audio_counts=audio,
    )
    return [p["type"] for p in built[0]["content"]]


VIDEO_MSG = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe vision and sound"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        ],
    }
]


def test_auto_extracted_audio_gets_a_template_placeholder():
    parts = _parts(VIDEO_MSG, frames={0: 4}, audio={0: 1})
    assert parts == ["text", "image", "image", "image", "image", "audio"]


def test_video_without_audio_track_adds_no_placeholder():
    parts = _parts(VIDEO_MSG, frames={0: 4}, audio={})
    assert parts == ["text", "image", "image", "image", "image"]


def test_second_video_part_does_not_duplicate_the_placeholder():
    two_videos = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,AAAA"},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,BBBB"},
                },
            ],
        }
    ]
    parts = _parts(two_videos, frames={0: 2}, audio={0: 1})
    assert (
        parts.count("audio") == 1
    ), "audio placeholder is consumed by the first video part"


def test_explicit_audio_part_still_renders_independently():
    msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64,AAAA"},
                },
                {"type": "text", "text": "what is this"},
            ],
        }
    ]
    parts = _parts(msg, frames={}, audio={})
    assert parts == ["audio", "text"]


# --- audio order across messages (review on #807) --------------------------
#
# The chat template renders audio placeholders message by message, so the
# audio list handed to mlx-vlm must follow message order. Explicit audio is
# collected first and a video soundtrack is added afterwards, so flattening by
# dict insertion order put a later message's explicit audio in front of an
# earlier message's soundtrack: each feature bound to the other message's slot.

VIDEO_THEN_AUDIO = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "watch this"},
            {"type": "video_url", "video_url": {"url": "clip.mov"}},
        ],
    },
    {"role": "assistant", "content": "ok"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "now listen"},
            {"type": "audio_url", "audio_url": {"url": "note.wav"}},
        ],
    },
]
SOUNDTRACK = "soundtrack-of-clip.mov"
EXPECTED_BY_MESSAGE = {0: SOUNDTRACK, 2: "note.wav"}


def test_audio_inputs_follow_message_order():
    explicit = {2: ["note.wav"]}
    extracted = {0: [SOUNDTRACK]}
    assert mllm._audio_inputs_in_message_order(explicit, extracted) == [
        SOUNDTRACK,
        "note.wav",
    ]
    # Explicit audio first in the conversation is unaffected.
    assert mllm._audio_inputs_in_message_order({0: ["a.wav"]}, {3: ["s"]}) == [
        "a.wav",
        "s",
    ]


def _slot_owners(chat_messages: list[dict]) -> list[int]:
    """Message index of every audio placeholder, in rendered order."""
    owners = []
    for msg_idx, message in enumerate(chat_messages):
        content = message.get("content")
        if isinstance(content, list):
            owners.extend(
                msg_idx
                for part in content
                if isinstance(part, dict) and part.get("type") == "audio"
            )
    return owners


@pytest.fixture
def omni_model(monkeypatch):
    """An omni model on the frames fallback, with media I/O stubbed out."""
    model = mllm.MLXMultimodalLM(model_name="local-omni-test")
    model._loaded = True
    model.model = object()
    # chat() tokenizes the prompt for its prefix-cache lookup.
    model.processor = SimpleNamespace(encode=lambda text: [1])
    model._cache_manager = None
    model._video_native = False
    model._video_native_with_audio = True
    model._prepare_video = lambda video, **kwargs: ["frame"]
    model._prepare_audio = lambda inputs: list(inputs)

    monkeypatch.setattr(mllm, "process_video_input", lambda v: v)
    monkeypatch.setattr(
        mllm, "extract_audio_from_video", lambda path: f"soundtrack-of-{path}"
    )

    seen: dict = {}

    def fake_get_chat_template(processor, chat_messages, **kwargs):
        seen["chat_messages"] = chat_messages
        return "prompt"

    class Chunk:
        text = "x"
        prompt_tokens = 1

    def fake_generate(*args, **kwargs):
        seen["audio"] = kwargs.get("audio")
        return SimpleNamespace(text="x", prompt_tokens=1, generation_tokens=1)

    def fake_stream_generate(*args, **kwargs):
        seen["audio"] = kwargs.get("audio")
        yield Chunk()

    fake_mlx_vlm = ModuleType("mlx_vlm")
    fake_mlx_vlm.generate = fake_generate
    fake_mlx_vlm.stream_generate = fake_stream_generate
    fake_prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    fake_prompt_utils.get_chat_template = fake_get_chat_template
    fake_models = ModuleType("mlx_vlm.models")
    fake_cache = ModuleType("mlx_vlm.models.cache")
    fake_cache.make_prompt_cache = lambda *args, **kwargs: None
    fake_models.cache = fake_cache
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", fake_prompt_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", fake_cache)
    return model, seen


def _assert_each_slot_gets_its_own_audio(seen: dict) -> None:
    owners = _slot_owners(seen["chat_messages"])
    assert owners == [0, 2]
    assert seen["audio"] == [EXPECTED_BY_MESSAGE[i] for i in owners]


def test_chat_binds_soundtrack_and_explicit_audio_to_their_messages(omni_model):
    model, seen = omni_model
    model.chat(messages=VIDEO_THEN_AUDIO, max_tokens=4)
    _assert_each_slot_gets_its_own_audio(seen)


def test_stream_chat_binds_soundtrack_and_explicit_audio_to_their_messages(
    omni_model,
):
    model, seen = omni_model
    list(model.stream_chat(messages=VIDEO_THEN_AUDIO, max_tokens=4))
    _assert_each_slot_gets_its_own_audio(seen)
