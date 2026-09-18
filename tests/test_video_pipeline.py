# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the native video path.

These run without MLX. vllm_mlx.models.mllm imports mlx lazily inside the
functions that need it, so the decode and preprocessing logic can be exercised
with a stand-in mlx_vlm — which is the point: the failures this file guards
against were an upstream module rename and a silently empty decode, and neither
needs a GPU to catch.

A second file's worth of behaviour — the actual model output — needs Apple
Silicon and is not attempted here.
"""

from __future__ import annotations

import base64
import sys
import types

import numpy as np
import pytest

from vllm_mlx.api.utils import is_mllm_model
from vllm_mlx.models.mllm import (
    MAX_FRAMES,
    MIN_FRAME_DIMENSION,
    VIDEO_MIME_TO_SUFFIX,
    assert_video_decoded,
    decode_base64_video,
)


def _frames(count: int, height: int = 64, width: int = 48) -> np.ndarray:
    """A (T, C, H, W) stack shaped the way mlx_vlm.utils.load_video returns."""
    return np.zeros((count, 3, height, width), dtype=np.uint8)


# --------------------------------------------------------------------------
# Decode failures must raise, not return empty
# --------------------------------------------------------------------------


def test_zero_frames_raises_and_names_the_layer():
    """An empty decode is the failure that reads as a model failure.

    The message has to say which layer failed, because the symptom a user sees
    is a fluent "I don't see any video" and the instinct is to blame the model.
    """
    with pytest.raises(ValueError, match="decode failure, not a model failure"):
        assert_video_decoded([], "clip.mov")


def test_implausible_frame_size_raises():
    tiny = _frames(4, height=MIN_FRAME_DIMENSION - 1, width=MIN_FRAME_DIMENSION - 1)
    with pytest.raises(ValueError, match="Implausible frame size"):
        assert_video_decoded(tiny, "clip.mov", height_axis=1, width_axis=2)


def test_malformed_frame_shape_raises():
    with pytest.raises(ValueError, match="unexpected shape"):
        assert_video_decoded([np.zeros(())], "clip.mov")


def test_valid_decode_reports_display_orientation():
    """Returned dims are used to spot a transposed read, so check the axes."""
    count, width, height = assert_video_decoded(
        _frames(16, height=2604, width=2160), "clip.mov", height_axis=1, width_axis=2
    )
    assert (count, width, height) == (16, 2160, 2604)


# --------------------------------------------------------------------------
# Container suffix: the MIME subtype is not the container name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mime,suffix",
    [
        ("video/quicktime", ".mov"),
        ("video/mp4", ".mp4"),
        ("video/x-matroska", ".mkv"),
        ("video/x-msvideo", ".avi"),
    ],
)
def test_base64_video_gets_a_real_container_suffix(mime, suffix):
    """Splitting the MIME type yields ".quicktime"; decoders infer from this."""
    assert VIDEO_MIME_TO_SUFFIX[mime] == suffix
    path = decode_base64_video(
        f"data:{mime};base64," + base64.b64encode(b"\0" * 16).decode()
    )
    assert path.endswith(suffix)


def test_unknown_video_mime_falls_back_to_mp4():
    payload = base64.b64encode(b"\0" * 16).decode()
    path = decode_base64_video(f"data:video/x-unheard-of;base64,{payload}")
    assert path.endswith(".mp4")


# --------------------------------------------------------------------------
# Multimodal model detection
# --------------------------------------------------------------------------


def test_local_config_beats_the_name(tmp_path):
    """A model whose name carries no VLM marker is still multimodal."""
    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3_5ForConditionalGeneration"],'
        ' "vision_config": {}, "video_token_id": 248057}'
    )
    assert is_mllm_model(str(tmp_path)) is True


def test_text_only_config_stays_text_only(tmp_path):
    (tmp_path / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    assert is_mllm_model(str(tmp_path)) is False


def test_name_patterns_remain_the_fallback():
    """With no config reachable, the legacy substring match still applies."""
    assert is_mllm_model("some-org/Qwen3-VL-4B-Instruct-3bit") is True
    assert is_mllm_model("some-org/definitely-a-text-model") is False


@pytest.fixture
def no_network(monkeypatch):
    """Any socket creation fails the test: these paths must stay offline."""
    import socket

    def _refuse(*args, **kwargs):
        raise AssertionError("network access attempted in a no-network test")

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """Redirect huggingface_hub's cache lookup to an isolated directory,
    keeping the real cache-layout parsing in the loop."""
    import functools

    import huggingface_hub

    real = huggingface_hub.try_to_load_from_cache
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        functools.partial(real, cache_dir=str(tmp_path)),
    )
    return tmp_path


def _write_hub_cached_config(cache_root, repo_id: str, config: str) -> None:
    """Lay out config.json the way huggingface_hub stores a cached repo."""
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    rev = "0" * 40
    (repo_dir / "snapshots" / rev).mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(rev)
    (repo_dir / "snapshots" / rev / "config.json").write_text(config)


def test_cached_neutral_id_is_detected_from_hub_config(no_network, hub_cache):
    """A neutral repo ID (no VL marker) already in the hub cache must be
    classified by its own config.json - the case that silently dropped video
    for Qwen3.8 - and without any network traffic."""
    _write_hub_cached_config(
        hub_cache,
        "mlx-community/Qwen-Neutral-Name-4bit",
        '{"architectures": ["Qwen3_5ForConditionalGeneration"],'
        ' "vision_config": {}, "video_token_id": 248057}',
    )
    assert is_mllm_model("mlx-community/Qwen-Neutral-Name-4bit") is True


def test_cached_text_only_id_stays_text_only(no_network, hub_cache):
    """The cached config governs in both directions: a text-only config keeps
    a model text-only even if a broad name pattern would have matched."""
    _write_hub_cached_config(
        hub_cache,
        "mlx-community/SomePixtral-Like-Name",
        '{"architectures": ["Qwen3ForCausalLM"]}',
    )
    assert is_mllm_model("mlx-community/SomePixtral-Like-Name") is False


def test_cold_neutral_id_falls_back_to_name_patterns_offline(no_network, hub_cache):
    """A neutral ID with nothing cached is decided by name patterns alone,
    before any download. This pins the current pre-download behaviour: the
    CLI's MLLM-versus-LLM routing decision for a cold neutral ID is made from
    the name, with no network attempted, and misroutes a neutrally-named VLM
    until its config is cached. Resolving hub metadata before that decision
    is a known follow-up; this test exists so the limitation is deliberate.
    """
    assert is_mllm_model("mlx-community/Totally-Uncached-Neutral-Model") is False
    assert is_mllm_model("mlx-community/Uncached-Qwen3-VL-Named-Model") is True


# --------------------------------------------------------------------------
# The pinned mlx-vlm contract
# --------------------------------------------------------------------------


@pytest.fixture
def fake_mlx_vlm(monkeypatch):
    """Stand in for mlx_vlm, recording how the native path calls it.

    The dependency that broke was a module rename: mlx_vlm.video_generate
    disappeared in 0.6 and every video request became a 500. Pinning the call
    shape here means a rename fails a test instead of production.
    """
    calls: dict[str, list] = {"load_video": []}

    def load_video(path, fps=None, max_frames=None, min_frames=None, frame_factor=None):
        calls["load_video"].append(
            {
                "path": path,
                "fps": fps,
                "max_frames": max_frames,
                "min_frames": min_frames,
                "frame_factor": frame_factor,
            }
        )
        # Distinct sampled rate per clip, so callers that collapse them show it.
        return _frames(8), 1.5 + len(calls["load_video"])

    utils = types.ModuleType("mlx_vlm.utils")
    utils.load_video = load_video
    package = types.ModuleType("mlx_vlm")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return calls


def _prepare(monkeypatch, native_messages, **kwargs):
    """Call _prepare_native_video_inputs against a stub model object."""
    from vllm_mlx.models.mllm import MLXMultimodalLM

    class Stub:
        _video_native = True
        _video_native_with_audio = False

        class processor:
            @staticmethod
            def apply_chat_template(messages, **kw):
                return "PROMPT"

        @staticmethod
        def _translate_messages_for_native_video(messages, fps, max_frames):
            return native_messages

    return MLXMultimodalLM._prepare_native_video_inputs(
        Stub(), [{"role": "user", "content": []}], **kwargs
    )


def test_native_path_calls_load_video_with_our_caps(fake_mlx_vlm, monkeypatch):
    """video_max_frames must reach the decoder: upstream's default cap is 768."""
    messages = [{"role": "user", "content": [{"type": "video", "video": "/tmp/a.mov"}]}]
    text, gen_kwargs = _prepare(
        monkeypatch, messages, video_fps=2.0, video_max_frames=MAX_FRAMES
    )

    assert text == "PROMPT"
    call = fake_mlx_vlm["load_video"][0]
    assert call["path"] == "/tmp/a.mov"
    assert call["fps"] == 2.0
    assert call["max_frames"] == MAX_FRAMES
    assert len(gen_kwargs["video"]) == 1


def test_mixed_image_video_audio_are_routed_separately(fake_mlx_vlm, monkeypatch):
    """Images stay paths, videos become arrays, audio stays paths."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/pic.png"},
                {"type": "video", "video": "/tmp/a.mov"},
                {"type": "audio", "audio": "/tmp/a.wav"},
            ],
        }
    ]
    _, gen_kwargs = _prepare(monkeypatch, messages)

    assert gen_kwargs["image"] == ["/tmp/pic.png"]
    assert gen_kwargs["audio"] == ["/tmp/a.wav"]
    assert len(gen_kwargs["video"]) == 1
    assert isinstance(gen_kwargs["video"][0], np.ndarray)


def test_multiple_videos_fail_closed(fake_mlx_vlm, monkeypatch):
    """More than one video per native request must be rejected, not mislabeled.

    mlx_vlm.generate takes a single scalar fps and fans it out to every video,
    so a request carrying two clips would label the second one's frames with
    the first one's sampled rate - and that value drives Qwen's interleaved
    timestamp tokens. Correct per-video timestamps mean calling the processor
    directly instead of going through mlx_vlm.generate; until that exists the
    path fails closed with the reason, before any decoding work is done.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/tmp/a.mov"},
                {"type": "video", "video": "/tmp/b.mov"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="one video per request"):
        _prepare(monkeypatch, messages)

    assert fake_mlx_vlm["load_video"] == [], "rejected before decoding starts"


def test_single_video_is_unaffected_by_the_multi_video_guard(fake_mlx_vlm, monkeypatch):
    messages = [{"role": "user", "content": [{"type": "video", "video": "/tmp/a.mov"}]}]
    _, gen_kwargs = _prepare(monkeypatch, messages)
    assert len(gen_kwargs["video"]) == 1


def test_empty_decode_stops_the_request(fake_mlx_vlm, monkeypatch):
    """A decoder returning nothing must raise before the model is called."""
    import vllm_mlx.models.mllm as mllm

    monkeypatch.setitem(
        sys.modules["mlx_vlm.utils"].__dict__,
        "load_video",
        lambda *a, **k: (_frames(0), 2.0),
    )
    messages = [{"role": "user", "content": [{"type": "video", "video": "/tmp/a.mov"}]}]
    with pytest.raises(ValueError, match="Decoded 0 frames"):
        _prepare(monkeypatch, messages)

    assert mllm.MAX_FRAMES  # module still importable after the failure


def test_missing_upstream_module_is_reported_not_swallowed(monkeypatch):
    """The 0.6 rename produced ModuleNotFoundError; it must stay an ImportError."""
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", None)
    messages = [{"role": "user", "content": [{"type": "video", "video": "/tmp/a.mov"}]}]
    with pytest.raises((ImportError, AttributeError, TypeError)):
        _prepare(monkeypatch, messages)


# --------------------------------------------------------------------------
# The real dependency, where it is installed
# --------------------------------------------------------------------------


def test_real_mlx_vlm_still_exposes_the_pinned_api():
    """Pins the two entry points this path depends on, on Apple Silicon.

    mlx_vlm.video_generate vanished in 0.6 and took the whole video path with
    it. If load_video or generate(video=...) moves again, this fails here
    rather than in a user's request.
    """
    # Guard on mlx, not mlx_vlm: mlx_vlm can be installed and still fail to
    # import when MLX is absent, which importorskip would surface as an error
    # rather than a skip.
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_vlm")
    import inspect

    from mlx_vlm.generate import generate
    from mlx_vlm.utils import load_video

    assert "video" in inspect.signature(generate).parameters
    load_params = inspect.signature(load_video).parameters
    for expected in ("fps", "max_frames", "min_frames", "frame_factor"):
        assert expected in load_params, f"load_video lost {expected}"
