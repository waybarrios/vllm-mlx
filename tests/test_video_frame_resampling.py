# SPDX-License-Identifier: Apache-2.0
"""Pre-sampled video frames must reach the model, not a re-sampled subset.

The server samples frames itself (at the request's video_fps) and hands the
array to the HF video processor. Qwen-VL video processors default to
do_sample_frames=True and sample again; without video metadata they assume a
24 fps source and clamp any short clip to 4 frames, i.e. 2 temporal positions
after the temporal patch fuses frames in pairs. The model then describes the
clip as a still image.

The helper is tested against stand-ins everywhere, and against the real
Qwen3-VL video processor where transformers and torchvision are installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_mlx.models.mllm import disable_video_frame_resampling


def test_disables_resampling_when_on():
    processor = SimpleNamespace(video_processor=SimpleNamespace(do_sample_frames=True))
    assert disable_video_frame_resampling(processor) is True
    assert processor.video_processor.do_sample_frames is False


def test_leaves_processor_alone_when_already_off():
    processor = SimpleNamespace(video_processor=SimpleNamespace(do_sample_frames=False))
    assert disable_video_frame_resampling(processor) is False
    assert processor.video_processor.do_sample_frames is False


@pytest.mark.parametrize(
    "processor",
    [
        SimpleNamespace(),  # image-only processor
        SimpleNamespace(video_processor=None),
        SimpleNamespace(video_processor=SimpleNamespace()),  # no such option
    ],
)
def test_no_video_processor_or_option_is_a_no_op(processor):
    assert disable_video_frame_resampling(processor) is False
    vp = getattr(processor, "video_processor", None)
    assert not hasattr(vp, "do_sample_frames")


def _temporal_positions(video_processor, n_frames: int) -> int:
    frames = np.zeros((n_frames, 3, 224, 224), dtype=np.uint8)
    out = video_processor(videos=[frames], return_tensors="np")
    return int(out["video_grid_thw"][0][0])


def test_real_qwen3_vl_processor_keeps_every_presampled_frame():
    pytest.importorskip("torchvision")
    qwen3_vl = pytest.importorskip(
        "transformers.models.qwen3_vl.video_processing_qwen3_vl"
    )
    video_processor = qwen3_vl.Qwen3VLVideoProcessor()
    processor = SimpleNamespace(video_processor=video_processor)

    # The bug: 16 pre-sampled frames collapse to 2 temporal positions.
    assert _temporal_positions(video_processor, 16) == 2

    assert disable_video_frame_resampling(processor) is True

    # Pairwise temporal fusion (temporal_patch_size=2) is expected; the
    # frames themselves must all survive.
    assert _temporal_positions(video_processor, 16) == 8
    assert _temporal_positions(video_processor, 32) == 16
