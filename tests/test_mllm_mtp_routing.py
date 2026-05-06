# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM + MTP per-request routing."""


def test_has_media_content_text_only():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    assert _has_media_content([{"role": "user", "content": "Hello"}]) is False


def test_has_media_content_with_image():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_with_local_image_part():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/frame.png"},
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_with_video():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "file:///tmp/v.mp4"}}
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_with_local_video_part():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/tmp/v.mp4"},
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_empty():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    assert _has_media_content([]) is False


def test_has_media_content_string_content():
    """String content (not list) should return False."""
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    assert _has_media_content([{"role": "user", "content": "Just text"}]) is False


def test_has_media_content_audio():
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
            ],
        }
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_multi_turn():
    """Media in earlier turns should still be detected."""
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        },
        {"role": "assistant", "content": "I see an image."},
        {"role": "user", "content": "Tell me more about it."},
    ]
    assert _has_media_content(messages) is True


def test_has_media_content_text_list():
    """List content with only text parts should return False."""
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
    ]
    assert _has_media_content(messages) is False


def test_batched_prepare_mllm_messages_converts_audio_url():
    from vllm_mlx.engine.batched import BatchedEngine

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/wav;base64,..."},
                },
                {"type": "text", "text": "Transcribe this"},
            ],
        }
    ]

    prepared = BatchedEngine._prepare_mllm_messages(messages)
    assert prepared[0]["content"][0] == {"type": "audio"}
    assert prepared[0]["content"][1]["type"] == "text"


def test_has_media_content_with_message_models():
    """Pydantic message models should follow the attribute-based content path."""
    from vllm_mlx.api.models import ContentPart, Message, VideoUrl
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [
        Message(
            role="user",
            content=[
                ContentPart(type="text", text="Describe this clip"),
                ContentPart(
                    type="video_url",
                    video_url=VideoUrl(url="https://example.com/video.mp4"),
                ),
            ],
        )
    ]

    assert _has_media_content(messages) is True


def test_has_media_content_none_content():
    """A schema-valid message with None content should not count as media."""
    from vllm_mlx.api.models import Message
    from vllm_mlx.api.utils import has_media_content as _has_media_content

    messages = [Message(role="system", content=None)]

    assert _has_media_content(messages) is False


# --- MLXMultimodalLM extraction method tests ---

from unittest.mock import MagicMock


def test_mllm_mtp_attempt_metadata_only_marks_real_attempts():
    from vllm_mlx.mllm_batch_generator import (
        MLLMBatchResponse,
        _mark_mtp_attempts_on_primary_responses,
    )

    attempted = {7: 1}
    attempted_response = MLLMBatchResponse(
        uid=7,
        request_id="req-7",
        token=11,
        logprobs=None,
    )
    skipped_response = MLLMBatchResponse(
        uid=9,
        request_id="req-9",
        token=21,
        logprobs=None,
    )

    _mark_mtp_attempts_on_primary_responses(
        [attempted_response, skipped_response],
        attempted,
    )

    assert attempted_response.mtp_attempted is True
    assert attempted_response.mtp_attempted_count == 1
    assert skipped_response.mtp_attempted is False
    assert skipped_response.mtp_attempted_count == 0


def test_mllm_scheduler_exposes_mtp_attempts_and_accepts_on_outputs():
    from vllm_mlx.mllm_batch_generator import MLLMBatchResponse
    from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler

    class _Tokenizer:
        clean_up_tokenization_spaces = False

        def decode(self, tokens):
            return "".join(str(token) for token in tokens)

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.uid_to_request_id = {7: "req-7"}
    scheduler.running = {"req-7": MLLMRequest(request_id="req-7", prompt="prompt")}
    scheduler.processor = _Tokenizer()
    scheduler._detokenizer_pool = {}
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0

    outputs, finished = scheduler._process_batch_responses(
        [
            MLLMBatchResponse(
                uid=7,
                request_id="req-7",
                token=11,
                logprobs=None,
                mtp_attempted=True,
                mtp_attempted_count=1,
            ),
            MLLMBatchResponse(
                uid=7,
                request_id="req-7",
                token=12,
                logprobs=None,
                finish_reason="stop",
                from_draft=True,
            ),
        ]
    )

    assert finished == {"req-7"}
    assert outputs[-1].mtp_drafts == 1
    assert outputs[-1].mtp_accepted == 1


def test_get_language_model():
    from vllm_mlx.models.mllm import MLXMultimodalLM

    mllm = MagicMock(spec=MLXMultimodalLM)
    inner_lm = MagicMock()
    mllm.model = MagicMock()
    mllm.model.language_model = inner_lm
    assert MLXMultimodalLM.get_language_model(mllm) is inner_lm


def test_get_tokenizer():
    from vllm_mlx.models.mllm import MLXMultimodalLM

    mllm = MagicMock(spec=MLXMultimodalLM)
    inner_tok = MagicMock()
    mllm.processor = MagicMock()
    mllm.processor.tokenizer = inner_tok
    assert MLXMultimodalLM.get_tokenizer(mllm) is inner_tok
