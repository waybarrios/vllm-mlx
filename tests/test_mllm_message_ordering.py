from vllm_mlx.models.mllm import _build_mllm_chat_messages


def test_mllm_chat_messages_preserve_text_only_shape():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Summarize the image."},
        {"role": "assistant", "content": "Ready."},
    ]

    chat_messages = _build_mllm_chat_messages(
        messages,
        all_image_urls=[],
        video_frame_counts={},
    )

    assert chat_messages == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are concise.",
                    "content": "You are concise.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize the image.",
                    "content": "Summarize the image.",
                }
            ],
        },
        {"role": "assistant", "content": "Ready."},
    ]


def test_mllm_chat_messages_preserve_image_text_order():
    all_image_urls = []
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare the left panel to "},
                {"type": "image_url", "image_url": {"url": "/tmp/chart.png"}},
                {"type": "text", "text": " and summarize the difference"},
            ],
        }
    ]

    chat_messages = _build_mllm_chat_messages(
        messages,
        all_image_urls=all_image_urls,
        video_frame_counts={},
    )

    assert chat_messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "compare the left panel to ",
                    "content": "compare the left panel to ",
                },
                {"type": "image"},
                {
                    "type": "text",
                    "text": " and summarize the difference",
                    "content": " and summarize the difference",
                },
            ],
        }
    ]
    assert all_image_urls == ["/tmp/chart.png"]


def test_mllm_chat_messages_preserve_audio_text_order():
    all_image_urls = []
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe "},
                {"type": "audio_url", "audio_url": {"url": "/tmp/audio.wav"}},
                {"type": "text", "text": " then identify the speaker"},
            ],
        }
    ]

    chat_messages = _build_mllm_chat_messages(
        messages,
        all_image_urls=all_image_urls,
        video_frame_counts={},
    )

    assert chat_messages[0]["content"] == [
        {"type": "text", "text": "transcribe ", "content": "transcribe "},
        {"type": "audio"},
        {
            "type": "text",
            "text": " then identify the speaker",
            "content": " then identify the speaker",
        },
    ]
    assert all_image_urls == []
