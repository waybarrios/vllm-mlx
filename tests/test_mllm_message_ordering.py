import json

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


def test_mllm_chat_messages_keep_assistant_tool_calls_without_text():
    # Issue #608: the standard OpenAI shape for an assistant tool-call turn is
    # {content: None, tool_calls: [...]} and it must not be dropped.
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Lima"}',
                    },
                }
            ],
            "reasoning_content": "User wants the weather.",
        }
    ]

    chat_messages = _build_mllm_chat_messages(
        messages,
        all_image_urls=[],
        video_frame_counts={},
    )

    assert chat_messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        # JSON argument strings are parsed to mappings for
                        # templates that iterate argument keys.
                        "arguments": {"city": "Lima"},
                    },
                }
            ],
            "reasoning_content": "User wants the weather.",
        }
    ]


def test_mllm_chat_messages_keep_tool_call_id_on_tool_messages():
    messages = [{"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 19}'}]

    chat_messages = _build_mllm_chat_messages(
        messages,
        all_image_urls=[],
        video_frame_counts={},
    )

    assert chat_messages == [
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": '{"temp_c": 19}',
                    "content": '{"temp_c": 19}',
                }
            ],
            "tool_call_id": "call_1",
        }
    ]


def test_mllm_chat_messages_keep_empty_tool_results():
    """Tools may return empty output; the anchored message must survive."""
    for empty_content in ("", None):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": empty_content}
        ]

        chat_messages = _build_mllm_chat_messages(
            messages,
            all_image_urls=[],
            video_frame_counts={},
        )

        assert chat_messages == [
            {"role": "tool", "content": "", "tool_call_id": "call_1"}
        ]


def test_mllm_chat_messages_keep_tool_exchange():
    fresh = [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": "What is the weather in Lima?"},
    ]
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Lima"}'},
    }
    continuation = fresh + [
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 19}'},
    ]

    fresh_out = _build_mllm_chat_messages(
        fresh, all_image_urls=[], video_frame_counts={}
    )
    cont_out = _build_mllm_chat_messages(
        continuation, all_image_urls=[], video_frame_counts={}
    )

    # The 4-message tool exchange must render strictly more prompt content
    # than the fresh exchange (issue #608: it rendered byte-identical).
    assert len(cont_out) == len(fresh_out) + 2
    assert json.dumps(cont_out, sort_keys=True) != json.dumps(fresh_out, sort_keys=True)
    assert len(json.dumps(cont_out)) > len(json.dumps(fresh_out))

    # Assistant tool_calls survive even with empty text content.
    assistant_msg = cont_out[2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"

    # Tool result keeps its anchor for template forward-scans.
    tool_msg = cont_out[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"


def test_mllm_chat_messages_still_drop_empty_messages_without_tool_calls():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
    ]

    chat_messages = _build_mllm_chat_messages(
        messages,
        all_image_urls=[],
        video_frame_counts={},
    )

    assert [msg["role"] for msg in chat_messages] == ["user"]
    # Kept messages carry only the rebuilt role/content keys.
    assert set(chat_messages[0]) == {"role", "content"}
