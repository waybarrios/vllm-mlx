# SPDX-License-Identifier: Apache-2.0
"""
Gradio Text-Only Chatbot Interface for vllm-mlx.

A fast, text-only chat interface for LLM models.
Use this for text conversations without image/video overhead.

Usage:
    # First start the server with a model:
    # Without a custom API model name (model path is the name used in the OpenAI API):
    vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
    # With a custom API model name ("default" is the name used in the OpenAI API):
    vllm-mlx serve --served-model-name default mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

    # Then run this app:
    vllm-mlx-text-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit

    # Or with vllm-mlx started with served model name is 'default', there is no need to use --served-model-name:
    vllm-mlx-text-chat --server-url http://localhost:8000 --port 7861

Note:
    Query the /v1/models endpoint with `curl` and `jq` to see available models and their names:
    ```bash
    curl http://localhost:8000/v1/models | jq ".data[0].id"
    ```
"""

import argparse

import gradio as gr
import requests


def create_chat_function(
    server_url: str,
    max_tokens: int,
    temperature: float,
    served_model_name: str = "default",
):
    """
    Create the chat function for Gradio ChatInterface.

    Args:
        server_url: URL of the vllm-mlx server
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        served_model_name: Model name to send in OpenAI-compatible requests

    Returns:
        Chat function compatible with gr.ChatInterface
    """

    def chat(message: str, history: list) -> str:
        """
        Process a text message and return response.

        Args:
            message: User's text message
            history: List of previous messages

        Returns:
            Assistant response text
        """
        # Build messages list for API
        messages = []

        # Add history
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from multimodal content
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                messages.append({"role": role, "content": content})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Send request to server
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": served_model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to server. Make sure vllm-mlx is running."
        except requests.exceptions.Timeout:
            return "Error: Timeout - server took too long to respond."
        except Exception as e:
            return f"Error: {str(e)}"

    return chat


def main():
    """Run the Gradio app."""
    parser = argparse.ArgumentParser(
        description="Gradio Text-Only Chat Interface for vllm-mlx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings
    vllm-mlx-text-chat --served-model-name <served-model name>

    # name OpenAI returns from /v1/models can be set with --served-model-name, for example:
    vllm-mlx-text-chat --served-model-name <served-model name>
    # example starting chat on the first model served on localhost:8000:
    vllm-mlx-text-chat --served-model-name $(curl http://localhost:8000/v1/models | jq ".data[0].id)

    # Connect to a different server
    vllm-mlx-text-chat --server-url http://localhost:9000

    # Create a public share link
    vllm-mlx-text-chat --share

Note: Make sure the vllm-mlx server is running:
    vllm-mlx serve --served-model-name default mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
        """,
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the vllm-mlx server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port for Gradio interface (default: 7861)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="default",
        help=(
            "Model name to send in /v1/chat/completions requests " "(default: default)"
        ),
    )
    args = parser.parse_args()

    print(f"Connecting to vllm-mlx server at: {args.server_url}")
    print(f"Starting Gradio text chat on port: {args.port}")

    # Create chat function
    chat_fn = create_chat_function(
        server_url=args.server_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        served_model_name=args.served_model_name,
    )

    # Create simple text ChatInterface
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="vllm-mlx Text Chat",
        description="Fast text-only chat with LLM models on Apple Silicon.",
        examples=[
            "Hello, who are you?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about programming.",
        ],
    )

    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
