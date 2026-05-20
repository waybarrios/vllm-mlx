# Client Compatibility

OpenAI-compatible desktop apps and agent tools usually need the same backend
contract: a `/v1` base URL, a model id that appears in `/v1/models`, and an API
key value when the client refuses empty credentials.

The checkout script `scripts/serve_client_profile.sh` maps common client names
onto conservative server profiles. The script does not prove that every feature
of a client is supported; it starts the backend with settings that are suitable
for a first connection test.

## Starting A Client Profile

```bash
scripts/serve_client_profile.sh generic-openai \
  mlx-community/Qwen3-4B-Instruct-2507-4bit
```

The script delegates to `scripts/serve_profile.sh`, binds to `127.0.0.1` by
default, and supplies a local API-key label unless `--api-key` is already
present. Override the label with `VLLM_MLX_CLIENT_API_KEY`.

## Profile Map

| Client profile | Backend profile | Default API key | Use case |
|---|---|---|---|
| `generic-openai` | `text-default` | `local-client` | Desktop or web clients that speak OpenAI chat completions |
| `generic-mllm` | `mllm-default` | `local-client` | Multimodal OpenAI-compatible clients |
| `goose-text` | `text-default` | `goose-local` | Goose text and streaming setup |
| `goose-tools` | `text-tools` | `goose-local` | Goose with tool calling enabled |
| `open-webui-text` | `text-default` | `openwebui-local` | Open WebUI text setup |
| `open-webui-mllm` | `mllm-default` | `openwebui-local` | Open WebUI image chat setup |
| `cherry-studio` | `text-default` | `cherrystudio-local` | Cherry Studio custom OpenAI provider |
| `chatbox` | `text-default` | `chatbox-local` | Chatbox built-in OpenAI provider |
| `librechat` | `text-default` | `librechat-local` | LibreChat OpenAI provider |
| `witsy` | `text-default` | `witsy-local` | Witsy OpenAI engine |
| `jan` | `text-default` | `jan-local` | Jan remote OpenAI-compatible engine |
| `anythingllm` | `text-default` | `anythingllm-local` | AnythingLLM generic OpenAI provider |
| `boltai` | `text-default` | `boltai-local` | BoltAI desktop setup |

## Client Settings

Use `http://127.0.0.1:8000/v1` as the base URL when the client asks for an
OpenAI API base. Some clients ask for the host without `/v1` and append the
path themselves; check the client's preview URL before saving.

Set the model to the exact served model id unless you intentionally run without
strict model-id checks and know the client sends `"default"`.

Use the API key printed or implied by the client profile. The default labels are
not secrets; they are compatibility values for local clients that require a
non-empty key field.

## Off-Host Access

Profiles bind to localhost by default. For a client running on another machine
or inside a container that cannot reach host localhost, make the exposure
explicit:

```bash
LISTEN_MODE=public scripts/serve_client_profile.sh generic-openai \
  mlx-community/Qwen3-4B-Instruct-2507-4bit
```

When a Dockerized client runs on the same Mac, the reachable base URL is often
`http://host.docker.internal:8000/v1`.

## Related Guides

- [OpenAI-Compatible Server](server.md)
- [Tool Calling](tool-calling.md)
- [Model And Serve Profile Matrix](model-profile-matrix.md)
