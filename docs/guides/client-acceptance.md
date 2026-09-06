# Client acceptance

The opt-in acceptance runner checks installed coding clients against a running
vllm-mlx server. It checks a complete tool interaction and a file edit through
the client's normal API adapter. It requires Python 3.10 or later on macOS or
Linux and uses only the Python standard library.

Run it from a repository checkout:

```bash
python -m scripts.client_acceptance --help
```

## Client matrix

| Client executable | API exercised | Coverage |
| --- | --- | --- |
| `opencode` | Chat Completions | Custom OpenAI-compatible provider |
| `pi` | Chat Completions | Custom provider configured in `models.json` |
| `codex` | Responses | Custom provider with the Responses transport |
| `claude` | Anthropic Messages | Local Anthropic-compatible gateway |
| `copilot` | Chat Completions | BYOK local provider with offline mode |
| `cline` | Chat Completions | CLI with isolated provider configuration |
| `openclaw` | Responses | Embedded `agent exec`, without messaging channels |

These are implemented test targets, not a claim that every client or model has
passed. Each run records its actual results and installed versions. Cline CLI
coverage does not establish editor-extension behavior, and OpenClaw embedded
coverage does not establish gateway or channel behavior.

OpenClaw uses Responses because that transport preserves standard tool-call
IDs through its replay path. Its Chat Completions path rewrites IDs, which
would prevent the runner from verifying exact call/result matches.

Continue and Aider are follow-up candidates. Cursor's native agent is outside
this local test matrix because its documented BYOK requests pass through Cursor
infrastructure. See [Cursor's BYOK documentation](https://cursor.com/help/models-and-usage/api-keys).

## Prepare the server and clients

Install the desired clients separately, using explicit versions. The runner
never installs packages or reuses your client login. It discovers executables
on `PATH`, probes `--version`, and can reject any version that differs from an
explicit `--expect-version CLIENT=VERSION` pin. A client release that lacks the
required flags or provider contract will fail its run.

Start vllm-mlx on Apple Silicon with a pinned local model snapshot and the tool
parser appropriate for that model. For example:

```bash
vllm-mlx serve /absolute/path/to/model-snapshot \
  --served-model-name acceptance-model \
  --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The parser in this example is for a compatible Qwen model. Choose the parser
for your own model using the [tool-calling guide](tool-calling.md).

Run the initial pair:

```bash
mkdir -p tmp/client-acceptance
export TMPDIR="$PWD/tmp/client-acceptance"
python -m scripts.client_acceptance \
  --clients opencode pi \
  --base-url http://127.0.0.1:8000/v1 \
  --model acceptance-model \
  --model-revision MODEL_COMMIT_OR_SHA256 \
  --server-revision SERVER_COMMIT \
  --report tmp/client-acceptance/report.json
```

Replace the revision placeholders with immutable 40-digit commit hashes,
64-digit digests, or `sha256:` followed by 64 digits. These values describe the
server and model you actually launched. They are recorded as operator-provided
provenance; the runner cannot independently attest to a running server's source
or model weights. The model ID must match `/v1/models` exactly.

Select the full matrix with:

```text
--clients opencode pi codex claude copilot cline openclaw
```

Add `--expect-version` once per client for reproducible comparisons and adjust
`--timeout` for the model's speed. The default budget is 180 seconds per client.
For an authenticated local server, set `VLLM_MLX_API_KEY` in the runner's
environment, or select another variable with `--api-key-env`. The upstream key
is held by the proxy. Each run gets a separate proxy credential, passed through
private client configuration files or environment variables, never command-line
arguments. The proxy checks that credential before forwarding requests. When the
upstream server has no key, client configurations receive an inert local key.

## What a passing run establishes

Each client receives a private temporary home, configuration directories, and
workspace. The workspace contains an input file holding a random token and an
output file containing `pending`. The prompt asks the client to read both files,
replace `pending` with the input token while preserving the final newline, and
answer with the token. The token is absent from the prompt.

A loopback proxy observes the client's API requests and streams upstream
responses through unchanged. A passing run requires all of the following:

- A matching result for every model-issued tool call, with at least one result carrying the token.
- At least two inference requests through the expected API, using the selected model.
- A completed streamed model response containing the token.
- Exact output bytes: the token followed by one LF newline, with the input unchanged.
- Successful client exit, no timeout, and no observed protocol errors.

The runner rejects symlink or non-regular fixture files. Reports contain
structured evidence counts and reason codes, without raw prompts, responses,
client transcripts, tokens, or authentication headers.

| Result | Meaning | Process exit |
| --- | --- | --- |
| `passed` | Every acceptance condition was observed | `0` only when all selected clients pass |
| `failed` | Launch/setup, process, protocol, or fixture validation failed | `1` if any client fails |
| `unavailable` | Missing executable, version mismatch/probe failure, or unavailable server/model | `2` when no client fails but some are unavailable |

Unselected clients are untested and omitted from the report. Unavailable
clients are never counted as passes. Reason codes and API evidence distinguish
missing prerequisites from unsuccessful tool execution; transcripts are not
retained, so detailed client debugging requires a separate manual reproduction.

Temporary directories and environment isolation are not an operating-system
sandbox. Clients retain their executable's host permissions, subject to their
configured sandbox and tool restrictions. Run acceptance on a dedicated trusted
host. Client model traffic goes through the local proxy, but this does not by
itself block unrelated client startup traffic. Install any required provider
packages beforehand and verify client-specific offline behavior where needed.

Codex's command tools receive the system executable search path while other
operator environment variables remain excluded. OpenClaw's startup respawning
and Node compile cache are disabled so the runner can terminate its process
group before removing the temporary profile.

## CI and platform coverage

Ordinary Linux CI runs portable runner, configuration, and HTTP/SSE fixture
tests. Those tests verify the harness; they do not establish MLX model or
installed-client acceptance.

The manually dispatched **Client acceptance** workflow runs only from `main`
on a trusted Apple Silicon self-hosted runner with the `client-acceptance`
label. Prepare the installed clients and a local server on port 8000 before
dispatch. Provide the exact server/model revisions and one version pin per
selected client. The workflow has no pull-request trigger, runs clients
sequentially, and uploads the structured report even after a failed run. It
does not install packages, start a model server, or connect messaging accounts.

Current provider contracts:
[OpenCode](https://opencode.ai/docs/providers/#custom-provider),
[pi](https://pi.dev/),
[Codex](https://developers.openai.com/codex/config-advanced),
[Claude Code](https://code.claude.com/docs/en/llm-gateway),
[Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models),
[Cline](https://docs.cline.bot/cli/cli-reference),
[OpenClaw](https://docs.openclaw.ai/providers/vllm).
