# locallmm — local AI stack

A terminal-driven local AI setup for Apple Silicon: a local MLX model, a
verify-correct proxy backed by a swappable cloud model, and the Hermes Agent
harness.

## Components

- **Engine** — `mlx_lm.server` runs Qwen3.6-35B-A3B (4-bit MLX) on `:8000`.
- **Verify-correct proxy** — `:8100`; drafts locally, then a cloud model verifies
  and corrects. Degrades to the local draft if no key / offline.
- **`llm`** — quick verified chat.
- **`hermes chat`** — the agent harness (tools, memory, multi-step).

## Setup

1. `cp .env.example .env` and fill in an API key (optional — works without one).
2. `bin/locallmm start` — starts both servers (first model load ~20-40s).
3. Add `bin/` to PATH — append to `~/.zshrc`:
   `export PATH="$HOME/Projects/locallmm/bin:$PATH"`

## Commands

| Command | Purpose |
|---|---|
| `locallmm start` / `stop` / `restart` / `status` | manage the servers |
| `llm` | interactive verified chat |
| `llm "a question"` | one-shot verified answer |
| `hermes chat` | agentic assistant |

## Using the local model in Hermes Agent

Hermes Agent's `lmstudio` provider is wired to the local MLX server via
`LM_BASE_URL` in `~/.hermes/.env`. Your existing Hermes default model is
untouched — to run the agent on the local model:

```
hermes chat --provider lmstudio --model mlx-community/Qwen3.6-35B-A3B-4bit-DWQ
```

Or switch to it mid-session with Hermes's `/model` command.

## Editor integration

Any editor with an OpenAI-compatible model setting (Continue.dev, Cursor,
Zed) can use the proxy at `http://127.0.0.1:8100/v1`. See
[docs/editor-integration.md](docs/editor-integration.md) for per-editor
config.

## Configuration (`.env`)

- `MLX_MODEL` — swap the local model (any MLX model on Hugging Face).
  Tested alternates:
  - `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` — code-tuned, same
    MoE shape as the default
  - `mlx-community/gemma-4-26b-a4b-it-4bit` — smaller, lower memory
- `VERIFIER_PROVIDER` / `VERIFIER_MODEL` — pick the cloud verifier and cost tier:
  - `anthropic` + `claude-opus-4-7` — max quality
  - `anthropic` + `claude-sonnet-4-6` — balanced (default)
  - `deepseek` + `deepseek-v4-flash` (cheapest) or `deepseek-v4-pro` (flagship reasoner)
  - `openai` + a GPT model — alternative
  - `openai-compatible` + `VERIFIER_BASE_URL` — any other OpenAI-compatible API

## Troubleshooting

- `mlx: not responding` → check `.run/mlx.log`; the model may still be loading.
- Out of memory → switch `MLX_MODEL` to the smaller Gemma 4 alternate.
- Answers tagged `⚠ unverified` → no verifier API key set; add one to `.env`.
- The MLX server **auto-restarts** if it crashes — `bin/mlx-supervisor` wraps it; crash history stays in `.run/mlx.log`.
- Heavy agentic use exhausting GPU memory? Raise the limit: `sudo bash macos/install-gpu-limit.sh`.
