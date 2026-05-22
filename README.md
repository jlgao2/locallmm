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

## Configuration (`.env`)

- `MLX_MODEL` — swap the local model (any MLX model on Hugging Face). Tested
  alternate: `mlx-community/gemma-4-26b-a4b-it-4bit`.
- `VERIFIER_PROVIDER` / `VERIFIER_MODEL` — pick the cloud verifier and cost tier:
  - `anthropic` + `claude-opus-4-7` — max quality
  - `anthropic` + `claude-sonnet-4-6` — balanced (default)
  - `deepseek` + `deepseek-v4-flash` — lowest cost
  - `openai` + a GPT model — alternative
  - `openai-compatible` + `VERIFIER_BASE_URL` — any other OpenAI-compatible API

## Troubleshooting

- `mlx: not responding` → check `.run/mlx.log`; the model may still be loading.
- Out of memory → switch `MLX_MODEL` to the smaller Gemma 4 alternate.
- Answers tagged `⚠ unverified` → no verifier API key set; add one to `.env`.
