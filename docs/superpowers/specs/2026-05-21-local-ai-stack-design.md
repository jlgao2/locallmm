# Local AI Stack — MLX + Hermes Agent + Cloud Verify-Correct

**Status:** Approved — 2026-05-21
**Machine:** Apple M5 Pro · 18 CPU / 20 GPU cores · 48 GB unified memory · macOS 26.4 · arm64 · 1.4 TB free

## 1. Goal

A cutting-edge, terminal-driven local AI setup that combines three things:

1. A fast **local model** for everyday chat and reasoning — private, free, offline-capable.
2. An **agent harness** (Hermes Agent) for tool-using, multi-step work.
3. A **cloud "verify & correct" layer** that improves the local model's answers, using a
   **swappable, cost-tiered** choice of cloud provider.

The local model always works on its own; the cloud layer is purely additive.

## 2. Architecture

```
 terminal
   │
   ├── llm ───────────────► verify-correct proxy ──┬─► mlx_lm.server :8000   (local draft)
   │                            (:8100)           └─► cloud verifier API    (verify + correct)
   │
   └── hermes chat ───────► mlx_lm.server :8000   (raw — fast agent loops)
                                │
                                └─ Qwen3.6-35B-A3B · 4-bit MLX  (loaded once, shared)

       hermes  ──fallback / `/model`──► cloud model
```

Three layers: **engine** (MLX) → **interfaces** (`llm`, `hermes chat`) → **cloud quality layer**
(verify-correct proxy).

## 3. Components

### 3.1 Engine — MLX

- Python virtual environment at `locallmm/.venv` with `mlx-lm` installed.
- `mlx_lm.server` serves an OpenAI-compatible API at `http://127.0.0.1:8000/v1`. Port 8000 is the
  address Hermes expects for an MLX backend.
- **Local model:** `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` — Qwen3.6-35B-A3B, a Mixture-of-Experts
  model (35 B total / ~3 B active per token), 4-bit DWQ MLX quant, ≈ 20 GB. Current recommended
  default for the 32–48 GB Apple-Silicon tier: large-model quality at small-model speed, with a
  built-in reasoning mode. The exact repo is confirmed at install time, with documented fallbacks.
- Served with ≥ 64 K context (Hermes's minimum). The model loads once into unified memory and is
  shared by every layer above; 4-bit keeps the footprint ≈ 20 GB and leaves ample headroom on
  48 GB for context and the other processes.

**Swapping the local model.** The model is a single config value (`MLX_MODEL`). Any MLX-format
model on Hugging Face drops in with one edit plus a download — the rest of the stack is
model-agnostic. Documented tested alternate: **`mlx-community/gemma-4-26b-a4b-it-4bit`**
(Gemma 4 26B-A4B, ≈ 15.6 GB) — smaller and lighter, though Qwen3.6 benchmarks ahead on knowledge,
coding, and agentic/tool-use, which is why it is the default.

### 3.2 Cloud quality layer — verify-correct proxy *(the only custom code)*

- `proxy/verify_proxy.py` — a small local service at `http://127.0.0.1:8100/v1`, itself
  OpenAI-compatible.
- **Request lifecycle:**
  1. Receive an OpenAI-format chat-completions request.
  2. Get a fast **draft** answer from the local MLX server (`:8000`).
  3. Send *{user question + local draft}* to the configured **cloud verifier** with a
     verify-correct instruction: return the best final answer — the draft unchanged if it is
     already correct, a corrected version if not.
  4. Stream the verifier's result back to the caller, tagged `✓ verified` or `✎ corrected`.
- **Swappable, cost-tiered verifier** — `proxy/verifier.py`, two adapters:
  - **Anthropic adapter** — native `anthropic` SDK, with prompt caching on the (fixed) verify
    instructions to cut cost. Models: Claude Opus / Sonnet / Haiku.
  - **OpenAI-compatible adapter** — `openai` SDK with a configurable `base_url`. Covers OpenAI
    (GPT), DeepSeek, OpenRouter, and any other OpenAI-compatible endpoint.
  - The provider is chosen by config — no code change to switch.
- **Cost tiers** (cheapest → priciest):

  | Tier | Example | Notes |
  |---|---|---|
  | Lowest cost | DeepSeek `deepseek-chat` | ~10–30× cheaper than Opus; strong |
  | Low | Claude Haiku / GPT mini-tier | very cheap, fast |
  | **Balanced — default** | **Claude Sonnet** (`claude-sonnet-4-6`) | best cost/quality for verification |
  | Max quality | Claude Opus (`claude-opus-4-7`) | frontier quality, highest cost |

- **Default verifier:** Claude Sonnet. Switching provider/model/tier is one edit in `.env`.
- **Graceful degradation:** if no API key is set for the selected provider, or the cloud call
  fails (network, auth, rate-limit, timeout), the proxy returns the **local draft** tagged
  `⚠ unverified (<reason>)`. The local model is never blocked by the cloud layer.

### 3.3 Interfaces

**`hermes chat` — the agent harness**

- Hermes Agent installed via its official installer; configuration at `~/.hermes/config.yaml`.
- Primary model = the **raw local MLX endpoint** (`:8000`) so multi-step agent loops stay fast
  and free.
- A cloud model is configured as a **fallback provider** and is reachable mid-session via
  Hermes's `/model` command. Hermes natively supports cheap providers (DeepSeek, OpenAI,
  OpenRouter) and fallback chains, so the cost-tiered-cloud requirement is satisfied natively
  on this side.
- `HERMES_STREAM_READ_TIMEOUT=1800` is set to tolerate slow local prefill on long contexts.

**`llm` — quick verified chat**

- `bin/llm` — a thin OpenAI-compatible chat client (shell function / small script) pointed at
  the **verify-correct proxy** (`:8100`).
- This is the daily "general chat & reasoning" driver; every answer is cloud-verified at the
  configured cost tier.

### 3.4 Operations

- `bin/locallmm` — a control script with `start` / `stop` / `status` that runs the MLX server
  and the verify-correct proxy as background processes. Optional macOS LaunchAgents for
  auto-start are documented but off by default.
- Secrets and configuration live in `locallmm/.env` (`chmod 600`, git-ignored). `.env.example`
  is committed as a template.
- `README.md` documents every command and every config key.
- The project is a git repository; the spec and all code are committed.

## 4. Configuration — `locallmm/.env`

```
# --- Local engine ---
MLX_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit-DWQ
MLX_PORT=8000
PROXY_PORT=8100

# --- Cloud verifier ---
VERIFIER_PROVIDER=anthropic        # anthropic | openai | deepseek | openai-compatible
VERIFIER_MODEL=claude-sonnet-4-6
VERIFIER_BASE_URL=                 # required only for openai-compatible

# --- API keys (set the one(s) you use) ---
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
```

## 5. Project layout

```
locallmm/
├── .venv/                       # Python env — gitignored, created by setup
├── .env                         # keys + config — gitignored, chmod 600
├── .env.example                 # committed template
├── .gitignore
├── README.md
├── bin/
│   ├── locallmm                 # start / stop / status for both servers
│   └── llm                      # quick verified-chat client → proxy
├── proxy/
│   ├── verify_proxy.py          # OpenAI-compatible verify-correct proxy
│   └── verifier.py              # multi-provider cloud-verifier adapter
└── docs/superpowers/specs/
    └── 2026-05-21-local-ai-stack-design.md
```

`~/.hermes/config.yaml` is managed by Hermes Agent, outside the project tree.

## 6. Error handling

- **No verifier API key** → proxy runs in passthrough mode (local draft returned unchanged);
  a one-time notice is logged. Local stack remains fully functional.
- **Cloud verifier call fails** (network / auth / rate-limit / timeout) → proxy returns the
  local draft tagged `⚠ unverified (<reason>)`. A user request is never hard-failed by a cloud
  problem.
- **MLX server unreachable** → proxy returns a clear error (no draft is possible);
  `bin/locallmm status` reports which server is down.
- **Model out-of-memory at load** → README troubleshooting: switch to a smaller quant or the
  Gemma 4 alternate, or reduce served context.
- **Hermes timeout on long-context prefill** → mitigated by `HERMES_STREAM_READ_TIMEOUT=1800`.

## 7. Testing & verification

- **Model smoke test** — load the model and generate; confirm output and record tokens/sec
  (done during model download).
- **Proxy:** (a) valid key → response carries a `verified`/`corrected` tag; (b) no key →
  passthrough returns the local draft; (c) invalid key → graceful degradation to draft with a
  warning tag.
- **Multi-provider:** exercise the Anthropic adapter and at least one OpenAI-compatible adapter
  (DeepSeek) to prove both code paths.
- **Hermes:** `hermes chat` connects to the local `:8000` endpoint and completes a simple
  multi-step task.
- **`llm`:** end-to-end verified chat works, including streaming.
- **`locallmm`:** `start` / `status` / `stop` behave correctly.

## 8. Out of scope (YAGNI)

- No GUI or web UI — terminal only.
- No multimodal / vision use, no RAG.
- No auto-start daemons by default (opt-in LaunchAgents only).
- Verifier selection is config-driven; no mid-chat verifier switching or per-request provider
  routing.

## 9. Resolved decisions

- **Local model:** Qwen3.6-35B-A3B, 4-bit MLX — live-confirmed as the 48 GB-tier default. Meta
  (Muse Spark is closed-weight; Llama 4 too large) and Kimi (1 T params, datacenter-only) ruled
  out by the 48 GB memory limit; Gemma 4 26B-A4B kept as a documented swappable alternate.
- **Engine:** MLX — fastest inference on Apple Silicon.
- **Hybrid pattern:** verify & correct.
- **Cloud verifier:** swappable across cost tiers; default Claude Sonnet.
- **Hermes endpoint:** raw `:8000` for fast agent loops; cloud model as fallback / `/model`.
