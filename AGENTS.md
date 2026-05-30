# AGENTS.md — locallmm

Local AI stack for Apple Silicon. MLX engine + verify-correct proxy + Hermes
harness. See `README.md` for user-facing docs.

## Layout

- `bin/locallmm` — start/stop/status for both servers
- `bin/mlx-supervisor` — auto-restarts mlx_lm.server on crash
- `bin/llm` — quick verified chat CLI (stdlib urllib)
- `proxy/verify_proxy.py` — FastAPI OpenAI-compatible proxy on :8100
- `proxy/verifier.py` — multi-provider cloud verifier (anthropic | openai | deepseek | openai-compatible)
- `macos/com.locallmm.gpulimit.plist` — LaunchDaemon to raise GPU memory cap
- `.env` — secrets and config (NEVER commit)

## Commands

- `bin/locallmm {start|stop|restart|status}`
- `cd proxy && pytest` — run proxy tests (no live LLM needed; uses mocks)
- `bin/llm "question"` — one-shot verified answer

## Conventions

- **Surgical changes only.** Touch what the task requires, nothing else.
- **No speculative abstractions** — direct code beats clever code here.
- **Secrets** live in `.env` (gitignored, chmod 600). Never `.env.example`.
- **MLX model swaps** go through `MLX_MODEL` in `.env`, never hard-coded.
- **Proxy must degrade gracefully** if the verifier is unreachable — return
  the local draft tagged `⚠ unverified`. Tests cover this.

## Don't touch without a reason

- `.run/` (PID + log files, runtime state)
- `.venv/` (regenerate from `requirements.txt` instead)
- `macos/com.locallmm.gpulimit.plist` (LaunchDaemon, needs sudo to reinstall)
