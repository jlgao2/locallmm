# Local AI Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a terminal-driven local AI stack on an Apple M5 Pro — a local MLX model, an OpenAI-compatible verify-correct proxy backed by a swappable cloud model, the Hermes Agent harness, and control/chat scripts.

**Architecture:** `mlx_lm.server` serves Qwen3.6-35B-A3B locally on `:8000`. A FastAPI proxy on `:8100` drafts with the local model then has a cloud model verify/correct it, degrading to the raw draft when the cloud is unavailable. `bin/llm` is a thin chat client for the proxy; Hermes Agent points at the raw `:8000` endpoint for fast agent loops. `bin/locallmm` starts/stops both servers.

**Tech Stack:** Python 3 (venv), `mlx-lm`, FastAPI + Uvicorn, `httpx`, `anthropic` SDK, `openai` SDK, Hermes Agent, bash.

**Spec:** `docs/superpowers/specs/2026-05-21-local-ai-stack-design.md`

**Paths:** Project root is `/Users/georgegao/Projects/locallmm` (referred to below as `$ROOT`). The venv is `$ROOT/.venv`.

---

## Task 1: Environment & configuration

**Files:**
- Verify (created by background subagent): `$ROOT/.venv/`, model in `~/.cache/huggingface/hub`
- Create: `$ROOT/requirements.txt`
- Create: `$ROOT/.env.example`
- Create: `$ROOT/.env`
- Modify: `$ROOT/.gitignore`
- Create dirs: `$ROOT/bin/`, `$ROOT/proxy/`

- [ ] **Step 1: Confirm the MLX engine deliverable**

The background subagent creates the venv, installs `mlx-lm` + `huggingface_hub`, and downloads the model. Confirm it finished:

Run: `ls $ROOT/.venv/bin/python && $ROOT/.venv/bin/python -c "import mlx_lm; print('mlx-lm', mlx_lm.__version__)" && ls ~/.cache/huggingface/hub | grep -i qwen3.6`
Expected: a Python path, an `mlx-lm <version>` line, and a `models--mlx-community--Qwen3.6-35B-A3B...` directory.
If the model directory is missing, wait for the subagent to finish before continuing.

- [ ] **Step 2: Install proxy + dev dependencies into the venv**

Run:
```bash
$ROOT/.venv/bin/pip install -U "fastapi" "uvicorn[standard]" "httpx" "anthropic" "openai" "pytest"
```
Expected: ends with `Successfully installed ...` listing fastapi, uvicorn, httpx, anthropic, openai, pytest.

- [ ] **Step 3: Create `requirements.txt`**

Create `$ROOT/requirements.txt`:
```
mlx-lm
huggingface_hub[cli]
fastapi
uvicorn[standard]
httpx
anthropic
openai
pytest
```

- [ ] **Step 4: Create `.env.example`**

Create `$ROOT/.env.example`:
```
# --- Local engine ---
MLX_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit-DWQ
MLX_PORT=8000
PROXY_PORT=8100

# --- Cloud verifier ---
VERIFIER_PROVIDER=anthropic        # anthropic | openai | deepseek | openai-compatible
VERIFIER_MODEL=claude-sonnet-4-6
VERIFIER_BASE_URL=                 # required only for openai-compatible
# VERIFIER_API_KEY=                # only for openai-compatible endpoints

# --- API keys (set the one(s) you use) ---
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
```

- [ ] **Step 5: Create the real `.env` and lock its permissions**

Run:
```bash
cp $ROOT/.env.example $ROOT/.env && chmod 600 $ROOT/.env && ls -l $ROOT/.env
```
Expected: `-rw-------` permissions on `.env`. (The user fills in an API key later; the stack runs without one.)

- [ ] **Step 6: Add `.run/` to `.gitignore`**

Append `.run/` to `$ROOT/.gitignore` so server PID/log files are not tracked. Final `.gitignore`:
```
.venv/
.env
__pycache__/
*.pyc
.DS_Store
*.log
.run/
```

- [ ] **Step 7: Create directories**

Run: `mkdir -p $ROOT/bin $ROOT/proxy && echo ok`
Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git -C $ROOT add requirements.txt .env.example .gitignore
git -C $ROOT commit -m "chore: project config and dependencies"
```

---

## Task 2: Cloud verifier adapter (`proxy/verifier.py`)

**Files:**
- Create: `$ROOT/proxy/verifier.py`
- Test: `$ROOT/proxy/test_verifier.py`

- [ ] **Step 1: Write the failing test**

Create `$ROOT/proxy/test_verifier.py`:
```python
import importlib
import verifier


def test_verifier_available_true_when_key_set(monkeypatch):
    monkeypatch.setenv("VERIFIER_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    importlib.reload(verifier)
    assert verifier.verifier_available() is True


def test_verifier_available_false_when_key_missing(monkeypatch):
    monkeypatch.setenv("VERIFIER_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    importlib.reload(verifier)
    assert verifier.verifier_available() is False


def test_parse_corrected():
    verdict, answer = verifier._parse("VERDICT: corrected\nThe fixed answer.")
    assert verdict == "corrected"
    assert answer == "The fixed answer."


def test_parse_ok():
    verdict, answer = verifier._parse("VERDICT: ok\nThe answer stands.")
    assert verdict == "verified"
    assert answer == "The answer stands."


def test_parse_no_marker_falls_back_to_verified():
    verdict, answer = verifier._parse("Just an answer with no marker.")
    assert verdict == "verified"
    assert answer == "Just an answer with no marker."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $ROOT/proxy && ../.venv/bin/pytest test_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verifier'`.

- [ ] **Step 3: Write the implementation**

Create `$ROOT/proxy/verifier.py`:
```python
"""Multi-provider cloud verifier for the verify-correct proxy.

Takes a user question plus a draft answer from the local model and returns a
(verdict, final_answer) pair produced by a configured cloud model.
"""
import os

VERIFY_SYSTEM = (
    "You are a verification layer. You receive a user's question and a DRAFT "
    "answer written by a small local model. Produce the best possible FINAL "
    "answer for the user.\n\n"
    "Rules:\n"
    "- If the draft is correct, complete, and clear, keep it essentially as-is.\n"
    "- If it has errors, omissions, or weak reasoning, fix them.\n"
    "- Output ONLY the final answer for the user — no meta commentary.\n"
    "- The FIRST line of your response must be exactly 'VERDICT: ok' (draft kept "
    "essentially unchanged) or 'VERDICT: corrected' (you changed it). Put the "
    "answer on the lines after."
)

_KEY_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai-compatible": "VERIFIER_API_KEY",
}


def _provider() -> str:
    return os.environ.get("VERIFIER_PROVIDER", "anthropic").strip().lower()


def _model() -> str:
    return os.environ.get("VERIFIER_MODEL", "claude-sonnet-4-6").strip()


def verifier_available() -> bool:
    """True when the selected provider has its API key set."""
    var = _KEY_VAR.get(_provider())
    return bool(var and os.environ.get(var))


def _user_msg(question: str, draft: str) -> str:
    return (
        f"<user_question>\n{question}\n</user_question>\n\n"
        f"<draft_answer>\n{draft}\n</draft_answer>"
    )


def _parse(raw: str) -> tuple[str, str]:
    """Split a 'VERDICT: ...' marked response into (verdict, answer)."""
    raw = raw.strip()
    head, _, rest = raw.partition("\n")
    head = head.strip().lower()
    if head.startswith("verdict: corrected"):
        return "corrected", rest.strip()
    if head.startswith("verdict: ok"):
        return "verified", rest.strip()
    return "verified", raw  # no marker -> assume the draft stood


def verify(question: str, draft: str) -> tuple[str, str]:
    """Return (verdict, answer); verdict is 'verified' or 'corrected'.

    Raises on transport/auth failure — the proxy catches it and degrades.
    """
    if _provider() == "anthropic":
        raw = _call_anthropic(question, draft)
    else:
        raw = _call_openai_compatible(_provider(), question, draft)
    return _parse(raw)


def _call_anthropic(question: str, draft: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=_model(),
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": VERIFY_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": _user_msg(question, draft)}],
    )
    return resp.content[0].text


def _call_openai_compatible(provider: str, question: str, draft: str) -> str:
    from openai import OpenAI

    kwargs = {"api_key": os.environ[_KEY_VAR[provider]]}
    if provider == "deepseek":
        kwargs["base_url"] = "https://api.deepseek.com"
    elif provider == "openai-compatible":
        base = os.environ.get("VERIFIER_BASE_URL", "").strip()
        if not base:
            raise RuntimeError("VERIFIER_BASE_URL is required for openai-compatible")
        kwargs["base_url"] = base
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": _user_msg(question, draft)},
        ],
    )
    return resp.choices[0].message.content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $ROOT/proxy && ../.venv/bin/pytest test_verifier.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git -C $ROOT add proxy/verifier.py proxy/test_verifier.py
git -C $ROOT commit -m "feat: multi-provider cloud verifier adapter"
```

---

## Task 3: Verify-correct proxy (`proxy/verify_proxy.py`)

**Files:**
- Create: `$ROOT/proxy/verify_proxy.py`
- Test: `$ROOT/proxy/test_proxy.py`

- [ ] **Step 1: Write the failing test**

Create `$ROOT/proxy/test_proxy.py`:
```python
import verify_proxy


def test_last_user_returns_latest_user_message():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert verify_proxy._last_user(messages) == "second"


def test_last_user_empty_when_no_user():
    assert verify_proxy._last_user([{"role": "assistant", "content": "x"}]) == ""


def test_envelope_shape():
    env = verify_proxy._envelope("hello")
    assert env["object"] == "chat.completion"
    assert env["choices"][0]["message"]["content"] == "hello"
    assert env["choices"][0]["finish_reason"] == "stop"


def test_sse_starts_with_role_and_ends_with_done():
    frames = list(verify_proxy._sse("hi there"))
    assert '"role": "assistant"' in frames[0]
    assert frames[-1] == "data: [DONE]\n\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $ROOT/proxy && ../.venv/bin/pytest test_proxy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_proxy'`.

- [ ] **Step 3: Write the implementation**

Create `$ROOT/proxy/verify_proxy.py`:
```python
"""OpenAI-compatible verify-correct proxy.

Drafts an answer with the local MLX model, then has a cloud model verify and
correct it. Falls back to the raw local draft whenever the cloud step is
unavailable or fails, so the local model always works on its own.
"""
import json
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from verifier import verify, verifier_available

MLX_URL = f"http://127.0.0.1:{os.environ.get('MLX_PORT', '8000')}/v1/chat/completions"
MLX_MODEL = os.environ.get("MLX_MODEL", "local")
MODEL_NAME = "locallmm-verify"

app = FastAPI(title="locallmm verify-correct proxy")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "verifier_configured": verifier_available()}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    try:
        draft = await _draft(messages)
    except Exception as exc:  # local engine unreachable / errored
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"local MLX server unreachable: {exc}"}},
        )

    question = _last_user(messages)
    if verifier_available():
        try:
            verdict, answer = verify(question, draft)
            tag = "✓ verified" if verdict == "verified" else "✎ corrected"
        except Exception as exc:
            answer, tag = draft, f"⚠ unverified (cloud error: {type(exc).__name__})"
    else:
        answer, tag = draft, "⚠ unverified (no verifier API key — local only)"

    final = f"{tag}\n\n{answer}"
    if stream:
        return StreamingResponse(_sse(final), media_type="text/event-stream")
    return JSONResponse(_envelope(final))


async def _draft(messages: list) -> str:
    payload = {
        "model": MLX_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 2048,
    }
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(MLX_URL, json=payload)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _last_user(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _envelope(content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


def _sse(content: str):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def frame(delta: dict, finish=None) -> str:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    yield frame({"role": "assistant"})
    step = 24
    for i in range(0, len(content), step):
        yield frame({"content": content[i:i + step]})
    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $ROOT/proxy && ../.venv/bin/pytest test_proxy.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C $ROOT add proxy/verify_proxy.py proxy/test_proxy.py
git -C $ROOT commit -m "feat: OpenAI-compatible verify-correct proxy"
```

---

## Task 4: Control script (`bin/locallmm`)

**Files:**
- Create: `$ROOT/bin/locallmm`

- [ ] **Step 1: Write the script**

Create `$ROOT/bin/locallmm`:
```bash
#!/usr/bin/env bash
# locallmm — start / stop / status for the local AI stack servers.
set -euo pipefail

ROOT="/Users/georgegao/Projects/locallmm"
VENV="$ROOT/.venv"
RUN="$ROOT/.run"
mkdir -p "$RUN"

set -a
[ -f "$ROOT/.env" ] && . "$ROOT/.env"
set +a
MLX_PORT="${MLX_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8100}"
MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit-DWQ}"

_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

start() {
  if _alive "$RUN/mlx.pid"; then
    echo "mlx:   already running"
  else
    echo "mlx:   starting mlx_lm.server ($MLX_MODEL) on :$MLX_PORT"
    nohup "$VENV/bin/mlx_lm.server" --model "$MLX_MODEL" \
      --host 127.0.0.1 --port "$MLX_PORT" >"$RUN/mlx.log" 2>&1 &
    echo $! >"$RUN/mlx.pid"
  fi
  if _alive "$RUN/proxy.pid"; then
    echo "proxy: already running"
  else
    echo "proxy: starting verify-correct proxy on :$PROXY_PORT"
    nohup "$VENV/bin/uvicorn" verify_proxy:app --app-dir "$ROOT/proxy" \
      --host 127.0.0.1 --port "$PROXY_PORT" >"$RUN/proxy.log" 2>&1 &
    echo $! >"$RUN/proxy.pid"
  fi
  echo "logs: $RUN/{mlx,proxy}.log   (mlx first load takes ~20-40s)"
}

stop() {
  for name in proxy mlx; do
    if _alive "$RUN/$name.pid"; then
      kill "$(cat "$RUN/$name.pid")" && echo "$name: stopped"
    else
      echo "$name: not running"
    fi
    rm -f "$RUN/$name.pid"
  done
}

status() {
  _alive "$RUN/mlx.pid"   && echo "mlx:   process up (pid $(cat "$RUN/mlx.pid"))"   || echo "mlx:   process down"
  _alive "$RUN/proxy.pid" && echo "proxy: process up (pid $(cat "$RUN/proxy.pid"))" || echo "proxy: process down"
  curl -fsS "http://127.0.0.1:$MLX_PORT/v1/models" >/dev/null 2>&1 \
    && echo "mlx:   responding on :$MLX_PORT" || echo "mlx:   not responding on :$MLX_PORT"
  curl -fsS "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 \
    && echo "proxy: responding on :$PROXY_PORT" || echo "proxy: not responding on :$PROXY_PORT"
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  *) echo "usage: locallmm {start|stop|restart|status}"; exit 1 ;;
esac
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x $ROOT/bin/locallmm && echo ok`
Expected: `ok`

- [ ] **Step 3: Start the stack and verify**

Run: `$ROOT/bin/locallmm start` then wait ~40s for the model to load, then `$ROOT/bin/locallmm status`
Expected: status shows both `process up` and both `responding`. If `mlx: not responding`, check `$ROOT/.run/mlx.log`.

- [ ] **Step 4: Commit**

```bash
git -C $ROOT add bin/locallmm
git -C $ROOT commit -m "feat: locallmm start/stop/status control script"
```

---

## Task 5: Quick chat client (`bin/llm`)

**Files:**
- Create: `$ROOT/bin/llm`

- [ ] **Step 1: Write the script**

Create `$ROOT/bin/llm`:
```python
#!/usr/bin/env python3
"""llm — quick verified chat against the local verify-correct proxy."""
import json
import os
import sys
import urllib.request

PORT = os.environ.get("PROXY_PORT", "8100")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def ask(messages: list) -> str:
    payload = json.dumps({"messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def ask_with_status(messages: list) -> str:
    sys.stdout.write("  (drafting + verifying…)")
    sys.stdout.flush()
    try:
        answer = ask(messages)
    finally:
        sys.stdout.write("\r" + " " * 32 + "\r")
        sys.stdout.flush()
    return answer


def main() -> None:
    history: list = []
    if len(sys.argv) > 1:  # one-shot mode
        history.append({"role": "user", "content": " ".join(sys.argv[1:])})
        print(ask_with_status(history))
        return
    print("locallmm chat — Ctrl-D or 'exit' to quit")
    while True:
        try:
            prompt = input("\n› ")
        except EOFError:
            print()
            break
        if prompt.strip() in {"exit", "quit"}:
            break
        if not prompt.strip():
            continue
        history.append({"role": "user", "content": prompt})
        try:
            answer = ask_with_status(history)
        except Exception as exc:
            print(f"error: {exc}\n(is the stack up? run: locallmm status)")
            history.pop()
            continue
        print(f"\n{answer}")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x $ROOT/bin/llm && echo ok`
Expected: `ok`

- [ ] **Step 3: Verify one-shot mode**

Run: `$ROOT/bin/llm "In one sentence, what is unified memory?"`
Expected: a one-sentence answer, prefixed with a `✓ verified`, `✎ corrected`, or `⚠ unverified` tag. (`⚠ unverified` is expected until an API key is added — that is correct degradation.)

- [ ] **Step 4: Commit**

```bash
git -C $ROOT add bin/llm
git -C $ROOT commit -m "feat: llm quick verified-chat client"
```

---

## Task 6: Hermes Agent install & configuration

**Files:**
- Create (via installer): `~/.hermes/` (Hermes home; outside the repo)
- Create: `$ROOT/hermes/config.reference.yaml` (committed reference copy)

- [ ] **Step 1: Install Hermes Agent**

Run: `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash`
Expected: installer completes; `hermes --version` (or `hermes --help`) works in a new shell. If `hermes` is not on `PATH`, note the path the installer reports.

- [ ] **Step 2: Point Hermes at the local MLX engine**

Ensure the stack is running (`$ROOT/bin/locallmm status` shows mlx responding), then run the model wizard:

Run: `hermes model`
In the wizard: choose **Custom endpoint**, base URL `http://127.0.0.1:8000/v1`, model name `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ`.

- [ ] **Step 3: Add the cloud fallback and timeout**

Edit `~/.hermes/config.yaml` so the local endpoint is primary and a cloud model is the fallback. Confirm the exact key names against what the wizard wrote in Step 2; the structure is:
```yaml
model:
  provider: custom            # the provider value the wizard wrote for the custom endpoint
  default: mlx-community/Qwen3.6-35B-A3B-4bit-DWQ
  base_url: http://127.0.0.1:8000/v1
fallback_providers:
  - provider: anthropic
    model: claude-sonnet-4-6
```
Then set the slow-prefill timeout in Hermes's env file `~/.hermes/.env` (create it if absent):
```
HERMES_STREAM_READ_TIMEOUT=1800
ANTHROPIC_API_KEY=
```

- [ ] **Step 4: Verify Hermes reaches the local model**

Run: `hermes chat` and send `Say "hermes online" and nothing else.`
Expected: the local model responds `hermes online`. Exit the chat. If it times out on first load, retry — the model is warm after the first request.

- [ ] **Step 5: Save a reference copy of the config**

Run: `mkdir -p $ROOT/hermes && cp ~/.hermes/config.yaml $ROOT/hermes/config.reference.yaml`
(Reference only — Hermes reads `~/.hermes/config.yaml`, not this copy.)

- [ ] **Step 6: Commit**

```bash
git -C $ROOT add hermes/config.reference.yaml
git -C $ROOT commit -m "docs: Hermes Agent config reference"
```

---

## Task 7: README, PATH setup & end-to-end verification

**Files:**
- Create: `$ROOT/README.md`

- [ ] **Step 1: Write the README**

Create `$ROOT/README.md`:
```markdown
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
3. Add `bin/` to PATH: append to `~/.zshrc`:
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
  - `deepseek` + `deepseek-chat` — lowest cost
  - `openai` + a GPT model — alternative
  - `openai-compatible` + `VERIFIER_BASE_URL` — any other OpenAI-compatible API

## Troubleshooting

- `mlx: not responding` → check `.run/mlx.log`; the model may still be loading.
- Out of memory → switch `MLX_MODEL` to the smaller Gemma 4 alternate.
- Answers tagged `⚠ unverified` → no verifier API key set; add one to `.env`.
```

- [ ] **Step 2: Add `bin/` to PATH**

Append to `~/.zshrc`: `export PATH="$HOME/Projects/locallmm/bin:$PATH"`
Run: `echo 'export PATH="$HOME/Projects/locallmm/bin:$PATH"' >> ~/.zshrc`
Then in a new shell verify: `which locallmm llm`
Expected: both resolve to `$ROOT/bin/...`.

- [ ] **Step 3: End-to-end verification — degradation path (no key)**

With `.env` having no API key, run: `locallmm restart`, wait for `locallmm status` to show both responding, then `llm "What is 17 * 23?"`
Expected: a correct answer (`391`) tagged `⚠ unverified (no verifier API key — local only)`.

- [ ] **Step 4: End-to-end verification — verified path (with key)**

Add a real key to `.env` (e.g. `ANTHROPIC_API_KEY=...`), `chmod 600 .env`, `locallmm restart`, then `llm "What is 17 * 23?"`
Expected: a correct answer tagged `✓ verified` or `✎ corrected`. Confirms the Anthropic adapter and the cloud round-trip.

- [ ] **Step 5: End-to-end verification — Hermes**

Run `hermes chat` and ask it to do a small multi-step task (e.g. "create a file /tmp/hermes_test.txt containing the current date").
Expected: Hermes uses the local model, performs the step, and `/tmp/hermes_test.txt` exists.

- [ ] **Step 6: Commit**

```bash
git -C $ROOT add README.md
git -C $ROOT commit -m "docs: README and usage guide"
```

---

## Self-review notes

- **Spec coverage:** Engine (Task 1) · verify-correct proxy + multi-provider verifier (Tasks 2-3) · cost-tiered swappable verifier (Task 2 `_call_*`, README) · `llm` (Task 5) · Hermes at raw `:8000` + cloud fallback + timeout (Task 6) · `locallmm` ops (Task 4) · `.env`/`.env.example` (Task 1) · README (Task 7) · error handling — local unreachable, cloud failure, no key (Task 3) · testing (Tasks 2,3 unit; Task 7 end-to-end). All spec sections mapped.
- **Model swap** documented in README (Task 7) and `.env.example` (Task 1).
- **Streaming:** the proxy calls the cloud verifier non-streaming, then re-chunks the final answer over SSE for streaming clients — simpler and robust; clients still render progressively.
- **Hermes config:** exact `config.yaml` key names are confirmed against the `hermes model` wizard output in Task 6 Step 3, since the wizard is the documented setup path.
```
