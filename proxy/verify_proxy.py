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
