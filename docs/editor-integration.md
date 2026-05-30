# Editor integration

Point any OpenAI-compatible editor at the verify-correct proxy
(`http://127.0.0.1:8100/v1`). Local drafts + cloud verification, same
ChatCompletions API.

The model id is whatever `MLX_MODEL` is set to (default
`mlx-community/Qwen3.6-35B-A3B-4bit-DWQ`).

## Continue.dev (VSCode / JetBrains)

Edit `~/.continue/config.yaml`:

```yaml
models:
  - name: Local Qwen (verified)
    provider: openai
    model: mlx-community/Qwen3.6-35B-A3B-4bit-DWQ
    apiBase: http://127.0.0.1:8100/v1
    apiKey: sk-local        # any non-empty string; proxy ignores it
    roles: [chat, edit, apply]
```

Reload Continue, pick the model from the dropdown.

## Cursor

Settings → **Models**:

1. Add a custom model name: `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ`
2. Enable **OpenAI API Key** with any non-empty value (e.g. `sk-local`).
3. Toggle **Override OpenAI Base URL** → `http://127.0.0.1:8100/v1`
4. Click **Verify**.

Note: Cursor's agent mode pings OpenAI even when overridden; for full
offline use, prefer Continue.dev or Zed.

## Zed

Edit `~/.config/zed/settings.json`:

```json
{
  "language_models": {
    "openai": {
      "version": "1",
      "api_url": "http://127.0.0.1:8100/v1",
      "available_models": [
        {
          "name": "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ",
          "max_tokens": 65536
        }
      ]
    }
  }
}
```

Set `OPENAI_API_KEY=sk-local` in Zed's keychain (any non-empty value works).
Pick the model from the assistant panel.

## Bypassing the verifier

To hit MLX directly (faster, no cloud round-trip), use port `8000` instead
of `8100`. Same API shape; replies are unverified.

## Sanity check

```
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"ping"}]}' \
  | jq -r '.choices[0].message.content'
```
