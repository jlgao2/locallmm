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
        max_tokens=8192,
        system=VERIFY_SYSTEM,
        messages=[{"role": "user", "content": _user_msg(question, draft)}],
    )
    return next(b.text for b in resp.content if b.type == "text")


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
