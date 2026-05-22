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
