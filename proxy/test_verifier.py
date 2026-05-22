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
