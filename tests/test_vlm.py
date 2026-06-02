"""Regression tests for the vlm client's request construction.

The client MUST send a `model` field. Without it the mlx-vlm server falls
back to a built-in default model (nanoLLaVA), which here downloads a second
model and crashes the request in its generation thread — so the call hangs.
"""
import importlib.util
import pathlib
from importlib.machinery import SourceFileLoader

_PATH = pathlib.Path(__file__).resolve().parents[1] / "bin" / "vlm"
_loader = SourceFileLoader("vlm_client", str(_PATH))
_spec = importlib.util.spec_from_loader("vlm_client", _loader)
vlm = importlib.util.module_from_spec(_spec)
_loader.exec_module(vlm)


def test_payload_names_a_model():
    payload = vlm.build_payload([{"role": "user", "content": "hi"}])
    assert payload["model"], "must name a model or the server uses a crashing default"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["stream"] is False


def test_split_tokens_separates_images_from_prompt(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n")
    text, images = vlm.split_tokens(
        ["describe", "this", str(img), "https://e.com/a.jpg"]
    )
    assert text == "describe this"
    assert images == [str(img), "https://e.com/a.jpg"]


def test_make_content_is_plain_string_without_images():
    assert vlm.make_content("hello", []) == "hello"


def test_make_content_attaches_image_url():
    content = vlm.make_content("what is this", ["/tmp/x.png"])
    assert {"type": "text", "text": "what is this"} in content
    assert {"type": "image_url", "image_url": {"url": "/tmp/x.png"}} in content
