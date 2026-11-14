"""ClaudeDetector against a stand-in client: no API call is made."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from PIL import Image

from ashline import detect
from ashline.detect import DetectionError
from ashline.detect_claude import ITEMS_SCHEMA, ClaudeDetector


class FakeMessages:
    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="general_harms") if self.stop_reason == "refusal" else None,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.reply)])


def fake_client(reply, stop_reason="end_turn"):
    plain, beta = FakeMessages(reply, stop_reason), FakeMessages(reply, stop_reason)
    return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta)), plain, beta


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "scan-1-f1.jpg"
    Image.new("RGB", (2048, 2048), (90, 90, 90)).save(path)
    return path


REPLY = json.dumps({"items": [
    {"label": "blue velvet three-seat sofa", "category": "furniture", "brand_guess": None, "confidence": 0.9,
     "box": [100, 500, 600, 900]},
    {"label": "oak bar stool", "category": "furniture", "brand_guess": "Acme", "confidence": 0.7,
     "box": [700, 600, 800, 950]},
    {"label": "half a box", "category": "decor", "brand_guess": None, "confidence": 0.9, "box": [1, 2]},
]})


def test_detect_returns_items_with_pixel_boxes(photo):
    client, plain, beta = fake_client(REPLY)
    items = ClaudeDetector(client=client).detect(photo, "Living Room")
    assert [i["label"] for i in items] == ["blue velvet three-seat sofa", "oak bar stool"]    # malformed box dropped
    assert items[0]["bbox"] == [205, 1024, 1229, 1843] and items[1]["brand_guess"] == "Acme"

    call = beta.calls[0]
    assert plain.calls == []                                   # the default model goes through the beta endpoint
    assert call["model"] == "claude-opus-5-5" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": ITEMS_SCHEMA}}
    assert "thinking" not in call and "temperature" not in call and "tool_choice" not in call
    image, text = call["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert "(taken in: Living Room)" in text["text"] and "0-1000 grid" in text["text"]


def test_models_without_server_side_fallback_use_the_plain_endpoint(photo):
    client, plain, beta = fake_client('{"items": []}')
    assert ClaudeDetector(model="claude-haiku-5-5", client=client).detect(photo) == []
    assert beta.calls == [] and "fallbacks" not in plain.calls[0] and "betas" not in plain.calls[0]


def test_refusal_and_truncation_are_errors_not_empty_lists(photo):
    client, _, _ = fake_client("", stop_reason="refusal")
    with pytest.raises(DetectionError, match="declined"):
        ClaudeDetector(client=client).detect(photo)
    client, _, _ = fake_client('{"items": [', stop_reason="max_tokens")
    with pytest.raises(DetectionError, match="cut off"):
        ClaudeDetector(client=client).detect(photo)


def test_schema_is_strict():
    item = ITEMS_SCHEMA["properties"]["items"]["items"]
    assert ITEMS_SCHEMA["additionalProperties"] is False and item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_default_detector_prefers_an_anthropic_key(monkeypatch, tmp_path):
    from ashline import config
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [tmp_path / ".env"])
    for name in ("ANTHROPIC_API_KEY", "ASHLINE_VLM_API_KEY", "ASHLINE_DETECTOR"):
        monkeypatch.delenv(name, raising=False)
    assert detect.default_detector_kind() == "florence"
    monkeypatch.setenv("ASHLINE_VLM_API_KEY", "k")
    assert detect.default_detector_kind() == "vlm"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert detect.default_detector_kind() == "claude"
    monkeypatch.setenv("ASHLINE_DETECTOR", "nonsense")
    with pytest.raises(DetectionError, match="claude, vlm or florence"):
        detect.get_detector()
