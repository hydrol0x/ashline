"""Item detection (build spec 6.3) with Claude, through the Anthropic SDK.

Uses your Anthropic API credits: set ANTHROPIC_API_KEY in .env. The reply is
constrained to a JSON schema (structured outputs), so there is no parsing of
free text. Output is the same item dicts as the other detectors.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import anthropic
from PIL import Image

from . import config
from .detect import CATEGORIES, TASK, DetectionError, parse_items

DEFAULT_MODEL = "claude-opus-5-5"

# Models whose safety classifiers can decline a request and that accept the
# server-side fallback, which re-runs a declined request on another model.
FALLBACK_MODELS = ("claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Specific noun phrase, e.g. blue velvet three-seat sofa"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "brand_guess": {"type": ["string", "null"],
                                    "description": "Brand if a logo or distinctive design shows, else null"},
                    "confidence": {"type": "number", "description": "0.0 to 1.0"},
                    "box": {"type": "array", "items": {"type": "number"},
                            "description": "[x0, y0, x1, y1] on a 0-1000 grid, origin top-left, x right, y down"},
                },
                "required": ["label", "category", "brand_guess", "confidence", "box"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

ROOM_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"room": {"type": ["string", "null"]}, "items": ITEMS_SCHEMA["properties"]["items"]},
    "required": ["room", "items"],
    "additionalProperties": False,
}

INSTRUCTIONS = """

List at most 30 items. Give each physical object its own entry with its own box: two matching chairs
are two entries, four bar stools are four entries, even when they look identical or overlap. Do not
split ONE object into several entries: a sofa partly hidden or cut off by the frame is one entry with
one box around all of it. Skip what people are wearing (clothes, earbuds, watches, glasses) and what they
hold in their hands; list furniture, electronics and other belongings of the home.
`box` is the item's bounding box as [x0, y0, x1, y1] on a 0-1000 grid: origin top-left, x to the
right, y down. An empty list is the right answer for a photo with no personal property in it."""


class ClaudeDetector:
    def __init__(self, model: str | None = None, effort: str | None = None, max_side: int = 1568,
                 client: anthropic.Anthropic | None = None):
        config.load_env()                              # so the SDK finds ANTHROPIC_API_KEY from .env
        self.model = model or config.env("ASHLINE_CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = effort or config.env("ASHLINE_CLAUDE_EFFORT", "medium")
        self.max_side = max_side
        self.client = client or anthropic.Anthropic()

    def detect(self, path: str | Path, room_hint: str | None = None, rooms: list[str] | None = None) -> list[dict]:
        """Items in one photo. For a family photo with no known room, pass the home's room names
        in `rooms`: the model says which one the photo shows and each item carries it as "room"."""
        image = Image.open(path).convert("RGB")
        width, height = image.size
        small = image.copy()
        small.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=85)

        room = f" (taken in: {room_hint})" if room_hint else ""
        schema, ask_room = ITEMS_SCHEMA, ""
        if not room_hint and rooms is not None:
            schema = ROOM_ITEMS_SCHEMA
            known = f" The rooms of this home are: {', '.join(rooms)}." if rooms else ""
            ask_room = (f"\n\nAlso give `room`: the room this photo shows.{known} Use one of those names when it "
                        "fits, a plain name such as \"Garage\" when none does, or null when the photo is not "
                        "inside or around a home.")
        request = dict(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8")}},
                {"type": "text", "text": TASK.format(room=room) + INSTRUCTIONS + ask_room},
            ]}],
        )
        try:
            if self.model.startswith(FALLBACK_MODELS):
                response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **request)
            else:
                response = self.client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise DetectionError("Anthropic rejected the key. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise DetectionError(f"This key may not use {self.model}: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise DetectionError(f"Unknown model {self.model!r}. Check ASHLINE_CLAUDE_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise DetectionError("Anthropic rate limit hit after retries. Wait a minute and run again; "
                                 "finished photos are cached.") from exc
        except anthropic.APIStatusError as exc:
            raise DetectionError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise DetectionError("Could not reach the Anthropic API. Check the network.") from exc

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise DetectionError(f"Claude declined this photo (category: {category}).")
        if response.stop_reason == "max_tokens":
            raise DetectionError("The item list was cut off at max_tokens.")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            reply = json.loads(text)
            items = reply.get("items", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            raise DetectionError(f"Unexpected reply from Claude: {text[:200]}") from exc
        parsed = parse_items(json.dumps(items), width, height)
        guess = reply.get("room") if isinstance(reply.get("room"), str) else None
        if guess and guess.strip():
            for item in parsed:
                item["room"] = guess.strip()
        return parsed
