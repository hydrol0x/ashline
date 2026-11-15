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
from .detect import CATEGORIES, SKIP_NEXT_ROOM, TASK, DetectionError, parse_items

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

# Family photos: each item can say it stands in another room than the one the photo was taken in.
_ITEM = ITEMS_SCHEMA["properties"]["items"]["items"]
_ITEM_IN_ROOM = {**_ITEM, "required": _ITEM["required"] + ["in_room"], "properties": {**_ITEM["properties"], "in_room": {
    "type": ["string", "null"],
    "description": "Null for an item in the room the photo was taken in. The other room's name for an item that "
                   "stands in a different room and is only seen from here."}}}
HOME_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": _ITEM_IN_ROOM}},
    "required": ["items"],
    "additionalProperties": False,
}
ROOM_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"room": {"type": ["string", "null"]}, "items": {"type": "array", "items": _ITEM_IN_ROOM}},
    "required": ["room", "items"],
    "additionalProperties": False,
}

IN_ROOM = """

Some photos look into another room: through a doorway, down a hallway, or across an open-plan space.
For each item give `in_room`: null when the item stands in the room the photo was taken in, or the
name of the room it really stands in when that is a different one. Decide by where the item stands,
not by where the camera is: a bed seen through an open door from the hallway belongs to the bedroom.
Use the room names of this home when they fit."""

INSTRUCTIONS = """

List at most 30 items. Give each physical object its own entry with its own box: two matching chairs
are two entries, four bar stools are four entries, even when they look identical or overlap. Do not
split ONE object into several entries: a sofa partly hidden or cut off by the frame is one entry with
one box around all of it. Skip what people are wearing (clothes, earbuds, watches, glasses) and what they
hold in their hands; list furniture, electronics and other belongings of the home. Skip food, drinks and
dishes in use: on a set table or a busy counter, list the table, the chairs and the appliances, not the
plates and bowls. When there are more than 30 things, keep the 30 most valuable.
`box` is the item's bounding box as [x0, y0, x1, y1] on a 0-1000 grid: origin top-left, x to the
right, y down. An empty list is the right answer for a photo with no personal property in it."""


def place_items(items: list[dict], photo_room: str | None, rooms: list[str] | None) -> list[dict]:
    """Give each item its room: the photo's, or its own `in_room` when it stands somewhere else.

    Each item gets "room" (where it stands) and "photo_room" (where the photo was taken); the
    two differ for an item seen through a doorway. A room name the home already has is used in
    the home's own spelling, so "dining area" and "Dining Area" are one room.
    """
    known = {r.strip().lower(): r for r in rooms or []}
    photo_room = known.get((photo_room or "").strip().lower(), photo_room)
    for item in items:
        other = item.pop("in_room", None)
        other = known.get(other.strip().lower(), other.strip()) if other else None
        if photo_room:
            item["photo_room"] = photo_room
        room = other if other and other.lower() != (photo_room or "").lower() else photo_room
        if room:
            item["room"] = room
    return items


class ClaudeDetector:
    def __init__(self, model: str | None = None, effort: str | None = None, max_side: int = 1568,
                 client: anthropic.Anthropic | None = None):
        config.load_env()                              # so the SDK finds ANTHROPIC_API_KEY from .env
        self.model = model or config.env("ASHLINE_CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = effort or config.env("ASHLINE_CLAUDE_EFFORT", "medium")
        self.max_side = max_side
        self.client = client or anthropic.Anthropic()

    def detect_short(self, path, room_hint=None, rooms=None) -> list[dict]:
        """The same request, asking for the 15 most valuable things only."""
        self._limit = 15
        try:
            return self.detect(path, room_hint, rooms)
        finally:
            self._limit = 30

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
        schema, ask_room, task = ITEMS_SCHEMA, "", TASK.format(room=room)
        if rooms is not None:                              # a family photo: rooms are named, not measured
            task = task.replace(SKIP_NEXT_ROOM, "")        # what stands through the doorway is listed, under its room
            known = f" The rooms of this home are: {', '.join(rooms)}." if rooms else ""
            if room_hint:
                schema, ask_room = HOME_ITEMS_SCHEMA, (f"\n\n{known.strip()}" if known else "") + IN_ROOM
            else:
                schema = ROOM_ITEMS_SCHEMA
                ask_room = (f"\n\nAlso give `room`: the room this photo was taken in.{known} Use one of those names "
                            "when it fits, a plain name such as \"Garage\" when none does, or null when the photo "
                            "is not inside or around a home." + IN_ROOM)
        request = dict(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8")}},
                {"type": "text", "text": task + INSTRUCTIONS.replace("30", str(getattr(self, "_limit", 30))) + ask_room},
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
            if not getattr(self, "_retrying", False):      # a very busy photo: one quicker, shorter pass
                self._retrying, effort = True, self.effort
                try:
                    self.effort = "low"
                    return self.detect_short(path, room_hint, rooms)
                finally:
                    self._retrying, self.effort = False, effort
            raise DetectionError("The item list was cut off at max_tokens.")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            reply = json.loads(text)
            items = reply.get("items", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            raise DetectionError(f"Unexpected reply from Claude: {text[:200]}") from exc
        parsed = parse_items(json.dumps(items), width, height)
        guess = reply.get("room") if isinstance(reply.get("room"), str) else None
        if rooms is None:                                  # a scan face: its room is known from the scan
            return parsed
        return place_items(parsed, (guess or "").strip() or room_hint, rooms)
