"""A floor-plan sketch for a home with no scan, drawn by Claude from the family's photos.

`layout.py` can always draw rooms as boxes of a typical size. This step improves that drawing:
Claude looks at the photos (or the stills of a walk-through video) in the order they were taken,
with the list of items found in each, and returns

- each room's rough width and depth in metres and where it sits, so rooms seen through a doorway
  or walked into next end up beside each other, and
- where in its room each item stood, as a position between 0 and 1 across and down the room.

The reply places rooms and items only. It cannot add, rename or remove an inventory line: ids
that are not on the inventory are ignored, and so are rooms that are not in the list given.
The sketch is kept in `cases/<case>/sketch.json` with a fingerprint of what it was drawn from,
and is drawn again only when the photos or items change.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json

from . import config, layout
from .rooms import tidy_room

MAX_PHOTOS = 16                 # stills sent to the model, spread evenly over what the family sent
MAX_SIDE = 768

SKETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "width_m": {"type": "number", "description": "Size along x, in metres"},
                    "depth_m": {"type": "number", "description": "Size along y, in metres"},
                    "x_m": {"type": "number", "description": "Left edge, metres from the plan's left"},
                    "y_m": {"type": "number", "description": "Top edge, metres from the plan's top"},
                },
                "required": ["name", "width_m", "depth_m", "x_m", "y_m"],
                "additionalProperties": False,
            },
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "u": {"type": "number", "description": "0 = the room's left wall, 1 = its right wall"},
                    "v": {"type": "number", "description": "0 = the room's top wall, 1 = its bottom wall"},
                },
                "required": ["id", "u", "v"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rooms", "items"],
    "additionalProperties": False,
}

PROMPT = """You are sketching a rough floor plan of a home for a contents inventory after a fire.
There is no survey or scan: only these {count} photos, shown above in the order they were taken.
{video}Photos taken one after another usually show the same room or the room next door.

The rooms of the home, as the inventory names them:
{rooms}

The items on the inventory, as: id | room | first seen in photo | item
{items}

Draw the plan as rectangles seen from above, x to the right and y down, in metres.
- Give every room in the list exactly once, using its name exactly as written.
- `width_m` and `depth_m`: your estimate of the room's size from what the photos show (floor
  boards, door widths of about 0.8 m, beds about 2 m long and counters about 0.6 m deep help).
- `x_m`, `y_m`: the room's top-left corner. Rooms must not overlap. Rooms that open onto each
  other, or that the camera walked between, share a wall. Start at 0, 0.
- For every item give `u` and `v`, its position inside its own room's rectangle: u from 0 at the
  left wall to 1 at the right wall, v from 0 at the top wall to 1 at the bottom wall. Keep it
  consistent with where you put the walls and neighbouring rooms: an item beside the doorway to
  the kitchen goes on the wall shared with the kitchen. Things against a wall get a value near
  0 or 1; things in the middle of the floor do not.

This is a sketch to help a family remember where things stood, so a sensible guess is better
than leaving something out."""


class SketchError(RuntimeError):
    pass


def chosen_photos(photos: list[dict]) -> list[dict]:
    """Up to MAX_PHOTOS family photos in the order taken, spread evenly."""
    ordered = sorted((p for p in photos if p.get("path")), key=lambda p: p.get("number") or 0)
    if len(ordered) <= MAX_PHOTOS:
        return ordered
    step = len(ordered) / MAX_PHOTOS
    return [ordered[int(n * step)] for n in range(MAX_PHOTOS)]


def sketch_request(case_id: str, store) -> dict | None:
    """What the model is asked: {prompt, photos, rooms, ids, digest}. None when there is nothing to draw."""
    lines = layout._lines(store, case_id)
    rooms = layout.room_order(store, case_id, lines)
    photos = chosen_photos(store.get_photos(case_id, source="photo"))
    if not rooms or not photos:
        return None
    number = {p["photo_id"]: p.get("number") for p in store.get_photos(case_id, source="photo")}
    shown = {p["photo_id"]: n for n, p in enumerate(photos, 1)}
    rows = []
    for item in lines:
        room = tidy_room(item.get("room_label"))
        if not room:
            continue
        seen = shown.get(item.get("photo_id"))
        where = f"photo {seen}" if seen else (f"a photo not shown (number {number.get(item.get('photo_id'))})"
                                              if item.get("photo_id") else "from memory, no photo")
        rows.append(f"{item['item_id']} | {room} | {where} | {item['label']}")
    clips = {p.get("batch_id") for p in photos if p.get("batch_id")}
    video = ("They are stills from a video of someone walking through the home, so the order follows the walk. "
             if clips else "")
    prompt = PROMPT.format(count=len(photos), video=video, rooms="\n".join(f"- {r}" for r in rooms),
                           items="\n".join(rows) or "(none yet)")
    digest = hashlib.sha256((prompt + "|".join(p["photo_id"] for p in photos)).encode()).hexdigest()[:16]
    return {"prompt": prompt, "photos": photos, "rooms": rooms,
            "ids": [int(r.split(" | ")[0]) for r in rows], "digest": digest}


def clean(reply: dict, rooms: list[str], ids: list[int]) -> dict:
    """Keep only rooms and items the inventory knows, with usable numbers."""
    known = {r.lower(): r for r in rooms}
    out_rooms = []
    for room in reply.get("rooms") or []:
        name = known.get((tidy_room(room.get("name")) or "").lower()) if isinstance(room, dict) else None
        if name and all(isinstance(room.get(k), (int, float)) for k in ("width_m", "depth_m", "x_m", "y_m")):
            out_rooms.append({"name": name, **{k: float(room[k]) for k in ("width_m", "depth_m", "x_m", "y_m")}})
    wanted = set(ids)
    out_items = [{"id": i["id"], "u": min(max(float(i["u"]), 0.0), 1.0), "v": min(max(float(i["v"]), 0.0), 1.0)}
                 for i in reply.get("items") or []
                 if isinstance(i, dict) and i.get("id") in wanted
                 and all(isinstance(i.get(k), (int, float)) for k in ("u", "v"))]
    return {"rooms": out_rooms, "items": out_items}


class ClaudeSketcher:
    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        from .detect_claude import DEFAULT_MODEL
        config.load_env()
        self.model = model or config.env("ASHLINE_CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = effort or config.env("ASHLINE_CLAUDE_EFFORT", "medium")
        self._client = client

    def draw(self, request: dict) -> dict:
        import anthropic
        from PIL import Image

        from .detect_claude import FALLBACK_BETA, FALLBACK_MODELS
        content = []
        for n, photo in enumerate(request["photos"], 1):
            image = Image.open(photo["path"]).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=80)
            content.append({"type": "text", "text": f"Photo {n}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": request["prompt"]})
        call = dict(model=self.model, max_tokens=16000,
                    output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SKETCH_SCHEMA}},
                    messages=[{"role": "user", "content": content}])
        client = self._client or anthropic.Anthropic()
        try:
            if self.model.startswith(FALLBACK_MODELS):
                response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **call)
            else:
                response = client.messages.create(**call)
        except anthropic.APIError as exc:
            raise SketchError(f"The plan sketch could not be drawn: {exc}") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise SketchError(f"The plan sketch was not finished ({response.stop_reason}).")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            return json.loads(text)
        except ValueError as exc:
            raise SketchError("The plan sketch came back unreadable.") from exc


def available() -> bool:
    """True when a sketch can be asked for: the setting allows it and there is a key."""
    setting = (config.env("ASHLINE_SKETCHER") or "auto").lower()
    return setting != "rules" and bool(config.env("ANTHROPIC_API_KEY"))


def refresh(case_id: str, store, sketcher=None, force: bool = False) -> dict | None:
    """Draw (or redraw) the sketch for a case that has no scan. Returns it, or None if not needed."""
    plan = layout.load_plan(case_id)
    if plan and plan.get("rooms"):
        return None                                         # the scan's own geometry is better
    request = sketch_request(case_id, store)
    if request is None:
        return None
    current = layout.load_sketch(case_id)
    if current and current.get("digest") == request["digest"] and not force:
        return current
    reply = (sketcher or ClaudeSketcher()).draw(request)
    drawn = {**clean(reply, request["rooms"], request["ids"]), "digest": request["digest"]}
    layout.save_sketch(case_id, drawn)
    return drawn
