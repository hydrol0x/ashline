"""scan_rooms and scan_photos: the inside of the home, from its Matterport scan.

Verified against the Model API on 14 Nov 2026:
- `panoLocations { anchor { room { id } } }` links a panorama to its room directly.
  Panoramas with no room were taken outdoors (yard, driveway).
- `rooms { tags }` carries "unfurnished" for empty rooms.
- `dimensionEstimates` is filled for rooms where the deprecated `dimensions` is
  missing (garages, closets). A height of 0 means "not measured".
- Image URLs are signed and expire, so they are downloaded straight away and
  never returned or stored.
"""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import requests
from PIL import Image

from .. import config

API = "https://api.matterport.com/api/models/graph"

Q_SCAN = """query Scan($id: ID!) { model(id: $id) {
  id description
  generatedDescription { summary }
  floors { id label }
  rooms { id floor { id } classifications { label confidence } tags
          dimensionEstimates(units: metric) { area width depth height }
          dimensions(units: metric) { areaFloor width depth height } }
  panoLocations { id label position { x y z } anchor { room { id } } }
} }"""

Q_PANOS = """query Panos($id: ID!, $res: String!) { model(id: $id) {
  panoLocations { id skybox(resolution: $res) { status children } }
} }"""

SKIP_LABELS = {"void", ""}
HORIZON_FACES = (1, 2, 3, 4)          # 0 is up, 5 is down
EXTERIOR = "Exterior"

# How much a room type is worth photographing for a contents inventory.
# Higher = picked first when the face budget is tight.
ROOM_PRIORITY = {
    "living room": 10, "family room": 10, "bedroom": 9, "primary bedroom": 9, "kitchen": 9,
    "office": 9, "dining room": 8, "garage": 8, "den": 8, "media room": 8, "game room": 8,
    "workshop": 7, "library": 7, "gym": 7, "basement": 6, "laundry room": 6, "utility room": 5,
    "closet": 5, "walk-in closet": 5, "pantry": 4, "bathroom": 3, "patio": 6, "balcony": 4,
    "hallway": 1, "stairs": 1, "entryway": 2, "foyer": 2,
}
BIG_ROOM_M2 = 30.0          # rooms this large get one panorama more than per_room
TYPICAL_ROOM_M2 = 10.0      # assumed when a room has no measured area
MIN_ROOM_M2 = 1.5           # smaller "rooms" are slivers of the floor plan


class MatterportError(RuntimeError):
    pass


class MatterportClient:
    """GraphQL over HTTP basic auth: the API key is the user name, the token the password."""

    def __init__(self, key: str | None = None, token: str | None = None, min_interval: float = 0.15):
        self.session = requests.Session()
        self.session.auth = (
            key or config.env("MATTERPORT_API_KEY", required=True),
            token or config.env("MATTERPORT_API_TOKEN", required=True),
        )
        self.min_interval = min_interval      # every team shares these keys: stay well under 10 rps
        self._last = 0.0

    def _wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def graph(self, query: str, **variables) -> dict:
        for attempt in range(4):
            self._wait()
            resp = self.session.post(API, json={"query": query, "variables": variables}, timeout=120)
            if resp.status_code in (429, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            break
        if resp.status_code in (401, 403):
            raise MatterportError(
                f"Matterport rejected the key (HTTP {resp.status_code}). Check MATTERPORT_API_KEY "
                "(user name) and MATTERPORT_API_TOKEN (password); they only work on api.matterport.com.")
        resp.raise_for_status()
        body = resp.json()
        for err in body.get("errors") or []:
            code = (err.get("extensions") or {}).get("code", "")
            if code == "not.found":
                raise MatterportError("Scan not found, or not shared with this key.")
            raise MatterportError(f"GraphQL error: {err.get('message')}")
        return body["data"]

    def download(self, url: str) -> bytes:
        self._wait()
        resp = requests.get(url, timeout=300)      # signed URL: no auth header
        resp.raise_for_status()
        return resp.content


# ----------------------------------------------------------------- rooms
def _positive(value) -> float | None:
    """Missing or zero measurements are unknown, not zero."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 2) if value > 0 else None


def room_label(room: dict) -> str:
    labels = [c.get("label") for c in room.get("classifications") or [] if c.get("label")]
    return " / ".join(dict.fromkeys(labels))


def parse_rooms(model: dict) -> tuple[list[dict], int]:
    """Labelled rooms of a scan, and how many unlabelled or Void rooms were skipped."""
    panos_in_room: dict[str, int] = {}
    for pano in model.get("panoLocations") or []:
        rid = ((pano.get("anchor") or {}).get("room") or {}).get("id")
        if rid:
            panos_in_room[rid] = panos_in_room.get(rid, 0) + 1

    floors = {f["id"]: f.get("label") for f in model.get("floors") or []}
    rooms, skipped = [], 0
    for room in model.get("rooms") or []:
        label = room_label(room)
        if label.lower() in SKIP_LABELS:
            skipped += 1
            continue
        est = room.get("dimensionEstimates") or {}
        old = room.get("dimensions") or {}
        tags = room.get("tags") or []
        rooms.append({
            "room_id": room["id"],
            "label": label,
            "area_m2": _positive(est.get("area")) or _positive(old.get("areaFloor")),
            "width": _positive(est.get("width")) or _positive(old.get("width")),
            "depth": _positive(est.get("depth")) or _positive(old.get("depth")),
            "height": _positive(est.get("height")) or _positive(old.get("height")),
            "floor": floors.get((room.get("floor") or {}).get("id")),
            # False when Matterport tagged the room empty, None when it gave no tags.
            "furnished": False if "unfurnished" in tags else (True if tags else None),
            "pano_count": panos_in_room.get(room["id"], 0),
        })
    rooms.sort(key=lambda r: -(r["area_m2"] or 0))
    # `name` tells rooms with the same label apart: Bedroom, Bedroom 2, Bedroom 3 (largest first).
    seen: dict[str, int] = {}
    for room in rooms:
        seen[room["label"]] = seen.get(room["label"], 0) + 1
        room["name"] = room["label"] if seen[room["label"]] == 1 else f"{room['label']} {seen[room['label']]}"
    return rooms, skipped


def scan_rooms(sid: str, client: MatterportClient | None = None) -> dict:
    """Room list of a scan: [{room_id, label, area_m2, width, depth, height, ...}] plus capture notes."""
    client = client or MatterportClient()
    model = client.graph(Q_SCAN, id=sid)["model"]
    rooms, skipped = parse_rooms(model)
    exterior = sum(1 for p in model.get("panoLocations") or []
                   if not ((p.get("anchor") or {}).get("room") or {}).get("id"))
    return {
        "sid": sid,
        "rooms": rooms,
        "skipped_rooms": skipped,
        # Free-text notes from the photographer, e.g. "home staged for open house".
        # Show them so the family knows what the scan shows.
        "capture_notes": (model.get("description") or "").strip() or None,
        "summary": ((model.get("generatedDescription") or {}).get("summary") or "").strip() or None,
        "pano_count": len(model.get("panoLocations") or []),
        "exterior_pano_count": exterior,
    }


# ------------------------------------------------------ panorama selection
def _xy(pano: dict) -> tuple[float, float]:
    pos = pano.get("position") or {}
    return float(pos.get("x") or 0.0), float(pos.get("y") or 0.0)


def _spread(panos: list[dict], count: int) -> list[dict]:
    """Up to `count` panoramas: the most central one first, then the farthest from those picked."""
    if count <= 0 or not panos:
        return []
    cx = sum(_xy(p)[0] for p in panos) / len(panos)
    cy = sum(_xy(p)[1] for p in panos) / len(panos)
    picked = [min(panos, key=lambda p: (math.dist(_xy(p), (cx, cy)), str(p.get("label"))))]
    while len(picked) < min(count, len(panos)):
        rest = [p for p in panos if p not in picked]
        picked.append(max(rest, key=lambda p: min(math.dist(_xy(p), _xy(q)) for q in picked)))
    return picked


def _priority(room: dict) -> float:
    names = [n.strip() for n in room["label"].lower().split("/")]
    return max(ROOM_PRIORITY.get(n, 4) for n in names)


def select_panos(model: dict, rooms: list[dict], per_room: int = 2, max_panos: int = 10,
                 exterior: int = 2, skip_unfurnished: bool = True,
                 only_rooms: list[str] | None = None) -> list[dict]:
    """Choose which panoramas to download.

    A scan has about 190 horizon faces, too many to process live. Keep up to
    `per_room` panoramas per labelled room, most valuable rooms first, within
    a budget of `max_panos`. Returns [{pano, room_id, room_label}].
    """
    by_room: dict[str | None, list[dict]] = {}
    for pano in model.get("panoLocations") or []:
        rid = ((pano.get("anchor") or {}).get("room") or {}).get("id")
        by_room.setdefault(rid, []).append(pano)

    wanted = None
    if only_rooms:
        wanted = {w.strip().lower() for w in only_rooms}

    def keep(room: dict) -> bool:
        if not by_room.get(room["room_id"]):
            return False
        if room["area_m2"] is not None and room["area_m2"] < MIN_ROOM_M2:
            return False
        if wanted is not None:
            return room["room_id"].lower() in wanted or any(
                w in room["label"].lower() for w in wanted)
        return not (skip_unfurnished and room["furnished"] is False)

    ranked = sorted((r for r in rooms if keep(r)),
                    key=lambda r: (-_priority(r), -(r["area_m2"] or 0), r["room_id"]))
    want_exterior = exterior if wanted is None or EXTERIOR.lower() in wanted else 0
    outside = _spread(by_room.get(None, []), want_exterior)
    budget = max(0, max_panos - len(outside))

    # Hand out the budget one panorama at a time to the room where the next
    # one is worth most: weight / (panoramas it already has + 1). Big, valuable
    # rooms get several views; hallways and bathrooms only if budget is left.
    def weight(room: dict) -> float:
        return _priority(room) * math.sqrt(max(room["area_m2"] or TYPICAL_ROOM_M2, 1.0))

    def cap(room: dict) -> int:
        bonus = 1 if (room["area_m2"] or 0) >= BIG_ROOM_M2 else 0
        return min(per_room + bonus, len(by_room[room["room_id"]]))

    counts = {r["room_id"]: 0 for r in ranked}
    while budget > 0:
        open_rooms = [r for r in ranked if counts[r["room_id"]] < cap(r)]
        if not open_rooms:
            break
        best = max(open_rooms, key=lambda r: weight(r) / (counts[r["room_id"]] + 1))
        counts[best["room_id"]] += 1
        budget -= 1

    chosen = []
    for room in ranked:
        for pano in _spread(by_room[room["room_id"]], counts[room["room_id"]]):
            chosen.append({"pano": pano, "room_id": room["room_id"], "room_label": room["name"]})
    for pano in outside:
        chosen.append({"pano": pano, "room_id": None, "room_label": EXTERIOR})
    return chosen


# ----------------------------------------------------------------- photos
def scan_photos(sid: str, case_id: str, rooms: list[str] | None = None, per_room: int = 2,
                max_faces: int = 40, resolution: str = "2k", exterior: int = 2,
                captured: str | None = None, client: MatterportClient | None = None) -> dict:
    """Download horizon faces of selected panoramas into the case folder.

    Each face is one photo tagged {sid, pano_id, face, room_id}. `rooms` filters
    by room id or label text ("kitchen", "exterior"). Files land in
    cases/<case>/photos/, which is git-ignored.
    """
    client = client or MatterportClient()
    model = client.graph(Q_SCAN, id=sid)["model"]
    room_list, _ = parse_rooms(model)
    chosen = select_panos(model, room_list, per_room=per_room,
                          max_panos=max(1, max_faces // len(HORIZON_FACES)),
                          exterior=exterior, only_rooms=rooms)
    if not chosen:
        return {"sid": sid, "photos": [], "panos_used": 0, "panos_total": len(model.get("panoLocations") or [])}

    # Signed URLs expire, so ask for them right before downloading.
    fresh = client.graph(Q_PANOS, id=sid, res=resolution)["model"]["panoLocations"]
    urls = {p["id"]: (p.get("skybox") or {}).get("children") or [] for p in fresh}

    out_dir = config.case_dir(case_id) / "photos"
    out_dir.mkdir(parents=True, exist_ok=True)
    photos, missing = [], 0
    for pick in chosen:
        pano = pick["pano"]
        children = urls.get(pano["id"]) or []
        pos = pano.get("position") or {}
        for face in HORIZON_FACES:
            if face >= len(children):
                missing += 1
                continue
            photo_id = f"scan-{pano.get('label') or pano['id']}-f{face}"
            path = out_dir / f"{photo_id}.jpg"
            if not path.exists():                          # cached per case
                image = Image.open(io.BytesIO(client.download(children[face]))).convert("RGB")
                image.save(path, quality=90)
            photos.append({
                "photo_id": photo_id,
                "path": str(path),
                "source": "scan",
                "sid": sid,
                "pano_id": pano["id"],
                "pano_label": pano.get("label"),
                "face": face,
                "room_id": pick["room_id"],
                "room_label": pick["room_label"],
                "x": pos.get("x"), "y": pos.get("y"), "z": pos.get("z"),
                "captured": captured,
            })
    return {
        "sid": sid,
        "photos": photos,
        "panos_used": len(chosen),
        "panos_total": len(model.get("panoLocations") or []),
        "faces_missing": missing,
    }


# ------------------------------------------------------- family message
def _plural(count: int, noun: str) -> str:
    if count == 1:
        return f"a {noun}"
    return f"{count} {noun}es" if noun.endswith("s") else f"{count} {noun}s"


def _month_year(date: str | None) -> str | None:
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    try:
        year, month = int(date[:4]), int(date[5:7])
        return f"{months[month - 1]} {year}"
    except (TypeError, ValueError, IndexError):
        return None


def listing_sentence(listing: dict | None) -> str:
    """'The sale listing describes a 3-bed, 2-bath home built in 1962.' or ''."""
    if not listing:
        return ""
    def num(value):
        return int(value) if float(value) == int(value) else value
    parts = []
    if listing.get("beds"):
        parts.append(f"{num(listing['beds'])}-bed")
    if listing.get("baths"):
        parts.append(f"{num(listing['baths'])}-bath")
    text = (", ".join(parts) + " home") if parts else "home"
    # Size is left out: a listing's figure can cover only the main house of a lot with several units.
    if listing.get("year_built"):
        text += f" built in {int(listing['year_built'])}"
    if text == "home":
        return ""
    return f" The sale listing describes a {text}."


def scan_confirmation_message(rooms: list[dict], captured: str | None,
                              capture_notes: str | None = None, listing: dict | None = None) -> str:
    """One short message asking the family whether to start from the scan."""
    counts: dict[str, int] = {}
    for room in rooms:
        if room["area_m2"] is not None and room["area_m2"] < MIN_ROOM_M2:
            continue                        # a sliver of the floor plan, not a room
        for name in room["label"].lower().split(" / "):
            counts[name] = counts.get(name, 0) + 1
    order = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "office", "garage"]
    parts = [_plural(counts[n], n) for n in order if counts.get(n)][:4]
    when = _month_year(captured)
    text = "I found a 3D scan of this address"
    if when:
        text += f" from {when}"
    if parts:
        text += " with " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0])
    about_listing = listing_sentence(listing)
    text += "." + about_listing + (" Use the scan as a starting point?" if about_listing else " Use it as a starting point?")
    if capture_notes:
        text += f' The photographer noted: "{capture_notes}".'
    text += " It may show staging or a previous owner's things, so nothing is added until you confirm it."
    return text
