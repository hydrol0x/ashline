"""A 2D plan of the home with a pin where each item was recorded.

Two sources, one output shape:

- **Scan** (`kind: "scan"`): the Matterport scan gives each room's outline in metres and where every
  panorama stood and which way it faced. An item seen in a panorama is placed where the bottom of
  its box meets the floor, by projecting that pixel from the camera onto the floor, and never
  beyond the room's wall. Verified on the Model API on 15 Nov 2026: cube face `f` (1-4) looks
  along `yaw - (f - 1) * 90 degrees`, yaw being the panorama rotation about the vertical axis.
- **Sketch** (`kind: "sketch"`): no scan. Rooms become rectangles of a typical size for their type,
  laid out in the order the photos or the video walked through them; a model that has looked at
  the photos can supply better sizes, neighbours and item positions (`sketch_claude.py`). It is a
  sketch, not a survey, and the page says so.

Output of `build()`: {kind, note, width, height, rooms: [{id, name, points, label_at, floor}],
pins: [{id, label, room, x, y, exact, ...}]}, all in metres with y pointing down, ready to draw.

The scan geometry is kept in `cases/<case>/plan.json`, which is private case data: floor plans
are never committed or published.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon

from . import config
from .rooms import tidy_room

CAMERA_HEIGHT = 1.45            # metres above the floor; a scan tripod
WALL_MARGIN = 0.93              # an item is never placed past this share of the way to the wall
THROUGH_DOOR = 6.0               # metres: how far an item seen in the next room is still placed by its floor point
FLOOR_GAP = 2.0                 # metres between floors drawn side by side
SKETCH_NOTE = "A sketch from your photos: rooms and positions are approximate, not to scale."
SCAN_NOTE = "Room outlines come from the 3D scan. Item positions are estimated from where each was seen."

Q_PLAN = """query Plan($id: ID!) { model(id: $id) {
  floors { id label sequence }
  rooms { id floor { id } boundary { edges { vertices { id position { x y } } } } }
  panoLocations { id position { x y z } rotation { x y z w } anchor { room { id } floor { id } } }
} }"""

# Typical room sizes in metres (width, depth) for the sketch, matched by words in the room's name.
ROOM_SIZES = [
    (("garage",), (6.0, 6.0)), (("living", "family", "great"), (5.2, 4.4)), (("primary", "master"), (4.4, 4.0)),
    (("kitchen",), (4.0, 3.4)), (("dining",), (3.6, 3.2)), (("bed",), (3.6, 3.2)), (("office", "study", "den"), (3.2, 3.0)),
    (("bath", "powder"), (2.4, 2.0)), (("laundry", "utility", "mud"), (2.2, 1.8)), (("closet", "pantry"), (1.6, 1.4)),
    (("hall", "entry", "foyer", "stairs"), (3.6, 1.3)), (("basement", "attic"), (6.0, 4.5)),
    (("patio", "deck", "yard", "porch", "balcony"), (4.5, 3.0)),
]
DEFAULT_SIZE = (3.4, 3.0)
# Things that usually stand in the middle of a room rather than against a wall.
MIDDLE_WORDS = {"rug", "coffee", "dining", "island", "ottoman", "bed", "table", "carpet"}


# ------------------------------------------------------------------- scan
def ring(edges: list[dict]) -> list[tuple[float, float]]:
    """A room's wall segments joined end to end into one outline."""
    where, links = {}, {}
    for edge in edges or []:
        a, b = (edge.get("vertices") or [None, None])[:2]
        if not a or not b:
            continue
        for v in (a, b):
            where[v["id"]] = (v["position"]["x"], v["position"]["y"])
        links.setdefault(a["id"], []).append(b["id"])
        links.setdefault(b["id"], []).append(a["id"])
    if len(where) < 3:
        return []
    start = next(iter(links))
    path, previous, current = [start], None, start
    while True:
        step = next((n for n in links[current] if n != previous and n not in path[1:]), None)
        if step is None or step == start:
            break
        path.append(step)
        previous, current = current, step
    if len(path) < len(where):                              # the walls do not form one loop: use their hull
        hull = Polygon(list(where.values())).convex_hull if len(where) >= 3 else None
        return list(hull.exterior.coords)[:-1] if hull is not None and hull.geom_type == "Polygon" else []
    return [where[v] for v in path]


def _yaw(q: dict | None) -> float:
    """Rotation about the vertical (z) axis, in radians, from a quaternion."""
    if not q:
        return 0.0
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def scan_plan(model: dict) -> dict:
    """Room outlines and camera spots from a Model API reply to `Q_PLAN`."""
    floors = sorted(model.get("floors") or [], key=lambda f: f.get("sequence") or 0)
    rooms = []
    for room in model.get("rooms") or []:
        points = ring(((room.get("boundary") or {}).get("edges")) or [])
        if len(points) >= 3:
            rooms.append({"id": room["id"], "floor": (room.get("floor") or {}).get("id"),
                          "points": [[round(x, 3), round(y, 3)] for x, y in points]})
    cameras = {}
    for pano in model.get("panoLocations") or []:
        anchor = pano.get("anchor") or {}
        cameras[pano["id"]] = {"x": pano["position"]["x"], "y": pano["position"]["y"], "yaw": _yaw(pano.get("rotation")),
                               "room": (anchor.get("room") or {}).get("id"), "floor": (anchor.get("floor") or {}).get("id")}
    return {"kind": "scan", "floors": [{"id": f["id"], "label": f.get("label") or ""} for f in floors],
            "rooms": rooms, "cameras": cameras}


def plan_path(case_id: str) -> Path:
    return config.case_dir(case_id) / "plan.json"


def fetch_scan_plan(sid: str, case_id: str, client=None) -> dict:
    """Ask the scan for its geometry and keep it with the case."""
    from .costar.matterport import MatterportClient
    plan = scan_plan((client or MatterportClient()).graph(Q_PLAN, id=sid)["model"])
    path = plan_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan))
    return plan


def load_plan(case_id: str) -> dict | None:
    path = plan_path(case_id)
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def ensure_scan_plan(case_id: str, store, client=None) -> None:
    """Fetch the scan's geometry once for a case that was seeded before plans existed. Never raises."""
    if (load_plan(case_id) or {}).get("rooms"):
        return
    sid = (store.get_home(case_id) or {}).get("sid")
    if not sid or (client is None and not config.env("MATTERPORT_API_KEY")):
        return
    try:
        fetch_scan_plan(sid, case_id, client)
    except Exception as exc:
        print(f"ashline: no plan geometry from the scan ({exc})")


def floor_spot(camera: dict, face: int, bbox: list, size: tuple[int, int], outline: Polygon | None,
               height: float = CAMERA_HEIGHT, others: list[Polygon] | None = None) -> tuple[float, float, bool]:
    """Where on the plan an object seen in one cube face stands: (x, y, exact).

    The bottom centre of the box is taken as the point where the object meets the floor. `exact`
    is False when that cannot be used (the object hangs on a wall, or the floor point would fall
    outside the room) and the wall in that direction is used instead.
    """
    angle = camera["yaw"] - (int(face or 1) - 1) * math.pi / 2
    forward = (math.cos(angle), math.sin(angle))
    right = (math.sin(angle), -math.cos(angle))
    x0, _y0, x1, y1 = bbox
    a = (x0 + x1) / size[0] - 1                             # -1 left edge .. +1 right edge
    b = 2 * y1 / size[1] - 1                                # -1 top .. +1 bottom; the horizon is 0
    ray = (forward[0] + a * right[0], forward[1] + a * right[1])
    length = math.hypot(*ray)
    unit = (ray[0] / length, ray[1] / length)
    reach = height / b * length if b > 0.08 else None       # metres along the floor to the object's foot
    wall = _to_wall(camera, unit, outline)
    exact = reach is not None and (wall is None or reach <= wall * WALL_MARGIN)
    if not exact and reach is not None and reach <= THROUGH_DOOR and b >= 0.25:
        # Seen through a doorway: the floor point lies in a neighbouring room.
        foot = Point(camera["x"] + unit[0] * reach, camera["y"] + unit[1] * reach)
        exact = any(room.contains(foot) for room in others or [])
    if exact:
        distance = reach
    elif wall is None or wall < 0.4:                         # outdoors, or the camera stood in a doorway
        distance = min(reach or 1.5, 1.5)
    else:
        distance = wall * WALL_MARGIN
    return camera["x"] + unit[0] * distance, camera["y"] + unit[1] * distance, exact


def _to_wall(camera: dict, unit: tuple[float, float], outline: Polygon | None) -> float | None:
    if outline is None:
        return None
    origin = Point(camera["x"], camera["y"])
    far = LineString([(camera["x"], camera["y"]), (camera["x"] + unit[0] * 60, camera["y"] + unit[1] * 60)])
    hit = far.intersection(outline.exterior)
    if hit.is_empty:
        return None
    points = [hit] if hit.geom_type == "Point" else [g for g in getattr(hit, "geoms", []) if g.geom_type == "Point"]
    distances = [origin.distance(p) for p in points if origin.distance(p) > 0.05]
    return min(distances) if distances else None


# ----------------------------------------------------------------- sketch
def room_size(name: str) -> tuple[float, float]:
    low = (name or "").lower()
    return next((size for words, size in ROOM_SIZES if any(w in low for w in words)), DEFAULT_SIZE)


def pack_rooms(rooms: list[dict]) -> list[dict]:
    """Rectangles placed in rows, in the order given, so rooms walked through in turn sit side by side."""
    total = sum(r["w"] * r["d"] for r in rooms)
    row_width = max(math.sqrt(total) * 1.5, max((r["w"] for r in rooms), default=0))
    x = y = row_depth = 0.0
    placed = []
    for room in rooms:
        if x > 0 and x + room["w"] > row_width + 0.01:
            x, y, row_depth = 0.0, y + row_depth, 0.0
        placed.append({**room, "x": x, "y": y})
        x += room["w"]
        row_depth = max(row_depth, room["d"])
    return placed


def _overlap(a: dict, b: dict) -> bool:
    return (min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]) > 0.05
            and min(a["y"] + a["d"], b["y"] + b["d"]) - max(a["y"], b["y"]) > 0.05)


def sketch_plan(names: list[str], drawn: dict | None = None) -> dict:
    """Rectangular rooms for a home with no scan.

    `names` are the rooms in the order to lay them out. `drawn` is an optional model sketch:
    {rooms: [{name, width_m, depth_m, x_m, y_m}]}. Its sizes are used when sensible; its positions
    only when no two rooms overlap, otherwise the rooms are packed in rows.
    """
    by_name = {}
    for room in (drawn or {}).get("rooms") or []:
        key = (tidy_room(room.get("name")) or "").lower()
        if key:
            by_name[key] = room
    rooms, positioned = [], True
    for name in names:
        guess = by_name.get(name.lower()) or {}
        w, d = guess.get("width_m"), guess.get("depth_m")
        if not (isinstance(w, (int, float)) and isinstance(d, (int, float)) and 0.8 <= w <= 15 and 0.8 <= d <= 15):
            w, d = room_size(name)
        room = {"name": name, "w": float(w), "d": float(d)}
        if isinstance(guess.get("x_m"), (int, float)) and isinstance(guess.get("y_m"), (int, float)):
            room.update(x=float(guess["x_m"]), y=float(guess["y_m"]))
        else:
            positioned = False
        rooms.append(room)
    if not positioned or any(_overlap(a, b) for n, a in enumerate(rooms) for b in rooms[n + 1:]):
        rooms = pack_rooms([{k: r[k] for k in ("name", "w", "d")} for r in rooms])
    return {"kind": "sketch", "floors": [], "cameras": {},
            "rooms": [{"id": f"sketch-{n}", "name": r["name"], "floor": None,
                       "points": [[r["x"], r["y"]], [r["x"] + r["w"], r["y"]],
                                  [r["x"] + r["w"], r["y"] + r["d"]], [r["x"], r["y"] + r["d"]]]}
                      for n, r in enumerate(rooms)]}


def wall_spots(count: int, middle: list[bool]) -> list[tuple[float, float]]:
    """Positions inside a unit room for `count` things in the order they were seen.

    Things that stand against walls go round the walls clockwise from the top left; things that
    stand in the middle (a rug, a coffee table) are spread across the centre.
    """
    along = [n for n in range(count) if not middle[n]]
    centre = [n for n in range(count) if middle[n]]
    spots: dict[int, tuple[float, float]] = {}
    inset = 0.14
    for k, n in enumerate(along):
        t = (k + 0.5) / len(along) * 4                       # 0..4 round the four walls
        side, f = int(t) % 4, t - int(t)
        f = inset + f * (1 - 2 * inset)
        spots[n] = [(f, inset), (1 - inset, f), (1 - f, 1 - inset), (inset, 1 - f)][side]
    for k, n in enumerate(centre):
        spots[n] = (0.5 + (k - (len(centre) - 1) / 2) * min(0.22, 0.5 / max(len(centre), 1)), 0.5)
    return [spots[n] for n in range(count)]


def _stands_in_middle(label: str) -> bool:
    return bool(MIDDLE_WORDS & set(re.findall(r"[a-z]+", (label or "").lower())))


# ------------------------------------------------------------------ build
def _lines(store, case_id: str) -> list[dict]:
    """Inventory lines (one per item, not per sighting), rejected ones left out."""
    return [i for i in store.get_items(case_id)
            if (i["cluster_id"] or i["item_id"]) == i["item_id"] and i["status"] != "rejected"]


def room_order(store, case_id: str, lines: list[dict]) -> list[str]:
    """Room names for a sketch: first in the order photos and video frames showed them, then the rest."""
    numbered = sorted((p for p in store.get_photos(case_id, source="photo") if p.get("room_label")),
                      key=lambda p: p.get("number") or 0)
    names = [tidy_room(p["room_label"]) for p in numbered]
    names += [tidy_room(r.get("name") or r.get("label")) for r in store.get_rooms(case_id)]
    names += [tidy_room(i.get("room_label")) for i in lines]
    seen, out = set(), []
    for name in names:
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def _image_size(path: str | None, cache: dict) -> tuple[int, int]:
    if path not in cache:
        try:
            from PIL import Image
            with Image.open(path) as image:
                cache[path] = image.size
        except Exception:
            cache[path] = (2048, 2048)
    return cache[path]


def _fill_points(outline: Polygon, count: int) -> list[tuple[float, float]]:
    """`count` spread-out points inside a room, for items whose exact spot is unknown."""
    if count <= 0:
        return []
    x0, y0, x1, y1 = outline.bounds
    step = max(math.sqrt(max(outline.area, 0.5) / (count * 1.6)), 0.25)
    inside = outline.buffer(-min(0.3, step / 2))
    inside = inside if not inside.is_empty else outline
    points, y = [], y0 + step / 2
    while y < y1:
        x = x0 + step / 2
        while x < x1:
            if inside.contains(Point(x, y)):
                points.append((x, y))
            x += step
        y += step
    if len(points) < count:
        centre = outline.representative_point()
        points += [(centre.x, centre.y)] * (count - len(points))
    stride = len(points) / count
    return [points[int(n * stride)] for n in range(count)]


def build(case_id: str, store, drawn: dict | None = None) -> dict:
    """The plan and the pins for a case, ready to draw. `drawn` is an optional model sketch."""
    lines = _lines(store, case_id)
    ensure_scan_plan(case_id, store)
    plan = load_plan(case_id)
    if not plan or not plan.get("rooms"):
        if drawn is None:
            drawn = load_sketch(case_id)
        plan = sketch_plan(room_order(store, case_id, lines), drawn)
    names = {r["room_id"]: r.get("name") or r.get("label") for r in store.get_rooms(case_id)}
    photos = {p["photo_id"]: p for p in store.get_photos(case_id)}

    # Floors side by side, each moved so the drawing starts at the origin; y flipped for the scan
    # (its y axis points up, a drawing's points down).
    flip = -1 if plan["kind"] == "scan" else 1
    floors = [f["id"] for f in plan.get("floors") or []] or [None]
    floors += sorted({r["floor"] for r in plan["rooms"] if r["floor"] not in floors}, key=str)
    shift, cursor, height = {}, 0.0, 0.0
    for floor in floors:
        points = [p for r in plan["rooms"] if r["floor"] == floor for p in r["points"]]
        if not points:
            continue
        xs, ys = [p[0] for p in points], [flip * p[1] for p in points]
        shift[floor] = (cursor - min(xs), -min(ys))
        cursor += max(xs) - min(xs) + FLOOR_GAP
        height = max(height, max(ys) - min(ys))

    def to_page(x: float, y: float, floor) -> tuple[float, float]:
        dx, dy = shift.get(floor, (0.0, 0.0))
        return round(x + dx, 2), round(flip * y + dy, 2)

    rooms, outlines = [], {}
    for room in plan["rooms"]:
        outline = Polygon(room["points"])
        if not outline.is_valid:
            outline = outline.buffer(0)
        outlines[room["id"]] = outline
        name = room.get("name") or names.get(room["id"]) or ""
        centre = outline.representative_point()
        rooms.append({"id": room["id"], "name": name, "floor": room["floor"], "area": round(outline.area, 1),
                      "points": [list(to_page(x, y, room["floor"])) for x, y in room["points"]],
                      "label_at": list(to_page(centre.x, centre.y, room["floor"]))})
    by_name = {(r["name"] or "").lower(): r for r in rooms if r["name"]}
    floor_of = {r["id"]: r["floor"] for r in rooms}

    pins, loose, sizes = [], {}, {}
    drawn_spots = {s.get("id"): s for s in (drawn or {}).get("items") or [] if isinstance(s, dict)}
    for item in lines:
        photo = photos.get(item.get("photo_id")) or {}
        camera = (plan.get("cameras") or {}).get(photo.get("pano_id"))
        bbox = item.get("bbox")
        bbox = json.loads(bbox) if isinstance(bbox, str) else bbox
        pin = {"id": item["item_id"], "label": item["label"], "category": item.get("category") or "other",
               "status": item["status"], "qty": item.get("qty") or 1,
               "value": round(float(item.get("price_mid") or 0) * (item.get("qty") or 1)),
               "room": tidy_room(item.get("room_label")) or "", "photo": bool(item.get("photo_id") and bbox)}
        if camera and bbox and len(bbox) == 4:
            room_id = camera.get("room") or item.get("room_id")
            same_floor = [o for rid, o in outlines.items() if floor_of.get(rid) == camera.get("floor")]
            x, y, exact = floor_spot(camera, photo.get("face") or 1, bbox, _image_size(photo.get("path"), sizes),
                                     outlines.get(room_id), others=same_floor)
            px, py = to_page(x, y, camera.get("floor") or floor_of.get(room_id))
            pins.append({**pin, "x": px, "y": py, "exact": exact, "room_id": room_id})
            continue
        room = (next((r for r in rooms if r["id"] == item.get("room_id")), None)
                or by_name.get(pin["room"].lower()))
        if room:
            loose.setdefault(room["id"], []).append(pin)
        else:
            pins.append({**pin, "x": None, "y": None, "exact": False, "room_id": None})

    # Items known only by their room: round the walls in the order seen (sketch), or spread over the floor (scan).
    for room in rooms:
        waiting = loose.get(room["id"]) or []
        if not waiting:
            continue
        outline = outlines[room["id"]]
        if plan["kind"] == "sketch":
            x0, y0, x1, y1 = outline.bounds
            spots = wall_spots(len(waiting), [_stands_in_middle(p["label"]) for p in waiting])
            for pin, (u, v) in zip(waiting, spots):
                spot = drawn_spots.get(pin["id"]) or {}
                if all(isinstance(spot.get(k), (int, float)) and 0 <= spot[k] <= 1 for k in ("u", "v")):
                    u, v = 0.08 + spot["u"] * 0.84, 0.08 + spot["v"] * 0.84
                px, py = to_page(x0 + u * (x1 - x0), y0 + v * (y1 - y0), room["floor"])
                pins.append({**pin, "x": px, "y": py, "exact": False, "room_id": room["id"]})
        else:
            for pin, (x, y) in zip(waiting, _fill_points(outline, len(waiting))):
                px, py = to_page(x, y, room["floor"])
                pins.append({**pin, "x": px, "y": py, "exact": False, "room_id": room["id"]})

    labels = {f["id"]: f.get("label") for f in plan.get("floors") or []}
    return {"kind": plan["kind"], "note": SCAN_NOTE if plan["kind"] == "scan" else SKETCH_NOTE,
            "width": round(max(cursor - FLOOR_GAP, 1.0), 2) if shift else 1.0, "height": round(max(height, 1.0), 2),
            "floors": [{"id": f, "label": labels.get(f) or "", "x": round(shift[f][0], 2)} for f in floors if f in shift]
            if len(shift) > 1 else [],
            "rooms": rooms, "pins": pins, "unplaced": sum(1 for p in pins if p["x"] is None)}


# ----------------------------------------------------- model sketch, kept
def sketch_path(case_id: str) -> Path:
    return config.case_dir(case_id) / "sketch.json"


def load_sketch(case_id: str) -> dict | None:
    path = sketch_path(case_id)
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def save_sketch(case_id: str, drawn: dict) -> None:
    path = sketch_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(drawn, indent=1))


# ------------------------------------------------------------ as a picture
CATEGORY_COLOURS = {"furniture": "#8a5a2b", "electronics": "#2f6fb0", "appliance": "#5b6b7a", "appliances": "#5b6b7a",
                    "kitchen": "#b0562f", "lighting": "#c79a1e", "decor": "#8c4fa0", "clothing": "#3d8f6d",
                    "tools": "#6b6b3a", "other": "#666666"}


def render_png(layout: dict, path: str | Path, width: int = 1400) -> Path:
    """The plan as an image, for sending in a chat."""
    from PIL import Image, ImageDraw, ImageFont
    pad = 50
    scale = (width - 2 * pad) / max(layout["width"], 1)
    height = int(layout["height"] * scale + 2 * pad + 40)
    image = Image.new("RGB", (width, height), "#fbfaf7")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()

    def at(x, y):
        return pad + x * scale, pad + y * scale

    for room in layout["rooms"]:
        draw.polygon([at(*p) for p in room["points"]], fill="#f1ede4", outline="#3b3a36", width=3)
    for pin in layout["pins"]:
        if pin["x"] is None:
            continue
        x, y = at(pin["x"], pin["y"])
        r = 7
        draw.ellipse([x - r, y - r, x + r, y + r], fill=CATEGORY_COLOURS.get(pin["category"], "#666666"),
                     outline="white", width=2)
    for room in layout["rooms"]:
        if room["name"]:
            x, y = at(*room["label_at"])
            box = draw.textbbox((0, 0), room["name"], font=font)
            draw.text((x - (box[2] - box[0]) / 2, y - 12), room["name"], fill="#1f1e1b", font=font)
    draw.text((pad, height - 34), layout["note"], fill="#6b6a65", font=small)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
