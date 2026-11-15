"""A 2D plan of the home with a pin where each item was recorded.

Two sources, one output shape:

- **Scan** (`kind: "scan"`): the Matterport scan gives each room's outline in metres, its doorways,
  and where every panorama stood and which way it faced. An item seen in a panorama is placed
  where the bottom of its box meets the floor, by projecting that pixel from the camera onto the
  floor. It is never placed beyond the room's wall, except in the next room when the line of
  sight passes through a doorway. Verified on the Model API on 15 Nov 2026: cube face `f` (1-4)
  looks along `yaw - (f - 1) * 90 degrees`, yaw being the panorama rotation about the vertical
  axis; a doorway's `relativeCenter` is measured from the wall segment's first vertex; and a
  doorway's `lowerElevation` is the floor height, which puts the cameras 1.41-1.45 m above it.
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
from .pricing import in_use
from .rooms import tidy_room

BASIS_WORDS = {"family": "yours", "paid": "paid", "online": "online", "estimate": "estimate"}   # which price a pin shows

CAMERA_HEIGHT = 1.43            # metres above the floor: what scan tripods measured at (1.41-1.45) when no floor height is known
DOOR_REACH = 0.2                # metres: a line of sight this close to a doorway counts as passing through it
WALL_MARGIN = 0.93              # an item is never placed past this share of the way to the wall
THROUGH_DOOR = 6.0               # metres: how far an item seen in the next room is still placed by its floor point
CAMERA_MARGIN = 1.2              # metres added round a room's camera spots when the scan has no walls for it
FLOOR_GAP = 2.0                 # metres between floors drawn side by side
SKETCH_NOTE = "A sketch from your photos: rooms and positions are approximate, not to scale."
SKETCH_ROOMS_NOTE = ("A sketch from your photos: rooms are approximate, not to scale. Things are listed by room; "
                     "a dot appears once its place has been worked out from the photos.")
SCAN_NOTE = "Room outlines come from the 3D scan. Item positions are estimated from where each was seen."
APPROX_NOTE = ("This scan has no measured walls, so rooms are drawn roughly from where the scanner stood. "
               "Item positions are estimated from where each was seen.")

Q_PLAN = """query Plan($id: ID!) { model(id: $id) {
  floors { id label sequence }
  rooms { id floor { id } boundary { edges { vertices { id position { x y } }
          openings { type width relativeCenter lowerElevation } } } }
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
# Large things get a footprint on the plan when it is zoomed in: (words in the label, short name,
# typical width, typical depth) in metres. With a scan the width is measured and only the depth
# is typical; in a sketch both are. First match wins, so "coffee table" comes before "table".
FOOTPRINTS = [
    (("sectional",), "Sectional", 2.7, 1.6), (("loveseat",), "Loveseat", 1.5, 0.9), (("sofa", "couch"), "Sofa", 2.1, 0.95),
    (("bunk bed",), "Bunk bed", 2.0, 1.0), (("crib",), "Crib", 1.4, 0.75), (("mattress", "bed"), "Bed", 1.6, 2.0),
    (("coffee table",), "Coffee table", 1.1, 0.6), (("dining table",), "Dining table", 1.7, 0.95),
    (("desk",), "Desk", 1.3, 0.65), (("dresser", "chest of drawers"), "Dresser", 1.2, 0.5),
    (("media console", "tv stand", "console", "credenza", "sideboard", "buffet"), "Console", 1.5, 0.45),
    (("bookcase", "bookshelf", "shelving"), "Shelves", 0.9, 0.35), (("wardrobe", "armoire"), "Wardrobe", 1.2, 0.6),
    (("refrigerator", "fridge"), "Fridge", 0.9, 0.75), (("range", "stove", "oven"), "Range", 0.76, 0.68),
    (("washing machine", "washer"), "Washer", 0.7, 0.7), (("dryer",), "Dryer", 0.7, 0.7), (("freezer",), "Freezer", 0.8, 0.7),
    (("piano",), "Piano", 1.5, 0.65), (("treadmill",), "Treadmill", 0.9, 1.9), (("recliner", "armchair"), "Armchair", 0.9, 0.9),
    (("table",), "Table", 1.2, 0.8), (("bench",), "Bench", 1.2, 0.45), (("cabinet",), "Cabinet", 0.9, 0.45),
]
NOT_FOOTPRINTS = ("lamp", "pillow", "cushion", "cover", "sheet", "blanket", "throw", "frame", "print", "tray", "mat",
                  "runner", "cloth", "toy", "remote", "top", "microwave", "toaster", "countertop", "dollhouse", "model")
#                 "bed pillow", "table lamp", "desk mat", "microwave oven" are not furniture standing on the floor
MIN_FOOTPRINT = 0.6             # metres: narrower things stay dots
WALL_SNAP = 0.7                 # metres beyond its own depth within which a thing is drawn standing against the wall


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
    """Room outlines and camera spots from a Model API reply to `Q_PLAN`.

    Checked on 15 Nov 2026: residential scans carry wall segments for most rooms, with one to four
    rooms per home left without (often outdoor areas); commercial scans often carry none. A room
    without walls is approximated from where the scan's cameras stood in it (`camera_rooms`),
    unless that rectangle would lie over a measured room. A plan with no measured room at all is
    marked `approx`.
    """
    floors = sorted(model.get("floors") or [], key=lambda f: f.get("sequence") or 0)
    cameras = {}
    for pano in model.get("panoLocations") or []:
        anchor = pano.get("anchor") or {}
        cameras[pano["id"]] = {"x": pano["position"]["x"], "y": pano["position"]["y"], "z": pano["position"].get("z"),
                               "yaw": _yaw(pano.get("rotation")),
                               "room": (anchor.get("room") or {}).get("id"), "floor": (anchor.get("floor") or {}).get("id")}
    rooms, missing, floor_z = [], [], {}
    for room in model.get("rooms") or []:
        edges = ((room.get("boundary") or {}).get("edges")) or []
        points = ring(edges)
        floor = (room.get("floor") or {}).get("id")
        if len(points) >= 3:
            found, heights = doorways(edges)
            if heights:
                floor_z[room["id"]] = sorted(heights)[len(heights) // 2]
            rooms.append({"id": room["id"], "floor": floor, "points": [[round(x, 3), round(y, 3)] for x, y in points],
                          "doors": found})
        else:
            missing.append({"id": room["id"], "floor": floor})
    for camera in cameras.values():                         # how high the lens stood above that room's floor
        z, floor_at = camera.pop("z", None), floor_z.get(camera["room"])
        if z is not None and floor_at is not None and 1.0 <= z - floor_at <= 1.9:
            camera["h"] = round(z - floor_at, 3)
    measured = [(r["floor"], Polygon(r["points"]).buffer(0)) for r in rooms]
    guessed = []
    for room in camera_rooms(missing, cameras):
        box = Polygon(room["points"])
        covered = sum(box.intersection(outline).area for floor, outline in measured if floor == room["floor"])
        if covered <= 0.2 * box.area:                       # not lying over rooms the scan did measure
            guessed.append(room)
    return {"kind": "scan", "approx": bool(guessed) and not rooms, "floors": [{"id": f["id"], "label": f.get("label") or ""} for f in floors],
            "rooms": rooms + guessed, "cameras": cameras}


def doorways(edges: list[dict]) -> tuple[list, list[float]]:
    """A room's doorways as wall segments [[x0, y0], [x1, y1]], and the floor heights they report."""
    found, heights = [], []
    for edge in edges or []:
        ends = edge.get("vertices") or []
        if len(ends) < 2:
            continue
        ax, ay = ends[0]["position"]["x"], ends[0]["position"]["y"]
        bx, by = ends[1]["position"]["x"], ends[1]["position"]["y"]
        length = math.hypot(bx - ax, by - ay)
        for opening in edge.get("openings") or []:
            if opening.get("type") != "doorway" or not length or opening.get("relativeCenter") is None:
                continue
            centre = float(opening["relativeCenter"])
            half = min(float(opening.get("width") or 0.8) / 2 / length, 0.5)
            t0, t1 = max(0.0, centre - half), min(1.0, centre + half)
            found.append([[round(ax + t0 * (bx - ax), 3), round(ay + t0 * (by - ay), 3)],
                          [round(ax + t1 * (bx - ax), 3), round(ay + t1 * (by - ay), 3)]])
            if opening.get("lowerElevation") is not None:
                heights.append(float(opening["lowerElevation"]))
    return found, heights


def camera_rooms(rooms: list[dict], cameras: dict, margin: float = CAMERA_MARGIN) -> list[dict]:
    """Rough rectangles for rooms the scan gives no walls for: the area its cameras covered, plus a margin.

    A scan operator sets the camera down every couple of metres and not against a wall, so the
    cameras of a room span most of it. Rectangles of neighbouring rooms are cut back where they
    would overlap. Rooms no camera stood in are left out.
    """
    boxes = []
    for room in rooms:
        spots = [(c["x"], c["y"]) for c in cameras.values() if c["room"] == room["id"]]
        if not spots:
            continue
        xs, ys = [p[0] for p in spots], [p[1] for p in spots]
        boxes.append({**room, "x0": min(xs) - margin, "x1": max(xs) + margin, "y0": min(ys) - margin, "y1": max(ys) + margin})
    for _ in range(2):
        for n, a in enumerate(boxes):
            for b in boxes[n + 1:]:
                if a["floor"] != b["floor"]:
                    continue
                over_x = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                over_y = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
                if over_x <= 0 or over_y <= 0:
                    continue
                axis, lo, hi = ("x", "x0", "x1") if over_x <= over_y else ("y", "y0", "y1")
                first, second = (a, b) if a[lo] + a[hi] <= b[lo] + b[hi] else (b, a)      # by centre along that axis
                cut = (max(a[lo], b[lo]) + min(a[hi], b[hi])) / 2
                first[hi], second[lo] = min(first[hi], cut), max(second[lo], cut)
    return [{"id": b["id"], "floor": b["floor"], "approx": True,
             "points": [[round(b["x0"], 3), round(b["y0"], 3)], [round(b["x1"], 3), round(b["y0"], 3)],
                        [round(b["x1"], 3), round(b["y1"], 3)], [round(b["x0"], 3), round(b["y1"], 3)]]}
            for b in boxes if b["x1"] - b["x0"] > 0.3 and b["y1"] - b["y0"] > 0.3]


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
               height: float | None = None, others: list[Polygon] | None = None,
               doors: list | None = None) -> tuple[float, float, bool]:
    """Where on the plan an object seen in one cube face stands: (x, y, exact).

    The bottom centre of the box is taken as the point where the object meets the floor. `exact`
    is False when that cannot be used (the object hangs on a wall or stands on furniture, so the
    floor point would fall outside the room) and the wall in that direction is used instead.
    A floor point in another room (`others`) is kept only when the line of sight passes through
    one of the camera's room's doorways (`doors`); a television on a console near a wall would
    otherwise be "seen" in the room behind that wall.
    """
    height = height or camera.get("h") or CAMERA_HEIGHT
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
    exact = reach is not None and (wall is None or reach <= wall)
    foot_beyond = False
    if not exact and reach is not None and reach <= THROUGH_DOOR and b >= 0.25 and doors:
        # Seen through a doorway: the floor point lies in a neighbouring room.
        foot = Point(camera["x"] + unit[0] * reach, camera["y"] + unit[1] * reach)
        sight = LineString([(camera["x"], camera["y"]), (foot.x, foot.y)])
        exact = foot_beyond = (any(sight.distance(LineString(door)) <= DOOR_REACH for door in doors)
                               and any(room.contains(foot) for room in others or []))
    if exact:
        distance = reach if wall is None or foot_beyond else min(reach, wall * WALL_MARGIN)
    elif wall is None or wall < 0.4:                         # outdoors, or the camera stood in a doorway
        distance = min(reach or 1.5, 1.5)
    else:
        distance = wall * WALL_MARGIN
    return camera["x"] + unit[0] * distance, camera["y"] + unit[1] * distance, exact


def typical(label: str) -> tuple[str, float, float] | None:
    """(short name, width, depth) for a large piece of furniture or an appliance, by its label; else None."""
    low = (label or "").lower()
    words = set(re.findall(r"[a-z]+", low))
    if words & set(NOT_FOOTPRINTS):
        return None
    for keys, short, width, depth in FOOTPRINTS:
        if any((key in low) if " " in key else (key in words) for key in keys):
            return short, width, depth
    return None


def seen_width(camera: dict, face: int, bbox: list, size: tuple[int, int], x: float, y: float) -> float:
    """How wide a thing is, in metres, from how wide it looks and how far away it stands."""
    angle = camera["yaw"] - (int(face or 1) - 1) * math.pi / 2
    forward = (math.cos(angle), math.sin(angle))
    ahead = (x - camera["x"]) * forward[0] + (y - camera["y"]) * forward[1]       # distance along the face's own axis
    return abs(ahead) * 2 * (bbox[2] - bbox[0]) / size[0]


def footprint(x: float, y: float, width: float, depth: float, outline: Polygon | None,
              toward: tuple[float, float] | None = None) -> list | None:
    """A rectangle for a thing whose front edge is centred on (x, y): four corners, or None.

    Near a wall it stands against that wall, as furniture does. Otherwise, with `toward` (the
    direction the camera looked along) it faces the camera; without, it lies square to the page.
    A rectangle that would leave the room is not drawn.
    """
    along, back, centre = (1.0, 0.0), (0.0, 1.0), None
    if toward:
        norm = math.hypot(*toward) or 1.0
        back = (toward[0] / norm, toward[1] / norm)
        along = (-back[1], back[0])
    if outline is not None:
        ring = list(outline.exterior.coords)
        nearest = min(((LineString([a, b]).distance(Point(x, y)), a, b) for a, b in zip(ring, ring[1:])), default=None)
        if nearest and nearest[0] <= depth + WALL_SNAP:
            gap, a, b = nearest
            length = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            along = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            t = (x - a[0]) * along[0] + (y - a[1]) * along[1]
            foot = (a[0] + t * along[0], a[1] + t * along[1])                 # the point on the wall behind the thing
            inward = (x - foot[0], y - foot[1])
            norm = math.hypot(*inward)
            inward = (inward[0] / norm, inward[1] / norm) if norm > 1e-6 else (-along[1], along[0])
            if not outline.contains(Point(foot[0] + inward[0] * 0.1, foot[1] + inward[1] * 0.1)):
                inward = (-inward[0], -inward[1])
            centre = (foot[0] + inward[0] * (depth / 2 + 0.04), foot[1] + inward[1] * (depth / 2 + 0.04))
            back = inward
    if centre is None:
        centre = (x + back[0] * depth / 2, y + back[1] * depth / 2) if toward else (x, y)
    corners = [(centre[0] + sx * along[0] * width / 2 + sy * back[0] * depth / 2,
                centre[1] + sx * along[1] * width / 2 + sy * back[1] * depth / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    if outline is not None and not outline.buffer(0.25).contains(Polygon(corners)):
        return None
    return corners


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


TOUR = "https://my.matterport.com/show/?m={sid}&play=1&qs=1&brand=0"      # the scan's own 3D viewer, as the sponsor's guide embeds it


def tour_url(sid: str | None, sweep: str | None = None) -> str | None:
    """The address of the walkable 3D scan, starting at one panorama when `sweep` (its number) is given."""
    if not sid or not re.fullmatch(r"[A-Za-z0-9]{6,20}", sid):
        return None
    return TOUR.format(sid=sid) + (f"&ss={sweep}" if sweep and str(sweep).isdigit() else "")


def sweep_of(photo: dict | None) -> str | None:
    """A scan photo's panorama number, from its id ("scan-13-f2" is panorama 13)."""
    found = re.fullmatch(r"scan-(\d+)-f\d", (photo or {}).get("photo_id") or "")
    return found.group(1) if found else None


def home_records(case_id: str, store) -> list[dict]:
    """What was found for this address from before the loss: [{what, detail}], for the family to see."""
    from .costar.matterport import _month_year
    home = store.get_home(case_id) or {}
    out = []
    if home.get("sid"):
        rooms = sum(1 for r in store.get_rooms(case_id) if r.get("source") == "scan")
        when = _month_year(home.get("scan_captured"))
        out.append({"what": "3D scan of the home", "detail": " · ".join(filter(None, [
            f"made in {when}" if when else None, f"{rooms} rooms" if rooms else None]))})
    if home.get("aerial_captured"):
        out.append({"what": "Aerial photo", "detail": f"taken in {_month_year(home['aerial_captured']) or home['aerial_captured']}"})
    listing = home.get("listing") if isinstance(home.get("listing"), dict) else None
    if listing:
        facts = [f"{listing['beds']:g} bed" if listing.get("beds") else None,
                 f"{listing['baths']:g} bath" if listing.get("baths") else None,
                 f"{int(listing['sqft']):,} sq ft" if listing.get("sqft") else None,
                 f"built {int(listing['year_built'])}" if listing.get("year_built") else None]
        out.append({"what": "Sale listing", "detail": " · ".join(f for f in facts if f)})
    return out


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

    rooms, outlines, doors_of = [], {}, {}
    for room in plan["rooms"]:
        outline = Polygon(room["points"])
        if not outline.is_valid:
            outline = outline.buffer(0)
        outlines[room["id"]] = outline
        name = room.get("name") or names.get(room["id"]) or ""
        centre = outline.representative_point()
        doors_of[room["id"]] = room.get("doors") or []
        rooms.append({"id": room["id"], "name": name, "floor": room["floor"], "area": round(outline.area, 1),
                      "points": [list(to_page(x, y, room["floor"])) for x, y in room["points"]],
                      "doors": [[list(to_page(x, y, room["floor"])) for x, y in door] for door in doors_of[room["id"]]],
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
               "room": tidy_room(item.get("room_label")) or "", "photo": bool(item.get("photo_id") and bbox),
               "basis": BASIS_WORDS.get(in_use(item), "")}
        if camera and bbox and len(bbox) == 4:
            room_id = camera.get("room") or item.get("room_id")
            same_floor = [o for rid, o in outlines.items() if floor_of.get(rid) == camera.get("floor")]
            x, y, exact = floor_spot(camera, photo.get("face") or 1, bbox, _image_size(photo.get("path"), sizes),
                                     outlines.get(room_id), others=same_floor, doors=doors_of.get(room_id))
            floor = camera.get("floor") or floor_of.get(room_id)
            px, py = to_page(x, y, floor)
            pin = {**pin, "x": px, "y": py, "exact": exact, "room_id": room_id if room_id in outlines else None,
                   "sweep": sweep_of(photo)}
            kind = typical(item["label"]) if exact else None      # only where the floor point is real
            if kind:
                wide = min(max(seen_width(camera, photo.get("face") or 1, bbox, _image_size(photo.get("path"), sizes), x, y), 0.3), 4.0)
                inside = next((o for o in same_floor if o.contains(Point(x, y))), outlines.get(room_id))
                shape = footprint(x, y, wide, kind[2], inside, toward=(x - camera["x"], y - camera["y"])) \
                    if wide >= MIN_FOOTPRINT else None
                if shape:
                    pin.update(shape=[list(to_page(sx, sy, floor)) for sx, sy in shape], short=kind[0], wide=round(wide, 1))
            pins.append(pin)
            continue
        room = (next((r for r in rooms if r["id"] == item.get("room_id")), None)
                or by_name.get(pin["room"].lower()))
        if room:
            loose.setdefault(room["id"], []).append(pin)
        else:
            pins.append({**pin, "x": None, "y": None, "exact": False, "room_id": None})

    # Items known only by their room get a dot only where a position is really known: a spot the
    # model gave after looking at the photos. Otherwise they are counted and listed with their
    # room and drawn nowhere; evenly spaced dots would look measured when they are not.
    for room in rooms:
        outline = outlines[room["id"]]
        x0, y0, x1, y1 = outline.bounds
        for pin in loose.get(room["id"]) or []:
            spot = drawn_spots.get(pin["id"]) or {}
            if plan["kind"] == "sketch" and all(isinstance(spot.get(k), (int, float)) and 0 <= spot[k] <= 1 for k in ("u", "v")):
                u, v = 0.08 + spot["u"] * 0.84, 0.08 + spot["v"] * 0.84
                x, y = x0 + u * (x1 - x0), y0 + v * (y1 - y0)
                px, py = to_page(x, y, room["floor"])
                pin = {**pin, "x": px, "y": py, "exact": False, "room_id": room["id"]}
                kind = typical(pin["label"])
                shape = footprint(x, y, kind[1], kind[2], outline) if kind else None
                if shape:                                   # a typical size for the kind of thing: as rough as the sketch
                    pin.update(shape=[list(to_page(sx, sy, room["floor"])) for sx, sy in shape], short=kind[0])
                pins.append(pin)
            else:
                pins.append({**pin, "x": None, "y": None, "exact": False, "room_id": room["id"]})

    labels = {f["id"]: f.get("label") for f in plan.get("floors") or []}
    dotted = sum(1 for p in pins if p["x"] is not None)
    note = ((SKETCH_NOTE if dotted else SKETCH_ROOMS_NOTE) if plan["kind"] == "sketch"
            else APPROX_NOTE if plan.get("approx") else SCAN_NOTE)
    return {"kind": plan["kind"], "note": note,
            "width": round(max(cursor - FLOOR_GAP, 1.0), 2) if shift else 1.0, "height": round(max(height, 1.0), 2),
            "floors": [{"id": f, "label": labels.get(f) or "", "x": round(shift[f][0], 2)} for f in floors if f in shift]
            if len(shift) > 1 else [],
            "rooms": rooms, "pins": pins, "dotted": dotted,
            "tour": tour_url((store.get_home(case_id) or {}).get("sid")) if plan["kind"] == "scan" else None,
            "records": home_records(case_id, store),
            "unplaced": sum(1 for p in pins if p["x"] is None and p["room_id"] is None)}


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
    for room in layout["rooms"]:                            # doorways: a gap in the wall
        for a, b in room.get("doors") or []:
            draw.line([at(*a), at(*b)], fill="#fbfaf7", width=5)
    for pin in layout["pins"]:
        if pin["x"] is None:
            continue
        x, y = at(pin["x"], pin["y"])
        r = 7
        draw.ellipse([x - r, y - r, x + r, y + r], fill=CATEGORY_COLOURS.get(pin["category"], "#666666"),
                     outline="white", width=2)
    held: dict = {}
    for pin in layout["pins"]:
        held[pin["room_id"]] = held.get(pin["room_id"], 0) + 1
    for room in layout["rooms"]:
        if room["name"]:
            x, y = at(*room["label_at"])
            box = draw.textbbox((0, 0), room["name"], font=font)
            draw.text((x - (box[2] - box[0]) / 2, y - 12), room["name"], fill="#1f1e1b", font=font)
            if held.get(room["id"]):
                count = f"{held[room['id']]} item{'s' if held[room['id']] != 1 else ''}"
                box = draw.textbbox((0, 0), count, font=small)
                draw.text((x - (box[2] - box[0]) / 2, y + 16), count, fill="#6b6a65", font=small)
    for floor in layout.get("floors") or []:
        if floor.get("label"):
            draw.text((at(floor["x"], 0)[0], pad - 30), f"Floor {floor['label']}", fill="#6b6a65", font=small)
    draw.text((pad, height - 34), layout["note"], fill="#6b6a65", font=small)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
