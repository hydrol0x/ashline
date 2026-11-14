"""Scan-specific additions to cross-photo de-duplication (build spec 6.4).

Neighbouring panoramas of one scan overlap heavily, so the same object is
detected several times. Two things are provided:

1. Priors for candidate pairing (`scan_prior`, `merge_threshold`): same
   room_id and panorama positions a few metres apart. Pairs from the same scan
   in the same room may auto-merge at a lower similarity than cross-family
   photos. The embedding-based matcher plugs these in.
2. `cluster_scan_items`: a text-and-geometry clustering used until crop
   embeddings exist. It works in three steps:
   a. An object cut by the seam between two faces of one panorama is listed
      once per face. Boxes that meet at the seam are joined into one object.
   b. Objects seen from different panoramas of the same room are linked when
      their labels describe the same thing (`same_thing`).
   c. Each linked group is one inventory line. Its quantity is the largest
      number of its objects seen from any single panorama, so four stools
      seen from two panoramas are four stools, not eight.
"""
from __future__ import annotations

import math
import re

from PIL import Image

from .detect import head_phrase

BASE_THRESHOLD = 0.90           # cross-family photos: only very confident pairs auto-merge
SCAN_ROOM_THRESHOLD = 0.75      # same scan, same room, panoramas close together
NEAR_M = 4.0

SEAM_EDGE = 0.03                # a box within 3% of a face's side edge is cut by the seam
SEAM_VERTICAL_OVERLAP = 0.5
SEAM_FRAGMENT = 0.12            # a box narrower than this share of the face is a cut-off fragment
SEAM_EDGE_MATCH = 0.06          # two wide halves must line up at top and bottom within this share
LINK_OVERLAP = 0.7              # share of the shorter label's words found in the other

SYNONYMS = {
    "couch": "sofa", "loveseat": "sofa", "sectional": "sofa", "settee": "sofa",
    "television": "tv", "fridge": "refrigerator", "barstool": "stool",
    "drape": "curtain", "carpet": "rug", "bookcase": "bookshelf", "shelve": "shelf",
    "picture": "photo", "wooden": "wood", "recliner": "reclining", "grey": "gray",
    "comforter": "bedspread", "quilt": "bedspread", "duvet": "bedspread", "clothe": "clothing",
}
# Colours that lighting or wording can turn into one another share a family.
COLOR_FAMILIES = [
    {"black", "charcoal", "gray", "silver"},
    {"white", "cream", "ivory", "beige"},
    {"brown", "tan", "bronze", "beige"},
    {"yellow", "gold"},
    {"red", "maroon", "orange"},
    {"pink", "magenta", "purple", "lavender"},
    {"blue", "navy", "teal"},
    {"green", "teal"},
]
COLORS = set().union(*COLOR_FAMILIES)
STOPWORDS = {"a", "an", "the", "and", "or", "with", "on", "in", "of", "for", "to", "at", "by", "from",
             "top", "side", "small", "large", "big", "second", "another", "pair", "set", "dark", "light"}
# Words that end a label without naming the thing: "curtain panel", "loveseat piece".
GENERIC_TAILS = {"panel", "piece", "set", "collection", "item", "unit", "print", "decor"}
QUANTITY_LEAD = re.compile(r"^(?:pair|set|stack|pile|bag|box|collection|package|bunch) of ")


def scan_prior(a: dict, b: dict) -> dict:
    """What the scan knows about two photos: same scan, same room, distance between panoramas."""
    same_scan = bool(a.get("pano_id") and b.get("pano_id")) and a.get("source") == b.get("source") == "scan"
    distance = None
    if same_scan and None not in (a.get("x"), a.get("y"), b.get("x"), b.get("y")):
        distance = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
    return {
        "same_scan": same_scan,
        "same_room": same_scan and a.get("room_id") == b.get("room_id"),
        "same_pano": same_scan and a.get("pano_id") == b.get("pano_id"),
        "pano_distance_m": distance,
    }


def merge_threshold(a: dict, b: dict, base: float = BASE_THRESHOLD) -> float | None:
    """Similarity needed to auto-merge items from photos a and b. None = never merge.

    Faces of one panorama look in different directions and do not overlap, so
    two detections from the same panorama are two objects.
    """
    prior = scan_prior(a, b)
    if prior["same_pano"]:
        return None
    if prior["same_scan"] and not prior["same_room"]:
        return base + 0.05          # different rooms of one scan: almost never the same object
    if prior["same_room"] and (prior["pano_distance_m"] is None or prior["pano_distance_m"] <= NEAR_M):
        return SCAN_ROOM_THRESHOLD
    return base


def _clean(label: str) -> str:
    """Lower-case label without remarks in brackets or punctuation: 'pillow (left)' -> 'pillow'."""
    text = re.sub(r"\([^)]*\)", " ", (label or "").lower())
    text = re.sub(r"[^a-z0-9/ ]", " ", text.replace("-", " "))
    return re.sub(r"\s+", " ", text).strip()


def _word(word: str) -> str:
    """Singular form with common synonyms folded together."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("xes", "ches", "shes", "sses")):
        word = word[:-2]                    # boxes -> box
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-1]                    # striped -> stripe
    return SYNONYMS.get(word, word)


def label_words(label: str) -> tuple[set, set]:
    """(content words, colour words) of a label."""
    words = {_word(w) for w in _clean(label).replace("/", " ").split() if len(w) > 1}
    words -= STOPWORDS | GENERIC_TAILS
    return {w for w in words if w not in COLORS}, {w for w in words if w in COLORS}


def item_key(label: str) -> str:
    """The noun a label is about: 'gray blackout curtain panel' -> 'curtain'."""
    text = _clean(label).split("/")[0].strip()          # "bedspread / comforter": the first name
    text = QUANTITY_LEAD.sub("", text)                  # "pair of sneakers" is about sneakers
    words = [_word(w) for w in head_phrase(text).split()]
    while len(words) > 1 and words[-1] in GENERIC_TAILS:
        words.pop()
    return words[-1] if words else ""


def same_thing(a: str, b: str) -> bool:
    """Do two labels, written from different viewpoints, describe the same object?

    They must be about the same noun, most words of the shorter label must
    appear in the longer one, and their colours must be compatible: lighting
    turns "black" into "dark gray", but a red bin is not a white bin.
    """
    words_a, colors_a = label_words(a)
    words_b, colors_b = label_words(b)
    if not words_a or not words_b:
        return False
    key_a, key_b = item_key(a), item_key(b)
    if not ((key_a and key_a in words_b) or (key_b and key_b in words_a)):
        return False
    if colors_a and colors_b and not any(colors_a & family and colors_b & family for family in COLOR_FAMILIES):
        return False
    return len(words_a & words_b) / min(len(words_a), len(words_b)) >= LINK_OVERLAP


def _width(photo: dict) -> int | None:
    if photo.get("width"):
        return int(photo["width"])
    try:
        with Image.open(photo["path"]) as image:
            return image.width
    except (KeyError, OSError):
        return None


def _meets_at_seam(left: dict, right: dict, width: int) -> bool:
    """Is `left` (cut by its face's right edge) the same object as `right` (cut by the next face's left edge)?

    A narrow sliver next to the seam is the cut-off end of the object on the
    other side. Two wide boxes are harder: they can be one object centred on
    the seam, or two similar things side by side (two sofas). They are joined
    only when their labels agree and their tops and bottoms line up.
    """
    a, b = left.get("bbox"), right.get("bbox")
    if not (a and b) or left.get("category") != right.get("category"):
        return False
    if a[2] < (1 - SEAM_EDGE) * width or b[0] > SEAM_EDGE * width:
        return False
    shared = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    if shorter <= 0 or shared / shorter < SEAM_VERTICAL_OVERLAP:
        return False
    if min(a[2] - a[0], b[2] - b[0]) < SEAM_FRAGMENT * width:
        return True
    lined_up = abs(a[1] - b[1]) <= SEAM_EDGE_MATCH * width and abs(a[3] - b[3]) <= SEAM_EDGE_MATCH * width
    return lined_up and same_thing(left["label"], right["label"])


def _best(members: list[dict]) -> dict:
    """The member whose wording should name the group: confident, complete view, specific."""
    return max(members, key=lambda i: (i.get("confidence") or 0, "(" not in i["label"], len(i["label"]), -i["item_id"]))


class _Groups:
    """Union-find over item ids."""

    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list]:
        out: dict = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def seam_objects(items: list[dict], photos: dict[str, dict]) -> list[list[dict]]:
    """Group detections of one panorama that are the same object cut by a face seam.

    Faces 1 to 4 turn right around the horizon, so the right edge of face k
    meets the left edge of face k+1, and face 4 wraps to face 1.
    """
    by_id = {i["item_id"]: i for i in items}
    groups = _Groups(by_id)
    by_face: dict[tuple, list[dict]] = {}
    for item in items:
        photo = photos.get(item["photo_id"]) or {}
        if photo.get("pano_id") and photo.get("face"):
            by_face.setdefault((photo["pano_id"], int(photo["face"])), []).append(item)
    for (pano, face), left_items in by_face.items():
        right_items = by_face.get((pano, face % 4 + 1), [])
        if not right_items:
            continue
        width = _width(photos[left_items[0]["photo_id"]])
        if not width:
            continue
        for left in left_items:
            for right in right_items:
                if _meets_at_seam(left, right, width):
                    groups.join(left["item_id"], right["item_id"])
    return [[by_id[i] for i in group] for group in groups.groups()]


def scan_objects(items: list[dict], photos: dict[str, dict]) -> list[dict]:
    """One entry per physical object seen from one panorama (seam-cut detections joined)."""
    objects = []
    for members in seam_objects(items, photos):
        best = _best(members)
        photo = photos.get(best["photo_id"]) or {}
        objects.append({"members": members, "best": best, "label": best["label"],
                        "category": best.get("category"), "room_id": best.get("room_id"),
                        "room_label": best.get("room_label") or photo.get("room_label"),
                        "pano": photo.get("pano_id") or best["photo_id"], "face": photo.get("face")})
    return objects


def group_by_rules(objects: list[dict]) -> list[list[int]]:
    """Link objects of one room seen from different panoramas when their labels match."""
    links = _Groups(range(len(objects)))
    for i, a in enumerate(objects):
        for j in range(i + 1, len(objects)):
            b = objects[j]
            # Two objects in one panorama are two objects.
            if a["pano"] != b["pano"] and same_thing(a["label"], b["label"]):
                links.join(i, j)
    return links.groups()


def cluster_scan_items(items: list[dict], photos: dict[str, dict], grouper=None) -> list[dict]:
    """Group scan detections into inventory lines: [{canonical_id, member_ids, qty, key, room_id}].

    `items` need item_id, photo_id, label, category, confidence, room_id, bbox.
    `photos` maps photo_id to its record (pano_id, face, path).
    `grouper(room_label, objects)` returns groups of indices into `objects` (all
    from one room), optionally as (indices, index_of_best). Default: the rules.

    Whatever the grouper says, the quantity of a line is the largest number of
    its objects seen from one panorama. A grouper can merge lines but can
    never raise a count above what one viewpoint actually showed.
    """
    objects = scan_objects(items, photos)
    by_room: dict = {}
    for index, obj in enumerate(objects):
        by_room.setdefault(obj["room_id"], []).append(index)

    clusters = []
    for room_id, indices in by_room.items():
        room = [objects[i] for i in indices]
        groups = grouper(room[0]["room_label"], room) if grouper else group_by_rules(room)
        for group in _complete(groups, len(room)):
            local, best_local = group
            per_pano: dict = {}
            for i in local:
                per_pano[room[i]["pano"]] = per_pano.get(room[i]["pano"], 0) + 1
            members = [m for i in local for m in room[i]["members"]]
            canonical = room[best_local]["best"] if best_local in local else _best(members)
            clusters.append({
                "canonical_id": canonical["item_id"],
                "member_ids": [m["item_id"] for m in members],
                "qty": max(per_pano.values()),
                "key": item_key(canonical["label"]),
                "room_id": room_id,
            })
    return clusters


def _complete(groups, count: int) -> list[tuple[list[int], int | None]]:
    """Make a grouping safe to use: every object exactly once, unknown indices dropped."""
    seen: set = set()
    out = []
    for group in groups or []:
        indices, best = group if isinstance(group, tuple) else (group, None)
        clean = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < count and i not in seen:
                seen.add(i)
                clean.append(i)
        if clean:
            out.append((clean, best))
    out.extend(([i], None) for i in range(count) if i not in seen)      # anything left out stays its own line
    return out
