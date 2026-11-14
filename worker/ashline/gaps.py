"""Voice-call questions from gaps in the case (build spec 6.7, plus the home twin).

Three gap sources come from the scan and the aerial image:
  (a) a scanned room with few confirmed items, weighted by room area;
  (b) every non-primary structure on the parcel;
  (c) yard findings the family has not confirmed yet.

Each question carries a weight: expected new items times a typical value per
item for that kind of room. The call asks the heaviest question next. Tone:
short, gentle, one question at a time.
"""
from __future__ import annotations

from .costar.matterport import _month_year
from .store import Store

# Typical contents per square metre and typical replacement value per item, by room
# type. Rough planning numbers, used only to order questions.
ROOM_PROFILE = {
    "garage": (1.0, 220), "workshop": (1.5, 200), "office": (1.5, 260), "kitchen": (2.5, 90),
    "living room": (0.9, 420), "family room": (0.9, 420), "bedroom": (1.1, 240),
    "dining room": (0.8, 300), "closet": (6.0, 60), "laundry room": (1.0, 250),
    "bathroom": (1.5, 40), "basement": (0.8, 200), "hallway": (0.2, 120),
}
DEFAULT_PROFILE = (0.8, 150)
STRUCTURE_ITEMS, STRUCTURE_VALUE = 15, 220      # an outbuilding usually means tools, bikes, storage
YARD_VALUE = {"car": 0, "shed": 900, "trampoline": 500, "patio furniture": 900, "grill": 450,
              "hot tub": 6000, "swimming pool": 800, "playground set": 900, "boat": 8000,
              "trailer": 3000, "deck": 700, "sport court": 400, "solar panels": 0}

PROMPTS = {
    "garage": "Was there a workbench, tools, bikes or storage bins?",
    "workshop": "What tools or equipment did you keep there?",
    "office": "Was there a computer, a printer, a desk chair?",
    "kitchen": "Think of the counters and cupboards: small appliances, cookware, dishes?",
    "living room": "Besides the seating, was there a TV, speakers, rugs or shelves?",
    "family room": "Besides the seating, was there a TV, speakers, rugs or shelves?",
    "bedroom": "Besides the bed, was there a dresser, lamps, a TV, things in the closet?",
    "dining room": "Was there a table and chairs, a sideboard, good china?",
    "closet": "What was kept in there: clothes, shoes, luggage, linens?",
    "laundry room": "Was there a washer and dryer, a vacuum, an ironing board?",
    "bathroom": "Anything of value there: a hair dryer, towels, a scale?",
    "basement": "What was stored down there?",
}


def _room_kind(label: str) -> str:
    names = [n.strip() for n in label.lower().split("/")]
    return max(names, key=lambda n: ROOM_PROFILE.get(n, DEFAULT_PROFILE)[0] * ROOM_PROFILE.get(n, DEFAULT_PROFILE)[1])


def room_questions(store: Store, case_id: str) -> list[dict]:
    confirmed: dict[str, int] = {}
    for item in store.get_items(case_id, status="confirmed", canonical_only=True):
        if item["room_id"]:
            confirmed[item["room_id"]] = confirmed.get(item["room_id"], 0) + (item["qty"] or 1)
    out = []
    for room in store.get_rooms(case_id):
        kind = _room_kind(room["label"])
        density, value = ROOM_PROFILE.get(kind, DEFAULT_PROFILE)
        area = room["area_m2"] or 10.0                      # unknown area: assume an average room
        have = confirmed.get(room["room_id"], 0)
        expected_new = max(0.0, density * area - have)
        if expected_new < 1:
            continue
        name = (room.get("name") or room["label"]).lower()
        if kind == "other":
            name = "one of the smaller rooms"                         # Matterport labels some rooms "Other"
        else:
            name = name if name[-1].isdigit() else f"the {name}"     # "bedroom 2", "the kitchen"
        lead = (f"I only have {have} thing{'s' if have != 1 else ''} from {name} so far."
                if have else f"I don't have anything from {name} yet.")
        out.append({
            "kind": "room", "ref": room["room_id"], "weight": round(expected_new * value),
            "text": f"{lead} {PROMPTS.get(kind, 'What do you remember being in there?')}",
        })
    return out


def structure_questions(store: Store, case_id: str) -> list[dict]:
    out = []
    structures = [f for f in store.get_findings(case_id, status="open") if f["kind"] == "structure"]
    for n, f in enumerate(structures, 1):
        when = _month_year(f["evidence_date"])
        seen = f"The aerial photo{' from ' + when if when else ''} shows"
        which = "a second building" if n == 1 else "another building"
        size = f["detail"].get("footprint_sqft")
        size_text = f" (about {size:,} sq ft)" if size else ""
        out.append({
            "kind": "structure", "ref": f["finding_id"], "weight": STRUCTURE_ITEMS * STRUCTURE_VALUE,
            "text": f"{seen} {which} on the lot{size_text}. Was that a garage or a workshop? What was in it?",
        })
    return out


def yard_questions(store: Store, case_id: str) -> list[dict]:
    open_yard = [f for f in store.get_findings(case_id, status="open") if f["kind"] == "yard"]
    by_label: dict[str, list[dict]] = {}
    for f in open_yard:
        by_label.setdefault(f["label"], []).append(f)
    out = []
    for label, group in by_label.items():
        value = YARD_VALUE.get(label, 300)
        if value <= 0:
            continue                    # cars and solar panels are not contents
        when = _month_year(group[0]["evidence_date"])
        seen = f"In the aerial photo{' from ' + when if when else ''} I can see what looks like"
        thing = f"a {label}" if len(group) == 1 else f"{len(group)} {label}s"
        follow = {"swimming pool": "Was there pool furniture or equipment with it?",
                  "deck": "Was there furniture or a grill on it?"}.get(label, "Was that yours?")
        out.append({"kind": "yard", "ref": [f["finding_id"] for f in group],
                    "weight": value * len(group), "text": f"{seen} {thing} out back. {follow}"})
    return out


def gap_questions(case_id: str, store: Store | None = None, limit: int | None = None) -> list[dict]:
    """The question queue for the voice call, heaviest first."""
    store = store or Store()
    queue = room_questions(store, case_id) + structure_questions(store, case_id) + yard_questions(store, case_id)
    queue.sort(key=lambda q: -q["weight"])
    return queue[:limit] if limit else queue
