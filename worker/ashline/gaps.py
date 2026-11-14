"""Voice-call questions from gaps in the case (build spec 6.7, plus the home twin).

Four gap sources come from the scan, the aerial image and the sale listing:
  (a) a scanned room with few confirmed items, weighted by room area;
  (b) every non-primary structure on the parcel;
  (c) yard findings the family has not confirmed yet;
  (d) things the listing mentions that the scan does not show (a garage, a
      basement, a washer and dryer, named appliances).

Each question carries a weight: expected new items times a typical value per
item for that kind of room. The call asks the heaviest question next. Tone:
short, gentle, one question at a time.
"""
from __future__ import annotations

from .costar.matterport import _month_year
from .dedupe import label_words
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
    pending: dict[str, int] = {}
    for item in store.get_items(case_id, status="suggested", canonical_only=True):
        if item["room_id"]:
            pending[item["room_id"]] = pending.get(item["room_id"], 0) + 1
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
        waiting = pending.get(room["room_id"], 0)
        if have:
            lead = f"I only have {have} confirmed thing{'s' if have != 1 else ''} from {name} so far."
        elif waiting:           # suggestions exist but the family has not looked at them yet
            lead = (f"I've spotted {waiting} possible thing{'s' if waiting != 1 else ''} in {name} "
                    "for you to check, and nothing is confirmed there yet.")
        else:
            lead = f"I don't have anything from {name} yet."
        out.append({
            "kind": "room", "ref": room["room_id"], "weight": round(expected_new * value),
            "text": f"{lead} {PROMPTS.get(kind, 'What do you remember being in there?')}",
        })
    return out


def _garage_phrase(finding: dict) -> str:
    spaces = finding["detail"].get("spaces")
    return f"a {spaces}-car garage" if spaces else "a garage"


def _scanned(store: Store, case_id: str, word: str) -> bool:
    return any(word in (room["label"] or "").lower() for room in store.get_rooms(case_id))


BUILDING_LABELS = ("garage", "guest house", "outdoor storage")       # what a listing can call an outbuilding


def _and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def structure_questions(store: Store, case_id: str) -> list[dict]:
    """Questions about the other buildings on the lot.

    Where the sale listing names what they are (a garage, a guest house), one
    question puts the three sources together: what the listing mentions, what
    the aerial shows, and what the scan did not go inside. It does not guess
    which building is which. Otherwise there is one question per building.
    """
    findings = store.get_findings(case_id)
    structures = [f for f in findings if f["kind"] == "structure" and f["status"] == "open"]
    if not structures:
        return []
    when = _month_year(structures[0]["evidence_date"])
    seen = f"the aerial photo{' from ' + when if when else ''} shows"
    listed = {f["label"]: f for f in findings if f["kind"] == "listing" and f["label"] in BUILDING_LABELS}

    if listed:
        mentions = (["a detached guest house"] if "guest house" in listed else []) \
            + ([_garage_phrase(listed["garage"])] if "garage" in listed else []) \
            + (["outdoor storage"] if "outdoor storage" in listed else [])
        sizes = [f"{f['detail']['footprint_sqft']:,}" for f in structures if f["detail"].get("footprint_sqft")]
        size_text = f" (about {_and(sizes)} sq ft)" if len(sizes) == len(structures) else ""
        count = len(structures)
        buildings = ("a second building" if count == 1 else
                     f"{ {2: 'two', 3: 'three', 4: 'four'}.get(count, count) } more buildings")
        text = f"The listing mentions {_and(mentions)}, and {seen} {buildings} on the lot{size_text}."
        if "garage" in listed and not _scanned(store, case_id, "garage"):
            text += " The 3D scan doesn't go inside a garage."
        if count == 1 and len(mentions) == 1:
            text += (" Was that the guest house? What was in it?" if "guest house" in listed
                     else " What was kept there: tools, bikes, storage?")
        else:
            text += " Which building was which, and what was in each?"
        return [{"kind": "structure", "ref": [f["finding_id"] for f in structures],
                 "weight": STRUCTURE_ITEMS * STRUCTURE_VALUE * count, "text": text}]

    out = []
    for n, f in enumerate(structures, 1):
        size = f["detail"].get("footprint_sqft")
        size_text = f" (about {size:,} sq ft)" if size else ""
        which = "a second building" if n == 1 else "another building"
        out.append({"kind": "structure", "ref": f["finding_id"], "weight": STRUCTURE_ITEMS * STRUCTURE_VALUE,
                    "text": f"{seen[0].upper() + seen[1:]} {which} on the lot{size_text}. "
                            "Was that a garage or a workshop? What was in it?"})
    return out


def listing_questions(store: Store, case_id: str) -> list[dict]:
    """Things the sale listing mentions that neither the scan nor the family has covered yet."""
    findings = store.get_findings(case_id)
    listing = {f["label"]: f for f in findings if f["kind"] == "listing" and f["status"] == "open"}
    if not listing:
        return []
    items = [i for i in store.get_items(case_id) if i["status"] != "rejected"]
    item_words = [label_words(i["label"])[0] for i in items]

    def have(name: str) -> bool:
        words = label_words(name)[0]
        return bool(words) and any(words <= w for w in item_words)

    out = []
    # With buildings on the aerial, the garage and guest house are asked about there, once.
    on_aerial = any(f["kind"] == "structure" for f in findings)
    if "garage" in listing and not on_aerial and not _scanned(store, case_id, "garage"):
        f = listing["garage"]
        out.append({"kind": "listing", "ref": f["finding_id"], "weight": STRUCTURE_ITEMS * STRUCTURE_VALUE,
                    "text": f"The listing mentions {_garage_phrase(f)}, but the 3D scan doesn't show inside it. "
                            "What was kept there: tools, bikes, storage?"})
    if "guest house" in listing and not on_aerial:
        out.append({"kind": "listing", "ref": listing["guest house"]["finding_id"],
                    "weight": STRUCTURE_ITEMS * STRUCTURE_VALUE,
                    "text": "The listing mentions a detached guest house. Was it furnished, and what was in it?"})
    if "outdoor storage" in listing and not on_aerial:
        out.append({"kind": "listing", "ref": listing["outdoor storage"]["finding_id"], "weight": 1800,
                    "text": "The listing mentions outdoor storage. What was kept in it?"})
    if "laundry" in listing and "washer and dryer" not in listing and not (have("washer") or have("dryer")):
        where = " outside the house" if listing["laundry"]["detail"].get("outside") else ""
        out.append({"kind": "listing", "ref": listing["laundry"]["finding_id"], "weight": 1400,
                    "text": f"The listing mentions laundry{where}, which the 3D scan doesn't show. "
                            "Was there a washer and dryer, and were they yours?"})
    if "patio" in listing and not any(f["kind"] == "yard" and f["label"] in ("patio furniture", "grill")
                                       for f in findings):
        out.append({"kind": "listing", "ref": listing["patio"]["finding_id"], "weight": 1350,
                    "text": "The listing mentions a patio. Was there outdoor furniture or a grill on it?"})
    if "basement" in listing and not _scanned(store, case_id, "basement"):
        out.append({"kind": "listing", "ref": listing["basement"]["finding_id"], "weight": 2400,
                    "text": "The listing mentions a basement that the 3D scan doesn't show. What was stored down there?"})
    if "storage" in listing:
        out.append({"kind": "listing", "ref": listing["storage"]["finding_id"], "weight": 1200,
                    "text": "The listing mentions storage space. Was anything kept there that we haven't covered?"})
    if "washer and dryer" in listing and not (have("washer") or have("dryer")):
        out.append({"kind": "listing", "ref": listing["washer and dryer"]["finding_id"], "weight": 1400,
                    "text": "The listing mentions a washer and dryer. Were they yours, and were they still there?"})
    if "spa" in listing and not any(f["kind"] == "yard" and f["label"] == "hot tub" for f in findings):
        out.append({"kind": "listing", "ref": listing["spa"]["finding_id"], "weight": YARD_VALUE["hot tub"],
                    "text": "The listing mentions a spa or hot tub. Was that yours?"})
    if "appliances" in listing:
        names = [n for n in listing["appliances"]["detail"].get("names", []) if not have(n)]
        names = [n for n in names if n.lower() not in ("washer", "dryer") or "washer and dryer" not in listing]
        if names:
            shown = names[:5]
            listed = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
            out.append({"kind": "listing", "ref": listing["appliances"]["finding_id"], "weight": 600 * len(shown),
                        "text": f"The listing also names these, which I haven't seen in the scan: {listed.lower()}. "
                                "Were those yours?"})
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
    queue = (room_questions(store, case_id) + structure_questions(store, case_id)
             + yard_questions(store, case_id) + listing_questions(store, case_id))
    queue.sort(key=lambda q: -q["weight"])
    return queue[:limit] if limit else queue
