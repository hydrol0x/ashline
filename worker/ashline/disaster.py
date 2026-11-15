"""What happened to the home, and what that means for the claim.

US only. One list of belongings can feed different claims depending on the disaster:

- fire, or anything else    one homeowners claim, at replacement cost (what it costs to buy new)
- flood                     a flood insurance claim, which pays depreciated value (actual cash value)
- hurricane or storm        usually two claims: rising water goes to flood insurance (depreciated value),
                            wind and rain go to the homeowners policy (replacement cost)

Families without insurance can ask FEMA for help with essential items only (appliances, furniture, a
computer), so those are marked. Depreciated values here are estimates from typical age and wear by
category, labelled as such everywhere they appear.
"""
from __future__ import annotations

import json
import re

from .rooms import tidy_room

FIRE, FLOOD, STORM, OTHER = "fire", "flood", "storm", "other"
NAMES = {FIRE: "fire", FLOOD: "flood", STORM: "hurricane or storm", OTHER: "disaster"}

# A claim is (key, title, value basis). "rcv" = replacement cost, "acv" = depreciated value.
CLAIMS = {
    FIRE: [("home", "Homeowners claim", "rcv")],
    FLOOD: [("flood", "Flood insurance claim", "acv")],
    STORM: [("flood", "Flood insurance claim (water damage)", "acv"),
            ("wind", "Homeowners claim (wind and rain damage)", "rcv")],
    OTHER: [("home", "Homeowners claim", "rcv")],
}

# Share of the new price an item of typical age is still worth. Rough, and shown as an estimate.
ACV_SHARE = {"electronics": 0.35, "appliances": 0.50, "furniture": 0.55, "clothing": 0.30, "kitchen": 0.50,
             "lighting": 0.55, "decor": 0.60, "art": 0.80, "tools": 0.60, "books": 0.40, "toys": 0.40,
             "sports": 0.45, "jewelry": 0.80, "other": 0.50}
ACV_NOTE = "Depreciated value is an estimate for an item of typical age and wear. Your adjuster sets the final figure."

ESSENTIAL = re.compile(r"\b(refrigerator|fridge|freezer|stove|range|oven|cooktop|microwave|washer|washing machine|"
                       r"dryer|dishwasher|water heater|furnace|air condition\w*|heater|bed|mattress|crib|sofa|couch|"
                       r"sectional|table|chair|dresser|wardrobe|computer|laptop|desktop)\b", re.I)


def parse_disaster(text: str) -> str | None:
    low = (text or "").lower()
    if re.search(r"hurricane|tropical storm|nor'?\s?easter|noreaster|storm|tornado|wind|typhoon|derecho", low):
        return STORM
    if re.search(r"flood|water|surge|river|levee|dam\b", low):
        return FLOOD
    if re.search(r"fire|burn|smoke|blaze|wildfire", low):
        return FIRE
    if re.search(r"earthquake|landslide|mudslide|explosion|something else|other", low):
        return OTHER
    return None


def parse_insured(text: str) -> str | None:
    low = (text or "").lower().strip(" .!")
    if re.fullmatch(r"(no|nope|none|not insured|no insurance|we don'?t|we do not|uninsured)( we don'?t)?", low):
        return "no"
    if re.fullmatch(r"(yes|yep|yeah|we do|insured|we have (it|insurance)|yes we do|homeowners|renters)( insurance)?", low):
        return "yes"
    if re.fullmatch(r"(not sure|unsure|don'?t know|i don'?t know|idk|maybe)", low):
        return "unsure"
    return None


def room_causes(case: dict) -> dict:
    try:
        return json.loads(case.get("room_causes") or "{}")
    except ValueError:
        return {}


def parse_flooded(text: str, rooms: list[str]) -> dict | None:
    """'the basement and kitchen' / 'all of them' / 'none' -> {room: "flood" | "wind"} for the known rooms."""
    low = (text or "").lower().strip(" .!")
    if re.fullmatch(r"(none|no|nothing flooded|no flooding|it didn'?t flood|nope)", low):
        return {r.lower(): "wind" for r in rooms}
    if re.fullmatch(r"(all|all of them|everything|the whole (house|home|place)|every room)", low):
        return {r.lower(): "flood" for r in rooms}
    named = {r.lower() for r in rooms if re.search(rf"\b{re.escape(r.lower())}s?\b", low)}
    loose = [tidy_room(p) for p in re.split(r",|;|\band\b|\n", re.sub(r"^(only |just |the )", "", low)) if p.strip()]
    if not named and not loose:
        return None
    out = {r.lower(): ("flood" if r.lower() in named else "wind") for r in rooms}
    for room in loose:
        if room and room.lower() not in out:
            out[room.lower()] = "flood"                    # a room they named that wasn't on the list
    return out


def item_cause(case: dict, item: dict) -> str | None:
    """flood / wind for a storm case; None when the disaster has a single claim."""
    disaster = case.get("disaster") or FIRE
    if disaster == FLOOD:
        return "flood"
    if disaster != STORM:
        return None
    if item.get("cause") in ("flood", "wind"):
        return item["cause"]
    room = (tidy_room(item.get("room_label")) or "").lower()
    return room_causes(case).get(room, "wind")             # unknown rooms: wind and rain until told otherwise


def claim_key(case: dict, item: dict) -> str:
    cause = item_cause(case, item)
    return cause if cause else "home"


def claims(case: dict) -> list[tuple[str, str, str]]:
    return CLAIMS.get(case.get("disaster") or FIRE, CLAIMS[FIRE])


def acv(price: float | None, category: str | None) -> float | None:
    if not price:
        return None
    return round(float(price) * ACV_SHARE.get((category or "other").lower(), ACV_SHARE["other"]))


def is_essential(item: dict) -> bool:
    return bool(ESSENTIAL.search(item.get("label") or ""))


def uses_sb495(case: dict) -> bool:
    """California SB 495 (the 60% advance without an inventory) is for declared-disaster total losses in
    California. The fire flow assumed it; other disasters only get it when the address says California."""
    address = (case.get("address") or "").lower()
    in_california = bool(re.search(r"\b(ca|calif\.?|california)\b", address))
    return (case.get("disaster") or FIRE) == FIRE or in_california
