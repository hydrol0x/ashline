"""Scan-specific additions to cross-photo de-duplication (build spec 6.4).

Neighbouring panoramas of one scan overlap heavily, so the same object is
detected several times. Two things are provided:

1. Priors for candidate pairing (`scan_prior`, `merge_threshold`): same
   room_id and panorama positions a few metres apart. Pairs from the same scan
   in the same room may auto-merge at a lower similarity than cross-family
   photos. The embedding-based matcher plugs these in.
2. `cluster_scan_items`: a text-only clustering used until crop embeddings
   exist. Within a room, items with the same head noun are one cluster whose
   quantity is the largest count seen from any single panorama.
"""
from __future__ import annotations

import math
import re

from .detect import head_phrase

BASE_THRESHOLD = 0.90           # cross-family photos: only very confident pairs auto-merge
SCAN_ROOM_THRESHOLD = 0.75      # same scan, same room, panoramas close together
NEAR_M = 4.0

SYNONYMS = {
    "couch": "sofa", "loveseat": "sofa", "sectional": "sofa", "settee": "sofa",
    "television": "tv", "fridge": "refrigerator", "barstool": "stool", "stools": "stool",
    "nightstand": "nightstand", "bedside table": "nightstand", "drapes": "curtain",
    "curtains": "curtain", "carpet": "rug", "bookcase": "bookshelf", "shelves": "shelf",
}


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


def item_key(label: str) -> str:
    """Head noun of a label, singular, with common synonyms folded together."""
    head = head_phrase(label)
    for phrase, canon in SYNONYMS.items():
        if re.search(rf"\b{phrase}\b", head):
            return canon
    words = head.split()
    if not words:
        return ""
    noun = words[-1]
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        noun = noun[:-1]
    return noun


def cluster_scan_items(items: list[dict], photos: dict[str, dict]) -> list[dict]:
    """Group scan detections into clusters: [{canonical_id, member_ids, qty, key, room_id}].

    `items` need item_id, photo_id, label, confidence, room_id. `photos` maps
    photo_id to its record (for pano_id).
    """
    groups: dict[tuple, list[dict]] = {}
    for item in items:
        key = item_key(item["label"])
        if key:
            groups.setdefault((item.get("room_id"), key), []).append(item)

    clusters = []
    for (room_id, key), members in groups.items():
        per_pano: dict[str, int] = {}
        for item in members:
            pano = (photos.get(item["photo_id"]) or {}).get("pano_id") or item["photo_id"]
            per_pano[pano] = per_pano.get(pano, 0) + 1
        canonical = max(members, key=lambda i: (i.get("confidence") or 0, len(i["label"]), -i["item_id"]))
        clusters.append({
            "canonical_id": canonical["item_id"],
            "member_ids": [i["item_id"] for i in members],
            "qty": max(per_pano.values()),
            "key": key,
            "room_id": room_id,
        })
    return clusters
