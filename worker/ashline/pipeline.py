"""Glue between photo sources and the detect step, with a per-case cache.

A photo is a dict: {photo_id, path, source, room_id, room_label, ...}. Scan
faces and family photos take the same path from here on.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .dedupe import dedupe_in_photo
from .detect import drop_fixtures


def detection_path(case_id: str, photo_id: str) -> Path:
    folder = config.case_dir(case_id) / "detections"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{photo_id}.json"


def detect_cached(case_id: str, photo: dict, detector=None, get_detector=None) -> list[dict]:
    """Detections for one photo, computed once per case.

    Pass `detector`, or `get_detector` to build it only when a photo is not
    cached yet (loading the local model takes a minute).
    """
    path = detection_path(case_id, photo["photo_id"])
    if path.exists():
        return dedupe_in_photo(drop_fixtures(json.loads(path.read_text())))
    if detector is None:
        detector = get_detector()
    items = detector.detect(photo["path"], photo.get("room_label"))
    path.write_text(json.dumps(items, indent=1))
    return dedupe_in_photo(drop_fixtures(items))
