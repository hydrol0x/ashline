"""One spelling per room, however the family typed it.

Photo captions and the detector name rooms freely ("just basement", "basement", "the Basement"),
which would split one room into several on the page and in the export. `tidy_room` drops filler
words and normalises case so they all read "Basement". Used when a caption is turned into a room,
and again when the page and the export group items, so older items named before this existed
also land in one room.
"""
from __future__ import annotations

import re

FILLER = re.compile(r"^(?:(?:just|only|this is|that's|thats|it's|its|here's|heres|in|from|the|our|my|"
                    r"mom's|dad's|a|an)\s+)+", re.I)


def tidy_room(label: str | None) -> str | None:
    """'just basement' / 'the Basement' / 'basement' -> 'Basement'. None or empty stays None."""
    text = re.sub(r"\s+", " ", (label or "").strip(" .!?"))
    text = FILLER.sub("", text).strip()
    return text[:1].upper() + text[1:] if text else None
