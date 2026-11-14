"""listing_record (optional): the CoStar residential listing behind a scan.

Kit layout, as listed by `kit_info()` on the real kit (14 Nov 2026): both
descriptions were right. `residential-listings.csv` is a small index that maps
a scan to its listing ids, the records themselves are JSONL under `data/`, and
field documentation sits in `models/` and `schemas/`.

The join code below follows that layout but has only been run on a made-up
kit of the same shape, not on the real records.

The kit is "hackathon use only". Never commit it or anything read from it.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile

import requests

from .. import config


class HttpFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, so one member of a
    remote zip can be read without downloading all 160 MB."""

    def __init__(self, url: str, block: int = 4 * 2 ** 20):
        self.url, self.pos, self.block, self.buf, self.buf_at = url, 0, block, b"", -1
        head = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self.buf_at <= self.pos and self.pos + n <= self.buf_at + len(self.buf)):
            end = min(self.size, self.pos + max(n, self.block)) - 1
            resp = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300)
            resp.raise_for_status()
            self.buf_at, self.buf = self.pos, resp.content
        out = self.buf[self.pos - self.buf_at: self.pos - self.buf_at + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_kit() -> zipfile.ZipFile | None:
    """The kit zip from COSTAR_KIT_ZIP (local path) or COSTAR_KIT_URL. None when not configured."""
    path, url = config.env("COSTAR_KIT_ZIP"), config.env("COSTAR_KIT_URL")
    password = config.env("COSTAR_ZIP_PASSWORD")
    if not password or not (path or url):
        return None
    kit = zipfile.ZipFile(path if path else HttpFile(url))
    kit.setpassword(password.encode())
    return kit


def _residential_members(kit: zipfile.ZipFile) -> list[str]:
    names = [n for n in kit.namelist() if n.lower().endswith((".csv", ".jsonl", ".ndjson"))]
    return [n for n in names if "residential" in n.lower() or "listing" in n.lower() or "homes" in n.lower()]


def _rows(kit: zipfile.ZipFile, member: str):
    with kit.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        if member.lower().endswith(".csv"):
            yield from csv.DictReader(text)
        else:
            for line in text:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def kit_info() -> dict:
    """What is in the kit: every file name, and the columns of the listing files."""
    kit = open_kit()
    if kit is None:
        return {"configured": False, "reason": "Set COSTAR_ZIP_PASSWORD and COSTAR_KIT_ZIP or COSTAR_KIT_URL."}
    info = {"configured": True, "files": kit.namelist()[:200], "listing_files": {}}
    for member in _residential_members(kit):
        first = next(_rows(kit, member), None)
        info["listing_files"][member] = sorted(first) if isinstance(first, dict) else None
    return info


INDEX_CSV = "residential-listings.csv"                       # MatterportModelId <-> listing ids
LISTINGS = "data/residential-listings.jsonl"                 # the full listing records
FEATURES = "data/residential-listing-features.jsonl"         # features and highlights per listing
# Parts of a listing record that are large and of no use for a contents inventory.
BULKY = {"media", "agents", "offices", "history", "permissionLevels", "openHouses", "mls", "attachments"}


def _same(value, wanted: set) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() in wanted


def _find(kit: zipfile.ZipFile, member: str, keys: tuple, wanted: set) -> dict | None:
    """First record of `member` whose value under one of `keys` is in `wanted`."""
    if member not in kit.namelist():
        return None
    for row in _rows(kit, member):
        if isinstance(row, dict) and any(_same(row.get(k), wanted) for k in keys):
            return row
    return None


def shape(value, path: str = "", out: list | None = None, limit: int = 200) -> list[str]:
    """Key paths and value types of a record, without any values: safe to share."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(value, dict):
        for key in sorted(value):
            shape(value[key], f"{path}.{key}" if path else key, out, limit)
    elif isinstance(value, list):
        out.append(f"{path}: list[{len(value)}]")
        if value:
            shape(value[0], f"{path}[]", out, limit)
    else:
        out.append(f"{path}: {type(value).__name__}")
    return out


def listing_record(homes_listing_id: str | None = None, sid: str | None = None) -> dict:
    """The CoStar residential listing behind a scan.

    Verified layout (kit of 14 Nov 2026): `residential-listings.csv` maps a scan
    (MatterportModelId) to ListingId and ResidentialListingId; the full record is
    in `data/residential-listings.jsonl` and its features in
    `data/residential-listing-features.jsonl`. Give the scan id, the listing id
    from the scan index, or both. Returns {found, ids, listing, features}.
    """
    kit = open_kit()
    if kit is None:
        return {"found": False, "reason": "CoStar kit not configured."}
    wanted = {str(v).strip() for v in (homes_listing_id, sid) if v}
    if not wanted:
        return {"found": False, "reason": "Give a scan id or a listing id."}
    try:
        index_row = _find(kit, INDEX_CSV, ("MatterportModelId", "ListingId", "ResidentialListingId"), wanted)
        ids = set(wanted) - ({str(sid)} if sid else set())
        if index_row:
            ids |= {str(index_row[k]).strip() for k in ("ListingId", "ResidentialListingId") if index_row.get(k)}
        listing = _find(kit, LISTINGS, ("residentialListingId", "id"), ids) if ids else None
        if listing is None and index_row is None:
            # An older or different kit layout: fall back to any listing file with a matching value.
            for member in _residential_members(kit):
                for row in _rows(kit, member):
                    if isinstance(row, dict) and any(_same(v, wanted) for v in row.values()):
                        return {"found": True, "file": member, "ids": sorted(wanted), "listing": row, "features": None}
            return {"found": False, "reason": "No listing for this scan or listing id in the kit."}
        if listing is not None:
            ids |= {str(listing[k]) for k in ("id", "residentialListingId") if listing.get(k) is not None}
        features = _find(kit, FEATURES, ("id", "residentialListingId"), ids)
    except RuntimeError as exc:         # zipfile raises RuntimeError on a wrong password
        return {"found": False, "reason": f"Could not read the kit: {exc}. Type the zip password by hand."}
    return {
        "found": True,
        "ids": {"index": index_row, "matched": sorted(ids)},
        "listing": {k: v for k, v in (listing or {}).items() if k not in BULKY} or None,
        "features": features,
    }


# Appliances a listing names that are part of the house, not contents.
BUILT_IN = {"dishwasher", "disposal", "garbage disposal", "range hood", "hood", "cooktop", "oven", "range",
            "water heater", "furnace", "exhaust fan", "vent hood"}


def _first(*values):
    """First value that is set and not zero."""
    for value in values:
        if value not in (None, "", 0, 0.0):
            return value
    return None


def summarize(record: dict) -> dict:
    """The parts of a listing record that matter for a contents inventory.

    Field names follow the real kit (checked with `ashline listing --shape`).
    Every value may be missing; missing stays None rather than a guess.
    """
    listing = record.get("listing") or {}
    features = record.get("features") or {}
    derived = features.get("derivedValues") or {}
    prop = listing.get("property") or {}
    reso = listing.get("resoValues") or {}
    structure = reso.get("structure") or {}
    costar = listing.get("costarFeatureValues") or {}

    baths = _first(derived.get("totalBathrooms"), prop.get("bathsMax"))
    if baths is None and prop.get("bathsFull"):
        baths = prop["bathsFull"] + 0.5 * (prop.get("bathsHalf") or 0)
    garage_spaces = _first(derived.get("totalGarageSpaces"), derived.get("totalPrivateGarageSpaces"),
                           structure.get("garageSpaces"))

    appliances = []
    for feature in features.get("features") or []:
        where = " ".join(str(feature.get(k) or "") for k in ("category", "subcategory1", "subcategory2")).lower()
        name = str(feature.get("name") or "").strip()
        if "applianc" in where and name and name.lower() not in BUILT_IN and name not in appliances:
            appliances.append(name)

    return {
        "beds": _first(derived.get("totalBedrooms"), prop.get("bedsMax")),
        "baths": baths,
        "sqft": _first(derived.get("homeSquareFeet"), prop.get("sqftMax")),
        "year_built": _first(derived.get("yearBuilt"), prop.get("yearBuilt"), structure.get("yearBuilt")),
        "stories": _first(derived.get("numberOfStories"), prop.get("stories"), structure.get("stories")),
        "lot_acres": _first(derived.get("lotSizeInAcres"), prop.get("lotSizeAcresMax")),
        "property_type": _first(derived.get("propertyType"), prop.get("mlsPropertyType")),
        "garage_spaces": int(garage_spaces) if garage_spaces else None,
        "list_date": (listing.get("listDate") or "")[:10] or None,
        "close_date": (listing.get("closeDate") or "")[:10] or None,
        "price": _first(listing.get("closePrice"), listing.get("currentPrice"), listing.get("originalPrice")),
        "has": {                                   # True only when the listing says so
            "garage": bool(garage_spaces or reso.get("hasGarage") or costar.get("hasGarage")),
            "carport": bool(structure.get("hasCarport")),
            "basement": bool(reso.get("hasBasement") or costar.get("hasBasement") or structure.get("hasBasement")),
            "storage": bool(reso.get("hasStorage")),
            "washer_dryer": bool(reso.get("hasWasherDryer")),
            "spa": bool(reso.get("hasSpa")),
            "fireplace": bool(reso.get("hasFireplace") or costar.get("hasFireplace") or structure.get("hasFireplace")),
        },
        "appliances": appliances[:12],
        "highlights": [str(h.get("text")) for h in (features.get("highlights") or []) if h.get("text")][:10],
        "remarks": (listing.get("remarks") or "").strip()[:800] or None,
    }
