"""home_lookup: survivor's address (or lat/lon) -> Matterport scan id, lat/lon, listing id.

The Model API has no geographic search, so every lookup goes through
models_index.csv. The file is fetched at runtime into a git-ignored cache and
must never be committed: it holds addresses and scan ids.
"""
from __future__ import annotations

import csv
import difflib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from .. import config

# USPS-style abbreviations, so "Avenue" and "Ave" compare equal.
_SUFFIX = {
    "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd", "drive": "dr",
    "road": "rd", "lane": "ln", "court": "ct", "place": "pl", "terrace": "ter",
    "circle": "cir", "highway": "hwy", "parkway": "pkwy", "trail": "trl", "way": "way",
    "square": "sq", "canyon": "cyn", "heights": "hts", "mount": "mt", "mountain": "mtn",
    "point": "pt", "crossing": "xing", "plaza": "plz", "alley": "aly", "walk": "walk",
}
_DIRECTION = {
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}
_SUFFIX_ABBR = set(_SUFFIX.values())
_DIR_ABBR = set(_DIRECTION.values())
_UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "no"}
_STATES = {"ca": "ca", "california": "ca", "tx": "tx", "texas": "tx"}

NEAREST_M = 30.0


@dataclass(frozen=True)
class ParsedAddress:
    number: str          # house number, "" when the text has none
    street: tuple        # normalized street-line tokens after the number
    unit: str            # "" when none
    postal_code: str     # "" when none
    rest: frozenset      # other tokens (city, state)
    loose: bool = False  # typed without commas, so a city name may trail the street


def split_street(tokens: tuple) -> tuple:
    """(leading direction, name tokens, suffix, trailing direction) of a street line.

    Only the ends are treated as direction or suffix: in "Sherwood Point Ln",
    "Point" is part of the name.
    """
    toks = list(tokens)
    pre = post = suffix = ""
    if len(toks) > 1 and toks[-1] in _DIR_ABBR:
        post = toks.pop()
    if len(toks) > 1 and toks[-1] in _SUFFIX_ABBR:
        suffix = toks.pop()
    if len(toks) > 1 and toks[0] in _DIR_ABBR:
        pre = toks.pop(0)
    return pre, tuple(toks), suffix, post


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("#", " unit ")
    text = re.sub(r"[^\w\s,]", "", text)
    out = []
    for tok in text.replace(",", " , ").split():
        out.append(_DIRECTION.get(tok, _SUFFIX.get(tok, tok)))
    return out


def parse_address(text: str) -> ParsedAddress:
    """Split free-text address into house number, street, unit and the rest."""
    toks = _tokens(text or "")
    segments, cur = [], []
    for tok in toks:
        if tok == ",":
            segments.append(cur)
            cur = []
        else:
            cur.append(tok)
    segments.append(cur)
    segments = [s for s in segments if s]
    if not segments:
        return ParsedAddress("", (), "", "", frozenset())
    loose = len(segments) == 1

    first, later = list(segments[0]), [t for seg in segments[1:] for t in seg]
    number = ""
    if first and re.fullmatch(r"\d+[a-z]?", first[0]):
        number = first.pop(0)

    unit = ""
    for i, tok in enumerate(first):
        if tok in _UNIT_WORDS and i + 1 < len(first):
            unit = first[i + 1]
            first = first[:i] + first[i + 2:]
            break

    # With no commas, zip/state trail the street line: move them out of it. A
    # trailing city name stays in the street line; _score() trims it per candidate.
    if len(segments) == 1:
        while first and (re.fullmatch(r"\d{5}", first[-1]) or first[-1] in _STATES
                         or first[-1] in ("us", "usa")):
            later.insert(0, first.pop())

    postal = ""
    rest = []
    it = iter(later)
    for tok in it:
        if tok in _UNIT_WORDS:
            unit = unit or next(it, "")
        elif re.fullmatch(r"\d{5}", tok):
            postal = tok
        elif tok in ("us", "usa"):
            continue
        else:
            rest.append(_STATES.get(tok, tok))
    # The index repeats the unit as its own segment ("..., 109, Los Angeles, ...").
    if unit:
        rest = [t for t in rest if t != unit]
    return ParsedAddress(number, tuple(first), unit, postal, frozenset(rest), loose)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class HomeIndex:
    """Residential rows of models_index.csv, loaded once at start-up."""

    def __init__(self, rows: list[dict]):
        self.rows = []
        for row in rows:
            if (row.get("segment") or "").strip().lower() != "residential":
                continue
            row = dict(row)
            row["_lat"], row["_lon"] = _float(row.get("lat")), _float(row.get("lon"))
            # Some rows have empty city/state/zip, so parse the address text itself.
            row["_parsed"] = parse_address(row.get("address") or row.get("name") or "")
            self.rows.append(row)
        self._by_number: dict[str, list[dict]] = {}
        for row in self.rows:
            self._by_number.setdefault(row["_parsed"].number, []).append(row)

    # ------------------------------------------------------------- loading
    @classmethod
    def from_csv(cls, path: str | Path) -> "HomeIndex":
        with open(path, newline="", encoding="utf-8") as fh:
            return cls(list(csv.DictReader(fh)))

    @classmethod
    def load(cls, refresh: bool = False) -> "HomeIndex":
        """Read the cached index, downloading it first when missing."""
        override = config.env("ASHLINE_INDEX_CSV")
        path = Path(override) if override else config.cache_dir() / "models_index.csv"
        if refresh or not path.is_file():
            resp = requests.get(config.INDEX_URL, timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        return cls.from_csv(path)

    # ------------------------------------------------------------ matching
    @staticmethod
    def _street_score(q_street: tuple, c_street: tuple) -> float:
        """10 = same line, 9 = same name with a direction or suffix left out, 0 = other name, -1 = contradiction."""
        if q_street == c_street:
            return 10.0
        q_pre, q_name, q_suf, q_post = split_street(q_street)
        c_pre, c_name, c_suf, c_post = split_street(c_street)
        if q_name != c_name:
            return 0.0
        for q, c in ((q_pre, c_pre), (q_suf, c_suf), (q_post, c_post)):
            if q and c and q != c:
                return -1.0                 # "N Main St" is not "S Main St"; "Oak St" is not "Oak Ave"
        return 9.0

    def _score(self, query: ParsedAddress, row: dict) -> float:
        """0 = no match. Higher = better. House number and street name must agree."""
        cand: ParsedAddress = row["_parsed"]
        if query.number != cand.number or not query.street or not cand.street:
            return 0.0
        streets = [query.street]
        if query.loose:                     # "12 Oak St Los Angeles": drop the trailing city
            trimmed = list(query.street)
            while len(trimmed) > 1 and trimmed[-1] in cand.rest:
                trimmed.pop()
            streets.append(tuple(trimmed))
            streets.append(query.street[:len(cand.street)])     # a neighbourhood name the index lacks
        scores = [self._street_score(st, cand.street) for st in streets]
        score = max(scores)
        if score <= 0 and min(scores) < 0:
            return 0.0                      # same name, contradicting direction or suffix
        if not score:
            q_name, c_name = split_street(streets[min(1, len(streets) - 1)])[1], split_street(cand.street)[1]
            ratio = difflib.SequenceMatcher(None, " ".join(q_name), " ".join(c_name)).ratio()
            # A near-miss on the street name only counts with a matching zip.
            if ratio < 0.85 or not (query.postal_code and query.postal_code == cand.postal_code):
                return 0.0
            score = 5.0 + ratio
        if query.postal_code and cand.postal_code:
            if query.postal_code != cand.postal_code:
                return 0.0
            score += 3.0
        if query.unit or cand.unit:
            if query.unit and cand.unit and query.unit != cand.unit:
                return 0.0
            score += 2.0 if query.unit == cand.unit else -1.0
        score += len(query.rest & cand.rest) * 0.5
        return score

    def by_address(self, text: str) -> list[dict]:
        """Rows that match the address text, best first."""
        query = parse_address(text)
        scored = [(self._score(query, row), row) for row in self._by_number.get(query.number, [])]
        scored = [(s, r) for s, r in scored if s > 0]
        # Same address scanned more than once: prefer the most recent capture.
        scored.sort(key=lambda sr: (sr[0], sr[1].get("original_created") or ""), reverse=True)
        return [r for _, r in scored]

    def nearest(self, lat: float, lon: float, max_m: float = NEAREST_M) -> list[tuple[float, dict]]:
        """(distance_m, row) for rows within max_m, nearest first."""
        hits = []
        for row in self.rows:
            if row["_lat"] is None or row["_lon"] is None:
                continue
            if abs(row["_lat"] - lat) > 0.01 or abs(row["_lon"] - lon) > 0.01:
                continue
            d = haversine_m(lat, lon, row["_lat"], row["_lon"])
            if d <= max_m:
                hits.append((d, row))
        hits.sort(key=lambda dr: (dr[0], -(int((dr[1].get("original_created") or "0")[:4] or 0))))
        return hits

    def by_sid(self, sid: str) -> dict | None:
        return next((r for r in self.rows if r.get("sid") == sid), None)


def _result(row: dict, match: str, distance_m: float | None = None, others: int = 0) -> dict:
    return {
        "found": True,
        "match": match,                       # "address", "nearest" or "sid"
        "sid": row.get("sid"),
        "address": row.get("address") or row.get("name"),
        "lat": row["_lat"],
        "lon": row["_lon"],
        "homes_listing_id": row.get("homes_listing_id") or None,
        "pano_count": _int(row.get("pano_count")),
        "room_count": _int(row.get("room_count")),
        "floor_count": _int(row.get("floor_count")),
        "original_created": row.get("original_created") or None,   # the capture date
        "distance_m": None if distance_m is None else round(distance_m, 1),
        "other_scans_at_address": others,
    }


_INDEX: HomeIndex | None = None


def get_index() -> HomeIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = HomeIndex.load()
    return _INDEX


def home_lookup(address: str | None = None, lat: float | None = None, lon: float | None = None,
                index: HomeIndex | None = None, sid: str | None = None) -> dict:
    """Find the 3D scan of a home by address text or by lat/lon.

    Returns {found: True, sid, lat, lon, homes_listing_id, pano_count, room_count,
    floor_count, original_created, ...} or {found: False, reason}. Most real
    addresses have no scan; Ashline then carries on with family photos only.
    """
    index = index or get_index()
    if sid:                                   # a known scan id, e.g. the demo home
        row = index.by_sid(sid.strip())
        if row:
            return _result(row, "sid")
        return {"found": False, "reason": "No residential scan with this id in the index."}
    if address and address.strip():
        rows = index.by_address(address)
        if rows:
            # A building with several scanned units: never seed from a neighbour's home.
            if rows[0]["_parsed"].unit and not parse_address(address).unit:
                units = sorted({r["_parsed"].unit for r in rows if r["_parsed"].unit})
                return {"found": False, "needs_unit": True, "units": units,
                        "reason": "There are scans for several units at this address. Which unit is it?"}
            return _result(rows[0], "address", others=len(rows) - 1)
    if lat is not None and lon is not None:
        hits = index.nearest(float(lat), float(lon))
        if hits:
            d, row = hits[0]
            return _result(row, "nearest", distance_m=d, others=len(hits) - 1)
    if not (address and address.strip()) and (lat is None or lon is None):
        return {"found": False, "reason": "Give an address or a lat/lon."}
    return {"found": False, "reason": "No 3D scan on file for this home. Continue with family photos."}
