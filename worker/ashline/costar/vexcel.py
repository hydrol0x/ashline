"""aerial_parcel: the home from above, from Vexcel aerial imagery.

Verified against the API on 14 Nov 2026:
- `property/extract` returns one feature PER IMAGERY COLLECTION (6 to 12 per LA
  parcel, 2020 to 2026), in no useful order, and `latest=true` does not narrow
  it. So `features[0]` is not the newest flight: sort by
  `metadata.ortho-date-estimate`.
- `ortho/extract` and the image request inside `elements/detect` accept a
  `collection`, so the flight can be chosen. Ashline takes the last flight
  BEFORE the loss date: that is what the home looked like when it was lost.
- Each feature already carries precomputed yard attributes (pool, hot tub,
  trampoline, playground, deck, vehicles, solar), which are more reliable than
  open-vocabulary detection for those things.
- Open-vocabulary detection (`sam3`) finds cars and sheds at this resolution.
  Small items (grill, patio set) are usually not found; treat hits as questions.
"""
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone

import requests
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, shape

from .. import config

API = "https://api.vexcelgroup.com/v2"
LAYERS = "urban,urban-r,wide-area,urban-g"
NO_PARCEL_HALF_M = 40            # 80 m square when the point has no parcel record

YARD_LABELS = ["car", "shed", "trampoline", "patio furniture", "grill", "hot tub",
               "swimming pool", "solar panels", "playground set", "boat", "trailer"]

# The hackathon key only answers inside these areas (lon/lat); elsewhere it returns 403.
GEOFENCE = shapely_wkt.loads(
    "MULTIPOLYGON (((-119.0134 33.0945, -116.8949 33.0945, -116.8949 34.3154, -119.0134 34.3154, -119.0134 33.0945)),"
    " ((-96.2185 29.0173, -94.5758 29.0173, -94.5758 30.3804, -96.2185 30.3804, -96.2185 29.0173)),"
    " ((-124.7437 38.0655, -118.9869 38.0655, -118.9869 41.9596, -124.7437 41.9596, -124.7437 38.0655)),"
    " ((-88.2145 30.8016, -88.3285 30.132, -86.2124 30.2269, -85.5125 29.6277, -84.8004 29.5039, -84.9062 30.7876, -88.2145 30.8016)),"
    " ((-74.6943406774712 40.31565454915983, -74.59812447568899 40.31565454915983, -74.59812447568899 40.3713908156042,"
    " -74.6943406774712 40.3713908156042, -74.6943406774712 40.31565454915983)))")

# Precomputed parcel attributes worth asking the family about: key -> plain words.
YARD_ATTRIBUTES = {
    "pool.detected": "swimming pool",
    "pool.types.hot-tub.detected": "hot tub",
    "trampoline.detected": "trampoline",
    "playground.detected": "playground set",
    "deck.detected": "deck",
    "sport-court.detected": "sport court",
    "vehicles.types.boat.detected": "boat",
}

ROOF_LABELS = {"solar panels"}
ANNOTATED_LONG_SIDE = 1400      # pixels; a parcel crop is only a few hundred pixels natively
RENDER_BUFFER_M = 6             # margin around the parcel in the rendered images

BOX_COLOR = "rgb(0,229,255)"
PARCEL_COLOR = "rgb(255,204,0)"


class VexcelError(RuntimeError):
    pass


class VexcelClient:
    def __init__(self, key: str | None = None):
        self.headers = {"X-API-KEY": key or config.env("VEXCEL_API_KEY", required=True)}

    def send(self, method: str, path: str, **kwargs) -> requests.Response:
        """Call the API; on 429 wait and retry with exponential backoff."""
        for attempt in range(6):
            resp = requests.request(method, f"{API}/{path}", headers=self.headers, **kwargs)
            if resp.status_code != 429 or attempt == 5:
                return resp
            time.sleep(min(2 ** attempt, 30) + random.random())
        return resp

    def property(self, lat: float, lon: float) -> list[dict]:
        """Parcel features for a point, one per imagery collection. [] when there is no parcel."""
        resp = self.send("GET", "property/extract", timeout=120, params={
            "wkt": f"POINT({lon} {lat})", "srid": 4326, "layer": LAYERS,     # WKT is lon then lat
            "include": "precomputed", "unit-system": "imperial"})
        if resp.status_code == 404:
            return []
        self._check(resp, "property/extract")
        return resp.json().get("features") or []

    def detect(self, area_wkt: str, image_request: dict, labels: list[str], threshold: float) -> dict:
        resp = self.send("POST", "elements/detect", timeout=600, json={
            "wkt": area_wkt, "srid": 4326, "buffer-meter": 0,
            "images": [{"endpoint": "ortho/extract", "request": image_request}],
            "model-input": {
                "elements": [{"label": label, "model": "sam3", "detection-threshold": threshold}
                             for label in labels],
                "include": ["bbox", "confidence"]},
        })
        self._check(resp, "elements/detect")
        return resp.json()

    def ortho(self, image_request: dict, drawing: list[dict] | None = None, buffer_m: int = 0) -> bytes:
        body = {**image_request, "buffer-meter": buffer_m, "image-format": "jpeg"}
        if drawing:
            body["drawing"] = drawing
        resp = self.send("POST", "ortho/extract", timeout=300, json=body)
        self._check(resp, "ortho/extract")
        return resp.content

    @staticmethod
    def _check(resp: requests.Response, what: str) -> None:
        if resp.status_code == 401:
            raise VexcelError("Vexcel rejected the key (401). Check VEXCEL_API_KEY.")
        if resp.status_code == 403:
            raise VexcelError("Vexcel answered 403: the point is outside the area the hackathon key covers.")
        if not resp.ok:
            raise VexcelError(f"Vexcel {what} failed ({resp.status_code}): {resp.text[:300]}")


# ------------------------------------------------------------- pure helpers
def in_geofence(lat: float, lon: float) -> bool:
    return GEOFENCE.contains(Point(lon, lat))


def square_around(lat: float, lon: float, half_m: float = NO_PARCEL_HALF_M) -> str:
    """WKT square (lon lat order) centred on a point, half_m metres from centre to edge."""
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * math.cos(math.radians(lat)))
    w, e, s, n = lon - dlon, lon + dlon, lat - dlat, lat + dlat
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _date(feature: dict) -> str:
    return ((feature.get("properties") or {}).get("metadata.ortho-date-estimate") or "")[:10]


def list_collections(features: list[dict]) -> list[dict]:
    """Imagery flights over the parcel, oldest first."""
    out = [{"collection": f["properties"].get("metadata.collection"),
            "layer": f["properties"].get("metadata.layer-name"),
            "date": _date(f) or None} for f in features]
    return sorted(out, key=lambda c: c["date"] or "")


def choose_feature(features: list[dict], before: str | None = None) -> tuple[dict | None, bool]:
    """Pick the flight to use: the last one before `before` (YYYY-MM-DD), else the newest.

    Returns (feature, after_loss). after_loss is True when a loss date was given
    but every flight is later, so the image shows the lot after the loss.
    """
    dated = sorted((f for f in features if _date(f)), key=_date)
    if not dated:
        return (features[0], False) if features else (None, False)
    if before:
        earlier = [f for f in dated if _date(f) < before[:10]]
        if earlier:
            return earlier[-1], False
        return dated[-1], True
    return dated[-1], False


def parse_structures(props: dict) -> list[dict]:
    out = []
    for i, s in enumerate(props.get("structures") or []):       # can be null
        area = s.get("footprint.area")
        out.append({
            "index": i,
            "primary": bool(s.get("primary-structure")),
            "footprint_sqft": round(area) if area else None,
            "roof_material": s.get("roof-characteristics.material.class"),
            "roof_shape": s.get("roof-characteristics.shape.value"),
            "solar": bool(s.get("roof-objects.solar.value")),
            "footprint_wkt": s.get("footprint.wkt"),
        })
    out.sort(key=lambda s: (not s["primary"], -(s["footprint_sqft"] or 0)))
    return out


def parse_attributes(props: dict) -> dict:
    """Precomputed yard findings that are true for this flight, in plain words."""
    found = {name: True for key, name in YARD_ATTRIBUTES.items() if props.get(key)}
    count = props.get("vehicles.types.automobile.count") or props.get("vehicles.count")
    if count:
        found["vehicles"] = int(count)
    if any(s.get("roof-objects.solar.value") for s in props.get("structures") or []):
        found["solar panels"] = True
    return found


def parse_detections(detect: dict, structures: list[dict]) -> list[dict]:
    """Flatten the detect response to [{n, label, confidence, bbox_wkt, on_structure}]."""
    footprints = []
    for s in structures:
        try:
            footprints.append((s["index"], s["primary"], shapely_wkt.loads(s["footprint_wkt"])))
        except Exception:
            continue
    out = []
    for feature in detect.get("features") or []:
        props = feature.get("properties") or {}
        for d in props.get("detections") or []:
            hit = {"label": props.get("label"), "confidence": round(float(d.get("confidence") or 0), 2),
                   "bbox_wkt": d.get("bbox"), "on_structure": None}
            on_main_roof = False
            try:
                centre = shapely_wkt.loads(d["bbox"]).centroid
                for index, primary, poly in footprints:
                    if poly.contains(centre):
                        hit["on_structure"] = index      # e.g. a "shed" that is a mapped outbuilding
                        on_main_roof = primary
                        break
            except Exception:
                pass
            # A "grill" or "car" on the roof of the house is roof equipment, not a yard item.
            if on_main_roof and hit["label"] not in ROOF_LABELS:
                continue
            out.append(hit)
    out.sort(key=lambda h: -h["confidence"])
    for n, hit in enumerate(out, 1):
        hit["n"] = n
    return out


def _capture_date(detect: dict, fallback: str | None) -> str | None:
    """Capture date of the image the detection ran on (epoch ms in the response)."""
    for image in detect.get("images") or []:
        stamp = (image.get("metadata") or {}).get("last-capture-date")
        if stamp:
            return datetime.fromtimestamp(stamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return fallback


def _render_size(detect: dict, area_wkt: str, buffer_m: float) -> dict:
    """Ask for an enlarged render so the labels are readable.

    The detection image is the bare parcel crop. The render adds `buffer_m` of
    margin on every side, so labels of boxes at the parcel edge are not cut
    off; its pixel size is worked out from the crop's size and extent.
    """
    for image in detect.get("images") or []:
        meta = image.get("metadata") or {}
        width, height = meta.get("width"), meta.get("height")
        if not (width and height):
            continue
        west, south, east, north = shapely_wkt.loads(area_wkt).bounds
        metres_wide = (east - west) * 111_320 * math.cos(math.radians((south + north) / 2))
        metres_high = (north - south) * 111_320
        if metres_wide <= 0 or metres_high <= 0:
            continue
        new_w = width * (metres_wide + 2 * buffer_m) / metres_wide
        new_h = height * (metres_high + 2 * buffer_m) / metres_high
        scale = max(1.0, ANNOTATED_LONG_SIDE / max(new_w, new_h))
        return {"destination-dimensions": f"{round(new_w * scale)},{round(new_h * scale)}"}
    return {}


# -------------------------------------------------------------------- tool
def aerial_parcel(lat: float, lon: float, case_id: str, before: str | None = None,
                  labels: list[str] | None = None, threshold: float = 0.5,
                  client: VexcelClient | None = None, refresh: bool = False) -> dict:
    """Parcel, structures, yard findings and an annotated aerial image for a point.

    `before` is the loss date (YYYY-MM-DD): imagery comes from the last flight
    before it. Results are cached in cases/<case>/aerial/ (git-ignored).
    Everything returned is a suggestion to ask the family about, never an item.
    """
    lat, lon = float(lat), float(lon)
    if not in_geofence(lat, lon):
        return {"covered": False, "source": "aerial",
                "reason": "No aerial imagery for this location with the hackathon key."}

    folder = config.case_dir(case_id) / "aerial"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "aerial.json"
    labels = labels or YARD_LABELS
    key = {"lat": round(lat, 6), "lon": round(lon, 6), "before": before, "labels": labels, "threshold": threshold}
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text())
        if cached.get("request") == key:
            return cached

    client = client or VexcelClient()
    features = client.property(lat, lon)
    feature, after_loss = choose_feature(features, before)

    if feature is None:
        # A point on a road or open land has no parcel record. Imagery and
        # detection accept any area, so carry on with a square around the point.
        area_wkt = square_around(lat, lon)
        props, structures, attributes = {}, [], {}
        image_request = {"layer": LAYERS, "srid": 4326, "wkt": area_wkt, "crop": "bounding_box"}
    else:
        props = feature["properties"]
        area_wkt = shape(feature["geometry"]).wkt
        structures, attributes = parse_structures(props), parse_attributes(props)
        # Detection and overlay must use the same source image, or boxes shift.
        image_request = {"layer": props.get("metadata.layer-name") or LAYERS, "srid": 4326,
                         "wkt": area_wkt, "crop": "bounding_box"}
        if props.get("metadata.collection"):
            image_request["collection"] = props["metadata.collection"]

    detect = client.detect(area_wkt, image_request, labels, threshold)
    detections = parse_detections(detect, structures)

    # The API draws the boxes itself: no photo pixels are edited by a generative model.
    drawing = [{"wkt": area_wkt, "outline-width": 3, "outline-color": PARCEL_COLOR}]
    for hit in detections:
        drawing.append({"wkt": hit["bbox_wkt"], "outline-color": BOX_COLOR, "outline-width": 3,
                        "label": f"{hit['n']} {hit['label']}", "label-color": BOX_COLOR, "label-size": 18})
    size = _render_size(detect, area_wkt, RENDER_BUFFER_M)
    annotated = folder / "annotated.jpg"
    annotated.write_bytes(client.ortho({**image_request, **size}, drawing, buffer_m=RENDER_BUFFER_M))
    plain = folder / "ortho.jpg"
    plain.write_bytes(client.ortho({**image_request, **size}, buffer_m=RENDER_BUFFER_M))

    result = {
        "covered": True,
        "source": "aerial",
        "request": key,
        "parcel_found": feature is not None,
        "parcel_wkt": area_wkt,
        "collection": props.get("metadata.collection"),
        "capture_date": _capture_date(detect, _date(feature) if feature else None),
        "after_loss": after_loss,              # True: no flight predates the loss date
        "collections": list_collections(features),
        "structures": [{k: v for k, v in s.items() if k != "footprint_wkt"} for s in structures],
        "pool": bool(props.get("pool.detected")),
        "attributes": attributes,
        "detections": detections,
        "annotated_path": str(annotated),
        "ortho_path": str(plain),
    }
    cache.write_text(json.dumps(result, indent=1))
    return result
