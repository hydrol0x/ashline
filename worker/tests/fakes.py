"""Made-up API payloads shaped like the real Matterport and Vexcel responses."""


def pano(label, x, y, room=None):
    out = {"id": f"pano{label}", "label": str(label), "position": {"x": x, "y": y, "z": 1.6}}
    if room:
        out["anchor"] = {"room": {"id": room}}
    return out


def room(rid, labels, area=None, tags=("living",), old=None):
    out = {"id": rid, "floor": {"id": "f1"}, "tags": list(tags),
           "classifications": [{"label": l, "confidence": 1.0} for l in labels]}
    if area is not None:
        out["dimensionEstimates"] = {"area": area, "width": 4.0, "depth": area / 4.0, "height": 0.0}
    if old:
        out["dimensions"] = old
    return out


MODEL = {
    "id": "fakeA000001",
    "description": " staged for open house ",
    "floors": [{"id": "f1", "label": "Floor 1"}],
    "rooms": [
        room("living", ["Living Room", "Dining Room"], 40.0),
        room("bed1", ["Bedroom"], 14.0, tags=("bedroom",)),
        room("bed2", ["Bedroom"], 11.0, tags=("bedroom", "unfurnished")),
        room("bath", ["Bathroom"], 4.0, tags=()),
        room("hall", ["Hallway"], 3.0),
        room("garage", ["Garage"], None, old={"areaFloor": 30.0, "width": 6.0, "depth": 5.0}),
        room("void", ["Void"], 2.0),
        room("nolabel", [], 5.0),
        room("sliver", ["Closet"], 0.8),
    ],
    "panoLocations": [
        pano(1, 0, 0, "living"), pano(2, 3, 0, "living"), pano(3, 6, 0, "living"), pano(4, 1.5, 0.2, "living"),
        pano(5, 0, 5, "bed1"), pano(6, 1, 5, "bed1"),
        pano(7, 4, 5, "bed2"),
        pano(8, 8, 5, "bath"),
        pano(9, 2, 3, "hall"),
        pano(10, 3, 9, "sliver"),
        pano(11, 0, -6), pano(12, 9, -6), pano(13, 4, -6),       # outdoors: no room
    ],
}


class FakeMatterport:
    """Stands in for MatterportClient: serves MODEL and tiny JPEG faces."""

    def __init__(self, model=MODEL):
        self.model, self.downloads = model, []

    def graph(self, query, **variables):
        if "skybox" in query:
            return {"model": {"panoLocations": [
                {"id": p["id"], "skybox": {"status": "available",
                                           "children": [f"https://signed.invalid/{p['id']}/{i}?t=secret" for i in range(6)]}}
                for p in self.model["panoLocations"]]}}
        return {"model": self.model}

    def download(self, url):
        import io
        from PIL import Image
        self.downloads.append(url)
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (120, 130, 140)).save(buf, format="JPEG")
        return buf.getvalue()


def parcel_feature(collection, layer, date, structures, **props):
    return {"type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": [[[[-118.4002, 34.0498], [-118.3998, 34.0498],
                                                                   [-118.3998, 34.0502], [-118.4002, 34.0502],
                                                                   [-118.4002, 34.0498]]]]},
            "properties": {"metadata.collection": collection, "metadata.layer-name": layer,
                           "metadata.ortho-date-estimate": f"{date}T00:00:00", "structures": structures, **props}}


def structure(primary, area, wkt):
    return {"primary-structure": primary, "footprint.area": area, "footprint.wkt": wkt,
            "roof-characteristics.material.class": "shingle", "roof-characteristics.shape.value": "hip",
            "roof-objects.solar.value": False}


HOUSE = "POLYGON ((-118.4001 34.0499, -118.3999 34.0499, -118.3999 34.0500, -118.4001 34.0500, -118.4001 34.0499))"
SHED = "POLYGON ((-118.4001 34.05012, -118.40005 34.05012, -118.40005 34.05018, -118.4001 34.05018, -118.4001 34.05012))"

FEATURES = [
    parcel_feature("fake-city-2024", "urban", "2024-08-06", [structure(False, 300.4, SHED), structure(True, 1500.6, HOUSE)],
                   **{"trampoline.detected": True, "vehicles.types.automobile.count": 2}),
    parcel_feature("fake-city-2026", "urban-r", "2026-07-07", None, **{"pool.detected": True}),
    parcel_feature("fake-city-2020", "urban", "2020-12-02", [structure(True, 1500.6, HOUSE)]),
]


def box(lon, lat, d=0.00002):
    return f"POLYGON (({lon - d} {lat - d}, {lon + d} {lat - d}, {lon + d} {lat + d}, {lon - d} {lat + d}, {lon - d} {lat - d}))"


DETECT = {
    "images": [{"image-id": "ortho-1", "metadata": {"last-capture-date": 1723583342000, "width": 200, "height": 600}}],
    "features": [
        {"properties": {"label": "car", "detections": [{"imageId": "ortho-1", "confidence": 0.93, "bbox": box(-118.39985, 34.05015)}]}},
        {"properties": {"label": "grill", "detections": [{"imageId": "ortho-1", "confidence": 0.52, "bbox": box(-118.4000, 34.04995)}]}},
        {"properties": {"label": "solar panels", "detections": [{"imageId": "ortho-1", "confidence": 0.75, "bbox": box(-118.40002, 34.04996)}]}},
        {"properties": {"label": "shed", "detections": [{"imageId": "ortho-1", "confidence": 0.6, "bbox": box(-118.40007, 34.05015)}]}},
        {"properties": {"label": "patio furniture", "detections": [{"imageId": "ortho-1", "confidence": 0.58, "bbox": box(-118.39990, 34.05010)}]}},
    ],
}


class FakeVexcel:
    def __init__(self, features=FEATURES, detect=DETECT):
        self.features, self._detect, self.calls = features, detect, []

    def property(self, lat, lon):
        self.calls.append(("property", lat, lon))
        return self.features

    def detect(self, area_wkt, image_request, labels, threshold):
        self.calls.append(("detect", image_request, labels, threshold))
        return self._detect

    def ortho(self, image_request, drawing=None, buffer_m=0):
        self.calls.append(("ortho", image_request, drawing))
        return b"\xff\xd8fakejpeg"
