import json

from ashline.costar.vexcel import (aerial_parcel, choose_feature, in_geofence, list_collections,
                                   parse_attributes, parse_structures, square_around)
from shapely import wkt

from fakes import DETECT, FEATURES, FakeVexcel

LAT, LON = 34.0500, -118.4000


def test_geofence():
    assert in_geofence(34.0459, -118.5265)          # Pacific Palisades
    assert in_geofence(34.1897, -118.1312)          # Altadena
    assert in_geofence(40.3440, -74.6514)           # Princeton campus
    assert not in_geofence(40.7128, -74.0060)       # New York
    assert not in_geofence(-118.4, 34.05)           # lat/lon swapped


def test_outside_geofence_makes_no_call():
    client = FakeVexcel()
    out = aerial_parcel(40.7128, -74.0060, "c", client=client)
    assert out["covered"] is False and client.calls == []


def test_choose_feature_takes_last_flight_before_the_loss():
    feature, after = choose_feature(FEATURES, before="2025-01-07")
    assert feature["properties"]["metadata.collection"] == "fake-city-2024" and after is False
    feature, after = choose_feature(FEATURES)                       # no loss date: newest, not features[0]
    assert feature["properties"]["metadata.collection"] == "fake-city-2026"
    feature, after = choose_feature(FEATURES, before="2019-01-01")  # nothing earlier: newest, flagged
    assert feature["properties"]["metadata.collection"] == "fake-city-2026" and after is True
    assert choose_feature([]) == (None, False)
    assert [c["date"] for c in list_collections(FEATURES)] == ["2020-12-02", "2024-08-06", "2026-07-07"]


def test_structures_and_attributes():
    props = FEATURES[0]["properties"]
    structures = parse_structures(props)
    assert [s["primary"] for s in structures] == [True, False]           # primary first
    assert structures[0]["footprint_sqft"] == 1501 and structures[1]["index"] == 0
    assert parse_structures({"structures": None}) == []                   # the API sends null
    assert parse_attributes(props) == {"trampoline": True, "vehicles": 2}


def test_square_fallback_is_80_m_and_lon_lat_order():
    poly = wkt.loads(square_around(LAT, LON))
    west, south, east, north = poly.bounds
    assert west < LON < east and south < LAT < north
    assert abs((north - south) * 111_320 - 80) < 0.5


def test_aerial_parcel_end_to_end(tmp_path):
    client = FakeVexcel()
    out = aerial_parcel(LAT, LON, "case1", before="2025-01-07", client=client)
    assert out["covered"] and out["parcel_found"] and out["collection"] == "fake-city-2024"
    assert out["capture_date"] == "2024-08-13" and out["after_loss"] is False
    assert len(out["collections"]) == 3 and out["attributes"] == {"trampoline": True, "vehicles": 2}

    labels = {d["label"]: d for d in out["detections"]}
    assert "grill" not in labels                        # on the roof of the house: roof equipment
    assert labels["solar panels"]["on_structure"] == 1  # allowed on a roof
    assert labels["shed"]["on_structure"] == 0          # sits on the mapped outbuilding
    assert labels["car"]["on_structure"] is None and labels["car"]["n"] == 1    # numbered by confidence

    detect_call = next(c for c in client.calls if c[0] == "detect")
    ortho_calls = [c for c in client.calls if c[0] == "ortho"]
    # Detection and overlay use the same collection and layer, or boxes would shift.
    assert detect_call[1]["collection"] == "fake-city-2024" and detect_call[1]["layer"] == "urban"
    assert all(c[1]["collection"] == "fake-city-2024" for c in ortho_calls)
    assert ortho_calls[0][1]["destination-dimensions"] == "467,1400"      # enlarged render
    drawn = [d.get("label") for d in ortho_calls[0][2]]
    assert drawn[0] is None and "1 car" in drawn                          # parcel outline, then boxes
    assert all("footprint_wkt" not in s for s in out["structures"])
    json.dumps(out)

    again = aerial_parcel(LAT, LON, "case1", before="2025-01-07", client=client)
    assert again == out and len([c for c in client.calls if c[0] == "property"]) == 1    # cached per case


def test_no_parcel_falls_back_to_a_square():
    client = FakeVexcel(features=[])
    out = aerial_parcel(LAT, LON, "case2", client=client)
    assert out["parcel_found"] is False and out["structures"] == [] and out["collection"] is None
    assert out["capture_date"] == "2024-08-13"          # read from the image the detection ran on
    assert "collection" not in next(c for c in client.calls if c[0] == "detect")[1]
    assert len(out["detections"]) == len(DETECT["features"])
