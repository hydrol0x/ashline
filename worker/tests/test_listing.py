"""listing_record against a made-up kit with the real kit's layout (file names and columns as
listed by `ashline kit-info`). The test kit is not encrypted: the standard library cannot write
encrypted zips."""
import json
import zipfile

import pytest

from ashline import cli
from ashline.costar.listing import kit_info, listing_record, shape

LISTINGS = [
    {"id": 501, "residentialListingId": 9001, "currentPrice": 750000, "listingStatus": "Sold",
     "property": {"bedrooms": 3, "bathrooms": 2, "livingArea": 1500, "yearBuilt": 1962},
     "remarks": "Detached garage and covered patio.", "media": [{"url": "x"}] * 40, "agents": [{"name": "n"}]},
    {"id": 502, "residentialListingId": 9002, "currentPrice": 410000, "property": {"bedrooms": 2}},
]
FEATURES = [{"id": 501, "features": ["Detached garage", "Fireplace"], "highlights": ["Covered patio"]}]


@pytest.fixture
def kit(tmp_path, monkeypatch):
    path = tmp_path / "kit.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("index.md", "# fake kit")
        z.writestr("residential-listings.csv",
                   "Address,City,ListingId,MatterportModelId,ResidentialListingId,State\n"
                   "12 Example Ave,Los Angeles,501,fakeA000001,9001,CA\n"
                   "12 Example St,Los Angeles,502,fakeB000001,9002,CA\n")
        z.writestr("data/residential-listings.jsonl", "\n".join(json.dumps(r) for r in LISTINGS))
        z.writestr("data/residential-listing-features.jsonl", "\n".join(json.dumps(r) for r in FEATURES))
        z.writestr("data/commercial-listing-summary.jsonl", json.dumps({"id": 9001, "propertyType": "Office"}))
        z.writestr("commercial-properties.csv", "PropertyId\n1\n")
    monkeypatch.setenv("COSTAR_KIT_ZIP", str(path))
    monkeypatch.setenv("COSTAR_ZIP_PASSWORD", "not-a-real-password")
    monkeypatch.delenv("COSTAR_KIT_URL", raising=False)
    return path


@pytest.mark.parametrize("args", [{"sid": "fakeA000001"}, {"homes_listing_id": "9001"},
                                  {"homes_listing_id": "501"}, {"homes_listing_id": "9001", "sid": "fakeA000001"}])
def test_join_by_scan_id_or_listing_id(kit, args):
    hit = listing_record(**args)
    assert hit["found"] and hit["listing"]["residentialListingId"] == 9001
    assert hit["listing"]["property"]["bedrooms"] == 3 and hit["listing"]["currentPrice"] == 750000
    assert "media" not in hit["listing"] and "agents" not in hit["listing"]          # bulky parts left out
    assert hit["features"]["features"] == ["Detached garage", "Fireplace"]
    assert hit["ids"]["index"]["MatterportModelId"] == "fakeA000001"
    assert hit["listing"].get("propertyType") != "Office"                              # not the commercial file


def test_listing_without_features_and_misses(kit):
    hit = listing_record(sid="fakeB000001")
    assert hit["found"] and hit["listing"]["id"] == 502 and hit["features"] is None
    assert listing_record("424242")["found"] is False
    assert listing_record(sid="nope")["found"] is False
    assert listing_record()["found"] is False


def test_shape_shows_fields_not_values(kit, capsys):
    cli.main(["listing", "--sid", "fakeA000001", "--shape"])
    out = capsys.readouterr().out
    assert "listing.property.bedrooms: int" in out and "features.features: list[2]" in out
    assert "750000" not in out and "garage" not in out.lower()
    assert shape({"a": [{"b": 1}], "c": None}) == ["a: list[1]", "a[].b: int", "c: NoneType"]


def test_kit_info_and_not_configured(kit, monkeypatch):
    info = kit_info()
    assert info["configured"] and "residential-listings.csv" in info["listing_files"]
    from ashline import config
    for name in ("COSTAR_KIT_ZIP", "COSTAR_KIT_URL", "COSTAR_ZIP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [])
    assert listing_record("9001") == {"found": False, "reason": "CoStar kit not configured."}
    assert kit_info()["configured"] is False
