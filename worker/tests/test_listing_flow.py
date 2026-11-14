"""The CoStar listing as the third data layer: summary, family message and combined questions.
The record below is made up; only its field names follow the real kit."""
import json
import sqlite3
import zipfile

import pytest

from ashline import seed
from ashline.costar import index as index_mod
from ashline.costar.listing import summarize
from ashline.costar.matterport import listing_sentence, scan_confirmation_message
from ashline.gaps import gap_questions
from ashline.store import Store

from fakes import FakeMatterport, FakeVexcel

LISTING = {"id": 501, "residentialListingId": 9001, "listDate": "2026-02-10T00:00:00Z", "closeDate": "2026-03-20",
           "closePrice": 750000.0, "currentPrice": 760000.0, "remarks": "Bright home with a detached garage.",
           "property": {"bedsMax": 3, "bathsFull": 2, "bathsHalf": 1, "sqftMax": 1480, "yearBuilt": 1960, "stories": 1},
           "resoValues": {"hasGarage": True, "hasWasherDryer": True, "hasBasement": False, "hasSpa": False,
                          "hasStorage": False, "structure": {"garageSpaces": 2.0, "hasCarport": False}},
           "costarFeatureValues": {"hasGarage": True, "hasFireplace": True},
           "media": [{"url": "x"}], "agents": [{"name": "n"}]}
FEATURES = {"id": 501,
            "derivedValues": {"homeSquareFeet": 1500, "totalBedrooms": 3, "totalBathrooms": 2, "yearBuilt": 1962,
                              "totalGarageSpaces": 2, "numberOfStories": 1, "lotSizeInAcres": 0.14,
                              "propertyType": "Single Family"},
            "features": [{"category": "Interior", "subcategory1": "Appliances", "subcategory2": "", "name": "Refrigerator"},
                         {"category": "Interior", "subcategory1": "Appliances", "subcategory2": "", "name": "Wine Cooler"},
                         {"category": "Interior", "subcategory1": "Appliances", "subcategory2": "", "name": "Dishwasher"},
                         {"category": "Exterior", "subcategory1": "Parking", "subcategory2": "", "name": "Detached Garage"}],
            "highlights": [{"kind": "feature", "text": "Detached two-car garage"}],
            "sections": []}


def test_summarize_picks_the_useful_fields():
    s = summarize({"listing": LISTING, "features": FEATURES})
    assert (s["beds"], s["baths"], s["sqft"], s["year_built"], s["garage_spaces"]) == (3, 2, 1500, 1962, 2)
    assert s["list_date"] == "2026-02-10" and s["price"] == 750000.0 and s["property_type"] == "Single Family"
    assert s["has"] == {"garage": True, "carport": False, "basement": False, "storage": False,
                        "washer_dryer": True, "spa": False, "fireplace": True}
    assert s["appliances"] == ["Refrigerator", "Wine Cooler"]            # the dishwasher is part of the house
    assert s["highlights"] == ["Detached two-car garage"] and s["remarks"].startswith("Bright home")


def test_summarize_survives_missing_parts():
    empty = summarize({"listing": None, "features": None})
    assert empty["beds"] is None and empty["appliances"] == [] and not any(empty["has"].values())
    only_listing = summarize({"listing": LISTING, "features": None})
    assert (only_listing["beds"], only_listing["baths"], only_listing["sqft"]) == (3, 2.5, 1480)   # 2 full + 1 half
    assert only_listing["garage_spaces"] == 2 and only_listing["year_built"] == 1960


def test_listing_sentence():
    assert listing_sentence({"beds": 3, "baths": 2.5, "sqft": 1480, "year_built": 1960}) == \
        " The sale listing describes a 3-bed, 2.5-bath home of 1,480 sq ft built in 1960."
    assert listing_sentence({"beds": 3}) == " The sale listing describes a 3-bed home."
    assert listing_sentence({}) == "" and listing_sentence(None) == ""
    text = scan_confirmation_message([], "2026-03-04", None, {"beds": 3, "baths": 2})
    assert "The sale listing describes a 3-bed, 2-bath home. Use the scan as a starting point?" in text
    assert "Use it as a starting point?" in scan_confirmation_message([], "2026-03-04")


class Sees:
    """Detector that sees a refrigerator in the living room and nothing else."""
    def detect(self, path, room_hint=None):
        if room_hint and "Living" in room_hint and str(path).endswith("f1.jpg"):
            return [{"label": "steel french door refrigerator", "category": "appliances", "confidence": 0.8,
                     "bbox": [0, 0, 9, 9]}]
        return []


@pytest.fixture
def kit(tmp_path, monkeypatch):
    path = tmp_path / "kit.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("residential-listings.csv",
                   "Address,City,ListingId,MatterportModelId,ResidentialListingId,State\n"
                   "12 Example Ave,Los Angeles,501,fakeA000001,9001,CA\n")
        z.writestr("data/residential-listings.jsonl", json.dumps(LISTING))
        z.writestr("data/residential-listing-features.jsonl", json.dumps(FEATURES))
    monkeypatch.setenv("COSTAR_KIT_ZIP", str(path))
    monkeypatch.setenv("COSTAR_ZIP_PASSWORD", "not-a-real-password")
    return path


@pytest.fixture
def offline(index, monkeypatch):
    monkeypatch.setattr(index_mod, "_INDEX", index)
    monkeypatch.setattr(seed.matterport, "MatterportClient", FakeMatterport)
    monkeypatch.setattr(seed.vexcel, "VexcelClient", FakeVexcel)
    monkeypatch.setattr(seed.detect_mod, "get_detector", Sees)


def test_seed_uses_all_three_layers(kit, offline):
    store = Store()
    result = seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)

    assert result["listing"] == {"found": True, "beds": 3, "baths": 2, "sqft": 1500, "year_built": 1962,
                                 "garage_spaces": 2, "list_date": "2026-02-10",
                                 "mentions": ["fireplace", "garage", "washer and dryer"],
                                 "appliances": ["Refrigerator", "Wine Cooler"]}
    assert store.get_home("fam1")["listing"]["sqft"] == 1500
    assert "The sale listing describes a 3-bed, 2-bath home of 1,500 sq ft built in 1962." in result["inside"]["message"]

    questions = {q["kind"]: q for q in reversed(gap_questions("fam1", store))}
    # Listing + aerial + what the scan did not cover, in one question. (The fake scan has a room
    # labelled Garage, so the "scan doesn't show inside it" sentence is left out here.)
    assert questions["structure"]["text"].startswith(
        "The listing mentions a 2-car garage, and the aerial photo from August 2024 shows a second building")
    listing_qs = [q["text"] for q in gap_questions("fam1", store) if q["kind"] == "listing"]
    assert "The listing mentions a washer and dryer. Were they yours, and were they still there?" in listing_qs
    # The refrigerator was seen in the scan, so only the wine cooler is asked about.
    assert any("wine cooler" in t and "refrigerator" not in t for t in listing_qs)
    assert not any("garage" in t for t in listing_qs)            # asked once, with the aerial
    assert not [i for i in store.get_items("fam1") if i["source"] not in ("scan",)]     # questions, never items


def test_listing_questions_stop_once_answered_or_seen(kit, offline):
    store = Store()
    seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)
    washer = next(f for f in store.get_findings("fam1") if f["label"] == "washer and dryer")
    store.set_finding_status("fam1", washer["finding_id"], "confirmed")
    store.add_item("fam1", {"label": "black wine cooler"}, source="voice")
    seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)   # re-seed
    texts = [q["text"] for q in gap_questions("fam1", store) if q["kind"] == "listing"]
    assert texts == []


def test_garage_only_in_the_listing(kit, offline, monkeypatch):
    """No outbuilding on the aerial and no garage in the scan: the listing alone raises it."""
    no_garage = dict(FakeMatterport().model)
    no_garage["rooms"] = [r for r in no_garage["rooms"] if r["id"] != "garage"]
    monkeypatch.setattr(seed.matterport, "MatterportClient", lambda: FakeMatterport(no_garage))
    monkeypatch.setattr(seed.vexcel, "VexcelClient", lambda: FakeVexcel(features=[]))
    store = Store()
    seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)
    texts = [q["text"] for q in gap_questions("fam1", store) if q["kind"] == "listing"]
    assert any(t.startswith("The listing mentions a 2-car garage, but the 3D scan doesn't show inside it.") for t in texts)


def test_no_kit_is_not_an_error(offline):
    store = Store()
    result = seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)
    assert result["listing"] == {"found": False, "reason": "CoStar kit not configured."}
    assert "sale listing" not in result["inside"]["message"] and "Use it as a starting point?" in result["inside"]["message"]
    assert store.get_home("fam1")["listing"] is None
    assert not [q for q in gap_questions("fam1", store) if q["kind"] == "listing"]


def test_database_from_an_earlier_version_gains_the_listing_column(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE home (case_id TEXT PRIMARY KEY, sid TEXT, lat REAL, lon REAL, parcel_wkt TEXT,"
               " homes_listing_id TEXT, scan_captured TEXT, capture_notes TEXT, aerial_collection TEXT, aerial_captured TEXT)")
    db.execute("INSERT INTO home (case_id, sid) VALUES ('old', 'fakeA000001')")
    db.commit(); db.close()
    store = Store(str(path))
    assert store.get_home("old")["sid"] == "fakeA000001" and store.get_home("old")["listing"] is None
    store.set_home("old", listing={"beds": 3})
    assert Store(str(path)).get_home("old")["listing"] == {"beds": 3}
