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
    assert {k for k, v in s["has"].items() if v} == {"garage", "washer_dryer", "fireplace"}
    assert s["features"] == {"Interior / Appliances": ["Refrigerator", "Wine Cooler", "Dishwasher"],
                             "Exterior / Parking": ["Detached Garage"]}
    assert s["appliances"] == ["Refrigerator", "Wine Cooler"]            # the dishwasher is part of the house
    assert s["highlights"] == ["Detached two-car garage"] and s["remarks"].startswith("Bright home")


def test_summarize_survives_missing_parts():
    empty = summarize({"listing": None, "features": None})
    assert empty["beds"] is None and empty["appliances"] == [] and not any(empty["has"].values())
    only_listing = summarize({"listing": LISTING, "features": None})
    assert (only_listing["beds"], only_listing["baths"], only_listing["sqft"]) == (3, 2.5, 1480)   # 2 full + 1 half
    assert only_listing["garage_spaces"] == 2 and only_listing["year_built"] == 1960


def test_listing_sentence():
    # Size is left out on purpose: it can describe only the main house of a multi-unit lot.
    assert listing_sentence({"beds": 3, "baths": 2.5, "sqft": 1480, "year_built": 1960}) == \
        " The sale listing describes a 3-bed, 2.5-bath home built in 1960."
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
    assert "The sale listing describes a 3-bed, 2-bath home built in 1962." in result["inside"]["message"]

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


def test_appliances_are_found_by_name_whatever_the_category_is_called():
    features = dict(FEATURES, features=[
        {"category": "Kitchen", "subcategory1": "Equipment", "name": "Wine Refrigerator"},
        {"category": "Laundry", "name": "Dryer"}, {"category": "Laundry", "name": "Washer"},
        {"category": "Kitchen", "name": "Range Hood"}, {"category": "Kitchen", "name": "Quartz Countertops"}])
    assert summarize({"listing": LISTING, "features": features})["appliances"] == ["Wine Refrigerator", "Dryer", "Washer"]


def test_guest_house_and_garage_with_two_buildings_on_the_aerial(kit, offline, monkeypatch, tmp_path):
    """The listing names a guest house and a garage; the aerial shows two more buildings. One
    question lays that out without guessing which building is which."""
    import zipfile as z
    from fakes import FEATURES as PARCEL, HOUSE, SHED, parcel_feature, structure
    features = dict(FEATURES, highlights=[{"kind": "feature", "text": "Detached Guest House"}])
    with z.ZipFile(kit, "w") as zf:
        zf.writestr("residential-listings.csv",
                    "Address,City,ListingId,MatterportModelId,ResidentialListingId,State\n"
                    "12 Example Ave,Los Angeles,501,fakeA000001,9001,CA\n")
        zf.writestr("data/residential-listings.jsonl", json.dumps(LISTING))
        zf.writestr("data/residential-listing-features.jsonl", json.dumps(features))
    other = SHED.replace("34.05012", "34.05020").replace("34.05018", "34.05026")
    parcel = [parcel_feature("fake-city-2024", "urban", "2024-08-06",
                             [structure(True, 1500, HOUSE), structure(False, 721, SHED), structure(False, 278, other)])]
    no_garage = dict(FakeMatterport().model)
    no_garage["rooms"] = [r for r in no_garage["rooms"] if r["id"] != "garage"]
    monkeypatch.setattr(seed.matterport, "MatterportClient", lambda: FakeMatterport(no_garage))
    monkeypatch.setattr(seed.vexcel, "VexcelClient", lambda: FakeVexcel(features=parcel))
    store = Store()
    seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)
    structure_qs = [q for q in gap_questions("fam1", store) if q["kind"] == "structure"]
    assert len(structure_qs) == 1 and len(structure_qs[0]["ref"]) == 2
    assert structure_qs[0]["text"] == (
        "The listing mentions a detached guest house and a 2-car garage, and the aerial photo from August 2024 "
        "shows two more buildings on the lot (about 721 and 278 sq ft). The 3D scan doesn't go inside a garage. "
        "Which building was which, and what was in each?")
    assert not [q for q in gap_questions("fam1", store) if q["kind"] == "listing" and "guest" in q["text"]]


def test_guest_house_wording_and_stale_cache(kit, offline):
    from ashline import config
    from ashline.costar.listing import SUMMARY_VERSION
    for text, expected in [("Detached Guest House", True), ("Permitted ADU", True), ("Maid or Guest Quarters", True),
                           ("Graduated Driveway", False), ("Recreation Room", False)]:
        features = dict(FEATURES, highlights=[{"kind": "feature", "text": text}])
        assert summarize({"listing": LISTING, "features": features})["has"]["guest_house"] is expected, text
    # A summary cached by an earlier version is rebuilt, not reused.
    cache = config.case_dir("fam1") / "listing.json"
    cache.write_text(json.dumps({"beds": 99, "has": {}}))
    result = seed.listing("fam1", "fakeA000001", store=Store())
    assert result["beds"] == 3 and json.loads(cache.read_text())["version"] == SUMMARY_VERSION


# Feature categories and names as the real kit writes them (a listing's feature vocabulary, not a home's data).
REAL_STYLE = [
    {"category": "Inside This Home", "subcategory1": "Additional Homes", "subcategory2": "ADU Guest House", "name": "Detached Guest House"},
    {"category": "Inside This Home", "subcategory1": "Interior Spaces", "subcategory2": "Spaces - Utility", "name": "Storage"},
    {"category": "Inside This Home", "subcategory1": "Outdoor Features", "subcategory2": "Other Structures", "name": "Outdoor Storage"},
    {"category": "Inside This Home", "subcategory1": "Outdoor Features", "subcategory2": "Patio", "name": "Concrete Porch or Patio"},
    {"category": "Inside This Home", "subcategory1": "Laundry", "subcategory2": "Laundry Location", "name": "Laundry Located Outside"},
    {"category": "Community Details", "subcategory1": "Amenities", "subcategory2": "Laundry", "name": "Laundry Facilities"},
    {"category": "Inside This Home", "subcategory1": "Kitchen", "subcategory2": "Appliances - Dishwasher", "name": "Dishwasher"},
    {"category": "Inside This Home", "subcategory1": "Parking", "subcategory2": "Doors", "name": "Two Garage Doors"},
]


def test_real_style_features():
    listing = dict(LISTING, resoValues={"hasGarage": True, "structure": {"garageSpaces": 2.0}})
    s = summarize({"listing": listing, "features": dict(FEATURES, features=REAL_STYLE, highlights=[])})
    assert {k for k, v in s["has"].items() if v} == {"garage", "guest_house", "storage", "outdoor_storage",
                                                     "laundry", "laundry_outside", "patio", "fireplace"}
    assert s["appliances"] == []                       # only a dishwasher is named, and that is part of the house
    assert s["features"]["Inside This Home / Outdoor Features / Other Structures"] == ["Outdoor Storage"]
    community_only = [f for f in REAL_STYLE if f["category"] == "Community Details"]
    assert not summarize({"listing": {}, "features": {"features": community_only}})["has"]["laundry"]
    labels = [f["label"] for f in seed.listing_findings(s)]
    assert labels == ["garage", "guest house", "outdoor storage", "storage", "patio", "laundry"]


def test_real_style_questions(kit, offline, monkeypatch):
    import zipfile as z
    listing = dict(LISTING, resoValues={"hasGarage": True, "structure": {"garageSpaces": 2.0}})
    with z.ZipFile(kit, "w") as zf:
        zf.writestr("residential-listings.csv",
                    "Address,City,ListingId,MatterportModelId,ResidentialListingId,State\n"
                    "12 Example Ave,Los Angeles,501,fakeA000001,9001,CA\n")
        zf.writestr("data/residential-listings.jsonl", json.dumps(listing))
        zf.writestr("data/residential-listing-features.jsonl",
                    json.dumps(dict(FEATURES, features=REAL_STYLE, highlights=[])))
    store = Store()
    result = seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)
    assert result["listing"]["mentions"] == ["fireplace", "garage", "guest house", "laundry", "outdoor storage",
                                             "patio", "storage"]
    queue = gap_questions("fam1", store)
    buildings = next(q["text"] for q in queue if q["kind"] == "structure")
    assert buildings.startswith("The listing mentions a detached guest house, a 2-car garage and outdoor storage, "
                                "and the aerial photo from August 2024 shows a second building on the lot")
    assert buildings.endswith("Which building was which, and what was in each?")
    texts = [q["text"] for q in queue if q["kind"] == "listing"]
    assert ("The listing mentions laundry outside the house, which the 3D scan doesn't show. "
            "Was there a washer and dryer, and were they yours?") in texts
    assert "The listing mentions storage space. Was anything kept there that we haven't covered?" in texts
    # The aerial already raised patio furniture as a yard finding, so the patio is not asked about twice.
    assert not any("patio" in t for t in texts)
    assert not any("outdoor storage" in t or "guest house" in t for t in texts)      # in the buildings question


def test_a_hair_dryer_does_not_answer_the_laundry_question(kit, offline):
    import zipfile as z
    listing = dict(LISTING, resoValues={"hasGarage": True, "structure": {"garageSpaces": 2.0}})
    with z.ZipFile(kit, "w") as zf:
        zf.writestr("residential-listings.csv",
                    "Address,City,ListingId,MatterportModelId,ResidentialListingId,State\n"
                    "12 Example Ave,Los Angeles,501,fakeA000001,9001,CA\n")
        zf.writestr("data/residential-listings.jsonl", json.dumps(listing))
        zf.writestr("data/residential-listing-features.jsonl",
                    json.dumps(dict(FEATURES, features=REAL_STYLE, highlights=[])))
    store = Store()
    seed.seed_home_twin("fam1", sid="fakeA000001", loss_date="2025-01-07", max_faces=16, store=store)

    def laundry_asked():
        return any("laundry" in q["text"] for q in gap_questions("fam1", store) if q["kind"] == "listing")

    for label in ("hair dryer / styling tool", "cordless pressure washer", "stainless steel dishwasher"):
        store.add_item("fam1", {"label": label}, source="voice")
        assert laundry_asked(), label
    store.add_item("fam1", {"label": "white front-load clothes dryer"}, source="voice")
    assert not laundry_asked()
