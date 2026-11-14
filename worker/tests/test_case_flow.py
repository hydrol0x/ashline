"""Seed -> store -> questions -> export, with fake APIs and a fake detector."""
import csv

import pytest

from ashline import seed
from ashline.costar import index as index_mod
from ashline.dedupe import cluster_scan_items, item_key, merge_threshold, scan_prior
from ashline.export import export_csv, export_rows
from ashline.gaps import gap_questions
from ashline.store import Store

from fakes import FakeMatterport, FakeVexcel

ADDRESS = "12 Example Ave, Los Angeles, CA 90001"


class FakeDetector:
    """Sees a sofa and two stools from every living-room panorama, a bed in bedrooms."""
    calls = 0

    def detect(self, path, room_hint=None):
        FakeDetector.calls += 1
        pano, face = str(path)[:-4].rsplit("scan-", 1)[1].split("-f")
        if room_hint and "Living" in room_hint:
            sofa = ({"label": "grey couch", "category": "furniture", "confidence": 0.4, "bbox": [0, 0, 9, 9]}
                    if int(pano) % 2 == 0 else
                    {"label": "green velvet sofa with brass legs", "category": "furniture",
                     "confidence": 0.5, "bbox": [0, 0, 9, 9]})
            return {1: [sofa],
                    2: [{"label": "oak bar stool", "category": "furniture", "confidence": 0.5, "bbox": [0, 0, 5, 5]},
                        {"label": "oak counter stools with leather seats", "category": "furniture",
                         "confidence": 0.5, "bbox": [6, 0, 9, 5]}]}.get(int(face), [])
        if room_hint == "Bedroom" and face == "1":
            return [{"label": "queen bed with white bedding", "category": "furniture", "confidence": 0.5, "bbox": [0, 0, 9, 9]}]
        return []


@pytest.fixture
def seeded(index, monkeypatch):
    monkeypatch.setattr(index_mod, "_INDEX", index)
    monkeypatch.setattr(seed.matterport, "MatterportClient", FakeMatterport)
    monkeypatch.setattr(seed.vexcel, "VexcelClient", FakeVexcel)
    monkeypatch.setattr(seed.detect_mod, "get_detector", FakeDetector)
    store = Store()
    result = seed.seed_home_twin("fam1", ADDRESS, loss_date="2025-01-07", max_faces=24, store=store)
    return store, result


def test_seed_fills_the_case(seeded):
    store, result = seeded
    assert result["lookup"]["sid"] == "fakeA000001"
    home = store.get_home("fam1")
    assert home["sid"] == "fakeA000001" and home["homes_listing_id"] == "9001"
    assert home["scan_captured"] == "2026-03-04" and home["aerial_captured"] == "2024-08-13"
    assert home["aerial_collection"] == "fake-city-2024" and home["parcel_wkt"].startswith("MULTIPOLYGON")
    assert home["capture_notes"] == "staged for open house"
    assert {r["label"] for r in store.get_rooms("fam1")} >= {"Living Room / Dining Room", "Bedroom", "Garage"}
    photos = store.get_photos("fam1", source="scan")
    assert len(photos) == 24 and all("http" not in p["path"] for p in photos)      # no signed URLs stored
    assert "Use it as a starting point?" in result["inside"]["message"]
    assert result["outside"]["capture_date"] == "2024-08-13" and result["outside"]["outbuildings"] == 1


def test_everything_enters_as_suggested_and_repeats_are_merged(seeded):
    store, result = seeded
    items = store.get_items("fam1")
    assert items and all(i["status"] == "suggested" and i["source"] == "scan" for i in items)
    assert all(i["evidence_date"] == "2026-03-04" for i in items)
    canon = {item_key(i["label"]): i for i in store.get_items("fam1", canonical_only=True)}
    # The living room was shot from 3 panoramas; each saw 1 sofa (or "couch") and 2 stools.
    assert canon["sofa"]["qty"] == 1 and canon["stool"]["qty"] == 2
    assert canon["sofa"]["label"].startswith("green velvet sofa with brass")        # most specific wording
    assert canon["bed"]["qty"] == 1
    assert result["inside"]["detections"] == len(items) > result["inside"]["suggested_items"] == len(canon)
    assert export_rows("fam1", store) == []                                          # nothing confirmed yet


def test_findings_are_questions_not_items(seeded):
    store, _ = seeded
    findings = store.get_findings("fam1")
    kinds = [(f["kind"], f["label"]) for f in findings]
    assert ("structure", "outbuilding") in kinds
    assert ("yard", "patio furniture") in kinds and ("yard", "trampoline") in kinds
    assert ("yard", "shed") not in kinds        # it is the mapped outbuilding: asked once, as a structure
    assert all(f["status"] == "open" and f["source"] == "aerial" for f in findings)
    assert not [i for i in store.get_items("fam1") if i["source"] == "aerial"]


def test_gap_questions_cover_rooms_structures_and_yard(seeded):
    store, _ = seeded
    queue = gap_questions("fam1", store)
    assert [q["weight"] for q in queue] == sorted((q["weight"] for q in queue), reverse=True)
    text = {q["kind"]: q["text"] for q in reversed(queue)}
    assert "second building" in text["structure"] and "August 2024" in text["structure"]
    assert "What was in it?" in text["structure"]
    yard = " ".join(q["text"] for q in queue if q["kind"] == "yard")
    assert "patio furniture" in yard and "trampoline" in yard and "car" not in yard.replace("carport", "")
    rooms = [q for q in queue if q["kind"] == "room"]
    assert rooms[0]["ref"] == "living"          # biggest valuable room with nothing confirmed
    # Suggestions from the scan are waiting there, so it must not claim to have nothing.
    assert "I've spotted 2 possible things in the living room / dining room for you to check" in rooms[0]["text"]
    assert "nothing is confirmed there yet" in rooms[0]["text"]
    garage = next(q for q in rooms if "garage" in q["text"])
    assert "I don't have anything from the garage yet" in garage["text"]
    assert any("garage" in q["text"] for q in rooms)        # no panorama inside, so it must be asked


def test_confirming_shrinks_the_queue_and_fills_the_export(seeded, tmp_path):
    store, _ = seeded
    before = {q["ref"]: q["weight"] for q in gap_questions("fam1", store) if q["kind"] == "room"}
    sofa = next(i for i in store.get_items("fam1", canonical_only=True) if item_key(i["label"]) == "sofa")
    stool = next(i for i in store.get_items("fam1", canonical_only=True) if item_key(i["label"]) == "stool")
    store.set_item_status("fam1", sofa["item_id"], "confirmed")
    store.set_item_status("fam1", stool["item_id"], "rejected")
    assert all(i["status"] == "confirmed" for i in store.get_items("fam1") if item_key(i["label"]) == "sofa")
    after = {q["ref"]: q["weight"] for q in gap_questions("fam1", store) if q["kind"] == "room"}
    assert after["living"] < before["living"]
    living = next(q for q in gap_questions("fam1", store) if q["ref"] == "living")
    assert living["text"].startswith("I only have 1 confirmed thing from the living room / dining room so far.")

    structure = next(f for f in store.get_findings("fam1") if f["kind"] == "structure")
    store.set_finding_status("fam1", structure["finding_id"], "confirmed")
    assert not [q for q in gap_questions("fam1", store) if q["kind"] == "structure"]

    rows = export_rows("fam1", store)
    assert len(rows) == 1 and rows[0]["source"] == "scan" and rows[0]["status"] == "confirmed"
    assert rows[0]["evidence"] == "3D scan of the home captured 2026-03-04"
    assert len(export_rows("fam1", store, include_suggested=True)) > 1
    path = export_csv("fam1", store)
    lines = path.read_text().splitlines()
    assert lines[0].split(",")[9:11] == ["source", "evidence"] and "not legal advice" in lines[-1]
    assert next(csv.DictReader(lines))["room"] == "Living Room / Dining Room"


def test_reseeding_keeps_family_answers(seeded):
    store, _ = seeded
    sofa = next(i for i in store.get_items("fam1", canonical_only=True) if item_key(i["label"]) == "sofa")
    store.set_item_status("fam1", sofa["item_id"], "confirmed")
    structure = next(f for f in store.get_findings("fam1") if f["kind"] == "structure")
    store.set_finding_status("fam1", structure["finding_id"], "rejected")
    calls = FakeDetector.calls
    seed.seed_home_twin("fam1", ADDRESS, loss_date="2025-01-07", max_faces=24, store=store)
    assert FakeDetector.calls == calls                                  # detections are cached per case
    assert [i for i in store.get_items("fam1", status="confirmed")]
    assert [f["status"] for f in store.get_findings("fam1") if f["kind"] == "structure"] == ["rejected"]


def test_no_scan_continues_cleanly(index, monkeypatch):
    monkeypatch.setattr(index_mod, "_INDEX", index)
    monkeypatch.setattr(seed.vexcel, "VexcelClient", FakeVexcel)
    store = Store()
    result = seed.seed_home_twin("fam2", "1 Nowhere Ln, Pacific Palisades, CA 90272", store=store)
    assert result["lookup"]["found"] is False and "inside" not in result and "outside" not in result
    # With a lat/lon the aerial half still runs: the burn areas have aerials but no scans.
    result = seed.seed_home_twin("fam2", "1 Nowhere Ln", lat=34.20, lon=-118.50, loss_date="2025-01-07", store=store)
    assert result["lookup"]["found"] is False and result["outside"]["covered"] is True
    assert store.get_rooms("fam2") == [] and store.get_findings("fam2")


def test_delete_case(seeded):
    store, _ = seeded
    store.delete_case("fam1")
    assert store.get_case("fam1") is None and not store.get_items("fam1") and not store.get_findings("fam1")


def test_store_rejects_unknown_source_and_status():
    store = Store()
    with pytest.raises(ValueError):
        store.add_item("c", {"label": "x"}, source="internet")
    with pytest.raises(ValueError):
        store.set_item_status("c", 1, "approved")


def test_scan_priors():
    a = {"source": "scan", "pano_id": "p1", "room_id": "r1", "x": 0, "y": 0}
    near = {"source": "scan", "pano_id": "p2", "room_id": "r1", "x": 3, "y": 0}
    far = {"source": "scan", "pano_id": "p3", "room_id": "r1", "x": 9, "y": 0}
    other_room = {"source": "scan", "pano_id": "p4", "room_id": "r2", "x": 1, "y": 0}
    family = {"source": "photo"}
    assert scan_prior(a, near) == {"same_scan": True, "same_room": True, "same_pano": False, "pano_distance_m": 3.0}
    assert merge_threshold(a, near) == 0.75 < merge_threshold(a, far) == 0.90 < merge_threshold(a, other_room)
    assert merge_threshold(a, dict(a, face=2)) is None      # two faces of one panorama never overlap
    assert merge_threshold(a, family) == 0.90 == merge_threshold(family, family)


def test_item_key_and_clusters():
    assert item_key("green velvet sofa with brass legs") == item_key("grey couch") == "sofa"
    assert item_key("oak counter stools with leather seats") == "stool"
    assert item_key("black french door refrigerator") == "refrigerator"
    items = [{"item_id": 1, "photo_id": "a", "label": "floor lamp", "confidence": 0.5, "room_id": "r1"},
             {"item_id": 2, "photo_id": "b", "label": "tall floor lamp", "confidence": 0.5, "room_id": "r2"}]
    photos = {"a": {"pano_id": "p1"}, "b": {"pano_id": "p2"}}
    assert len(cluster_scan_items(items, photos)) == 2      # same thing in two rooms is two things
