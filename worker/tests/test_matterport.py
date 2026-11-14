from collections import Counter

from ashline.costar import matterport
from ashline.costar.matterport import parse_rooms, scan_confirmation_message, scan_photos, scan_rooms, select_panos

from fakes import MODEL, FakeMatterport


def test_scan_rooms_skips_void_and_unlabelled_and_keeps_unknowns_unknown():
    out = scan_rooms("fakeA000001", FakeMatterport())
    labels = [r["label"] for r in out["rooms"]]
    assert "Void" not in labels and "" not in labels and out["skipped_rooms"] == 2
    assert labels[0] == "Living Room / Dining Room"            # largest first, both classifications kept
    by = {r["room_id"]: r for r in out["rooms"]}
    assert by["living"]["area_m2"] == 40.0 and by["living"]["height"] is None     # 0.0 means not measured
    assert by["garage"]["area_m2"] == 30.0 and by["garage"]["width"] == 6.0       # falls back to old field
    assert by["bed2"]["furnished"] is False and by["bed1"]["furnished"] is True and by["bath"]["furnished"] is None
    assert by["living"]["pano_count"] == 4 and by["garage"]["pano_count"] == 0
    assert out["capture_notes"] == "staged for open house"
    assert out["pano_count"] == 13 and out["exterior_pano_count"] == 3


def test_select_panos_budget_goes_to_big_valuable_rooms():
    rooms, _ = parse_rooms(MODEL)
    picks = select_panos(MODEL, rooms, per_room=2, max_panos=6, exterior=1)
    count = Counter(p["room_id"] for p in picks)
    assert sum(count.values()) == 6
    assert count["living"] == 3            # 40 m2: one more than per_room
    assert count["bed1"] == 2 and count[None] == 1
    assert "bed2" not in count             # tagged unfurnished
    assert "sliver" not in count and "garage" not in count      # under 1.5 m2; no panorama inside
    assert "hall" not in count and "bath" not in count          # lowest value, budget ran out


def test_select_panos_spreads_views_within_a_room():
    rooms, _ = parse_rooms(MODEL)
    picks = select_panos(MODEL, rooms, per_room=1, max_panos=10, exterior=0)
    living = [p["pano"]["label"] for p in picks if p["room_id"] == "living"]
    assert len(living) == 2 and {"2", "4"} & set(living)        # one near the centre...
    assert {"1", "3"} & set(living)                              # ...and one far from it


def test_select_panos_room_filter():
    rooms, _ = parse_rooms(MODEL)
    picks = select_panos(MODEL, rooms, only_rooms=["bedroom"])
    assert {p["room_id"] for p in picks} == {"bed1", "bed2"}     # an explicit ask overrides "unfurnished"
    picks = select_panos(MODEL, rooms, only_rooms=["exterior"], exterior=2)
    assert [p["room_label"] for p in picks] == ["Exterior", "Exterior"]


def test_scan_photos_saves_horizon_faces_and_never_returns_urls(tmp_path):
    client = FakeMatterport()
    out = scan_photos("fakeA000001", "case1", max_faces=8, exterior=0, captured="2026-03-04", client=client)
    assert out["panos_used"] == 2 and len(out["photos"]) == 8
    assert {p["face"] for p in out["photos"]} == {1, 2, 3, 4}    # not up (0) or down (5)
    assert all(url.rsplit("/", 1)[1][0] in "1234" for url in client.downloads)
    photo = out["photos"][0]
    assert photo["source"] == "scan" and photo["sid"] == "fakeA000001" and photo["room_id"] == "living"
    assert photo["captured"] == "2026-03-04"
    assert "signed.invalid" not in repr(out)
    again = scan_photos("fakeA000001", "case1", max_faces=8, exterior=0, client=client)
    assert len(client.downloads) == 8 and len(again["photos"]) == 8      # cached per case


def test_confirmation_message():
    rooms, _ = parse_rooms(MODEL)
    text = scan_confirmation_message(rooms, "2026-03-04", "staged for open house")
    assert "from March 2026" in text and "2 bedrooms" in text and "a bathroom" in text
    assert "Use it as a starting point?" in text and "staged for open house" in text
    assert "until you confirm" in text
    assert "I found a 3D scan of this address." in scan_confirmation_message([], None)
    assert [r["name"] for r in rooms if r["label"] == "Bedroom"] == ["Bedroom", "Bedroom 2"]


def test_missing_key_is_a_clear_error(monkeypatch):
    import pytest
    from ashline import config
    monkeypatch.delenv("MATTERPORT_API_KEY", raising=False)
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [])
    with pytest.raises(config.MissingKey, match="MATTERPORT_API_KEY"):
        matterport.MatterportClient()
