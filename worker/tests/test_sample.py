"""Copying a ready-made home from another data folder."""
import json

import pytest
from PIL import Image

from ashline import config, layout, sample
from ashline.store import Store


def a_home_elsewhere(root):
    """A small home in another data folder: two photos, a merged repeat, a sketch with dots."""
    root.mkdir(parents=True, exist_ok=True)
    there = Store(str(root / "ashline.sqlite"))
    there.set_case("old", name="The Example home", invite_code="HOME-1111", owner_token="own", member_token="mem")
    folder = root / "old" / "photos"
    folder.mkdir(parents=True)
    for n in (1, 2):
        Image.new("RGB", (400, 300), (90, 90, 90)).save(folder / f"p{n}.jpg")
    there.add_photos("old", [{"photo_id": f"p{n}", "path": f"/somewhere/else/cases/old/photos/p{n}.jpg", "source": "photo",
                              "number": n, "room_label": "Living room"} for n in (1, 2)])
    for _ in range(3):                                           # so the numbers there differ from the numbers here
        there.add_item("old", {"label": "filler", "category": "other"}, source="voice")
    sofa = there.add_item("old", {"label": "blue sofa", "category": "furniture", "bbox": [10, 10, 200, 200],
                                  "room_label": "Living room"}, source="photo", photo=there.get_photo("old", "p1"))
    again = there.add_item("old", {"label": "navy couch", "category": "furniture", "bbox": [5, 5, 150, 150],
                                   "room_label": "Living room"}, source="photo", photo=there.get_photo("old", "p2"))
    there.set_cluster("old", sofa, [sofa, again], 1)
    (root / "old" / "sketch.json").write_text(json.dumps({
        "rooms": [{"name": "Living room", "width_m": 4, "depth_m": 4, "x_m": 0, "y_m": 0}],
        "items": [{"id": sofa, "u": 0.5, "v": 0.5}], "digest": "old"}))
    return sofa


def test_a_home_is_copied_with_new_numbers_and_its_plan(tmp_path):
    old_sofa = a_home_elsewhere(tmp_path / "there")
    here = Store()
    here.set_case("mine", name="Ours", invite_code="HOME-1111", owner_token="own", member_token="other")
    here.add_item("mine", {"label": "our lamp", "category": "lighting"}, source="voice")
    got = sample.import_case(tmp_path / "there", "old", "demo-home")
    assert got["items"] == 5 and got["photos"] == 2 and got["name"] == "The Example home"
    assert got["invite_code"] != "HOME-1111" and got["owner_token"] != "own" and got["member_token"] == "mem"
    lines = here.get_items("demo-home", canonical_only=True)
    sofa = next(i for i in lines if i["label"] == "blue sofa")
    assert sofa["item_id"] != old_sofa and len(here.get_sightings("demo-home", sofa["item_id"])) == 2
    photo = here.get_photo("demo-home", "p1")
    assert photo["path"].startswith(str(config.case_dir("demo-home"))) and Image.open(photo["path"]).size == (400, 300)
    drawn = layout.build("demo-home", here)
    pin = next(p for p in drawn["pins"] if p["id"] == sofa["item_id"])
    assert drawn["kind"] == "sketch" and pin["x"] is not None                 # its dot followed it to the new number
    assert [i["label"] for i in here.get_items("mine")] == ["our lamp"]       # the home already here is untouched
    with pytest.raises(sample.ImportError_):
        sample.import_case(tmp_path / "there", "old", "demo-home")            # not twice
    with pytest.raises(sample.ImportError_):
        sample.import_case(tmp_path / "there", "nope")
    with pytest.raises(sample.ImportError_):
        sample.import_case(tmp_path / "nowhere", "old")
