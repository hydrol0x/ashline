"""The 2D plan: real outlines from a scan, a sketch without one, and a pin per inventory line."""
import json
import math

import pytest
from PIL import Image
from shapely.geometry import Polygon

from ashline import layout, sketch_claude
from ashline.store import Store
from test_chat import MOM, Family, photo, texts


def edge(a, b):
    return {"vertices": [{"id": f"{a[0]},{a[1]}", "position": {"x": a[0], "y": a[1]}},
                         {"id": f"{b[0]},{b[1]}", "position": {"x": b[0], "y": b[1]}}]}


def box_room(room_id, x0, y0, x1, y1, floor="f1"):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges = [edge(corners[n], corners[(n + 1) % 4]) for n in (2, 0, 3, 1)]          # deliberately out of order
    return {"id": room_id, "floor": {"id": floor}, "boundary": {"edges": edges}}


def quaternion(yaw):
    return {"x": 0, "y": 0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}


MODEL = {
    "floors": [{"id": "f1", "label": "1", "sequence": 0}],
    "rooms": [box_room("living", 0, 0, 6, 4), box_room("kitchen", 6, 0, 9, 4)],
    "panoLocations": [{"id": "p1", "position": {"x": 3, "y": 2, "z": 1.5}, "rotation": quaternion(0),
                       "anchor": {"room": {"id": "living"}, "floor": {"id": "f1"}}}],
}


def test_wall_segments_join_into_one_outline():
    points = layout.ring(box_room("r", 0, 0, 6, 4)["boundary"]["edges"])
    assert len(points) == 4 and Polygon(points).area == pytest.approx(24)
    assert layout.ring([]) == []


def test_scan_plan_reads_rooms_and_cameras():
    plan = layout.scan_plan(MODEL)
    assert [r["id"] for r in plan["rooms"]] == ["living", "kitchen"]
    assert plan["cameras"]["p1"]["yaw"] == pytest.approx(0) and plan["cameras"]["p1"]["room"] == "living"
    assert layout.scan_plan({"rooms": [{"id": "x", "boundary": None}]})["rooms"] == []      # no geometry: no room


def test_a_scan_without_walls_gets_rough_rooms_from_its_camera_spots():
    def pano(pid, x, y, room):
        return {"id": pid, "position": {"x": x, "y": y, "z": 1.5}, "rotation": quaternion(0),
                "anchor": {"room": {"id": room}, "floor": {"id": "f1"}}}
    model = {"floors": [{"id": "f1", "label": "1", "sequence": 0}],
             "rooms": [{"id": "big", "floor": {"id": "f1"}, "boundary": {"edges": []}},
                       {"id": "small", "floor": {"id": "f1"}, "boundary": {"edges": []}},
                       {"id": "unseen", "floor": {"id": "f1"}, "boundary": None}],
             "panoLocations": [pano("a", 0, 0, "big"), pano("b", 4, 0, "big"), pano("c", 4, 3, "big"),
                               pano("d", 6, 1, "small")]}
    plan = layout.scan_plan(model)
    assert plan["approx"] and [r["id"] for r in plan["rooms"]] == ["big", "small"]          # no camera, no room
    big, small = (Polygon(r["points"]) for r in plan["rooms"])
    assert big.intersection(small).area < 1e-6                                              # cut back, not overlapping
    assert all(big.contains(Polygon([(x, y), (x + .01, y), (x, y + .01)])) for x, y in ((0, 0), (4, 0), (4, 3)))
    assert small.bounds[0] == pytest.approx(5.0) and big.bounds[2] == pytest.approx(5.0)    # the wall between them
    assert not layout.scan_plan(MODEL)["approx"]
    # Next to measured rooms: a rough room is kept beside them (a patio), dropped if it would lie over one.
    mixed = {**MODEL, "rooms": MODEL["rooms"] + [{"id": "patio", "floor": {"id": "f1"}, "boundary": {"edges": []}},
                                                  {"id": "ghost", "floor": {"id": "f1"}, "boundary": {"edges": []}}],
             "panoLocations": MODEL["panoLocations"] + [pano("p", 4, 6, "patio"), pano("g", 7, 2, "ghost")]}
    plan = layout.scan_plan(mixed)
    assert [r["id"] for r in plan["rooms"]] == ["living", "kitchen", "patio"] and not plan["approx"]


def test_an_item_is_placed_where_its_foot_meets_the_floor():
    camera = {"x": 3.0, "y": 2.0, "yaw": 0.0}
    room = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    # Face 1 looks along +x. The box's bottom edge is three quarters of the way down: 45 degrees below... half way
    # between horizon and bottom edge is b = 0.5, so the floor point is height / 0.5 ahead.
    x, y, exact = layout.floor_spot(camera, 1, [900, 800, 1148, 1536], (2048, 2048), room, height=1.0)
    assert exact and (x, y) == (pytest.approx(5.0), pytest.approx(2.0))
    # Face 2 is a quarter turn to the right: it looks along -y.
    x, y, exact = layout.floor_spot(camera, 2, [900, 800, 1148, 2048], (2048, 2048), room, height=1.0)
    assert exact and (x, y) == (pytest.approx(3.0), pytest.approx(1.0))
    # Something on the right of face 1 is to the right of the camera's line of sight.
    x, y, _ = layout.floor_spot(camera, 1, [1536, 800, 2048, 2048], (2048, 2048), room, height=1.0)
    assert x == pytest.approx(4.0) and y == pytest.approx(1.25)


def test_a_picture_on_the_wall_goes_to_the_wall_not_through_it():
    camera = {"x": 3.0, "y": 2.0, "yaw": 0.0}
    room = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    x, y, exact = layout.floor_spot(camera, 1, [900, 300, 1148, 700], (2048, 2048), room)     # above the horizon
    assert not exact and x == pytest.approx(3 + 3 * layout.WALL_MARGIN) and y == pytest.approx(2.0)
    x, y, exact = layout.floor_spot(camera, 1, [900, 900, 1148, 1100], (2048, 2048), room)    # "floor point" 20 m away
    assert not exact and x < 6
    # Seen through a doorway into the kitchen next door: the floor point there is kept...
    kitchen = Polygon([(6, 0), (9, 0), (9, 4), (6, 4)])
    seen = dict(camera=camera, face=1, bbox=[900, 900, 1148, 1400], size=(2048, 2048), outline=room, others=[room, kitchen])
    x, y, exact = layout.floor_spot(**seen, doors=[[[6, 1.6], [6, 2.4]]])
    assert exact and 6 < x < 9
    # ...but not through a solid wall: a doorway elsewhere on that wall, or no doorway at all.
    x, y, exact = layout.floor_spot(**seen, doors=[[[6, 3.2], [6, 3.9]]])
    assert not exact and x < 6
    assert layout.floor_spot(**seen, doors=[])[0] < 6 and layout.floor_spot(**seen)[0] < 6
    # Standing right at the wall is still in the room, drawn just inside it.
    x, y, exact = layout.floor_spot(camera, 1, [900, 800, 1148, 1380], (2048, 2048), room, height=1.0)     # 2.87 m ahead
    assert exact and x == pytest.approx(3 + 3 * layout.WALL_MARGIN)


def test_doorways_and_camera_height_come_from_the_scan():
    wall = edge((6, 0), (6, 4))
    wall["openings"] = [{"type": "doorway", "width": 0.8, "relativeCenter": 0.25, "lowerElevation": 0.5},
                        {"type": "window", "width": 1.2, "relativeCenter": 0.8, "lowerElevation": 1.4}]
    room = box_room("living", 0, 0, 6, 4)
    room["boundary"]["edges"][3] = wall                               # the wall from (6, 0) to (6, 4)
    model = {**MODEL, "rooms": [room], "panoLocations": [{**MODEL["panoLocations"][0], "position": {"x": 3, "y": 2, "z": 1.92}}]}
    plan = layout.scan_plan(model)
    assert plan["rooms"][0]["doors"] == [[[6, 0.6], [6, 1.4]]]         # measured from the wall's first vertex; no windows
    assert plan["cameras"]["p1"]["h"] == pytest.approx(1.42)          # lens height above that room's floor
    assert "h" not in layout.scan_plan(MODEL)["cameras"]["p1"]         # no doorway, no floor height: the default applies


def test_only_large_furniture_and_appliances_get_a_footprint():
    assert layout.typical("gray leather three-seat power reclining sofa")[0] == "Sofa"
    assert layout.typical("oak coffee table") == ("Coffee table", 1.1, 0.6)          # before the plain "table"
    assert layout.typical("stainless steel French-door refrigerator")[0] == "Fridge"
    for small in ("white bed pillow", "brass table lamp", "black countertop microwave oven", "desk mat", "light gray table top",
                  "red stand mixer", "framed print"):
        assert layout.typical(small) is None, small


def test_a_footprint_stands_against_the_wall_it_is_near_and_stays_in_the_room():
    room = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    # A sofa whose front edge is 1 m in front of the bottom wall (y = 0): drawn with its back on that wall.
    corners = layout.footprint(3.0, 1.0, 2.0, 0.9, room, toward=(0, -1))
    xs, ys = sorted(c[0] for c in corners), sorted(c[1] for c in corners)
    assert (xs[0], xs[-1]) == (pytest.approx(2.0), pytest.approx(4.0))               # 2 m along the wall
    assert ys[0] == pytest.approx(0.04) and ys[-1] == pytest.approx(0.94)            # 0.9 m deep, against the wall
    # In the middle of a big room it faces the camera instead.
    hall = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    corners = layout.footprint(10.0, 10.0, 2.0, 1.0, hall, toward=(1, 0))
    assert sorted(round(c[0], 2) for c in corners) == [10.0, 10.0, 11.0, 11.0]
    assert sorted(round(c[1], 2) for c in corners) == [9.0, 9.0, 11.0, 11.0]
    # A thing too big for where it stands is not drawn through the walls.
    assert layout.footprint(3.0, 1.0, 9.0, 0.9, room, toward=(0, -1)) is None


def test_width_is_measured_from_how_wide_a_thing_looks_and_how_far_it_stands():
    camera = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    # Face 1 looks along +x with a 90 degree view: at 2 m ahead the face is 4 m wide, so half the image is 2 m.
    assert layout.seen_width(camera, 1, [512, 0, 1536, 2048], (2048, 2048), 2.0, 0.0) == pytest.approx(2.0)
    assert layout.seen_width(camera, 1, [512, 0, 1536, 2048], (2048, 2048), 4.0, 1.0) == pytest.approx(4.0)


@pytest.fixture
def scanned(tmp_path):
    store = Store()
    store.upsert_case("c1")
    store.set_rooms("c1", [{"room_id": "living", "label": "living room", "name": "Living room", "area_m2": 24},
                           {"room_id": "kitchen", "label": "kitchen", "name": "Kitchen", "area_m2": 12}])
    layout.plan_path("c1").parent.mkdir(parents=True, exist_ok=True)
    layout.plan_path("c1").write_text(json.dumps(layout.scan_plan(MODEL)))
    path = tmp_path / "face.jpg"
    Image.new("RGB", (2048, 2048)).save(path)
    shot = {"photo_id": "scan-1-f1", "path": str(path), "source": "scan", "room_id": "living",
            "room_label": "Living room", "pano_id": "p1", "face": 1, "x": 3, "y": 2, "z": 1.5}
    store.add_photos("c1", [shot])
    sofa = store.add_item("c1", {"label": "grey sofa", "category": "furniture", "bbox": [900, 800, 1148, 1792]},
                          source="scan", photo=shot)
    store.update_item("c1", sofa, price_mid=900)
    mixer = store.add_item("c1", {"label": "stand mixer", "category": "kitchen", "room_label": "Kitchen"}, source="voice")
    lost = store.add_item("c1", {"label": "bicycle", "category": "sports"}, source="voice")
    return store, sofa, mixer, lost


def test_build_draws_the_scan_with_a_pin_per_line(scanned):
    store, sofa, mixer, lost = scanned
    drawn = layout.build("c1", store)
    assert drawn["kind"] == "scan" and (drawn["width"], drawn["height"]) == (9, 4)
    assert [(r["name"], r["area"]) for r in drawn["rooms"]] == [("Living room", 24.0), ("Kitchen", 12.0)]
    pins = {p["id"]: p for p in drawn["pins"]}
    assert pins[sofa]["exact"] and pins[sofa]["room_id"] == "living" and pins[sofa]["value"] == 900
    assert pins[sofa]["x"] == pytest.approx(3 + layout.CAMERA_HEIGHT / 0.75, abs=0.01) and pins[sofa]["y"] == pytest.approx(2.0)
    # 248 of 2048 pixels wide at 1.9 m is a 0.46 m thing: too narrow for a footprint, so it stays a dot.
    assert "shape" not in pins[sofa]
    # Known only by its room: counted with that room, but no dot, since nobody knows where it stood.
    assert pins[mixer]["room_id"] == "kitchen" and pins[mixer]["x"] is None and not pins[mixer]["exact"]
    # No room at all: listed as not placed.
    assert pins[lost]["x"] is None and pins[lost]["room_id"] is None and drawn["unplaced"] == 1 and drawn["dotted"] == 1
    store.set_item_status("c1", sofa, "rejected")
    assert sofa not in {p["id"] for p in layout.build("c1", store)["pins"]}


def test_a_thing_seen_from_a_room_the_scan_gives_no_outline_for_is_not_lost(scanned):
    store, sofa, mixer, lost = scanned
    plan = layout.scan_plan(MODEL)
    plan["cameras"]["p1"]["room"] = "void"                       # the camera's room has no walls in the scan
    layout.plan_path("c1").write_text(json.dumps(plan))
    drawn = layout.build("c1", store)
    pin = next(p for p in drawn["pins"] if p["id"] == sofa)
    assert pin["x"] is not None and pin["room_id"] is None        # drawn, and counted as outside the rooms


def test_the_scan_can_be_walked_and_what_was_on_file_is_listed(scanned):
    store, sofa, mixer, lost = scanned
    assert layout.build("c1", store)["tour"] is None and layout.build("c1", store)["records"] == []      # no scan id on file
    store.set_home("c1", sid="AbC123xyz90", scan_captured="2026-05-29", aerial_captured="2024-08-13",
                   listing=json.dumps({"beds": 4, "baths": 2.5, "sqft": 2140, "year_built": 1958}))
    drawn = layout.build("c1", store)
    assert drawn["tour"] == "https://my.matterport.com/show/?m=AbC123xyz90&play=1&qs=1&brand=0"
    assert drawn["records"] == [
        {"what": "3D scan of the home", "detail": "made in May 2026 · 2 rooms"},
        {"what": "Aerial photo", "detail": "taken in August 2024"},
        {"what": "Sale listing", "detail": "4 bed · 2.5 bath · 2,140 sq ft · built 1958"}]
    pin = next(p for p in drawn["pins"] if p["id"] == sofa)
    assert pin["sweep"] == "1"                                      # seen in panorama 1: the viewer can start there
    assert layout.tour_url("AbC123xyz90", "13").endswith("&ss=13")
    assert layout.tour_url("not a scan id; drop table", "1") is None and layout.tour_url(None) is None

    from fastapi.testclient import TestClient

    from ashline.web import create_app
    store.set_case("c1", owner_token="tok-owner-123456789012345", member_token="tok-member-12345678901234", name="Test home")
    web = TestClient(create_app(lambda: store))
    assert web.get(f"/p/tok-owner-123456789012345/item/{sofa}.json").json()["tour"].endswith("m=AbC123xyz90&play=1&qs=1&brand=0&ss=1")
    assert web.get(f"/p/tok-owner-123456789012345/item/{mixer}.json").json()["tour"] is None       # never seen in the scan
    page = web.get("/p/tok-owner-123456789012345/plan").text
    assert "The home in 3D" in page and "On file for this address" in page


def test_the_drawing_is_not_mirrored(scanned):
    """The scan's y axis points up, a page's points down: a room at the top of the scan is drawn at the top."""
    store = scanned[0]
    plan = layout.scan_plan({**MODEL, "rooms": MODEL["rooms"] + [box_room("porch", 0, 4, 6, 6)]})
    layout.plan_path("c1").write_text(json.dumps(plan))
    rooms = {r["id"]: r for r in layout.build("c1", store)["rooms"]}
    assert rooms["porch"]["label_at"][1] < rooms["living"]["label_at"][1]


def test_without_a_scan_rooms_are_sketched_in_the_order_seen():
    f = Family().setup()
    f.say(MOM, photos=[photo(656), photo(640)])              # kitchen first, then the living room
    f.wait()
    drawn = layout.build(f.case_id, f.store)
    assert drawn["kind"] == "sketch" and "approximate" in drawn["note"]
    names = [r["name"] for r in drawn["rooms"]]
    assert names[:2] == ["Kitchen", "Living room"] and "Garage" in names and len(names) == len(set(names))
    rooms = {r["name"]: Polygon(r["points"]) for r in drawn["rooms"]}
    assert all(not a.intersection(b).area > 0.01 for n, a in rooms.items() for m, b in rooms.items() if n != m)
    assert rooms["Garage"].area > rooms["Kitchen"].area
    # Without a look at the photos nobody knows where in the room a thing stood: it is listed with
    # its room and gets no dot, rather than a tidy made-up position.
    by_name = {r["name"]: r["id"] for r in drawn["rooms"]}
    assert all(p["x"] is None and p["room_id"] == by_name[p["room"]] for p in drawn["pins"])
    assert drawn["dotted"] == 0 and drawn["unplaced"] == 0 and "a dot appears once" in drawn["note"]


def test_a_model_sketch_sets_sizes_neighbours_and_item_spots():
    f = Family().setup()
    f.say(MOM, photos=[photo(656), photo(640)])
    f.wait()
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])

    class Sketcher:
        def draw(self, request):
            self.request = request
            rooms = [{"name": n, "width_m": 4.0, "depth_m": 3.0, "x_m": 4.0 * k, "y_m": 0.0}
                     for k, n in enumerate(request["rooms"])]
            rooms[0].update(name="kitchen", width_m=5.0, depth_m=2.5)        # case differs; too wide: overlaps the next
            return {"rooms": rooms + [{"name": "Wine cellar", "width_m": 9, "depth_m": 9, "x_m": 50, "y_m": 0}],
                    "items": [{"id": sofa["item_id"], "u": 1.0, "v": 0.5}, {"id": 999999, "u": 0.5, "v": 0.5}]}

    sketcher = Sketcher()
    drawn = sketch_claude.refresh(f.case_id, f.store, sketcher)
    assert f"{sofa['item_id']} | Living room | photo 2 | blue velvet sofa" in sketcher.request["prompt"]
    assert [r["name"] for r in drawn["rooms"]][:2] == ["Kitchen", "Living room"]           # the inventory's own names
    assert "Wine cellar" not in {r["name"] for r in drawn["rooms"]}                         # no invented rooms
    assert [i["id"] for i in drawn["items"]] == [sofa["item_id"]]                           # no invented items

    built = layout.build(f.case_id, f.store)
    rooms = {r["name"]: Polygon(r["points"]) for r in built["rooms"]}
    assert rooms["Kitchen"].area == pytest.approx(12.5)                                     # the model's size is used
    assert all(not a.intersection(b).area > 0.01 for n, a in rooms.items() for m, b in rooms.items() if n != m)
    pin = next(p for p in built["pins"] if p["id"] == sofa["item_id"])
    x0, y0, x1, y1 = rooms["Living room"].bounds
    assert pin["x"] > x0 + 0.8 * (x1 - x0)                                                   # against the right wall
    # Only the thing the model placed has a dot; the rest are still listed with their rooms.
    assert built["dotted"] == 1 and all(p["x"] is None for p in built["pins"] if p["id"] != sofa["item_id"])
    assert "positions are approximate" in built["note"]

    # Nothing changed: the kept sketch is reused without asking again.
    class Never:
        def draw(self, request):
            raise AssertionError("asked again")
    assert sketch_claude.refresh(f.case_id, f.store, Never())["digest"] == drawn["digest"]


def test_the_map_command_sends_a_picture_and_the_link():
    f = Family().setup()
    assert "nothing to put on a plan yet" in texts(f.say(MOM, "map"))[0]
    f.say(MOM, photos=[photo(640)])
    f.wait()
    out = f.say(MOM, "show me the floor plan")
    assert len(out) == 1 and out[0].image.endswith("plan.png") and Image.open(out[0].image).width == 1400
    assert "3 items where they were recorded" in out[0].text and "approximate" in out[0].text
    assert out[0].text.endswith("/plan") and "https://ashline.example/p/" in out[0].text


def test_plan_page(monkeypatch):
    from fastapi.testclient import TestClient

    from ashline.web import create_app
    f = Family().setup()
    f.say(MOM, photos=[photo(640)])
    f.wait()
    client = TestClient(create_app(lambda: f.store))
    token = f.store.get_case(f.case_id)["owner_token"]
    page = client.get(f"/p/{token}/plan")
    assert page.status_code == 200 and "Where things were" in page.text and "blue velvet sofa" in page.text
    data = client.get(f"/p/{token}/plan.json").json()
    assert data["kind"] == "sketch" and len(data["pins"]) == 3
    assert f'href="/p/{token}/plan"' in client.get(f"/p/{token}").text
    assert client.get("/p/not-a-token/plan").status_code == 404
