"""The inventory page and its endpoints, through FastAPI's test client."""
import io

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from PIL import Image

from ashline import config
from ashline.store import Store
from ashline.web import _ago, create_app


@pytest.fixture
def case(tmp_path):
    store = Store()
    store.set_case("fam1", name="The Example home", owner_token="owner-token", member_token="member-token",
                   loss_date="2026-10-20")
    store.add_member("fam1", "+15550001", "Mom", "owner")
    store.add_member("fam1", "+15550002", "Maya")
    photo = config.case_dir("fam1") / "photos" / "p1.jpg"
    photo.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), (120, 110, 100)).save(photo)
    store.add_photos("fam1", [{"photo_id": "p1", "path": str(photo), "source": "photo", "number": 1,
                               "taken": "2019-11-28", "added_by": "Maya"}])
    p1 = store.get_photo("fam1", "p1")
    sofa = store.add_item("fam1", {"label": "blue velvet sofa", "category": "furniture", "bbox": [100, 200, 500, 500],
                                   "room_label": "Living room"}, source="photo", photo=p1, added_by="Maya")
    again = store.add_item("fam1", {"label": "navy couch", "category": "furniture", "bbox": [50, 100, 400, 400],
                                    "room_label": "Living room"}, source="photo", photo=p1, added_by="Dad")
    store.set_cluster("fam1", sofa, [sofa, again], 1)
    lamp = store.add_item("fam1", {"label": "floor lamp", "category": "lighting", "bbox": [600, 100, 700, 500],
                                   "room_label": "Living room", "qty": 2}, source="photo", photo=p1, added_by="Maya")
    saw = store.add_item("fam1", {"label": "table saw", "category": "tools", "room_label": "Garage"},
                         source="voice", added_by="Mom")
    gone = store.add_item("fam1", {"label": "not ours", "category": "decor"}, source="photo", photo=p1)
    for item_id, mid in ((sofa, 1200), (lamp, 90), (saw, 600)):
        store.update_item("fam1", item_id, price_low=mid * 0.6, price_mid=mid, price_high=mid * 1.5,
                          price_source="claude estimate")
    store.set_item_status("fam1", sofa, "confirmed")
    store.set_item_status("fam1", saw, "confirmed")
    store.set_item_status("fam1", gone, "rejected")
    return TestClient(create_app()), store, {"sofa": sofa, "lamp": lamp, "saw": saw, "gone": gone}


def test_owner_page_shows_meter_items_and_export(case):
    client, _, ids = case
    html = client.get("/p/owner-token").text
    assert "The Example home · 3 items" in html and "2 family members contributing" in html
    assert "$1,800" in html and "documented · 2 items" in html                 # no coverage limit yet
    assert "What is the contents coverage limit" in html                        # asked, with a skip
    assert "Blue velvet sofa" in html and "Table saw" in html and "not ours" not in html.lower()
    assert f'/p/owner-token/crop/{ids["sofa"]}.jpg' in html and "from memory" in html
    assert "2 × $54–$135" in html                                               # two lamps, priced per unit
    assert "Download PDF for insurer" in html and "noindex" in html
    assert html.count('class="item suggested"') == 1 and html.count('class="item confirmed"') == 2


def test_member_page_has_no_export_or_owner_questions(case):
    client, _, ids = case
    html = client.get("/p/member-token").text
    assert "Blue velvet sofa" in html and "Download PDF" not in html and "coverage limit" not in html
    assert client.get("/p/member-token/export.csv").status_code == 403
    assert client.post(f"/p/member-token/item/{ids['lamp']}", json={"action": "remove"}).status_code == 403
    assert client.post("/p/member-token/coverage", json={"limit": 1}).status_code == 403
    assert client.get("/p/not-a-token").status_code == 404
    assert client.get(f"/p/member-token/item/{ids['lamp']}.json").json()["editable"] is False


def test_coverage_turns_the_meter_into_progress(case):
    client, store, _ = case
    assert client.post("/p/owner-token/coverage", json={"limit": "$10,000"}).json() == {"ok": True}
    html = client.get("/p/owner-token").text
    assert "of $4,000 documented" in html and "coverage limit on the policy" not in html
    assert "Proof of loss due in" in html or store.get_case("fam1")["loss_date"] < "2026-08-01"
    assert client.post("/p/owner-token/coverage", json={"limit": "0"}).status_code == 400
    store.set_case("fam1", coverage_limit=None)
    client.post("/p/owner-token/coverage", json={"skip": True})
    assert "coverage limit on the policy" not in client.get("/p/owner-token").text


def test_item_detail_and_images(case):
    client, _, ids = case
    detail = client.get(f"/p/owner-token/item/{ids['sofa']}.json").json()
    assert detail["title"] == "Blue velvet sofa" and detail["editable"] and len(detail["sightings"]) == 2
    assert [s["by"] for s in detail["sightings"]] == ["Maya", "Dad"] and detail["badges"]["photo"] == 2
    assert "rough allowance" in detail["price_basis"]                            # no calibration file: no coverage claim
    for url in (detail["sightings"][0]["image"], f"/p/owner-token/crop/{ids['sofa']}.jpg"):
        response = client.get(url)
        assert response.status_code == 200 and Image.open(io.BytesIO(response.content)).width > 100
    memory = client.get(f"/p/owner-token/item/{ids['saw']}.json").json()
    assert memory["memory_only"] and memory["sightings"][0]["image"] is None and memory["needs_proof"]
    assert client.get(f"/p/owner-token/crop/{ids['saw']}.jpg").status_code == 404
    assert client.get("/p/owner-token/item/99999.json").status_code == 404


def test_owner_actions(case):
    client, store, ids = case
    lamp = ids["lamp"]
    for body in ({"action": "confirm"}, {"action": "rename", "value": "brass floor lamp"},
                 {"action": "qty", "value": 3}, {"action": "price", "value": 150}):
        assert client.post(f"/p/owner-token/item/{lamp}", json=body).status_code == 200
    item = store.get_item("fam1", lamp)
    assert (item["status"], item["label"], item["qty"]) == ("confirmed", "brass floor lamp", 3)
    assert (item["price_low"], item["price_mid"], item["price_high"], item["price_source"]) == (150, 150, 150, "set by family")
    assert "Price set by the family." in client.get(f"/p/owner-token/item/{lamp}.json").json()["price_basis"]
    assert client.post(f"/p/owner-token/item/{lamp}", json={"action": "remove"}).status_code == 200
    assert store.get_item("fam1", lamp)["status"] == "rejected" and "Brass floor lamp" not in client.get("/p/owner-token").text
    assert store.last_event("fam1")["action"] == "item.remove"
    assert store.last_event("fam1")["payload"]["before"]["status"] == "confirmed"       # enough to undo
    assert client.post(f"/p/owner-token/item/{lamp}", json={"action": "explode"}).status_code == 400
    assert client.post(f"/p/owner-token/item/{lamp}", json={"action": "price", "value": 0}).status_code == 400


def test_exports(case):
    client, _, _ = case
    csv_text = client.get("/p/owner-token/export.csv").text
    lines = csv_text.splitlines()
    assert lines[0].startswith("room,category,item,qty,brand_model,age,rcv_low,rcv_mid,rcv_high,source,evidence")
    assert len(lines) == 4 and "floor lamp" not in csv_text                      # confirmed items only
    assert "Living room,furniture,blue velvet sofa,1,,,720.0,1200.0,1800.0,photo,Family photo" in csv_text
    assert "From memory (voice call), needs proof" in csv_text and '"Dad, Maya"' in csv_text
    xlsx = client.get("/p/owner-token/export.xlsx")
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"
    from openpyxl import load_workbook
    book = load_workbook(io.BytesIO(xlsx.content))
    assert book.sheetnames == ["Inventory", "By category"] and book["Inventory"]["C2"].value == "blue velvet sofa"
    pdf = client.get("/p/owner-token/export.pdf")
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-" and len(pdf.content) > 3000
    assert client.get("/p/owner-token/export.docx").status_code == 404


def test_ago():
    from datetime import datetime, timezone
    now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    assert [_ago(t, now) for t in ("2026-11-14T11:59:30+00:00", "2026-11-14T11:40:00+00:00", "2026-11-14T09:00:00+00:00",
                                   "2026-11-12T12:00:00+00:00", None)] == ["just now", "20 min ago", "3 hours ago", "2 days ago", "just now"]
