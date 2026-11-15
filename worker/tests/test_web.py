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
    assert "<h1>The Example home</h1>" in html and '<span class="count">3 items</span>' in html
    assert "Mom and Maya are on this list" in html                              # the members, by name
    assert "$1,800" in html and "documented · 2 items" in html                 # no coverage limit yet
    assert "What is the contents coverage limit" in html                        # asked, with a skip
    assert "Blue velvet sofa" in html and "Table saw" in html and f'data-removed="1" hidden data-id="{ids['gone']}"' in html
    assert "Removed<span>1</span>" in html                                      # only under the Removed tab
    assert f'/p/owner-token/crop/{ids["sofa"]}.jpg' in html and '<span class="tag memory">From memory</span>' in html
    assert "2 × $54–$135" in html                                               # two lamps, priced per unit
    assert "View PDF for insurer" in html and "noindex" in html
    assert html.count('class="item suggested"') == 1 and html.count('class="item confirmed"') == 2


def test_page_data_has_room_rows_names_and_a_small_plan(case):
    from ashline.web import page_data
    _, store, ids = case
    data = page_data("fam1", store, "owner")
    assert data["room_rows"] == [{"name": "Living room", "count": 2, "waiting": 1},   # the lamp waits for an OK
                                 {"name": "Garage", "count": 1, "waiting": 0}]
    assert data["waiting"] == 1 and data["member_names"] == ["Mom", "Maya"]
    assert data["counts"] == {"backed": 1, "memory": 1, "pending": 1}                 # sofa by photo, saw from memory
    plan = data["plan"]
    named = {r["name"] for r in plan["rooms"] if r["name"]}
    assert named <= {"Living room", "Garage"} and named and len(plan["box"].split()) == 4
    assert all(len(r["points"].split()) >= 3 for r in plan["rooms"])
    sofa = next(i for i in data["items"] if i["id"] == ids["sofa"])
    assert sofa["shot_from"] == "Family photo" and sofa["proof"] == ""


def test_owner_page_draws_the_small_plan_and_filters(case):
    client, _, ids = case
    html = client.get("/p/owner-token").text
    assert 'id="mini-plan"' in html and 'data-room="Living room" tabindex="0"' in html
    assert 'href="/p/owner-token/plan">Open the full plan' in html and "See where things were" not in html
    assert '<em>1 to check</em>' in html and 'data-status="suggested"' in html
    where = client.get(f"/p/owner-token/item/{ids['sofa']}.json").json()["where"]
    assert where is None or set(where) == {"room_id", "x", "y"}


def test_page_renders_with_no_items_and_no_plan(case, monkeypatch):
    client, store, _ = case
    store.set_case("empty1", name="An empty home", owner_token="empty-token", member_token="empty-member")
    html = client.get("/p/empty-token").text
    assert "Nothing here yet" in html and 'id="mini-plan"' not in html and "All rooms<span>0</span>" in html
    from ashline import layout
    monkeypatch.setattr(layout, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no plan")))
    html = client.get("/p/owner-token").text
    assert "Blue velvet sofa" in html and 'id="mini-plan"' not in html and "Where things were" in html


def test_member_page_has_no_export_or_owner_questions(case):
    client, _, ids = case
    html = client.get("/p/member-token").text
    assert "data-removed" not in html and "Removed<span>" not in html             # members never see removed items
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
    assert store.get_item("fam1", lamp)["status"] == "rejected"
    assert f'data-removed="1" hidden data-id="{lamp}"' in client.get("/p/owner-token").text
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
    assert "From memory, needs proof" in csv_text and '"Dad, Maya"' in csv_text
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


def test_inbound_runs_the_conversation(case, monkeypatch):
    import base64
    client, store, _ = case
    first = client.post("/inbound", json={"sender_phone": "+15557777", "sender_name": "Ana", "text": "hi"}).json()
    assert first["messages"][0]["to"] == "+15557777" and first["messages"][0]["text"].startswith("I'm Ashline.")
    assert client.post("/inbound", json={"text": "hi"}).status_code == 400

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (100, 100, 100)).save(buf, format="JPEG")
    store.add_member("fam1", "+15550002", "Maya")
    store.set_chat("+15550002", "fam1")
    sent = client.post("/inbound", json={"chat_id": "+15550002", "sender_phone": "+15550002", "text": "",
                                         "attachments": [{"filename": "a.jpg", "bytes_b64": base64.b64encode(buf.getvalue()).decode()},
                                                         {"bytes_b64": "!!not base64!!"}]}).json()
    assert [m["text"] for m in sent["messages"]] == ["Got it, looking at it."]
    assert store.get_photo("fam1", number=2)["status"] == "queued"
    assert client.post("/tick").json() == {"messages": []}                       # still inside the quiet window

    listed = client.post("/inbound", json={"sender_phone": "+15550002", "text": "list"}).json()["messages"][0]
    assert listed["text"].endswith("/p/member-token") and "image_b64" not in listed
    shown = client.post("/inbound", json={"sender_phone": "+15550002", "text": "show photo 1"}).json()["messages"][0]
    assert Image.open(io.BytesIO(base64.b64decode(shown["image_b64"]))).size == (800, 600)

    monkeypatch.setenv("ASHLINE_INBOUND_SECRET", "s3cret")
    assert client.post("/inbound", json={"sender_phone": "+1", "text": "hi"}).status_code == 401
    assert client.post("/tick").status_code == 401
    ok = client.post("/inbound", json={"sender_phone": "+15550002", "text": "help"}, headers={"X-Ashline-Secret": "s3cret"})
    assert ok.status_code == 200


def test_room_names_typed_differently_are_one_room():
    from ashline.rooms import tidy_room
    assert {tidy_room(r) for r in ("just basement", "Basement", "basement", "the Basement.")} == {"Basement"}
    assert tidy_room("") is None and tidy_room(None) is None



def test_landing_page_shows_the_number(monkeypatch):
    from fastapi.testclient import TestClient
    from ashline.web import create_app
    monkeypatch.setenv("ASHLINE_NUMBER", "+16285550100")
    page = TestClient(create_app()).get("/")
    assert page.status_code == 200 and "(628) 555-0100" in page.text and 'href="sms:+16285550100&body=hi"' in page.text


def test_prices_typed_with_symbols():
    from ashline.web import parse_price
    assert parse_price("$450") == 450 and parse_price("1,200") == 1200 and parse_price("1.2k") == 1200
    assert parse_price("about 80 dollars") == 80 and parse_price("no idea") is None


def test_preview_pages_for_owner_only(case):
    client, _, _ = case
    sheet = client.get("/p/owner-token/view/sheet")
    assert sheet.status_code == 200 and "Blue velvet sofa" in sheet.text and "Table saw" in sheet.text
    assert "/p/owner-token/export.xlsx" in sheet.text and "/p/owner-token/export.csv" in sheet.text
    assert "1 more is waiting for your OK" in sheet.text                       # the lamp is still unconfirmed
    pdf = client.get("/p/owner-token/view/pdf")
    assert pdf.status_code == 200 and "/static/pdfjs/pdf.min.mjs" in pdf.text and "export.pdf?inline=1" in pdf.text
    assert client.get("/p/member-token/view/sheet").status_code == 403
    assert client.get("/p/owner-token/view/other").status_code == 404
    inline = client.get("/p/owner-token/export.pdf?inline=1")
    assert inline.headers["content-type"] == "application/pdf" and inline.headers["content-disposition"].startswith("inline")
    assert client.get("/p/owner-token/export.pdf").headers["content-disposition"].startswith("attachment")
    js = client.get("/static/pdfjs/pdf.worker.min.mjs")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/static/pdfjs/../web.py").status_code == 404
    page = client.get("/p/owner-token").text
    assert "/p/owner-token/view/pdf" in page and "/p/owner-token/view/sheet" in page


def test_brand_assets_are_served(case):
    client, _, _ = case
    for path, kind in (("/static/brand.css", "text/css"), ("/static/img/logo.png", "image/png"),
                       ("/static/fonts/urbanist-latin-500-normal.woff2", "font/woff2"), ("/favicon.ico", "image/png")):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith(kind), path
    assert client.get("/static/../web.py").status_code == 404
    assert client.get("/static/fonts/LICENSE-lora").status_code == 404           # only known file types
    assert '/static/brand.css' in client.get("/p/owner-token").text and '/static/brand.css' in client.get("/").text


def test_asset_urls_carry_a_version_and_errors_are_not_cached(case):
    import re
    client, _, _ = case
    page = client.get("/p/owner-token").text
    v = re.search(r"/static/brand\.css\?v=(\w+)", page).group(1)
    css = client.get(f"/static/brand.css?v={v}").text
    assert f"/static/fonts/urbanist-latin-500-normal.woff2?v={v}" in css and f"/static/img/swoosh.jpg?v={v}" in css
    missing = client.get("/static/nope.css")
    assert missing.status_code == 404 and missing.headers["cache-control"] == "no-store"


def test_coverage_limit_reads_k_and_million_and_refuses_what_it_cannot_read(case):
    client, store, _ = case
    for typed, want in (("400k", 400_000), ("$1.5 million", 1_500_000), ("120,000", 120_000)):
        assert client.post("/p/owner-token/coverage", json={"limit": typed}).status_code == 200
        assert store.get_case("fam1")["coverage_limit"] == want
    for typed in ("abc", "0", "-500", "nan", "inf", "2000000000", "1" * 400):
        res = client.post("/p/owner-token/coverage", json={"limit": typed})
        assert res.status_code == 400, typed
        assert "could not read" in res.json()["detail"]
    assert store.get_case("fam1")["coverage_limit"] == 120_000                  # left as it was
    for day in ("2026-13-45", "soon"):
        assert client.post("/p/owner-token/coverage", json={"limit": "100k", "loss_date": day}).status_code == 400
    assert store.get_case("fam1")["coverage_limit"] == 120_000


def test_bodies_that_are_not_a_json_object_are_refused(case):
    client, _, ids = case
    for url in (f"/p/owner-token/item/{ids['sofa']}", "/p/owner-token/coverage"):
        for body in (b"not json", b"[1,2]", b'"x"'):
            assert client.post(url, content=body, headers={"content-type": "application/json"}).status_code == 400


def test_a_huge_item_id_is_not_found(case):
    client, _, _ = case
    huge = "1" * 23
    assert client.post(f"/p/owner-token/item/{huge}", json={"action": "confirm"}).status_code == 404
    assert client.get(f"/p/owner-token/item/{huge}.json").status_code == 404


def test_prices_and_quantities_out_of_range_are_refused(case):
    client, store, ids = case
    url = f"/p/owner-token/item/{ids['lamp']}"
    for typed in ("1" * 400, "-50", "0", "0.5", "20000000", "abc"):
        assert client.post(url, json={"action": "price", "value": typed}).status_code == 400, typed
    assert store.get_item("fam1", ids["lamp"])["price_mid"] == 90
    assert client.post(url, json={"action": "price", "value": "$10 million"}).status_code == 200
    for typed in ("-5", "0", "1e400", "1000", "2.5", "", "three"):
        assert client.post(url, json={"action": "qty", "value": typed}).status_code == 400, typed
    assert store.get_item("fam1", ids["lamp"])["qty"] == 2
    assert client.post(url, json={"action": "qty", "value": 7}).status_code == 200
    assert store.get_item("fam1", ids["lamp"])["qty"] == 7
    assert client.get("/p/owner-token").status_code == 200


def test_a_price_set_by_hand_keeps_the_receipt_as_proof(case):
    from ashline.meter import meter
    from ashline.pricing import record_price
    client, store, ids = case
    saw = ids["saw"]                                                            # from memory only
    record_price(store, "fam1", saw, "paid", {"amount": 540, "each": 540, "merchant": "Home Depot", "bank_id": "b1"})
    assert meter("fam1", store)["memory"] == 0
    assert client.post(f"/p/owner-token/item/{saw}", json={"action": "price", "value": "700"}).status_code == 200
    assert store.get_item("fam1", saw)["price_source"] == "set by family"
    m = meter("fam1", store)
    assert m["memory"] == 0 and m["backed"] == 1200 + 700
    detail = client.get(f"/p/owner-token/item/{saw}.json").json()
    assert detail["needs_proof"] is False and detail["badges"]["receipt"] == 1
    assert "Card record" in client.get("/p/owner-token").text
