"""The list page's live meter and small wording fixes."""
import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ashline.store import Store
from ashline.web import create_app


@pytest.fixture
def case():
    store = Store()
    store.set_case("fam1", name="The Example home", owner_token="owner-token", member_token="member-token")
    sofa = store.add_item("fam1", {"label": "blue velvet sofa", "category": "furniture", "room_label": "Living room"},
                          source="voice", added_by="Maya")
    lamp = store.add_item("fam1", {"label": "floor lamp", "category": "lighting", "room_label": "Living room"},
                          source="voice", added_by="Maya")
    for item_id, mid in ((sofa, 1200), (lamp, 90)):
        store.update_item("fam1", item_id, price_low=mid * 0.6, price_mid=mid, price_high=mid * 1.5,
                          price_source="claude estimate")
    store.set_item_status("fam1", sofa, "confirmed")
    return TestClient(create_app()), store, {"sofa": sofa, "lamp": lamp}


def test_meter_json_follows_an_answer_and_matches_the_page(case):
    client, _, ids = case
    before = client.get("/p/owner-token/meter.json").json()
    assert before["documented"] == "$1,200" and before["figure_note"] == "documented · 1 item"
    assert before["pending"] == {"amount": "$90", "count": 1, "shown": True} and before["waiting"] == 1
    html = client.get("/p/owner-token").text                       # the same strings the page renders
    assert '<span data-m="documented">$1,200</span>' in html and "documented · 1 item<" in html
    assert f'class="pending" style="width: {before["bar"]["pending"]}%"' in html

    assert client.post(f"/p/owner-token/item/{ids['lamp']}", json={"action": "confirm"}).status_code == 200
    after = client.get("/p/member-token/meter.json").json()        # a member's link reads it too
    assert after["documented"] == "$1,290" and after["figure_note"] == "documented · 2 items"
    assert after["confirmed"] == 2 and after["waiting"] == 0
    assert after["pending"]["shown"] is False and after["pending"]["count"] == 0
    assert after["memory"]["count"] == 2 and after["bar"]["pending"] == "0.00"


def test_meter_json_wrong_token_is_404(case):
    client, _, _ = case
    assert client.get("/p/nope/meter.json").status_code == 404
