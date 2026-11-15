"""Card purchases from the mock bank matched to inventory lines. No network: a fake Nessie."""
import json

import pytest

from ashline import bank
from ashline.meter import meter
from test_chat import MAYA, MOM, Family, photo, texts


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.content = b"x" if body is not None else b""
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeNessie:
    """Answers the documented endpoints from a small in-memory bank."""

    def __init__(self):
        self.merchants = {"m1": {"_id": "m1", "name": "IKEA"}, "m2": {"_id": "m2", "name": "Best Buy"}}
        self.accounts = [{"_id": "acc1", "type": "Credit Card", "customer_id": "cust1"}]
        self.purchases = {"acc1": [
            {"_id": "p1", "merchant_id": "m1", "purchase_date": "2024-03-12", "amount": 1149.0, "status": "completed",
             "description": "Blue velvet 3-seat sofa"},
            {"_id": "p2", "merchant_id": "m1", "purchase_date": "2024-03-12", "amount": 38.5, "status": "completed",
             "description": "Swedish meatballs and coffee"},
            {"_id": "p3", "merchant_id": "m2", "purchase_date": "2023-11-24", "amount": 219.99, "status": "cancelled",
             "description": "Brass floor lamp"},
            {"_id": "p4", "merchant_id": "m2", "purchase_date": "2023-11-25", "amount": 189.0, "status": "completed",
             "description": "Floor lamp, brass finish"},
            {"_id": "p5", "merchant_id": "m2", "purchase_date": "2022-01-02", "amount": 4.99, "status": "completed",
             "description": "Sofa cleaning wipes"},
        ]}
        self.calls, self.posted = [], []

    def request(self, method, url, params=None, json=None, timeout=None):
        path = url.split("nessieisreal.com", 1)[1]
        self.calls.append((method, path, params))
        if params.get("key") != "test-key":
            return FakeResponse(401, {"code": 401})
        if method == "POST":
            made = {"_id": f"new{len(self.posted) + 1}", **json}
            self.posted.append((path, json))
            if path.endswith("/purchases"):
                self.purchases.setdefault(path.split("/")[2], []).append(made)
            if path == "/merchants":
                self.merchants[made["_id"]] = made
            return FakeResponse(201, {"code": 201, "objectCreated": made})
        if path in ("/accounts", "/customers/cust1/accounts"):
            return FakeResponse(200, self.accounts)
        parts = path.strip("/").split("/")
        if parts[0] == "accounts" and parts[-1] == "purchases":
            return FakeResponse(200, self.purchases.get(parts[1], []))
        if parts[0] == "merchants":
            found = self.merchants.get(parts[1])
            return FakeResponse(200 if found else 404, found)
        return FakeResponse(404, None)


@pytest.fixture
def nessie():
    return FakeNessie()


@pytest.fixture
def client(nessie):
    return bank.NessieClient(key="test-key", session=nessie)


def test_purchases_are_read_with_the_merchant_name_and_the_key(client, nessie):
    rows = bank.purchases(client)
    assert [r["id"] for r in rows] == ["p1", "p2", "p4", "p5"]                 # newest first, cancelled left out
    assert rows[0] == {"id": "p1", "date": "2024-03-12", "amount": 1149.0, "description": "Blue velvet 3-seat sofa",
                       "merchant": "IKEA", "account": "acc1"}
    assert all(params == {"key": "test-key"} for _, _, params in nessie.calls)
    with pytest.raises(bank.BankError, match="rejected the key"):
        bank.purchases(bank.NessieClient(key="wrong", session=nessie))


def test_no_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("NESSIE_API_KEY", raising=False)
    with pytest.raises(bank.config.MissingKey, match="NESSIE_API_KEY"):
        bank.NessieClient()


def test_a_purchase_matches_the_item_it_names_once():
    sofa = {"item_id": 1, "label": "blue velvet sofa", "price_mid": 900, "qty": 1}
    other_sofa = {"item_id": 2, "label": "grey leather couch", "price_mid": 1200, "qty": 1}
    lamp = {"item_id": 3, "label": "brass floor lamp", "price_mid": 150, "qty": 1}
    tv = {"item_id": 4, "label": "wall-mounted flat-screen television", "brand_guess": "Samsung", "price_mid": 700, "qty": 1}
    bought = [{"id": "a", "description": "Blue velvet 3-seat sofa", "amount": 1149.0, "merchant": "IKEA", "date": "2024-03-12"},
              {"id": "b", "description": "Sofa cleaning wipes", "amount": 4.99, "merchant": "Target", "date": None},
              {"id": "c", "description": "Floor lamp, brass finish", "amount": 189.0, "merchant": "", "date": None},
              {"id": "d", "description": "Samsung 65in QLED TV", "amount": 899.0, "merchant": "Best Buy", "date": None},
              {"id": "e", "description": "Gift card", "amount": 100.0, "merchant": "Best Buy", "date": None}]
    found = {m["item"]["item_id"]: m["purchase"]["id"] for m in bank.match_purchases([sofa, other_sofa, lamp, tv], bought)}
    assert found == {1: "a", 3: "c", 4: "d"}            # 'tv' is a television; wipes are not a sofa; one sofa bought
    assert bank.score(sofa, bought[1]) == 0             # $5 is not a $900 sofa
    # One shared word is not enough when the amount is far from the estimate; materials do not count as shared.
    recliner = {"item_id": 5, "label": "black leather recliner armchair", "price_mid": 350, "qty": 1}
    loveseat = {"id": "f", "description": "Gray leather reclining loveseat", "amount": 828.0, "merchant": "IKEA", "date": None}
    assert bank.score(recliner, loveseat) == 0
    assert bank.score({**recliner, "price_mid": 800}, loveseat) == 0          # close in price, still only "leather"
    assert bank.words("Stainless steel French-door refrigerators") == {"stainless", "steel", "french", "door", "refrigerator"}
    assert bank.describe({"item": sofa, "purchase": bought[0]}) == \
        "blue velvet sofa: $1,149 at IKEA on 2024-03-12 (the list had about $900)"


def test_checking_the_bank_in_chat_proposes_then_applies_and_undoes(nessie, client):
    f = Family().setup()
    f.engine.bank = client
    f.say(MOM, photos=[photo(640)])
    f.wait()
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    estimate = sofa["price_mid"]
    assert "Only the person who started" in texts(f.say(MAYA, "check my bank"))[0]

    out = texts(f.say(MOM, "check my bank"))
    assert len(out) == 1 and out[0].startswith("I found 2 card purchases that match things on your list:")
    assert "1. Blue velvet sofa: $1,149 at IKEA on 2024-03-12" in out[0] and "Brass floor lamp: $189 at Best Buy" in out[0]
    assert f.store.get_item(f.case_id, sofa["item_id"])["price_mid"] == estimate          # nothing changes before a yes

    out = texts(f.say(MOM, "yes"))
    assert out[0].startswith("Done. 2 items now carry the price you paid, marked as a bank record.")
    after = f.store.get_item(f.case_id, sofa["item_id"])
    assert after["price_mid"] == after["price_low"] == after["price_high"] == 1149.0
    assert after["price_source"] == "bank record: IKEA"
    assert json.loads(after["price_ref"])[0]["bank_id"] == "p1"
    assert "found none" in texts(f.say(MOM, "check my bank"))[0]                           # already used
    # A purchase that is already one item's price is not offered for another item.
    twin = f.store.add_item(f.case_id, {"label": "blue velvet sofa", "category": "furniture", "room_label": "Den"},
                            source="voice")
    f.store.update_item(f.case_id, twin, price_mid=1000)
    assert "found none" in texts(f.say(MOM, "check my bank"))[0]
    f.store.set_item_status(f.case_id, twin, "rejected")

    assert texts(f.say(MOM, "undo")) == ["Undone: those prices are back to the estimates."]
    back = f.store.get_item(f.case_id, sofa["item_id"])
    assert back["price_mid"] == estimate and back["price_source"] != "bank record: IKEA"


def test_saying_no_leaves_the_prices(client):
    f = Family().setup()
    f.engine.bank = client
    f.say(MOM, photos=[photo(640)])
    f.wait()
    f.say(MOM, "check my bank")
    assert texts(f.say(MOM, "no")) == ["Okay, the prices stay as they are."]
    assert not any((i.get("price_source") or "").startswith("bank") for i in f.store.get_items(f.case_id))


def test_without_a_key_the_chat_says_bank_records_are_not_connected(monkeypatch):
    monkeypatch.delenv("NESSIE_API_KEY", raising=False)
    f = Family().setup()
    assert texts(f.say(MOM, "check my bank")) == ["Bank records aren't connected yet, so I can't look at purchases."]


def test_a_card_purchase_counts_as_proof_for_a_remembered_item(client):
    f = Family().setup()
    item = f.store.add_item(f.case_id, {"label": "blue velvet sofa", "category": "furniture", "room_label": "Living room"},
                            source="voice")
    f.store.update_item(f.case_id, item, price_low=600, price_mid=900, price_high=1300, price_source="claude estimate")
    f.store.set_item_status(f.case_id, item, "confirmed")
    assert meter(f.case_id, f.store)["memory"] == 900 and meter(f.case_id, f.store)["backed"] == 0
    matches = bank.find_matches(f.case_id, f.store, client)
    assert bank.apply_matches(f.store, f.case_id, matches) == 1
    assert meter(f.case_id, f.store)["backed"] == 1149 and meter(f.case_id, f.store)["memory"] == 0

    from ashline.export import export_rows
    assert export_rows(f.case_id, f.store)[0]["evidence"] == \
        "From memory, with a card purchase at IKEA on 2024-03-12 ($1,149.00)"

    from fastapi.testclient import TestClient

    from ashline.web import create_app
    token = f.store.get_case(f.case_id)["owner_token"]
    detail = TestClient(create_app(lambda: f.store)).get(f"/p/{token}/item/{item}.json").json()
    assert detail["price_basis"] == "What the family paid, from their card purchases."
    assert detail["price_evidence"] == [{"text": "$1,149 at IKEA on 2024-03-12: Blue velvet 3-seat sofa", "url": ""}]
    assert detail["badges"]["receipt"] == 1 and not detail["needs_proof"]


def test_demo_purchases_can_be_written_to_the_mock_bank(nessie, client):
    f = Family().setup()
    f.say(MOM, photos=[photo(640)])
    f.wait()
    made = bank.seed_demo(f.case_id, f.store, client, top=2)
    assert [p for p, _ in nessie.posted][:2] == ["/customers", f"/customers/{made['customer_id']}/accounts"]
    assert len(made["purchases"]) == 2 and all(row["amount"] > 0 for row in made["purchases"])
    matches = bank.match_purchases([i for i in f.store.get_items(f.case_id, canonical_only=True)],
                                   bank.purchases(client, account_id=made["account_id"]))
    assert len(matches) == 2                                  # the demo purchases match the items they were made from


def test_a_price_that_was_paid_or_found_is_not_replaced_by_an_estimate(client):
    from ashline.price_search import apply_price, price_check
    from ashline.pricing import DefaultPricer, price_items
    f = Family().setup()
    f.engine.bank = client
    f.say(MOM, photos=[photo(640)])
    f.wait()
    f.say(MOM, "check my bank")
    f.say(MOM, "yes")
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    table = next(i for i in f.store.get_items(f.case_id) if "coffee table" in i["label"])
    found = {"price": 240.0, "low": 200.0, "high": 260.0, "retailer": "Wayfair", "url": "https://example.com/t"}
    assert apply_price(f.store, f.case_id, table["item_id"], found).startswith("Recorded $240")
    assert "left as it is" in apply_price(f.store, f.case_id, sofa["item_id"], found)       # paid beats found

    f.say(MOM, photos=[photo(656)])                             # a later burst re-prices the estimates only
    f.wait()
    price_items(f.case_id, f.store, DefaultPricer())
    after = {i["item_id"]: i for i in f.store.get_items(f.case_id)}
    assert after[sofa["item_id"]]["price_mid"] == 1149.0 and after[sofa["item_id"]]["price_source"] == "bank record: IKEA"
    assert after[table["item_id"]]["price_mid"] == 240.0 and after[table["item_id"]]["price_source"] == "web: Wayfair"

    class Searcher:
        def __init__(self):
            self.asked = []

        def search(self, description, brand=None):
            self.asked.append(description)
            return [{"title": description, "price": 99.0, "retailer": "Target", "url": "https://example.com/x"}]
    searcher = Searcher()
    price_check(f.case_id, f.store, top=50, searcher=searcher)
    assert not any("sofa" in a or "coffee table" in a for a in searcher.asked)
