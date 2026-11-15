"""Bank records: what the family actually paid for things, from their card purchases.

Capital One's Nessie API (the hackathon's mock bank) lists an account's purchases with the
merchant, the date, the amount and a description. A purchase that matches an item on the
inventory is better than any estimate: it is the price the family paid and a record of the
purchase, both of which an insurer asks for.

    purchases(...)        every purchase on the customer's accounts
    match_purchases(...)  which purchase is which inventory line
    apply_matches(...)    put the amounts on the items, with the purchase kept as the reference

Matching is deliberately strict. A purchase is proposed for an item only when its description
names the item (words in common, and the amount is plausible for it). The family confirms
before any price changes, and one purchase is never used for two items.

Nessie holds mock data, so `seed_demo` can first write a few purchases that match a case's
most valuable items; that is a demo helper and says so.

Docs: http://api.nessieisreal.com/documentation. The key goes in the `key` query parameter.
Probed on the live service on 15 Nov 2026 with an invalid key (no real key was available): reads
answer 200 with an empty list rather than an error, writes answer 401 "Invalid API key.", an
unknown id answers 404, and a purchase needs a 24-character merchant id, a date and a status.
The keyed calls follow the documented v2 endpoints and are covered by tests with a fake.
"""
from __future__ import annotations

import json
import re

import requests

from . import config
from .pricing import money, nice

API = "https://api.nessieisreal.com"
SOURCE_PREFIX = "bank record"
STOP = {"the", "a", "an", "and", "with", "of", "for", "in", "on", "to", "from", "by", "new", "set", "pack", "inch", "in",
        "size", "large", "small", "medium", "big", "black", "white", "gray", "grey", "brown", "blue", "red", "green",
        "dark", "light", "partly", "visible", "only", "model", "series", "edition", "color", "colour"}
# Materials help confirm a match but say nothing about which object it is: "leather" alone matches nothing.
MATERIALS = {"leather", "wood", "wooden", "metal", "steel", "stainless", "glass", "plastic", "fabric", "velvet",
             "cotton", "brass", "oak", "walnut", "pine", "chrome", "ceramic", "marble", "wool", "linen"}
# The same thing under two names.
SAME = {"tv": "television", "couch": "sofa", "loveseat": "sofa", "sectional": "sofa", "fridge": "refrigerator",
        "laptop": "computer", "notebook": "computer", "macbook": "computer", "recliner": "armchair", "hoover": "vacuum",
        "cooker": "stove", "range": "stove", "oven": "stove", "monitor": "display", "bike": "bicycle",
        "washing": "washer", "nightstand": "bedside", "bedframe": "bed"}


class BankError(RuntimeError):
    pass


class NessieClient:
    def __init__(self, key: str | None = None, base: str | None = None, session=None):
        config.load_env()
        self.key = key or config.env("NESSIE_API_KEY")
        if not self.key:
            raise config.MissingKey("NESSIE_API_KEY is not set. Get a free key at http://api.nessieisreal.com "
                                    "and put it in .env.")
        self.base = (base or config.env("NESSIE_API_URL") or API).rstrip("/")
        self.session = session or requests.Session()
        self._merchants: dict = {}

    def _call(self, method: str, path: str, body: dict | None = None):
        try:
            resp = self.session.request(method, f"{self.base}{path}", params={"key": self.key}, json=body, timeout=30)
        except requests.RequestException as exc:
            raise BankError(f"Could not reach the bank API: {exc}") from exc
        if resp.status_code == 401:
            raise BankError("The bank API rejected the key. Check NESSIE_API_KEY in .env.")
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise BankError(f"Bank API error {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else None

    def get(self, path: str):
        return self._call("GET", path)

    def post(self, path: str, body: dict) -> dict:
        reply = self._call("POST", path, body) or {}
        return reply.get("objectCreated") or reply

    def accounts(self, customer_id: str | None = None) -> list[dict]:
        return self.get(f"/customers/{customer_id}/accounts" if customer_id else "/accounts") or []

    def account_purchases(self, account_id: str) -> list[dict]:
        return self.get(f"/accounts/{account_id}/purchases") or []

    def merchant(self, merchant_id: str | None) -> dict:
        if not merchant_id:
            return {}
        if merchant_id not in self._merchants:
            try:
                self._merchants[merchant_id] = self.get(f"/merchants/{merchant_id}") or {}
            except BankError:
                self._merchants[merchant_id] = {}
        return self._merchants[merchant_id]


def purchases(client: NessieClient | None = None, customer_id: str | None = None,
              account_id: str | None = None) -> list[dict]:
    """Every purchase that was not cancelled: [{id, date, amount, description, merchant, account}].

    Looks at one account (`account_id` or NESSIE_ACCOUNT_ID), else at one customer's accounts
    (`customer_id` or NESSIE_CUSTOMER_ID), else at every account the key can see.
    """
    client = client or NessieClient()
    account_id = account_id or config.env("NESSIE_ACCOUNT_ID")
    customer_id = customer_id or config.env("NESSIE_CUSTOMER_ID")
    account_ids = [account_id] if account_id else [a["_id"] for a in client.accounts(customer_id)]
    if not account_ids:                                     # the live API answers a wrong key with an empty list
        raise BankError("The bank shows no accounts for this key. Check NESSIE_API_KEY, or create demo data "
                        "with `ashline bank --case <case> --seed-demo 5`.")
    out = []
    for account in account_ids:
        for row in client.account_purchases(account):
            if (row.get("status") or "").lower() == "cancelled":
                continue
            try:
                amount = float(row.get("amount") or 0)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            out.append({"id": row.get("_id"), "date": row.get("purchase_date"), "amount": amount,
                        "description": (row.get("description") or "").strip(),
                        "merchant": (client.merchant(row.get("merchant_id")).get("name") or "").strip(),
                        "account": account})
    out.sort(key=lambda p: p.get("date") or "", reverse=True)
    return out


# ---------------------------------------------------------------- matching
def words(text: str) -> set[str]:
    """The telling words of a label or a purchase description, singular, synonyms folded."""
    out = set()
    for token in re.findall(r"[a-z][a-z0-9]+|\d{2,}", (text or "").lower().replace("-", " ")):
        if token in STOP:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        out.add(SAME.get(token, token))
    return out


def score(item: dict, purchase: dict) -> float:
    """0-1: how well a purchase's description names an inventory line."""
    said = words(purchase["description"])
    if not said:
        return 0.0
    label = words(item["label"]) | words(item.get("brand_guess") or "")
    common = said & label
    if not common - MATERIALS:
        return 0.0
    value = len(common) / max(min(len(said), len(label)), 1)
    brand = words(item.get("brand_guess") or "")
    if brand and brand & said:
        value += 0.2
    # A $12 purchase is not the sofa, and a $2,000 one is not the lamp.
    mid, qty = float(item.get("price_mid") or 0), int(item.get("qty") or 1)
    if mid:
        # With a single word in common ("sofa") the amount has to be close to the estimate to count.
        low, high = (0.25, 4) if len(common) > 1 else (0.5, 2)
        if not any(low <= purchase["amount"] / expected <= high for expected in (mid * qty, mid)):
            return 0.0
    return min(value, 1.0)


def match_purchases(items: list[dict], bought: list[dict], threshold: float = 0.5) -> list[dict]:
    """Best purchase for each item, each purchase used once: [{item, purchase, score}], best first."""
    pairs = sorted(((score(i, p), n, m) for n, i in enumerate(items) for m, p in enumerate(bought)), reverse=True)
    used_items, used_purchases, out = set(), set(), []
    for value, n, m in pairs:
        if value < threshold:
            break
        if n in used_items or m in used_purchases:
            continue
        used_items.add(n)
        used_purchases.add(m)
        out.append({"item": items[n], "purchase": bought[m], "score": round(value, 2)})
    return out


def find_matches(case_id: str, store, client: NessieClient | None = None) -> list[dict]:
    """Purchases that match this case's inventory lines and are not already used as their price."""
    everything = [i for i in store.get_items(case_id, canonical_only=True) if i["status"] != "rejected"]
    used = set()                                            # purchases that already are some item's price
    for item in everything:
        if (item.get("price_source") or "").startswith(SOURCE_PREFIX):
            try:
                used |= {ref.get("bank_id") for ref in json.loads(item.get("price_ref") or "[]")}
            except (ValueError, AttributeError):
                pass
    items = [i for i in everything if i.get("price_source") != "set by family"
             and not (i.get("price_source") or "").startswith(SOURCE_PREFIX)]
    return match_purchases(items, [p for p in purchases(client) if p["id"] not in used])


def describe(match: dict) -> str:
    purchase, item = match["purchase"], match["item"]
    where = f" at {purchase['merchant']}" if purchase["merchant"] else ""
    when = f" on {purchase['date']}" if purchase.get("date") else ""
    qty = int(item.get("qty") or 1)
    before = (f" (the list had about {money(item['price_mid'] * qty)}{f' for all {qty}' if qty > 1 else ''})"
              if item.get("price_mid") else "")
    return f"{item['label']}: {money(purchase['amount'])}{where}{when}{before}"


def apply_matches(store, case_id: str, matches: list[dict], member_id: int | None = None) -> int:
    """Use what was paid as each matched item's price. One undo reverses the whole batch."""
    before, done = [], 0
    for match in matches:
        item = store.get_item(case_id, match["item"]["item_id"])
        purchase = match["purchase"]
        if not item or item.get("price_source") == "set by family":
            continue
        qty = int(item.get("qty") or 1)
        each = nice(purchase["amount"] / qty) if qty > 1 else round(purchase["amount"], 2)
        before.append({"item_id": item["item_id"], **{k: item.get(k) for k in (
            "price_low", "price_mid", "price_high", "price_source", "price_ref")}})
        merchant = purchase["merchant"] or "card purchase"
        store.update_item(case_id, item["item_id"], price_low=each, price_mid=each, price_high=each,
                          price_source=f"{SOURCE_PREFIX}: {merchant[:60]}",
                          price_ref=json.dumps([{"title": purchase["description"], "price": purchase["amount"],
                                                 "retailer": merchant, "date": purchase.get("date"),
                                                 "bank_id": purchase.get("id"), "url": ""}]))
        done += 1
    if before:
        store.log_event(case_id, "bank", {"before": before}, member_id)
    return done


# -------------------------------------------------------------- demo data
DEMO_MERCHANTS = {"electronics": "Best Buy", "appliances": "Home Depot", "furniture": "IKEA", "kitchen": "Target",
                  "bedding": "Target", "tools": "Home Depot", "sports": "REI", "outdoor": "Home Depot",
                  "lighting": "IKEA", "decor": "Target", "office": "Staples", "toys": "Target", "clothing": "Target",
                  "other": "Amazon"}


def seed_demo(case_id: str, store, client: NessieClient | None = None, top: int = 5, name: str = "Ashline Demo") -> dict:
    """Write demo purchases for a case's most valuable items into the mock bank. Demo helper only.

    Creates a customer with one credit card account, the merchants, and one purchase per item at
    a price near the estimate. Returns {customer_id, account_id, purchases}; put the ids in .env
    as NESSIE_CUSTOMER_ID / NESSIE_ACCOUNT_ID so only this account is read.
    """
    client = client or NessieClient()
    first, _, last = name.partition(" ")
    customer = client.post("/customers", {"first_name": first, "last_name": last or "Family", "address": {
        "street_number": "1", "street_name": "Demo Street", "city": "Princeton", "state": "NJ", "zip": "08540"}})
    account = client.post(f"/customers/{customer['_id']}/accounts", {
        "type": "Credit Card", "nickname": "Everyday card", "rewards": 0, "balance": 0})
    items = [i for i in store.get_items(case_id, canonical_only=True) if i["status"] != "rejected" and i.get("price_mid")]
    items.sort(key=lambda i: -(i["price_mid"] * (i.get("qty") or 1)))
    merchants, made = {}, []
    for n, item in enumerate(items[:top]):
        shop = DEMO_MERCHANTS.get(item.get("category") or "other", "Amazon")
        if shop not in merchants:
            merchants[shop] = client.post("/merchants", {"name": shop, "category": item.get("category") or "retail",
                                                         "address": {"street_number": "100", "street_name": "Main Street",
                                                                     "city": "Princeton", "state": "NJ", "zip": "08540"},
                                                         "geocode": {"lat": 40.35, "lng": -74.66}})["_id"]
        qty = int(item.get("qty") or 1)
        amount = round(item["price_mid"] * qty * (0.92 + 0.05 * (n % 4)) - 0.01, 2)      # near, not equal to, the estimate
        client.post(f"/accounts/{account['_id']}/purchases", {
            "merchant_id": merchants[shop], "medium": "balance", "purchase_date": f"{2023 + n % 3}-{1 + (n * 5) % 12:02d}-15",
            "amount": amount, "status": "completed", "description": (f"{qty} x " if qty > 1 else "") + item["label"][:80]})
        made.append({"item": item["label"], "amount": amount, "merchant": shop})
    return {"customer_id": customer["_id"], "account_id": account["_id"], "purchases": made}
