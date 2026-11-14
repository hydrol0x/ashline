import json
from datetime import date
from types import SimpleNamespace

import pytest

from ashline import config, pricing
from ashline.meter import meter, meter_line, sb495
from ashline.pricing import (ClaudePricer, DefaultPricer, money, money_range, nice, price_items, price_range)
from ashline.store import Store


def test_money_formats():
    assert [money(v) for v in (45, 8400, 96000, 160000, 1_250_000)] == ["$45", "$8,400", "$96k", "$160k", "$1.2M"]
    assert money_range(400, 750) == "$400–$750" and money_range(4000, 7000) == "$4–7k"
    assert money_range(1500, 2500) == "$1.5–2.5k" and money_range(700, 1100) == "$700–$1,100"
    assert [nice(v) for v in (43, 377, 1234, 4567, 23456)] == [45, 380, 1250, 4600, 23500]


def test_range_is_uncalibrated_until_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    assert pricing.load_calibration() == {"q": pricing.DEFAULT_Q, "by_category": {}, "calibrated": False}
    low, high = price_range(1000)
    assert low < 1000 < high and round(high / 1000, 2) == round(1000 / low, 2) or abs(high * low - 1e6) < 1e5
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "calibration.json").write_text(json.dumps(
        {"q": 0.2, "by_category": {"electronics": 0.1}, "n": 120, "coverage": 0.9}))
    cal = pricing.load_calibration()
    assert cal["calibrated"] and price_range(1000, "furniture", cal) == (800, 1200)
    assert price_range(1000, "electronics", cal) == (900, 1100)


def reply(prices):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps({"prices": prices}))])


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.beta = SimpleNamespace(messages=self)
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def seeded():
    store = Store()
    store.upsert_case("fam1")
    ids = [store.add_item("fam1", {"label": "blue velvet sofa", "category": "furniture"}, source="photo"),
           store.add_item("fam1", {"label": "floor lamp", "category": "lighting", "qty": 2}, source="photo"),
           store.add_item("fam1", {"label": "table saw", "category": "tools"}, source="voice"),
           store.add_item("fam1", {"label": "not ours", "category": "decor"}, source="photo")]
    store.set_item_status("fam1", ids[3], "rejected")
    return store, ids


def test_claude_pricer_prices_lines_and_caches(tmp_path):
    pytest.importorskip("anthropic")
    store, ids = seeded()
    client = FakeClient(reply([{"id": 0, "mid": 1234}, {"id": 1, "mid": 82}, {"id": 2, "mid": 610}, {"id": 9, "mid": 5}]))
    out = price_items("fam1", store, ClaudePricer(client=client))
    assert out == {"priced": 3, "source": "claude estimate", "calibrated": False}
    sofa = store.get_item("fam1", ids[0])
    assert sofa["price_mid"] == 1250 and sofa["price_low"] < 1250 < sofa["price_high"]
    assert store.get_item("fam1", ids[3])["price_mid"] is None            # rejected items are not priced
    prompt = client.calls[0]["messages"][0]["content"]
    assert "0 | furniture | blue velvet sofa | -" in prompt and "Never inflate" in prompt
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"

    assert price_items("fam1", store, ClaudePricer(client=FakeClient()))["priced"] == 0      # nothing left to price
    # Same label seen again later: answered from the cache, no second request.
    new = store.add_item("fam1", {"label": "Blue velvet sofa", "category": "furniture"}, source="photo")
    assert price_items("fam1", store, ClaudePricer(client=FakeClient()))["priced"] == 1
    assert store.get_item("fam1", new)["price_mid"] == 1250


def test_family_price_is_kept_and_failures_fall_back():
    pytest.importorskip("anthropic")
    store, ids = seeded()
    store.update_item("fam1", ids[0], price_low=3000, price_mid=3000, price_high=3000, price_source="set by family")

    class Broken:
        source = "claude estimate"
        def prices(self, items):
            raise pricing.PricingError("boom")

    out = price_items("fam1", store, Broken())
    assert "pricing failed: boom" in out["source"]
    assert store.get_item("fam1", ids[0])["price_mid"] == 3000            # the family's figure stands
    assert store.get_item("fam1", ids[2])["price_source"] == "category default"
    assert DefaultPricer().prices([{"item_id": 1, "category": "tools"}]) == {1: 90.0}


def test_sb495_rule():
    assert sb495(400_000) == {"limit": 400_000, "advance": 240_000, "remaining": 160_000}
    assert sb495(700_000)["advance"] == 350_000 and sb495(700_000)["remaining"] == 350_000      # the cap
    assert sb495(0)["remaining"] == 0


def test_meter_separates_backed_memory_and_pending():
    store, ids = seeded()
    for item_id, mid in zip(ids[:3], (1000, 100, 600)):
        store.update_item("fam1", item_id, price_low=mid * 0.6, price_mid=mid, price_high=mid * 1.5)
    store.set_item_status("fam1", ids[0], "confirmed")       # photo-backed
    store.set_item_status("fam1", ids[2], "confirmed")       # from memory
    m = meter("fam1", store)
    assert (m["items"], m["backed"], m["memory"], m["documented"]) == (2, 1000, 600, 1600)
    assert (m["pending_items"], m["pending"]) == (1, 200)                 # 2 lamps at $100, still a suggestion
    assert m["target"] is None and m["percent"] is None and m["days_left"] is None
    assert meter_line(m) == "2 items documented so far, about $1,600. 1 more item ($200) is waiting for an OK."

    store.set_case("fam1", coverage_limit=10_000, loss_date="2026-10-01")
    m = meter("fam1", store, today=date(2026, 11, 14))
    assert (m["advance"], m["target"], m["percent"]) == (6000, 4000, 40)
    assert m["due"] == "2027-01-09" and m["days_left"] == 56
    assert meter_line(m).startswith("You're at $1,600 of $4,000 documented.") and "due in 56 days" in meter_line(m)
    assert meter("fam1", store, today=date(2027, 6, 1))["days_left"] is None      # past: no countdown
