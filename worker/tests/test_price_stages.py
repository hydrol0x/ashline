"""The online price lookup in stages: look, search, verify, price."""
import json
from types import SimpleNamespace as NS

from PIL import Image

from ashline import price_search
from ashline.price_search import ClaudeSearcher, find_price, judge, lower_median
from test_assistant import fam  # noqa: F401  (a fixture)

SPEC = {"kind": "dining chair", "size_class": "", "material": "wood and metal", "color": "black", "style": "industrial",
        "quality_tier": "budget", "brand": "", "model": "", "brand_read": False, "is_set": False, "set_count": 1,
        "cannot_see": "", "enough_to_price": True,
        "queries": ["black metal frame dining chair wood seat", "industrial dining chair wood back"]}

CANDIDATES = [
    {"title": "Room Essentials dining chair, set of 2", "price": 60.0, "retailer": "Target",
     "url": "https://www.target.com/p/1", "units": 2},
    {"title": "Miki dining chair set of 2", "price": 170.0, "retailer": "Wayfair", "url": "https://www.wayfair.com/2", "units": 2},
    {"title": "Hercules metal chair", "price": 76.0, "retailer": "Walmart", "url": "https://www.walmart.com/3", "units": 1},
    {"title": "Mercana designer chair", "price": 798.0, "retailer": "Mercana", "url": "https://mercana.com/4", "units": 1},
    {"title": "Made up chair", "price": 50.0, "retailer": "Nowhere", "url": "https://invented.example/5", "units": 1},
]

VERDICTS = [{"index": 0, "keep": True, "reason": "same kind, set of 2", "units": 2},
            {"index": 1, "keep": True, "reason": "same kind, set of 2", "units": 2},
            {"index": 2, "keep": True, "reason": "same kind", "units": 1},
            {"index": 3, "keep": False, "reason": "designer piece, item is budget", "units": 1}]


class Client:
    """An Anthropic client at the SDK level: answers the three stages by the shape of the request."""

    def __init__(self, spec=SPEC, candidates=CANDIDATES, verdicts=VERDICTS, pause=False, fail=None):
        self.calls, self.spec, self.candidates, self.verdicts, self.pause, self.fail = [], spec, candidates, verdicts, pause, fail
        self.messages = NS(create=self.create)
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if "tools" in kw:
            stage = "search"
        elif "verdicts" in json.dumps(kw.get("output_config", {}).get("format", {})):
            stage = "verify"
        else:
            stage = "look"
        if stage == "search" and kw["tools"][0]["max_uses"] == 3:
            # the single-call fallback
            return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps({"matches": [
                {"title": "Chair", "price": 99, "retailer": "Shop", "url": "https://s.example/c"}]}))])
        if stage == self.fail:
            raise RuntimeError("boom")
        if stage == "look":
            return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(self.spec))])
        if stage == "verify":
            return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps({"verdicts": self.verdicts}))])
        results = NS(type="web_search_tool_result", content=[NS(type="web_search_result", url=u) for u in (
            "https://www.target.com/x", "https://www.wayfair.com/y", "https://walmart.com/z", "https://mercana.com/w")])
        if self.pause and len([c for c in self.calls if "tools" in c]) == 1:
            return NS(stop_reason="pause_turn", content=[NS(type="server_tool_use"), results])
        return NS(stop_reason="end_turn", content=[results, NS(type="text", text=json.dumps({"candidates": self.candidates}))])


def stages(client):
    return [("search" if "tools" in c else "verify" if "verdicts" in json.dumps(c.get("output_config", {})) else "look")
            for c in client.calls]


def test_the_lookup_runs_in_stages_and_prices_from_the_median_of_kept_listings():
    client = Client(pause=True)
    found = find_price("wood-back dining chair with black metal frame", None, ClaudeSearcher(client=client),
                       image=Image.new("RGB", (900, 900), (30, 30, 30)))
    assert stages(client) == ["look", "search", "search", "verify"]               # a paused search is resumed
    # 4 candidates came from sites the search returned (the invented link is gone); 3 kept, sets divided.
    assert (found["checked"], found["dropped"]) == (4, 1)
    assert sorted(m["price"] for m in found["matches"]) == [30.0, 76.0, 85.0]
    assert (found["price"], found["low"], found["high"]) == (76.0, 30.0, 85.0)     # the median, then the spread
    assert found["matches"][0]["title"] == "Hercules metal chair"                  # closest to the median first
    assert found["matches"][1]["per_unit"] == "a set of 2 for $170, divided by 2"
    assert found["queries"] == SPEC["queries"] and found["spec"]["kind"] == "dining chair" and found["verified"]


def test_the_request_shapes():
    client = Client()
    ClaudeSearcher(client=client).lookup("dining chair", None, image=Image.new("RGB", (900, 600)))
    look, search, verify = client.calls
    for call in client.calls:
        assert call["model"] == "claude-opus-5-5" and call["max_tokens"] == 16000
        assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert look["output_config"]["format"]["type"] == "json_schema"
    assert look["output_config"]["format"]["schema"] is price_search.SPEC_SCHEMA and look["output_config"]["effort"] == "medium"
    assert [b["type"] for b in look["messages"][0]["content"]] == ["image", "text"]
    assert search["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]
    assert "format" not in search["output_config"]                               # web search and its citations stay unconstrained
    assert "black metal frame dining chair wood seat" in search["messages"][0]["content"]
    assert verify["output_config"]["format"]["schema"] is price_search.VERIFY_SCHEMA and "tools" not in verify
    image, text = verify["messages"][0]["content"]
    assert image["type"] == "image" and "Room Essentials" in text["text"] and "web_search" not in text["text"]
    # The verify call sees the crop, the spec and the candidates, never the search transcript or an estimate.
    assert len(verify["messages"]) == 1


def test_one_kept_listing_is_not_enough_unless_the_brand_and_model_were_read():
    one = [{"index": 2, "keep": True, "reason": "same", "units": 1}]
    found = find_price("chair", None, ClaudeSearcher(client=Client(verdicts=one)))
    assert found["found"] is False and found["reason"] == "Not enough evidence: 1 of the 4 listings looked at was the same kind of thing."
    read = {**SPEC, "brand": "Flash", "model": "Hercules", "brand_read": True}
    found = find_price("chair", None, ClaudeSearcher(client=Client(spec=read, verdicts=one)))
    assert found["found"] and found["price"] == 76.0


def test_a_hidden_thing_is_not_searched():
    client = Client(spec={**SPEC, "kind": "bed frame", "enough_to_price": False, "cannot_see": "the frame is hidden by bedding"})
    found = find_price("bed with white bedding", None, ClaudeSearcher(client=client))
    assert stages(client) == ["look"]
    assert found["found"] is False and found["reason"] == "Not enough of it can be seen to price it (the frame is hidden by bedding)."


def test_a_failed_stage_falls_back_to_the_single_search():
    for stage in ("look", "search", "verify"):
        client = Client(fail=stage)
        found = find_price("chair", None, ClaudeSearcher(client=client))
        assert found["found"] and found["price"] == 99 and "verified" not in found
    bad = Client(spec={**SPEC, "queries": []})                                    # an unusable spec is a failure too
    assert find_price("chair", None, ClaudeSearcher(client=bad))["price"] == 99


def test_median_and_judge():
    assert lower_median([90.0]) == 90.0 and lower_median([90.0, 120.0]) == 90.0 and lower_median([5, 1, 3]) == 3
    assert lower_median([10, 20, 30, 40]) == 20
    out = judge(SPEC, CANDIDATES[:2], [{"keep": True, "units": 2}, {"keep": False, "units": 1}])
    assert out["matches"] == [{"title": CANDIDATES[0]["title"], "price": 30.0, "retailer": "Target",
                               "url": CANDIDATES[0]["url"], "per_unit": "a set of 2 for $60, divided by 2"}]
    assert (out["checked"], out["dropped"]) == (2, 1) and "queries" not in out["spec"]
    assert price_search.parse_candidates('{"candidates": [{"title": "x", "price": "$1,200", "url": "https://a.example", "units": 2},'
                                         '{"title": "no link", "price": 5, "url": ""}]}') == [
        {"title": "x", "price": 1200.0, "units": 2, "retailer": "online retailer", "url": "https://a.example"}]


def test_what_was_looked_at_is_kept_with_the_online_price(fam):
    from ashline.price_search import apply_price
    from ashline.pricing import price_sources
    lamp = fam.ids["brass floor lamp"]
    found = find_price("brass floor lamp", None, ClaudeSearcher(client=Client()))
    assert apply_price(fam.store, fam.case_id, lamp, found).startswith("Recorded $75 for the brass floor lamp")
    online = price_sources(fam.store.get_item(fam.case_id, lamp))["online"]
    assert (online["checked"], online["dropped"], online["queries"][0]) == (4, 1, SPEC["queries"][0])
    assert len(online["matches"]) == 3 and online["spec"]["kind"] == "dining chair"


def test_a_watcher_sees_each_step_and_a_failure_names_its_step():
    lines = []
    searcher = ClaudeSearcher(model="claude-opus-5-5", client=Client())
    searcher.trace = lines.append
    assert find_price("wood dining chair", None, searcher)["found"]
    seen = "\n".join(lines)
    assert "look:" in seen and "queries:" in seen and "search: " in seen and "keep $" in seen and "drop $" in seen
    broken = ClaudeSearcher(model="claude-opus-5-5", client=Client(fail="verify"))
    broken.search = lambda *a, **k: []                      # the single search finds nothing either
    result = find_price("wood dining chair", None, broken)
    assert not result["found"] and "the verify step failed" in result["reason"]
