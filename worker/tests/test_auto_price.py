"""The price agent looking items up online without being asked, after new items come in."""
from ashline import price_search
from ashline.pricing import DefaultPricer
from test_chat import MAYA, MOM, Family, photo, texts


class Searcher:
    """Finds a listing for anything with 'sofa' or 'lamp' in its name; records what it was shown."""

    def __init__(self):
        self.asked = []

    def search(self, description, brand=None, image=None):
        self.asked.append((description, image.size if image is not None else None))
        if "sofa" in description:
            return [{"title": "Velvet 3-seat sofa", "price": 1199.0, "retailer": "Acme Home", "url": "https://shop.example/sofa"},
                    {"title": "Similar sofa", "price": 1150.0, "retailer": "Other", "url": "https://other.example/s"}]
        if "lamp" in description:
            return [{"title": "Brass floor lamp", "price": 189.0, "retailer": "LampCo", "url": "https://lamp.example/l"}]
        return []


def family(searcher, minimum="50"):
    f = Family().setup()
    f.engine.searcher, f.engine.background = searcher, False          # inline, so the test can look straight away
    return f


def test_new_items_are_looked_up_after_a_burst_and_mentioned_once(monkeypatch):
    monkeypatch.setenv("ASHLINE_AUTO_PRICE_MIN", "50")
    searcher = Searcher()
    f = family(searcher)
    f.say(MAYA, photos=[photo(640)])                                  # sofa, lamp, coffee table
    out = texts(f.wait())                                             # the digest, then what the agent found
    items = {i["label"]: i for i in f.store.get_items(f.case_id)}
    sofa, lamp, table = items["blue velvet sofa"], items["brass floor lamp"], items["oak coffee table"]
    assert (sofa["price_mid"], sofa["price_low"], sofa["price_high"], sofa["price_source"]) == (1200, 1150, 1200, "web: Acme Home")
    assert lamp["price_source"] == "web: LampCo" and not (table["price_source"] or "").startswith("web")   # nothing found: estimate stays
    assert len(searcher.asked) == 3 and all(size is not None for _, size in searcher.asked)   # each lookup carried the photo crop

    news = [t for t in out if t.startswith("I looked up")]
    assert len(news) == 1 and news[0].startswith("I looked up 2 of the bigger items online to firm up the prices:")
    assert "• Blue velvet sofa: $1,200 at Acme Home (I had guessed about" in news[0] and "LampCo" in news[0]
    assert out.index(news[0]) > 0                                     # after the digest, not instead of it
    assert texts(f.wait(5)) == []                                     # said once
    # A lookup nobody asked for is not what "undo" reverses.
    assert f.store.get_item(f.case_id, sofa["item_id"])["price_source"] == "web: Acme Home"      # the lookup did happen
    assert (f.store.last_event(f.case_id) or {}).get("action") != "item.price"


def test_small_items_and_prices_the_family_trusts_are_left_alone(monkeypatch):
    monkeypatch.setenv("ASHLINE_AUTO_PRICE_MIN", "300")
    searcher = Searcher()
    f = family(searcher)
    f.say(MAYA, photos=[photo(640)])
    f.wait()
    assert {d for d, _ in searcher.asked} == {"blue velvet sofa", "oak coffee table"}      # the lamp is under $300
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    f.store.update_item(f.case_id, sofa["item_id"], price_mid=2000, price_low=2000, price_high=2000, price_source="set by family")
    assert price_search.auto_price(f.case_id, f.store, searcher=searcher, minimum=0) != []       # the lamp
    assert f.store.get_item(f.case_id, sofa["item_id"])["price_mid"] == 2000
    assert price_search.auto_price(f.case_id, f.store, searcher=searcher, minimum=0) == []       # already looked up or kept


class ModelPricer:
    """Stands in for the model's estimate: a real estimate, not a fixed category figure."""
    source = "claude estimate"

    def prices(self, items):
        return {i["item_id"]: {"blue velvet sofa": 900.0, "brass floor lamp": 120.0}.get(i["label"], 200.0) for i in items}


def test_a_found_price_far_from_the_estimate_is_not_used_unasked():
    class Wild:
        def search(self, description, brand=None, image=None):
            return [{"title": "Designer sofa", "price": 9800.0, "retailer": "Luxe", "url": "https://luxe.example/s"}]
    f = family(Wild())
    f.engine.pricer = ModelPricer()
    f.say(MAYA, photos=[photo(640)])
    out = texts(f.wait())
    assert not any((i.get("price_source") or "").startswith("web") for i in f.store.get_items(f.case_id))
    assert not any(t.startswith("I looked up") for t in out)
    # Asked for by a person, the same lookup is recorded: they can see it and judge.
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    found = price_search.find_price(sofa["label"], None, Wild())
    assert price_search.apply_price(f.store, f.case_id, sofa["item_id"], found).startswith("Recorded $9,800")


def test_it_can_be_turned_off_and_does_nothing_without_a_key(monkeypatch):
    searcher = Searcher()
    monkeypatch.setenv("ASHLINE_AUTO_PRICE", "off")
    f = family(searcher)
    f.say(MAYA, photos=[photo(640)])
    f.wait()
    assert searcher.asked == []
    monkeypatch.delenv("ASHLINE_AUTO_PRICE")
    f.engine.searcher = None                                           # no searcher given and no key in the test environment
    f.say(MAYA, photos=[photo(656)])
    f.wait()
    assert not any((i.get("price_source") or "").startswith("web") for i in f.store.get_items(f.case_id))


def test_the_photo_goes_to_the_search_model():
    """ClaudeSearcher sends the crop as an image block with the instruction to match what it shows."""
    from types import SimpleNamespace

    from PIL import Image

    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(
                type="text", text='{"matches": [{"title": "Sofa", "price": 999, "retailer": "Shop", "url": "https://s.example/x"}]}')])

    searcher = price_search.ClaudeSearcher(client=SimpleNamespace(messages=Messages()))
    found = searcher.search("blue velvet sofa", None, image=Image.new("RGB", (1600, 900), (40, 60, 120)))
    assert found[0]["price"] == 999
    image, text = calls[0]["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert "The photo above shows the item" in text["text"] and "blue velvet sofa" in text["text"]
    searcher.search("blue velvet sofa")                               # without a photo: plain text, as before
    assert isinstance(calls[1]["messages"][0]["content"], str)


def test_news_is_handed_out_once_even_to_two_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(price_search.config, "case_dir", lambda case_id: tmp_path / case_id)
    price_search.add_news("c1", [{"item_id": 1, "label": "sofa", "price": 1200, "retailer": "Acme", "chat_id": "a"}])
    price_search.add_news("c1", [{"item_id": 2, "label": "lamp", "price": 190, "retailer": "LampCo", "chat_id": "a"}])
    assert not list((tmp_path / "c1" / "price_news").glob("*.part"))          # nothing half-written is left visible
    first, second = price_search.take_news("c1"), price_search.take_news("c1")
    assert [row["item_id"] for row in first] == [1, 2] and second == []


class Staged:
    """A searcher that looks in stages: returns the kept listings it is given for every item."""

    def __init__(self, prices, read=False):
        self.prices, self.read, self.asked = prices, read, []

    def lookup(self, description, brand=None, image=None):
        self.asked.append(description)
        spec = {"kind": description, "brand": "Acme" if self.read else "", "model": "X1" if self.read else "",
                "brand_read": self.read, "enough_to_price": True}
        return {"matches": [{"title": f"listing {n}", "price": p, "retailer": f"Shop {n}", "url": f"https://s{n}.example/"}
                            for n, p in enumerate(self.prices)],
                "spec": spec, "queries": [description], "checked": len(self.prices) + 2, "dropped": 2}

    def search(self, description, brand=None, image=None):
        raise AssertionError("the staged lookup worked, so the single search is not used")


def chair_case(source="category default", mid=450.0, label="wood-back dining chair", f=None):
    if f is None:
        f = Family().setup()
        f.engine.searcher, f.engine.background = None, False
    item = f.store.add_item(f.case_id, {"label": label, "category": "furniture", "room_label": "Kitchen"},
                            source="photo", added_by="Maya")
    f.store.set_cluster(f.case_id, item, [item], 1)
    f.store.update_item(f.case_id, item, price_low=mid * 0.6, price_mid=mid, price_high=mid * 1.5, price_source=source)
    return f, item


def test_the_450_chair_becomes_about_90():
    """Three checked listings that agree replace the fixed furniture figure, however far below it."""
    f, chair = chair_case()
    out = price_search.auto_price(f.case_id, f.store, [chair], Staged([80.0, 92.0, 99.0]), minimum=0)
    item = f.store.get_item(f.case_id, chair)
    assert out and (item["price_mid"], item["price_low"], item["price_high"]) == (90, 80, 100)


def test_two_agreeing_listings_are_used_whatever_the_estimate():
    f, chair = chair_case("claude estimate", 900.0)                       # a real estimate, 10x off
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([85.0, 150.0]), minimum=0)
    assert f.store.get_item(f.case_id, chair)["price_mid"] == 85          # two listings: the lower


def test_two_listings_that_disagree_fall_back_to_the_3x_rule():
    f, chair = chair_case("claude estimate", 900.0)
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([85.0, 400.0]), minimum=0) == []
    _, chair = chair_case("claude estimate", 200.0, f=f)
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([85.0, 400.0]), minimum=0)


def test_one_listing_keeps_the_3x_rule_except_against_a_fixed_figure():
    f, chair = chair_case("claude estimate", 900.0)
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([90.0], read=True), minimum=0) == []
    _, chair = chair_case("claude estimate", 200.0, f=f)
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([90.0], read=True), minimum=0)
    _, chair = chair_case("category default", 450.0, f=f)
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([90.0], read=True), minimum=0)
    # A lone listing for a brand nobody read is not enough evidence, so nothing is recorded at all.
    _, chair = chair_case("category default", 450.0, f=f)
    assert price_search.auto_price(f.case_id, f.store, [chair], Staged([90.0]), minimum=0) == []
    assert f.store.get_item(f.case_id, chair)["price_mid"] == 450


def test_reach_more_items_and_cheaper_ones(monkeypatch):
    assert (price_search.AUTO_TOP, price_search.AUTO_MIN, price_search.AUTO_AT_ONCE) == (12, 60.0, 3)
    f, first = chair_case("claude estimate", 40.0, label="toaster")               # under $60: left alone
    ids = [first] + [chair_case("claude estimate", 70.0 + n, label=f"thing {n}", f=f)[1] for n in range(14)]
    staged = Staged([90.0, 95.0])
    price_search.auto_price(f.case_id, f.store, ids, staged)
    assert len(staged.asked) == 12 and "toaster" not in staged.asked and "thing 0" not in staged.asked   # the 12 dearest
    monkeypatch.setenv("ASHLINE_AUTO_PRICE_TOP", "1")
    monkeypatch.setenv("ASHLINE_AUTO_PRICE_MIN", "30")
    again = Staged([30.0, 35.0])
    price_search.auto_price(f.case_id, f.store, ids, again)
    assert again.asked == ["thing 1"]                                              # what is left, dearest first


def test_a_fixed_furniture_figure_is_looked_up_whatever_the_minimum():
    f, sofa = chair_case("category default", 450.0, label="sofa")
    _, lamp = chair_case("claude estimate", 450.0, label="side table", f=f)
    staged = Staged([300.0, 320.0])
    price_search.auto_price(f.case_id, f.store, [sofa, lamp], staged, minimum=1000)
    assert staged.asked == ["sofa"]


def test_at_most_three_lookups_at_once():
    import threading
    import time
    lock, now = threading.Lock(), {"running": 0, "most": 0}

    class Slow(Staged):
        def lookup(self, description, brand=None, image=None):
            with lock:
                now["running"] += 1
                now["most"] = max(now["most"], now["running"])
            time.sleep(0.05)
            with lock:
                now["running"] -= 1
            return super().lookup(description, brand, image)
    f, first = chair_case("claude estimate", 100.0, label="thing a")
    ids = [first] + [chair_case("claude estimate", 100.0, label=f"thing {n}", f=f)[1] for n in range(7)]
    price_search.auto_price(f.case_id, f.store, ids, Slow([90.0, 95.0]), minimum=0)
    assert now["most"] == 3
