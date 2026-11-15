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


def test_a_found_price_far_from_the_estimate_is_not_used_unasked():
    class Wild:
        def search(self, description, brand=None, image=None):
            return [{"title": "Designer sofa", "price": 9800.0, "retailer": "Luxe", "url": "https://luxe.example/s"}]
    f = family(Wild())
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
