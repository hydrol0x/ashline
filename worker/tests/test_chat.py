"""A family's conversation with Ashline, start to finish, with a fake detector and no network."""
import io
import re
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from ashline.chat import Engine, Inbound, extract_things, normalize_photo, parse_money, split_rooms
from ashline.pricing import DefaultPricer
from ashline.store import Store

MOM, MAYA, DAD, GROUP = "+15550001", "+15550002", "+15550003", "group-rivera"
T0 = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)

# What the fake detector "sees", chosen by the photo's width.
SCENES = {
    640: ("Living room", [("blue velvet sofa", "furniture", [50, 200, 400, 420]),
                          ("brass floor lamp", "lighting", [450, 100, 520, 400]),
                          ("oak coffee table", "furniture", [200, 380, 420, 470])]),
    648: ("Living room", [("blue velvet sofa with cushions", "furniture", [80, 220, 500, 430]),
                          ("upright piano", "other", [10, 100, 200, 400])]),
    656: ("Kitchen", [("red stand mixer", "kitchen", [100, 100, 220, 260])]),
    664: ("Living room", [("floral armchair sofa", "furniture", [300, 200, 600, 440])]),      # a second "sofa"?
    672: (None, []),                                                                           # a blurry photo
}


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, path, room_hint=None, rooms=None):
        self.calls += 1
        room, things = SCENES[Image.open(path).width]
        return [{"label": label, "category": category, "brand_guess": None, "confidence": 0.8, "bbox": bbox,
                 **({"room": room} if room else {})} for label, category, bbox in things]


def photo(width, color=(120, 110, 100)):
    buf = io.BytesIO()
    Image.new("RGB", (width, 480), color).save(buf, format="JPEG")
    return (f"IMG_{width}.jpg", buf.getvalue())


class CategoryPricer(DefaultPricer):
    """The fixed category figures only, so the chat tests' sums stay put (the typical-price table
    is tested in test_pricing_meter)."""

    def prices(self, items):
        from ashline.pricing import CATEGORY_DEFAULT
        return {i["item_id"]: float(CATEGORY_DEFAULT.get(i.get("category") or "other", 50)) for i in items}

    def source_for(self, item):
        return self.source


class Family:
    """Sends messages and advances the clock."""

    def __init__(self):
        self.store, self.detector, self.now = Store(), FakeDetector(), T0
        self.engine = Engine(self.store, detector=self.detector, pricer=CategoryPricer(),
                             public_url="https://ashline.example", number="+1 555 010 0000")

    def say(self, who, text="", photos=(), chat=None, name=None):
        self.now += timedelta(seconds=5)
        return self.engine.handle(Inbound(chat or who, who, text, list(photos), is_group=chat is not None,
                                          sender_name=name, now=self.now))

    def wait(self, seconds=61):
        self.now += timedelta(seconds=seconds)
        return self.engine.tick(self.now)

    def setup(self):
        self.say(MOM, "hi", name="Mom")
        self.say(MOM, "The Rivera home")
        self.say(MOM, "skip")
        out = self.say(MOM, "living room, kitchen, 3 bedrooms and a garage")
        self.code = re.search(r"HOME-\d{4}", out[0].text).group(0)
        self.case_id = self.store.get_member(MOM)["case_id"]
        self.say(MAYA, self.code, name="Maya")
        return self


@pytest.fixture
def fam():
    return Family().setup()


def texts(out):
    return [o.text for o in out]


def test_owner_onboarding_takes_three_messages():
    f = Family()
    assert texts(f.say(MOM, "hi", name="Mom")) == [
        "I'm Ashline. I help families rebuild the list of belongings insurance asks for after a fire, flood or storm. It takes a few "
        "minutes at a time, and you can stop whenever. What should I call your home?"]
    assert texts(f.say(MOM, "The Rivera home")) == [
        "What was the address? I'll check whether there's a 3D scan or a sale listing of the home to start from. "
        "You can say skip."]
    assert texts(f.say(MOM, "skip")) == ["Which rooms did it have? A rough list is fine."]
    out = f.say(MOM, "living room, kitchen, 3 bedrooms and a garage")
    code = re.search(r"HOME-\d{4}", out[0].text).group(0)
    assert out[0].text.startswith("Thank you. Your code is ") and f"text {code} to this number to join" in out[0].text
    assert out[1].text == (f"We're using Ashline to rebuild the Rivera home inventory. Text {code} to +1 555 010 0000, "
                           "then send any photos from inside the house, from any year.")
    case = f.store.get_case(f.store.get_member(MOM)["case_id"])
    assert case["name"] == "The Rivera home" and case["invite_code"] == code and len(case["owner_token"]) > 20
    assert [r["name"] for r in f.store.get_rooms(case["case_id"])] == [
        "Living room", "Kitchen", "Bedroom", "Bedroom 2", "Bedroom 3", "Garage"]
    assert f.store.get_member(MOM)["role"] == "owner"
    # Coverage limit and date of loss are not asked now.
    assert case["coverage_limit"] is None and not any("coverage" in t for t in texts(out))


def test_relative_joins_with_the_code(fam):
    member = fam.store.get_member(MAYA)
    assert member["case_id"] == fam.case_id and member["role"] == "contributor" and member["name"] == "Maya"
    assert texts(fam.say(DAD, fam.code.lower(), name="Dad")) == [
        "You've joined the Rivera home. Send any photos from inside the house: holidays, birthdays, video calls, "
        "real estate listings, anything."]
    assert "already part of the Rivera home" in fam.say(DAD, fam.code)[0].text
    assert "don't recognise that code" in fam.say("+15559999", "HOME-0000")[0].text
    assert len(fam.store.get_members(fam.case_id)) == 3


def test_one_photo_comes_back_annotated(fam):
    assert texts(fam.say(MAYA, "Thanksgiving 2019", [photo(640)])) == ["Got it, looking at it."]
    assert fam.wait(20) == []                                     # still inside the quiet window
    out = fam.wait(45)
    assert len(out) == 1 and out[0].chat_id == MAYA
    assert out[0].image.endswith("annotated/fam-1.jpg") and Image.open(out[0].image).size == (640, 480)
    assert out[0].text == ("Found 3 items · 3 new\n\n1. Blue velvet sofa · $290–$700\n"
                           "2. Brass floor lamp · $50–$130\n3. Oak coffee table · $290–$700")
    items = fam.store.get_items(fam.case_id)
    assert all(i["status"] == "suggested" and i["source"] == "photo" and i["added_by"] == "Maya" for i in items)
    assert {i["room_label"] for i in items} == {"Living room"}
    assert fam.store.get_photo(fam.case_id, number=1)["status"] == "done"
    assert fam.wait() == []                                       # nothing more to say


def test_a_room_named_with_the_photo_keeps_items_apart(fam):
    fam.say(MAYA, "", [photo(640)])                               # detector: living room, a sofa
    fam.wait()
    fam.say(MAYA, "this is the bedroom", [photo(648)])            # detector also says living room, another sofa
    fam.wait()
    sofas = [i for i in fam.store.get_items(fam.case_id, canonical_only=True) if "sofa" in i["label"]]
    assert sorted(i["room_label"] for i in sofas) == ["Bedroom", "Living room"]   # two sofas, not one
    assert fam.store.get_photo(fam.case_id, number=2)["room_label"] == "Bedroom"
    fam.say(MAYA, "dorm room", [photo(664)])                      # a room that isn't on the list still counts
    fam.wait()
    assert fam.store.get_photo(fam.case_id, number=3)["room_label"] == "Dorm room"


def test_a_burst_gets_one_ack_and_one_digest(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    first = fam.say(MAYA, "", [photo(648), photo(656)])
    assert texts(first) == ["Got it, looking through them."]
    assert fam.say(MAYA, "", [photo(672)]) == []                  # same burst: no second ack
    out = fam.wait()
    assert out[0].text.startswith("Got 3 photos from Maya.\n3 items found · 2 new · 1 already listed\n\n"
                                  "Biggest finds:\n")
    assert "• Red stand mixer · $30–$70" in out[0].text and out[0].text.endswith(
        "\n\nFull list: https://ashline.example/p/" + fam.store.get_case(fam.case_id)["member_token"])
    sofa = next(i for i in fam.store.get_items(fam.case_id, canonical_only=True) if "sofa" in i["label"])
    assert len(fam.store.get_sightings(fam.case_id, sofa["item_id"])) == 2          # the same sofa, seen twice
    assert fam.detector.calls == 4


def test_blurry_photo_gets_no_complaint(fam):
    fam.say(MAYA, "", [photo(672)])
    assert texts(fam.wait()) == ["Couldn't make out items in this one, that's fine."]
    assert fam.say(MAYA, "", [("notes.txt", b"not an image")]) == []                # ignored quietly


def test_same_item_question_and_undo(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    fam.say(DAD, fam.code, name="Dad")
    fam.say(DAD, "", [photo(664), photo(656)])
    out = fam.wait()
    question = [o for o in out if not o.text.startswith("Photo ")][-1]
    assert question.text == "Same sofa? 1 = same, 2 = different." and question.image.endswith(".jpg")
    assert Image.open(question.image).height == 600
    lines = lambda: len([i for i in fam.store.get_items(fam.case_id, canonical_only=True)])
    before = lines()
    assert texts(fam.say(DAD, "1")) == ["Thanks, counted once."]
    assert lines() == before - 1
    assert texts(fam.say(DAD, "undo")) == ["Undone: those are two separate items again."]
    assert lines() == before
    assert fam.say(DAD, "1")[0].text.startswith("I'm not sure what you'd like")     # the question is closed


def test_unanswered_question_expires_and_both_stay(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    fam.say(MAYA, "", [photo(664), photo(656)])
    fam.wait()
    assert len(fam.store.get_questions(fam.case_id)) == 1
    fam.wait(5 * 3600)
    assert fam.store.get_questions(fam.case_id) == []
    assert len([i for i in fam.store.get_items(fam.case_id, canonical_only=True) if "sofa" in i["label"]]) == 2


def test_group_chat_is_quiet_by_default(fam):
    assert fam.say(MAYA, "is this thing on", chat=GROUP) == []                       # links the group to the case
    assert fam.say(DAD, "I'll be late tonight", chat=GROUP, name="Dad") == []
    assert fam.store.get_member(DAD)["case_id"] == fam.case_id                       # joined by being in the chat
    assert fam.say(MAYA, "Remember Grandma's piano?", chat=GROUP) == []              # heard, not answered
    assert fam.say(MAYA, "ok", chat=GROUP) == []                                     # no pending question: silence
    assert texts(fam.say(DAD, "how much so far", chat=GROUP)) == ["0 items documented so far, about $0."]
    assert "Send me photos from inside the home" in fam.say(DAD, "Ashline, what are you", chat=GROUP)[0].text
    assert fam.say("+15558888", "hello", chat="some-other-group") == []              # not Ashline's group

    fam.say(MAYA, "", [photo(640), photo(656)], chat=GROUP)
    out = fam.wait()
    assert out[0].chat_id == GROUP and out[0].text.startswith("Got 2 photos from Maya.")
    assert out[1].text == "I heard a piano mentioned. Add it to the list? Reply yes or no."
    assert "Grandma" not in " ".join(texts(out))                                     # never quotes the family
    assert texts(fam.say(DAD, "yes", chat=GROUP)) == [
        'Added the piano, marked "from memory". A photo of it would make it stronger.']
    piano = next(i for i in fam.store.get_items(fam.case_id) if i["label"] == "piano")
    assert piano["source"] == "voice" and piano["status"] == "suggested" and piano["added_by"] == "Dad"


def test_commands(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    case = fam.store.get_case(fam.case_id)
    assert texts(fam.say(MAYA, "list")) == [f"Here is everything so far: https://ashline.example/p/{case['member_token']}"]
    assert texts(fam.say(MOM, "show me everything")) == [f"Here is everything so far: https://ashline.example/p/{case['owner_token']}"]

    out = fam.say(MOM, "how much so far")
    assert out[0].text == "0 items documented so far, about $0.\n3 more items ($980) are waiting for an OK."
    assert "contents coverage limit" in out[1].text and "skip" in out[1].text
    assert texts(fam.say(MOM, "$400k")) == [
        "You're at $0 of $160k documented.\n3 more items ($980) are waiting for an OK."]
    assert len(fam.say(MAYA, "how much so far")) == 1                                # relatives are not asked

    shown = fam.say(MAYA, "show photo 1")
    assert shown[0].image.endswith("fam-1.jpg") and shown[0].text.startswith("1. Blue velvet sofa")
    assert "don't have a photo 7" in fam.say(MAYA, "show photo 7")[0].text
    assert len(fam.say(MAYA, "show the living room")) == 1
    assert "don't have photos of the garage" in fam.say(MAYA, "show the garage")[0].text

    assert texts(fam.say(MAYA, "2 isn't ours")) == ["Removed the brass floor lamp. Say undo if that was a mistake."]
    assert texts(fam.say(MAYA, "undo")) == ["Undone: it's back on the list."]
    assert texts(fam.say(MAYA, "undo")) == ["There's nothing to undo."]
    assert texts(fam.say(MAYA, "2 is a Pottery Barn arc lamp")) == ["Updated: Pottery Barn arc lamp · $50–$130."]
    assert "don't have a number 9" in fam.say(MAYA, "remove 9")[0].text

    assert "Only Mom can export" in fam.say(MAYA, "export")[0].text
    assert "Nothing is confirmed yet" in fam.say(MOM, "export")[0].text
    assert "• how much so far" in fam.say(MAYA, "help")[0].text


def test_review_ok_and_export(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    out = fam.say(MOM, "review")
    assert out[0].text.startswith("3 new items from Maya, about $980.\n\n1. Blue velvet sofa · $290–$700\n"
                                  "2. Oak coffee table · $290–$700\n3. Brass floor lamp · $50–$130\n\n"
                                  'Reply OK to add them all, or tell me what\'s wrong ("remove 3").\nSee them: ')
    assert texts(fam.say(MOM, "remove 3")) == ["Removed the brass floor lamp. Say undo if that was a mistake."]
    assert texts(fam.say(MOM, "OK")) == ["Added 2 items. 2 items documented so far, about $900."]
    assert [i["status"] for i in fam.store.get_items(fam.case_id, canonical_only=True)].count("confirmed") == 2
    assert texts(fam.say(MOM, "review")) == ["Nothing new is waiting for your OK."]

    out = fam.say(MOM, "export")
    assert "2 items, about $900" in out[0].text and "not legal or insurance advice" in out[0].text
    assert [p.rsplit(".", 1)[1] for p in out[0].files] == ["pdf", "xlsx"]
    assert all(open(p, "rb").read(2) in (b"%P", b"PK") for p in out[0].files)

    assert texts(fam.say(MOM, "undo")) == ["Undone: those 2 items are back to waiting for an OK."]


def test_daily_review_goes_to_the_owner_once(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    fam.now = fam.now.replace(hour=18, minute=0)
    out = fam.engine.tick(fam.now)
    assert len(out) == 1 and out[0].chat_id == MOM and out[0].text.startswith("3 new items from Maya")
    assert fam.engine.tick(fam.now + timedelta(minutes=5)) == []                     # once a day
    for day in range(1, 7):                                                          # nothing new: not the same list again
        assert fam.engine.tick(fam.now + timedelta(days=day)) == []
    nudge = texts(fam.engine.tick(fam.now + timedelta(days=7)))
    assert nudge == ["3 things are still waiting for your OK. No rush. Say review whenever you're ready."]
    assert fam.engine.tick(fam.now + timedelta(days=8)) == []                        # at most one a week
    lamp = next(i for i in fam.store.get_items(fam.case_id, canonical_only=True) if "lamp" in i["label"])
    fam.store.update_item(fam.case_id, lamp["item_id"], reviewed=0)                 # as if it had just come in
    fresh = fam.engine.tick(fam.now + timedelta(days=9))                             # something new: the review comes again
    assert len(fresh) == 1 and fresh[0].text.startswith("1 new item from Maya") and '("remove 1")' in fresh[0].text
    fam.say(MOM, "pause")
    fam.store.set_case(fam.case_id, last_review_at=None)
    assert fam.engine.tick(fam.now + timedelta(days=2)) == []                        # paused: nothing


def test_pause_until_they_write_again(fam):
    assert texts(fam.say(MOM, "not today")) == ["Okay. I'll stay quiet until you write again."]
    assert fam.store.get_member(MOM)["paused"] == 1
    fam.say(MOM, "help")
    assert fam.store.get_member(MOM)["paused"] == 0


def test_memory_by_text_then_ok(fam):
    out = fam.say(MOM, "call me")
    assert out[0].text.startswith("I can't call from this number yet, so let's do it here, one question at a time.")
    assert "living room" in out[1].text                                              # the biggest gap first
    nxt = fam.say(MOM, "a big TV, two armchairs and a bunch of records")
    assert len(nxt) == 1 and "?" in nxt[0].text
    fam.say(MOM, "I don't remember")
    out = fam.say(MOM, "that's enough")
    assert out[0].text.startswith('That was really helpful. From our chat: 3 items, about ')
    assert 'marked "from memory"' in out[0].text and "1 big tv · 2 armchairs · 3 records, assorted" in out[0].text
    items = [i for i in fam.store.get_items(fam.case_id) if i["source"] == "voice"]
    assert [(i["label"], i["qty"], i["status"]) for i in items] == [
        ("big tv", 1, "suggested"), ("armchairs", 2, "suggested"), ("records, assorted", 1, "suggested")]
    assert fam.say(MOM, "OK")[0].text.startswith("Added 3 items.")
    assert fam.store.get_member(MOM)["state"] == {}

    fam.say(MOM, "call me")
    assert texts(fam.say(MOM, "stop")) == ["That's fine. Thank you for trying. We can pick it up another day."]


def test_a_mistyped_code_does_not_start_a_new_home(fam):
    cases = len(fam.store.list_cases())
    for words in ("HOME-12345", "home 99", "my code is home-123"):
        assert texts(fam.say(DAD, words, name="Dad"))[0].startswith("That code doesn't look quite right.")
    assert fam.store.get_member(DAD) is None and len(fam.store.list_cases()) == cases
    assert texts(fam.say(DAD, f"home {fam.code[5:]}", name="Dad"))[0].startswith("You've joined")
    fam.say("+15550009", "hello?", name="Gran")                    # a plain first message still starts a home...
    assert len(fam.store.list_cases()) == cases + 1
    assert texts(fam.say("+15550009", fam.code))[0].startswith("You've joined")
    assert len(fam.store.list_cases()) == cases                    # ...and the empty one goes when she joins
    assert fam.store.get_member("+15550009")["case_id"] == fam.case_id


def test_a_memory_chat_skips_command_words_keeps_prices_and_can_be_undone(fam):
    voice = lambda: [i for i in fam.store.get_items(fam.case_id) if i["source"] == "voice" and i["status"] != "rejected"]
    fam.say(MOM, "call me")
    for words in ("exprot", "menu", "ok"):
        assert "?" in fam.say(MOM, words)[-1].text
    assert voice() == []
    fam.say(MOM, "a piano worth 3000 and 2 bikes")
    fam.say(MOM, "an old quilt")
    assert texts(fam.say(MOM, "undo")) == ["Undone: I've taken that off the list."]
    piano = next(i for i in voice() if "piano" in i["label"])
    assert (piano["label"], piano["price_mid"], piano["price_source"]) == ("piano", 3000, "set by family")
    out = fam.say(MOM, "stop")
    assert out[0].text.startswith("That was really helpful. From our chat: 2 items, about $3,")
    assert "quilt" not in out[0].text
    fam.say(MOM, "OK")
    assert fam.store.get_item(fam.case_id, piano["item_id"])["price_mid"] == 3000


def test_delete_my_case(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    assert fam.say(MAYA, "delete my case")[0].text.startswith("I'm not sure")        # only the owner can
    assert "Reply yes to delete" in fam.say(MOM, "delete my case")[0].text
    assert texts(fam.say(MOM, "no wait")) == ["Okay, nothing was deleted."]
    fam.say(MOM, "delete my case")
    assert "has been deleted" in fam.say(MOM, "yes")[0].text
    assert fam.store.get_case(fam.case_id) is None and fam.store.get_items(fam.case_id) == []
    assert fam.store.get_members(fam.case_id) == []


def test_parse_date():
    from ashline.chat import parse_date
    assert [parse_date(t) for t in ("January 2025", "Jan 7, 2025", "it was the 7th of january 2025", "1/7/2025",
                                    "2025-01-07", "last winter", "skip", "13/40/2025")] == [
        "2025-01-01", "2025-01-07", "2025-01-07", "2025-01-07", "2025-01-07", None, None, None]


def test_helpers():
    assert split_rooms("Living room, kitchen, two bathrooms and a garage") == [
        "Living room", "Kitchen", "Bathroom", "Bathroom 2", "Garage"]
    assert [parse_money(t) for t in ("$400,000", "400k", "1.2 million", "about 250000", "none")] == [
        400000, 400000, 1200000, 250000, None]
    assert extract_things("Yes, a table saw, two bikes and a bunch of tools") == [
        ("table saw", 1), ("bikes", 2), ("tools, assorted", 1)]
    assert extract_things("no") == [] and extract_things("my grandmother's quilt") == [("grandmother's quilt, assorted", 1)]
    buf = io.BytesIO()
    image = Image.new("RGB", (300, 200))
    exif = image.getexif()
    exif[0x0132] = "2019:11:28 16:20:00"
    image.save(buf, format="JPEG", exif=exif)
    assert normalize_photo(buf.getvalue())["taken"] == "2019-11-28"
    assert normalize_photo(b"nope") is None


class ScanDetector:
    """For scan faces: a bed in every bedroom view, a sofa and two stools in the living room."""
    def detect(self, path, room_hint=None, rooms=None):
        if str(path).endswith("f1.jpg") and room_hint and "Living" in room_hint:
            return [{"label": "green velvet sofa", "category": "furniture", "confidence": 0.8, "bbox": [0, 0, 30, 30]},
                    {"label": "oak bar stool", "category": "furniture", "confidence": 0.8, "bbox": [31, 0, 40, 30]},
                    {"label": "oak bar stool", "category": "furniture", "confidence": 0.8, "bbox": [41, 0, 50, 30]}]
        if str(path).endswith("f1.jpg") and room_hint == "Bedroom":
            return [{"label": "queen bed", "category": "furniture", "confidence": 0.8, "bbox": [0, 0, 30, 30]}]
        return []


def test_onboarding_from_the_scan_of_the_address(index, monkeypatch):
    from ashline import seed
    from ashline.costar import index as index_mod
    from fakes import FakeMatterport, FakeVexcel
    monkeypatch.setattr(index_mod, "_INDEX", index)
    monkeypatch.setattr(seed.matterport, "MatterportClient", FakeMatterport)
    monkeypatch.setattr(seed.vexcel, "VexcelClient", FakeVexcel)
    monkeypatch.setattr(seed.detect_mod, "get_detector", ScanDetector)

    f = Family()
    f.say(MOM, "hi", name="Mom")
    f.say(MOM, "Rivera home")
    out = f.say(MOM, "12 Example Ave, Los Angeles")
    assert out[0].text == ("I found a 3D scan of this address from March 2026 with 14 rooms. Use it as a starting "
                           "point? It may show staging or a previous owner's things, so nothing is added until you "
                           "confirm it.")
    assert texts(f.say(MOM, "yes")) == ["When did it happen? A month and year is enough. It lets me use an aerial "
                                        "photo of the home from before it. You can say skip."]
    out = f.say(MOM, "January 2025")
    assert out[0].text.startswith("Thank you. I'll go through the scan now") and "Your code is HOME-" in out[1].text
    case_id = f.store.get_member(MOM)["case_id"]
    assert f.store.get_case(case_id)["loss_date"] == "2025-01-01"
    assert f.store.get_items(case_id) == []                                # nothing yet: it runs on the next tick

    out = f.wait(5)
    assert out[0].chat_id == MOM
    assert out[0].text.startswith("I've gone through the scan: 3 possible things in 2 rooms, about $")
    assert "None of it is on the list until you say so." in out[0].text and "/p/" in out[0].text
    assert out[1].text.startswith("The aerial photo from August 2024 shows a second building on the lot")
    items = f.store.get_items(case_id, canonical_only=True)
    assert all(i["status"] == "suggested" and i["source"] == "scan" and i["price_mid"] for i in items)
    assert [r["name"] for r in f.store.get_rooms(case_id)][:2] == ["Living Room / Dining Room", "Garage"]
    assert f.store.get_home(case_id)["sid"] == "fakeA000001"
    assert f.wait(5) == []                                                 # the job runs once

    review = f.say(MOM, "review")[0].text
    assert review.startswith("3 new items from the 3D scan, about $") and "1. 2 × oak bar stool" in review
    assert f.say(MOM, "ok")[0].text.startswith("Added 3 items.")


def test_no_scan_for_the_address_is_normal(index, monkeypatch):
    from ashline.costar import index as index_mod
    monkeypatch.setattr(index_mod, "_INDEX", index)
    f = Family()
    f.say(MOM, "hi")
    f.say(MOM, "Rivera home")
    assert texts(f.say(MOM, "1 Nowhere Ln, Pacific Palisades")) == [
        "I don't have a scan of that address, which is normal. Which rooms did it have? A rough list is fine."]
    assert f.store.get_case(f.store.get_member(MOM)["case_id"])["address"] == "1 Nowhere Ln, Pacific Palisades"

    g = Family()
    g.engine.lookup = lambda address: {"found": True, "sid": "x", "original_created": "2026-03-04", "room_count": 9}
    g.say(DAD, "hi")
    g.say(DAD, "Rivera home")
    g.say(DAD, "12 Example Ave")
    assert texts(g.say(DAD, "no thanks")) == ["That's fine, we'll leave it. Which rooms did it have? A rough list is fine."]


def test_a_long_review_goes_one_room_at_a_time(fam):
    for n in range(40):
        room = "Garage" if n < 25 else "Kitchen"
        item_id = fam.store.add_item(fam.case_id, {"label": f"thing {n}", "category": "tools", "room_label": room},
                                     source="scan", added_by="scan")
        fam.store.update_item(fam.case_id, item_id, price_low=50, price_mid=100 if room == "Garage" else 10, price_high=150)
    first = fam.say(MOM, "review")[0].text
    assert first.startswith("Garage: (1 more room after this.) 25 new items from the 3D scan, about $2,500.")
    assert fam.say(MOM, "ok")[0].text.startswith("Added 25 items.")
    second = fam.say(MOM, "review")[0].text
    assert second.startswith("15 new items from the 3D scan, about $150.")       # few enough for one message
    assert len([i for i in fam.store.get_items(fam.case_id) if i["status"] == "confirmed"]) == 25


def test_family_sets_a_price_and_splits_a_merge(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    fam.say(MAYA, "", [photo(648), photo(656)])            # the sofa is seen again and merged
    fam.wait()
    fam.say(MOM, "review")
    numbered = fam.store.get_chat(MOM)["context"]["list"]
    sofa_n = next(n for n, i in numbered.items() if "sofa" in fam.store.get_item(fam.case_id, i)["label"])
    assert texts(fam.say(MOM, f"{sofa_n} was $2,400")) == [
        "Set the blue velvet sofa to $2,400, marked as a price set by the family."]
    sofa = fam.store.get_item(fam.case_id, numbered[sofa_n])
    assert (sofa["price_low"], sofa["price_mid"], sofa["price_high"], sofa["price_source"]) == (2400, 2400, 2400, "set by family")
    assert texts(fam.say(MOM, "undo")) == ["Undone: that item is back to how it was."]
    assert fam.store.get_item(fam.case_id, numbered[sofa_n])["price_source"] == "category default"

    lines = lambda: len(fam.store.get_items(fam.case_id, canonical_only=True))
    before = lines()
    assert texts(fam.say(MOM, f"split {sofa_n}")) == ["Split into 2 separate items. Say undo if that was a mistake."]
    assert lines() == before + 1
    assert texts(fam.say(MOM, "undo")) == ["Undone: those are one item again."] and lines() == before
    other = next(n for n in numbered if n != sofa_n)
    assert "nothing to split" in fam.say(MOM, f"split {other}")[0].text


def test_one_directed_ask_a_day(fam):
    fam.say(MAYA, "", [photo(640), photo(648)])
    out = fam.wait()
    assert re.fullmatch(r"Maya, there's nothing from the (kitchen|bedroom|garage)( \d)? yet\. "
                        r"Any birthday or holiday photos taken there\?", [t for t in texts(out) if not t.startswith("Photo ")][-1])
    fam.say(MAYA, "", [photo(656), photo(672)])
    out = fam.wait()
    assert not any("Any birthday" in t for t in texts(out))                       # already asked today
    fam.now += timedelta(days=1)
    fam.say(MAYA, "", [photo(664), photo(672)])
    asked = [t for t in texts(fam.wait()) if "photos taken there" in t]
    assert asked == [] or "kitchen only has 1 item so far" in asked[0] or "bedroom" in asked[0]


def test_deadline_reminders_and_weekly_nudge(fam):
    fam.store.set_case(fam.case_id, coverage_limit=400000, loss_date="2026-10-15")       # due 2027-01-23
    at = lambda d: datetime.fromisoformat(d + "T18:00:00+00:00")
    assert fam.engine.tick(at("2026-11-20")) == []                                       # 64 days left, seen this week
    out = fam.engine.tick(at("2026-11-24"))                                              # 60 days left
    assert texts(out) == ["A gentle reminder: proof of loss is due in 60 days. You're at $0 of $160k documented."]
    assert fam.engine.tick(at("2026-11-24")) == []                                       # once
    assert "due in 30 days" in fam.engine.tick(at("2026-12-24"))[0].text
    assert fam.engine.tick(at("2026-12-27")) == []                                       # reminded 3 days ago
    nudge = fam.engine.tick(at("2026-12-31"))                                            # a week of silence
    assert texts(nudge) == ["You're at $0 of $160k. Even a few more photos would help, whenever you feel up to it."]
    assert fam.engine.tick(at("2027-01-02")) == []                                       # at most one a week
    fam.say(MOM, "pause")
    assert fam.engine.tick(at("2027-01-13")) == []                                       # 10 days left, but paused


def test_short_answers_with_both_words_count():
    from ashline.chat import yes_or_no
    assert yes_or_no("1 same") == (True, False) and yes_or_no("2, different") == (False, True)
    assert yes_or_no("yes same") == (True, False) and yes_or_no("not the same") == (False, True)
    assert yes_or_no("same couch as the kitchen one") == (False, False)


def test_the_same_photo_twice_adds_nothing(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    before = len(fam.store.get_items(fam.case_id))
    assert texts(fam.say(MAYA, "", [photo(640)])) == ["I already have that one (photo 1), so nothing new was added."]
    assert fam.wait() == [] and len(fam.store.get_items(fam.case_id)) == before


def test_one_object_listed_twice_in_a_photo_is_folded():
    from ashline.dedupe import dedupe_in_photo
    items = [{"label": "gray sofa", "bbox": [0, 0, 400, 200], "confidence": 0.9},
             {"label": "gray fabric sofa", "bbox": [20, 10, 300, 190], "confidence": 0.6},   # the same sofa again
             {"label": "oak chair", "bbox": [500, 0, 600, 200], "confidence": 0.8},
             {"label": "oak chair", "bbox": [610, 0, 710, 200], "confidence": 0.8},          # a second chair
             {"label": "oak chair", "bbox": [560, 20, 650, 210], "confidence": 0.7}]         # a third, partly behind
    kept = dedupe_in_photo(items)
    assert [i["label"] for i in kept] == ["gray sofa", "oak chair", "oak chair", "oak chair"]


def test_free_text_is_mapped_to_a_real_command_or_a_reply():
    from ashline.intent import valid_command
    asked = []

    def fake(message, context=""):
        asked.append((message, context))
        table = {"can u send me the lsit pls": {"command": "list", "reply": None},
                 "how r u": {"command": None, "reply": "Doing fine, thanks for asking."},
                 "get rid of all my stuff in here": {"command": "delete my case", "reply": None},
                 "make me a sandwich": {"command": valid_command("make sandwich"), "reply": None}}
        return table[message]

    fam = Family().setup()
    fam.engine._interpreter = fake
    out = fam.say(MOM, "can u send me the lsit pls")
    assert "/p/" in out[0].text                                         # the list link, as if "list" was typed
    assert texts(fam.say(MOM, "how r u")) == ["Doing fine, thanks for asking."]
    assert "Reply yes to delete" in fam.say(MOM, "get rid of all my stuff in here")[0].text
    assert fam.store.get_case(fam.case_id) is not None                  # only asked; nothing deleted yet
    assert texts(fam.say(MOM, "no wait")) == ["Okay, nothing was deleted."]
    assert "I'm not sure" in fam.say(MOM, "make me a sandwich")[0].text  # an invented command is refused
    assert "owns the case" in asked[0][1]


def test_without_a_key_only_exact_commands_work(fam):
    assert "I'm not sure" in fam.say(MOM, "can u send me the lsit pls")[0].text



def test_plain_requests_to_delete_still_ask_first(fam):
    assert "Reply yes to delete" in fam.say(MOM, "Can you please delete my case")[0].text
    assert "deleted" in fam.say(MOM, "Yes really delete my case")[0].text
    assert fam.store.get_case(fam.case_id) is None



def test_room_captions_are_tidied_and_reused(fam):
    fam.say(MAYA, "just basement", [photo(640)])
    fam.wait()
    fam.say(MAYA, "basement", [photo(648)])
    fam.wait()
    fam.say(MAYA, "this is the Basement", [photo(656)])
    fam.wait()
    rooms = {fam.store.get_photo(fam.case_id, number=n)["room_label"] for n in (1, 2, 3)}
    assert rooms == {"Basement"}



def test_photos_that_could_not_be_fetched_or_opened_are_mentioned(fam):
    from ashline.chat import Inbound
    fam.now += timedelta(seconds=5)
    out = fam.engine.handle(Inbound(MAYA, MAYA, "", [photo(640)], missing=2, now=fam.now))
    assert texts(out) == ["Got it, looking at it.", "I couldn't open 2 of the photos. Could you send them again?"]
    assert texts(fam.say(MAYA, "", [("IMG_9.jpg", b"\xff\xd8 broken jpeg")])) == [
        "I couldn't open one of the photos. Could you send it again?"]


def test_rooms_answer_in_a_sentence():
    assert split_rooms("it only has basement") == ["Basement"]
    assert split_rooms("we had a kitchen, 2 bedrooms and our garage") == ["Kitchen", "Bedroom", "Bedroom 2", "Garage"]


def test_saying_these_are_new_unmatches_items_from_the_last_photo(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    fam.say(MAYA, "", [photo(648)])                               # its sofa gets matched to the first one
    out = fam.wait()
    legend = next(o.text for o in out if o.image)
    assert "already listed" in legend and legend.endswith('Reply "new" and I\'ll list it separately.')
    before = len(fam.store.get_items(fam.case_id, canonical_only=True))

    reply = fam.say(MAYA, "These were not previously added")
    assert reply[0].text.startswith("Got it, those are new. Added")
    assert len(fam.store.get_items(fam.case_id, canonical_only=True)) == before + 1
    sofas = [i for i in fam.store.get_items(fam.case_id, canonical_only=True) if "sofa" in i["label"]]
    assert len(sofas) == 2 and all(i["price_mid"] for i in sofas)            # the new one is priced too

    assert "back on the lines" in fam.say(MAYA, "undo")[0].text
    assert len(fam.store.get_items(fam.case_id, canonical_only=True)) == before
    numbers = fam.engine._context(MAYA)["photo_items"]
    sofa_n = next(n for n, i in numbers.items() if "sofa" in fam.store.get_item(fam.case_id, i)["label"])
    assert fam.say(MAYA, f"{sofa_n} is new")[0].text.startswith("Got it, those are new.")   # just that one, by number
    other_n = next(n for n in numbers if n != sofa_n)
    assert "already its own item" in fam.say(MAYA, f"{other_n} is new")[0].text


@pytest.mark.parametrize("words", ["new", "they're new", "those are different", "not the same", "these weren't added",
                                   "no, they are not on the list", "2 is new"])
def test_ways_of_saying_new(words):
    from ashline.chat import NEW_ITEMS
    assert re.fullmatch(NEW_ITEMS, words) or re.fullmatch(r"#?(\d+)\s+(?:is|was|are)\s+(?:new|different)", words)


def test_an_item_seen_through_a_doorway_is_filed_where_it_stands(fam):
    """A dining table glimpsed from the kitchen is the dining room's table, not a second one."""
    class Doorway:
        def detect(self, path, room_hint=None, rooms=None):
            table = {"label": "oak dining table", "category": "furniture", "brand_guess": None, "confidence": 0.8,
                     "bbox": [300, 200, 600, 440]}
            if Image.open(path).width == 656:                       # taken in the kitchen, looking through the doorway
                return [{"label": "red stand mixer", "category": "kitchen", "brand_guess": None, "confidence": 0.8,
                         "bbox": [100, 100, 220, 260], "room": "Kitchen", "photo_room": "Kitchen"},
                        {**table, "room": "Dining room", "photo_room": "Kitchen"}]
            return [{**table, "room": "Dining room", "photo_room": "Dining room"}]      # taken in the dining room

    fam.engine.detector = Doorway()
    fam.say(MAYA, photos=[photo(656), photo(640)])
    fam.wait()
    photos = {p["number"]: p["room_label"] for p in fam.store.get_photos(fam.case_id, source="photo")}
    assert photos == {1: "Kitchen", 2: "Dining room"}                # each photo keeps the room it was taken in
    lines = [i for i in fam.store.get_items(fam.case_id) if i["cluster_id"] in (None, i["item_id"])]
    assert sorted((i["label"], i["room_label"]) for i in lines) == [
        ("oak dining table", "Dining room"), ("red stand mixer", "Kitchen")]
    table = next(i for i in lines if "table" in i["label"])
    assert len(fam.store.get_sightings(fam.case_id, table["item_id"])) == 2 and table["qty"] == 1


def test_hurricane_family_gets_two_claims_and_essentials():
    from ashline.export import claim_groups, export_pdf, export_rows, export_xlsx
    from ashline.web import page_data
    f = Family()
    f.say(MOM, "hi", name="Mom")
    f.say(MOM, "The Rivera home")
    f.say(MOM, "skip")
    out = f.say(MOM, "living room, kitchen, basement")
    assert out[-1].text.startswith("One more question: what happened to the home?")
    case_id = f.store.get_member(MOM)["case_id"]
    assert "two claims" in f.say(MOM, "Hurricane Milton")[0].text
    reply = f.say(MOM, "the basement and kitchen")[0].text
    assert reply.startswith("Got it: ") and "Basement" in reply and "Kitchen" in reply and "Living room" not in reply
    assert "Do you have insurance" in reply
    assert "FEMA can help replace essentials" in f.say(MOM, "no")[0].text
    case = f.store.get_case(case_id)
    assert (case["disaster"], case["insured"]) == ("storm", "no")

    f.say(MOM, "kitchen", [photo(656)])                           # red stand mixer, kitchen: flooded
    f.wait()
    f.say(MOM, "living room", [photo(640)])                       # sofa, lamp, table: wind and rain
    f.wait()
    for item in f.store.get_items(case_id, canonical_only=True):
        f.store.set_item_status(case_id, item["item_id"], "confirmed")
    rows = export_rows(case_id, f.store)
    groups = {g["key"]: g for g in claim_groups(f.store.get_case(case_id), rows)}
    assert [r["item"] for r in groups["flood"]["rows"]] == ["red stand mixer"]
    assert groups["flood"]["acv"] and groups["flood"]["acv"] < groups["flood"]["new"]      # depreciated, lower
    assert len(groups["wind"]["rows"]) == 3 and groups["wind"]["acv"] == 0
    assert any(r["essential"] for r in rows)                                               # the sofa and table

    data = page_data(case_id, f.store, "owner")
    assert [c["key"] for c in data["claims"]] == ["flood", "wind"] and data["show_essentials"]
    from openpyxl import load_workbook
    book = load_workbook(export_xlsx(case_id, f.store))
    assert {"By claim", "Flood claim", "Wind and rain claim", "Essentials (FEMA)"} <= set(book.sheetnames)
    assert export_pdf(case_id, f.store).read_bytes().startswith(b"%PDF")

    assert "goes on the flood claim" in f.say(MOM, "the living room flooded")[0].text      # change a room later
    rows = export_rows(case_id, f.store)
    assert all(r["_claim"] == "flood" for r in rows)


def test_setup_question_never_blocks_a_family_that_moves_on(fam):
    out = fam.say(MOM, "how much so far")                         # the "what happened?" question is still open
    assert "$" in out[0].text or "items" in out[0].text            # answered as a normal command
    assert fam.store.get_member(MOM)["state"].get("step") != "disaster"     # the question was dropped
    assert fam.store.get_case(fam.case_id).get("disaster") is None # still the fire flow


def test_one_unreadable_photo_does_not_stop_the_others_and_is_retried(fam):
    from ashline import detect as detect_mod
    real = fam.detector.detect
    calls = {"bad": 0}

    def flaky(path, room_hint=None, rooms=None):
        if Image.open(path).width == 680:
            calls["bad"] += 1
            if calls["bad"] == 1:
                raise detect_mod.DetectionError("The item list was cut off at max_tokens.")
            return [{"label": "walnut dresser", "category": "furniture", "confidence": 0.8, "bbox": [10, 10, 300, 300]}]
        return real(path, room_hint, rooms)
    fam.detector.detect = flaky
    fam.say(MAYA, "", [photo(640), photo(680)])
    out = fam.wait()
    text = "\n".join(o.text for o in out)
    assert "Blue velvet sofa" in text or "blue velvet sofa" in text.lower()      # the good photo was read
    assert "couldn't read one of the photos" in text and "max_tokens" not in text  # plain words, no internals
    fam.say(MAYA, "", [photo(656)])                                               # the next burst retries it
    fam.wait()
    labels = [i["label"] for i in fam.store.get_items(fam.case_id)]
    assert "walnut dresser" in labels and calls["bad"] == 2


def test_claude_detector_retries_a_cut_off_reply_with_a_shorter_list(tmp_path):
    from types import SimpleNamespace as NS
    from ashline.detect_claude import ClaudeDetector
    replies = [NS(stop_reason="max_tokens", content=[], stop_details=None),
               NS(stop_reason="end_turn", stop_details=None,
                  content=[NS(type="text", text='{"items": [{"label": "sofa", "category": "furniture", "confidence": 0.9, "box": [0, 0, 500, 500]}]}')])]
    asked = []
    client = NS(messages=NS(create=lambda **kw: (asked.append(kw), replies.pop(0))[1]))
    det = ClaudeDetector(model="claude-sonnet-test", client=client) if "client" in ClaudeDetector.__init__.__code__.co_varnames else None
    if det is None:
        pytest.skip("detector takes no client")
    path = tmp_path / "p.jpg"
    Image.new("RGB", (640, 480)).save(path)
    items = det.detect(path)
    assert [i["label"] for i in items] == ["sofa"]
    assert "at most 15 items" in asked[1]["messages"][0]["content"][1]["text"] and asked[1]["output_config"]["effort"] == "low"


def listed(fam):
    fam.say(MAYA, photos=[photo(640)])                            # 1 sofa, 2 lamp, 3 coffee table
    fam.wait()


def test_a_relative_undoes_only_their_own_change_and_the_owner_any(fam):
    listed(fam)
    fam.say(MOM, "review")
    mine, hers = fam.engine._context(MOM)["list"], fam.engine._context(MAYA)["list"]
    status = lambda item_id: fam.store.get_item(fam.case_id, item_id)["status"]
    fam.say(MOM, "remove 1")
    assert texts(fam.say(MAYA, "undo"))[0].startswith("There's nothing of yours to undo.")
    assert status(mine["1"]) == "rejected"
    other = next(n for n, i in hers.items() if i != mine["1"])
    fam.say(MAYA, f"remove {other}")
    last = next(n for n, i in mine.items() if i not in (mine["1"], hers[other]))
    fam.say(MOM, f"remove {last}")
    assert texts(fam.say(MAYA, "undo")) == ["Undone: it's back on the list."]          # her change, not Mom's
    assert status(hers[other]) != "rejected" and status(mine[last]) == "rejected"
    fam.say(MAYA, f"remove {other}")
    assert texts(fam.say(MOM, "undo")) == ["Undone: it's back on the list."]           # the owner undoes anyone's
    assert status(hers[other]) != "rejected"


def test_a_rename_keeps_a_price_the_family_set_and_undo_restores_it(fam):
    listed(fam)
    fam.say(MAYA, "price 2 150")
    assert texts(fam.say(MAYA, "2 is a tiffany lamp")) == ["Updated: tiffany lamp · $150."]
    lamp = fam.engine._context(MAYA)["list"]["2"]
    fam.say(MAYA, "1 is a grey couch")                            # an estimate is looked up again for a new name
    assert texts(fam.say(MAYA, "undo")) == ["Undone: the name is back to what it was."]
    assert texts(fam.say(MAYA, "undo")) == ["Undone: the name is back to what it was."]
    item = fam.store.get_item(fam.case_id, lamp)
    assert (item["label"], item["price_mid"], item["price_source"]) == ("brass floor lamp", 150, "set by family")


def test_room_and_quantity_changes_are_not_renames_and_can_be_undone(fam):
    listed(fam)
    table = fam.engine._context(MAYA)["list"]["3"]
    assert texts(fam.say(MAYA, "3 was in the kitchen")) == [
        "Moved the oak coffee table to the kitchen. Say undo if that was a mistake."]
    item = fam.store.get_item(fam.case_id, table)
    assert (item["label"], item["room_label"]) == ("oak coffee table", "Kitchen")
    fam.say(MAYA, "move 3 to the garage")
    assert fam.store.get_item(fam.case_id, table)["room_label"] == "Garage"
    for words, qty in (("3 x2", 2), ("qty 3 4", 4), ("there were 3 of number 3", 3)):
        assert texts(fam.say(MAYA, words)) == [f"Got it: {qty} of the oak coffee table. Say undo if that was a mistake."]
        assert fam.store.get_item(fam.case_id, table)["qty"] == qty
    for _ in range(3):
        assert texts(fam.say(MAYA, "undo")) == ["Undone: the number is back to what it was."]
    assert texts(fam.say(MAYA, "undo")) == ["Undone: it's back in the room it was in."]
    assert texts(fam.say(MAYA, "undo")) == ["Undone: it's back in the room it was in."]
    item = fam.store.get_item(fam.case_id, table)
    assert (item["label"], item["qty"], item["room_label"]) == ("oak coffee table", 1, "Living room")


@pytest.mark.parametrize("text", ["632 N Detroit St, Los Angeles, CA 90036", "address 632 N Detroit St, Los Angeles, CA",
                                  "Our address is 12 Oak Street, Pasadena"])
def test_an_address_after_setup_is_saved(fam, text):
    fam.engine.lookup = lambda address: {"found": False, "reason": "no scan"}
    out = fam.say(MOM, text)
    assert out[0].text.startswith("Saved the address")
    assert fam.store.get_case(fam.case_id)["address"].startswith(("632 N Detroit", "12 Oak"))


def test_an_address_with_a_scan_offers_it_without_repeating_setup(fam):
    fam.engine.lookup = lambda address: {"found": True, "sid": "abc", "room_count": 6, "original_created": "2021-05-01"}
    out = fam.say(MOM, "632 N Detroit St, Los Angeles, CA 90036")
    assert "I found a 3D scan" in out[0].text
    assert texts(fam.say(MOM, "no")) == ["That's fine, we'll leave it."]          # no rooms question again
    maya = fam.say(MAYA, "1 Elm St, Austin, TX")                                     # only the owner can set it
    assert not any(o.text.startswith("Saved the address") for o in maya)
    assert fam.store.get_case(fam.case_id)["address"].startswith("632 N Detroit")


@pytest.mark.parametrize("reply,used", [("yes! use it as a starting point'", True), ("yes, use it as a starting point", True),
                                        ("sure", True), ("no thanks", False), ("skip", False)])
def test_scan_offer_understands_natural_answers(fam, reply, used):
    fam.engine.lookup = lambda address: {"found": True, "sid": "abc", "room_count": 9, "original_created": "2026-10-01"}
    fam.say(MOM, "632 N Detroit St, Los Angeles, CA 90036")
    out = fam.say(MOM, reply)
    assert out[0].text.startswith("When did it happen?") == used


def test_unclear_scan_answer_is_asked_again_and_use_the_scan_works_later(fam):
    fam.engine.lookup = lambda address: {"found": True, "sid": "abc", "room_count": 9, "original_created": "2026-10-01"}
    fam.say(MOM, "632 N Detroit St, Los Angeles, CA 90036")
    assert "Yes or no is fine" in fam.say(MOM, "hmm what does that mean")[0].text
    fam.say(MOM, "no")
    assert fam.say(MOM, "actually, use the scan")[0].text.startswith("When did it happen?")


def test_scan_and_date_answers_work_in_the_family_group(fam):
    fam.engine.lookup = lambda address: {"found": True, "sid": "abc", "room_count": 9, "original_created": "2026-10-01"}
    group = "group-rivera"
    fam.say(MOM, fam.code, chat=group, name="Mom")                           # the group is linked to the home
    assert "I found a 3D scan" in fam.say(MOM, "632 N Detroit St, Los Angeles, CA 90036", chat=group)[0].text
    assert fam.say(MOM, "yes! use it as a starting point", chat=group)[0].text.startswith("When did it happen?")
    out = fam.say(MOM, "july 2025", chat=group)
    assert out[0].text.startswith("Thank you. I'll go through the scan now")
    assert fam.store.get_case(fam.case_id)["loss_date"] == "2025-07-01"


@pytest.mark.parametrize("text,when", [("it happened in July 2025", "2025-07-01"), ("the fire was in january 2025", "2025-01-01"),
                                       ("date of loss: 7/2025", "2025-07-01"), ("date July 2025", "2025-07-01")])
def test_the_date_can_be_given_any_time(fam, text, when):
    out = fam.say(MOM, text)
    assert out[0].text.startswith("Saved: it happened in") or "insurance" in out[0].text   # or it answered "what happened?"
    assert fam.store.get_case(fam.case_id)["loss_date"] == when


def test_a_failed_scan_can_be_retried_and_the_error_is_plain(fam):
    from ashline import seed
    fam.engine.lookup = lambda address: {"found": True, "sid": "abc", "room_count": 9, "original_created": "2026-10-01"}
    calls = []

    def boom(case_id, **kw):
        calls.append(1)
        raise RuntimeError("MATTERPORT_API_KEY is not set. Put it in .env (see .env.example).")
    real, seed.seed_home_twin = seed.seed_home_twin, boom
    try:
        fam.say(MOM, "632 N Detroit St, Los Angeles, CA 90036")
        fam.say(MOM, "yes")
        fam.say(MOM, "july 2025")
        out = fam.wait(5)
        text = " ".join(o.text for o in out)
        assert "couldn't go through the scan" in text and "MATTERPORT" not in text and "try again" in text
        assert fam.say(MOM, "try again")[0].text.startswith("Trying the scan again")
        fam.wait(5)
        assert len(calls) == 2
    finally:
        seed.seed_home_twin = real
