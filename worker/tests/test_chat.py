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


class Family:
    """Sends messages and advances the clock."""

    def __init__(self):
        self.store, self.detector, self.now = Store(), FakeDetector(), T0
        self.engine = Engine(self.store, detector=self.detector, pricer=DefaultPricer(),
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
        "I'm Ashline. I help families rebuild the list of belongings insurance asks for after a fire. It takes a few "
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
    assert out[0].text == ("Found 3 items, 3 new to the list. 1 blue velvet sofa ($290–$700) · "
                           "2 brass floor lamp ($50–$130) · 3 oak coffee table ($290–$700)")
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
    assert out[0].text.startswith("Got 3 photos from Maya. Found 3 items, 2 new to the list. Merged 1 I'd already seen. "
                                  "Biggest finds: ")
    assert "red stand mixer ($30–$70)" in out[0].text and out[0].text.endswith(
        "Full list: https://ashline.example/p/" + fam.store.get_case(fam.case_id)["member_token"])
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
    assert "Send photos from inside the home" in fam.say(DAD, "Ashline, what are you", chat=GROUP)[0].text
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
    assert out[0].text == "0 items documented so far, about $0. 3 more items ($980) are waiting for an OK."
    assert "contents coverage limit" in out[1].text and "skip" in out[1].text
    assert texts(fam.say(MOM, "$400k")) == [
        "You're at $0 of $160k documented. 3 more items ($980) are waiting for an OK."]
    assert len(fam.say(MAYA, "how much so far")) == 1                                # relatives are not asked

    shown = fam.say(MAYA, "show photo 1")
    assert shown[0].image.endswith("fam-1.jpg") and shown[0].text.startswith("1 blue velvet sofa")
    assert "don't have a photo 7" in fam.say(MAYA, "show photo 7")[0].text
    assert len(fam.say(MAYA, "show the living room")) == 1
    assert "don't have photos of the garage" in fam.say(MAYA, "show the garage")[0].text

    assert texts(fam.say(MAYA, "2 isn't ours")) == ["Removed the brass floor lamp. Say undo if that was a mistake."]
    assert texts(fam.say(MAYA, "undo")) == ["Undone: it's back on the list."]
    assert texts(fam.say(MAYA, "undo")) == ["There's nothing to undo."]
    assert texts(fam.say(MAYA, "2 is a Pottery Barn arc lamp")) == ["Updated: Pottery Barn arc lamp ($50–$130)."]
    assert "don't have a number 9" in fam.say(MAYA, "remove 9")[0].text

    assert "Only Mom can export" in fam.say(MAYA, "export")[0].text
    assert "Nothing is confirmed yet" in fam.say(MOM, "export")[0].text
    assert "list, how much so far" in fam.say(MAYA, "help")[0].text


def test_review_ok_and_export(fam):
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    out = fam.say(MOM, "review")
    assert out[0].text.startswith('3 new items from Maya, about $980. Reply OK to add all, or tell me what\'s wrong '
                                  '("remove 4"). 1 blue velvet sofa · 2 oak coffee table · 3 brass floor lamp. See them: ')
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
    assert fam.engine.tick(fam.now + timedelta(days=1)) != []                        # unanswered: rolls into tomorrow
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
    assert texts(f.say(MOM, "yes")) == ["When was the fire? A month and year is enough. It lets me use an aerial "
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
    assert review.startswith("3 new items from the 3D scan, about $") and "2 × " not in review
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
