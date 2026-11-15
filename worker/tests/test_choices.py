"""The questions that come with tappable answers: which ones, and that a tap's reply text does what typing it does."""
import pytest

from ashline import bank
from test_bank import FakeNessie
from test_chat import DAD, GROUP, MAYA, MOM, Family, photo, texts
from test_receipts import Reader, picture

SAME = [("Same", "1"), ("Different", "2")]
PRICES = [("Use these prices", "yes"), ("Keep as is", "no")]
ADD = [("Add it", "yes"), ("Leave it off", "no")]


@pytest.mark.parametrize("tap, answer", [("Same", "Thanks, counted once."), ("Different", "Thanks, kept as two.")])
def test_same_item_question(tap, answer):
    f = Family().setup()
    f.say(MAYA, "", [photo(640)])
    f.wait()
    f.say(DAD, f.code, name="Dad")
    f.say(DAD, "", [photo(664), photo(656)])
    question = next(o for o in f.wait() if o.text.startswith("Same sofa?"))
    assert question.choices == SAME and question.text == "Same sofa? 1 = same, 2 = different."
    assert texts(f.say(DAD, dict(SAME)[tap])) == [answer]


def test_heard_item_question():
    f = Family().setup()
    f.say(MAYA, "is this thing on", chat=GROUP)
    f.say(MAYA, "Remember Grandma's piano?", chat=GROUP)
    f.say(MAYA, "", [photo(640), photo(656)], chat=GROUP)
    heard = next(o for o in f.wait() if o.text.startswith("I heard a piano"))
    assert heard.choices == ADD and heard.text == "I heard a piano mentioned. Add it to the list? Reply yes or no."
    assert texts(f.say(DAD, "Ashline " + dict(ADD)["Add it"], chat=GROUP, name="Dad"))[0].startswith("Added the piano")


def test_bank_price_offer():
    f = Family().setup()
    f.engine.bank = bank.NessieClient(key="test-key", session=FakeNessie())
    f.say(MOM, photos=[photo(640)])
    f.wait()
    offer = f.say(MOM, "check my bank")[0]
    assert offer.choices == PRICES and offer.text.endswith("Reply yes or no.")
    assert texts(f.say(MOM, dict(PRICES)["Keep as is"])) == ["Okay, the prices stay as they are."]


def test_receipt_price_offer_only_when_something_matches():
    f = Family().setup()
    f.engine.receipt_reader = Reader()
    f.say(MOM, photos=[photo(640)])
    f.wait()
    read = f.say(MOM, "receipt", photos=[picture()])[0]
    assert read.choices == PRICES
    assert texts(f.say(MOM, dict(PRICES)["Use these prices"]))[0].startswith("Done. 1 item now carries the price you paid")
    again = f.say(MOM, "receipt", photos=[picture()])[0]             # the sofa is priced now: nothing to offer
    assert "matches things on your list" not in again.text and again.choices == []


def test_other_messages_have_no_buttons():
    f = Family().setup()
    assert all(o.choices == [] for o in f.say(MOM, "how much so far") + f.say(MOM, "help"))
