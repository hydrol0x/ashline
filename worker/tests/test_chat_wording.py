"""Small conversation fixes: everyday wording, typos, greetings, skipped setup answers, things remembered."""
import pytest

from test_chat import DAD, MAYA, MOM, Family, fam, photo, texts  # noqa: F401  (fam is a fixture)


@pytest.mark.parametrize("words", ["show me the list", "show the list", "show list", "the list", "my list",
                                   "what's on the list", "What's on the list?"])
def test_asking_for_the_list_sends_the_list(fam, words):
    out = fam.say(MOM, words)
    assert len(out) == 1 and out[0].text.startswith("Here is everything so far: https://ashline.example/p/")


def test_the_coverage_question_is_asked_once_per_home(fam):
    first = fam.say(MOM, "how much so far")
    assert len(first) == 2 and "coverage limit" in first[1].text
    fam.say(MOM, "list")                                        # not an answer: the question lapses
    for words in ("how much", "total?", "how much so far"):
        out = fam.say(MOM, words)
        assert len(out) == 1 and "coverage" not in out[0].text
    assert fam.store.get_case(fam.case_id)["coverage_skipped"] == 0       # the page can still ask for it


@pytest.mark.parametrize("unsure", ["idk", "I don't know", "dont know", "not sure", "no idea", "?", "skip"])
def test_not_knowing_a_setup_answer_skips_the_question(unsure):
    f = Family()
    f.say(MOM, "hi", name="Mom")
    assert "address" in f.say(MOM, unsure)[0].text
    assert "Which rooms" in f.say(MOM, unsure)[0].text
    out = f.say(MOM, unsure)
    assert "rebuild our home's inventory" in out[1].text and "what happened" in out[-1].text
    case = f.store.get_case(f.store.get_member(MOM)["case_id"])
    assert case["name"] == "Our home" and not case["address"] and not case["rooms_note"]
    assert f.store.get_rooms(case["case_id"]) == []
    out = f.say(MOM, unsure)                                    # the disaster question too
    assert texts(out) == ["That's fine, we can leave it for now. Send photos from inside the home whenever you're ready."]
    assert f.store.get_member(MOM)["state"] == {}
    assert texts(f.say(MOM, "fire")) == ["Got it, a fire."]     # a bare word still answers it later
    assert f.store.get_case(case["case_id"])["disaster"] == "fire"


@pytest.mark.parametrize("word,kind", [("fire", "fire"), ("Wildfire", "fire"), ("flood", "flood"),
                                       ("hurricane", "storm"), ("a tornado", "storm"), ("storm", "storm")])
def test_a_bare_disaster_word_answers_the_question(word, kind):
    f = Family().setup()
    out = f.say(MOM, word)
    assert f.store.get_case(f.case_id)["disaster"] == kind and out[0].text.startswith("I'm sorry")


def test_a_bare_disaster_word_answers_the_question_after_it_was_dropped():
    f = Family().setup()
    f.say(MOM, "list")                                          # not an answer: the question is dropped
    assert f.store.get_member(MOM)["state"] == {}
    assert texts(f.say(MOM, "flood")) == ["Got it, a flood."]
    assert f.store.get_case(f.case_id)["disaster"] == "flood"


@pytest.mark.parametrize("greeting", ["hi", "Hello", "hey!", "good morning", "hi there 👋"])
def test_a_greeting_says_where_things_stand(fam, greeting):
    fam.say(MOM, "list")                                        # past the setup question
    out = fam.say(MOM, greeting)
    assert texts(out) == ["Hi, Mom. Nothing is on the list yet. Send a photo from inside the home, from any year, "
                          "and I'll start it."]


def test_a_greeting_counts_what_is_waiting(fam):
    fam.say(MOM, "list")
    fam.say(MAYA, "", [photo(640)])
    fam.wait()
    assert texts(fam.say(MOM, "hi")) == ["Hi, Mom. 3 things are waiting for your OK. Say review to go through them, "
                                         "or send a photo any time."]
    assert texts(fam.say(MAYA, "hello")) == ["Hi, Maya. The list has 3 things on it so far. Send a photo any time "
                                             "to add more."]


@pytest.mark.parametrize("words", ["🙂", "😢😢", "", "..."])
def test_emoji_or_nothing_gets_one_kind_line(fam, words):
    fam.say(MOM, "list")
    assert texts(fam.say(MOM, words)) == ["I'm here whenever you're ready. Send a photo any time."]


@pytest.mark.parametrize("typo,starts", [("hlep", "Send me photos"), ("lsit", "Here is everything so far"),
                                         ("exprot", "Nothing is confirmed yet"), ("reveiw", "Nothing new is waiting"),
                                         ("undoo", "There's nothing to undo"),
                                         ("chek my bank", "Bank records aren't connected yet")])
def test_a_command_one_letter_off_still_runs(fam, typo, starts, monkeypatch):
    fam.say(MOM, "list")
    from ashline import bank, config

    def no_key(*a, **k):
        raise config.MissingKey("NESSIE_API_KEY")
    monkeypatch.setattr(bank, "find_matches", no_key)
    out = fam.say(MOM, typo)
    assert out[0].text.startswith(starts), out[0].text


def test_a_word_that_only_looks_like_a_command_is_left_alone(fam):
    fam.say(MOM, "list")
    assert fam.say(MOM, "lost")[0].text.startswith("I'm not sure")


@pytest.mark.parametrize("words,label,qty,price,room", [
    ("we had a piano", "piano", 1, None, None),
    ("we also had a piano worth 3000", "piano", 1, 3000, None),
    ("I had 2 bikes", "bikes", 2, None, None),
    ("there was a grandfather clock in the hall", "grandfather clock", 1, None, "Hall"),
])
def test_something_remembered_outside_a_memory_chat_is_suggested(fam, words, label, qty, price, room):
    fam.say(MOM, "list")
    out = fam.say(MOM, words)
    items = fam.store.get_items(fam.case_id, canonical_only=True)
    assert [(i["label"], i["qty"], i["status"]) for i in items] == [(label, qty, "suggested")]
    assert items[0]["source"] == "voice" and items[0]["room_label"] == room
    if price:
        assert items[0]["price_mid"] == price
    assert out[0].text.startswith("Added from memory: ") and "waits for your OK" in out[0].text
    assert "undo" in out[0].text
    assert fam.say(MOM, "undo")[0].text.startswith("Undone")
    assert fam.store.get_item(fam.case_id, items[0]["item_id"])["status"] == "rejected"


def test_a_relative_remembering_something_waits_for_the_owner(fam):
    out = fam.say(MAYA, "we had a piano")
    assert "waits for Mom's OK" in out[0].text


@pytest.mark.parametrize("words", ["did we have a piano?", "did we have a piano", "we had a fire", "list"])
def test_questions_and_commands_are_not_taken_as_things_remembered(fam, words):
    fam.say(MOM, "skip")
    fam.say(MOM, words)
    assert fam.store.get_items(fam.case_id) == []
