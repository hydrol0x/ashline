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
