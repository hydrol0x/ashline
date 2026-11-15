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
