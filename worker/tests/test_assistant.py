"""The LangChain chat agent and the online price lookup, driven by a scripted model and a fake
searcher: no model or search call is made."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ashline import assistant
from ashline.chat import Engine, Inbound
from ashline.price_search import ClaudeSearcher, apply_price, find_price, parse_matches, price_check
from ashline.pricing import DefaultPricer
from ashline.store import Store

MOM, MAYA = "+15550001", "+15550002"
NOW = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class FakeSearcher:
    def __init__(self, matches):
        self.matches, self.asked = matches, []

    def search(self, description, brand=None):
        self.asked.append((description, brand))
        return self.matches


MATCHES = [{"title": "Acme velvet 3-seat sofa", "price": 1399.0, "retailer": "Acme Home", "url": "https://shop.example/sofa"},
           {"title": "Similar velvet sofa", "price": 1150.0, "retailer": "Other Store", "url": "https://other.example/s"},
           {"title": "Premium velvet sofa", "price": 1800.0, "retailer": "Third", "url": "https://third.example/s"}]


@pytest.fixture
def fam():
    store = Store()
    engine = Engine(store, pricer=DefaultPricer(), public_url="https://ashline.example")
    for who, name, text in ((MOM, "Mom", "hi"), (MOM, None, "Example home"), (MOM, None, "skip"),
                            (MOM, None, "living room and kitchen")):
        engine.handle(Inbound(who, who, text, sender_name=name, now=NOW))
    case_id = store.get_member(MOM)["case_id"]
    code = store.get_case(case_id)["invite_code"]
    engine.handle(Inbound(MAYA, MAYA, code, sender_name="Maya", now=NOW))
    ids = {}
    for label, category, room, mid in (("blue velvet sofa", "furniture", "Living room", 900),
                                       ("brass floor lamp", "lighting", "Living room", 80),
                                       ("red stand mixer", "kitchen", "Kitchen", 300)):
        ids[label] = store.add_item(case_id, {"label": label, "category": category, "room_label": room}, source="photo",
                                    added_by="Maya")
        store.set_cluster(case_id, ids[label], [ids[label]], 1)
        store.update_item(case_id, ids[label], price_low=mid * 0.6, price_mid=mid, price_high=mid * 1.5,
                          price_source="claude estimate")
    return NS(store=store, engine=engine, case_id=case_id, ids=ids)


def say(fam, who, text, script, group=False):
    fam.engine.assistant_model = Scripted(messages=iter(script))
    return fam.engine.handle(Inbound("family-group" if group else who, who, text, is_group=group, now=NOW))


def test_answers_a_question_from_the_inventory(fam):
    out = say(fam, MAYA, "what have we got from the living room?", [
        call("search_inventory", {"room": "living room"}),
        AIMessage(content="Two things so far: the blue velvet sofa and the brass floor lamp.")])
    assert [o.text for o in out] == ["Two things so far: the blue velvet sofa and the brass floor lamp."]
    numbered = fam.store.get_chat(MAYA)["context"]["list"]
    assert numbered == {"1": fam.ids["blue velvet sofa"], "2": fam.ids["brass floor lamp"]}      # usable in "remove 2"
    talk = fam.store.get_chat(MAYA)["context"]["talk"]
    assert [t["role"] for t in talk] == ["user", "assistant"] and "living room" in talk[0]["text"]


def test_tools_report_only_what_is_stored(fam):
    tools, _ = assistant.build_tools(fam.engine, Inbound(MAYA, MAYA, "x"), fam.store.get_member(MAYA),
                                     fam.store.get_case(fam.case_id), NOW, [])
    search, overview, progress, home, questions = tools[:5]
    found = search(query="sofa")
    assert found == [{"number": 1, "item": "blue velvet sofa", "room": "Living room", "quantity": 1,
                      "estimate": "$540–$1,350", "price_from": "claude estimate", "status": "suggested",
                      "evidence": "1 photo"}]
    assert search(query="piano") == []
    summary = overview()
    assert summary["rooms"]["Living room"] == {"items": 2, "confirmed": 0, "value": 980}
    assert summary["waiting_for_ok"] == 3 and summary["family_members"] == ["Mom", "Maya"]
    assert "0 items documented so far" in progress() and "3 more items ($1,280)" in progress()
    assert home()["rooms"] == ["Living room", "Kitchen"] and home()["listing"] is None
    assert questions(2) and all("?" in q for q in questions(2))


def test_a_remembered_item_is_only_ever_a_suggestion(fam):
    out = say(fam, MAYA, "we also had grandma's upright piano in the den", [
        call("remember_item", {"description": "upright piano", "room": "Den", "quantity": 1}),
        AIMessage(content="I've added the upright piano as a suggestion marked from memory. It waits for Mom's OK.")])
    assert "waits for Mom's OK" in out[0].text
    piano = next(i for i in fam.store.get_items(fam.case_id) if i["label"] == "upright piano")
    assert (piano["status"], piano["source"], piano["added_by"], piano["room_label"]) == ("suggested", "voice", "Maya", "Den")
    assert piano["price_mid"] and fam.store.last_event(fam.case_id)["action"] == "add"       # priced, and undoable


def test_commands_run_through_the_same_checks(fam):
    out = say(fam, MAYA, "can you send the spreadsheet to the insurance people", [
        call("run_command", {"command": "export"}), AIMessage(content="DONE")])
    assert len(out) == 1 and "Only Mom can export" in out[0].text                # a relative still cannot export
    out = say(fam, MOM, "ok get rid of everything", [
        call("run_command", {"command": "delete my case"}), AIMessage(content="DONE")])
    assert "Reply yes to delete" in out[0].text and fam.store.get_case(fam.case_id)      # asks first; nothing deleted
    fam.engine.handle(Inbound(MOM, MOM, "no", now=NOW))
    out = say(fam, MOM, "please wipe the database tables", [
        call("run_command", {"command": "drop table items"}), AIMessage(content="I can't do that.")])
    assert [o.text for o in out] == ["I can't do that."] and len(fam.store.get_items(fam.case_id)) == 3


def test_the_plan_and_bank_records_are_commands_the_agent_can_run(fam):
    out = say(fam, MAYA, "where was everything in the house?", [
        call("run_command", {"command": "map"}), AIMessage(content="DONE")])
    assert len(out) == 1 and out[0].image.endswith("plan.png") and out[0].text.endswith("/plan")
    assert "3 items where they were recorded" in out[0].text
    out = say(fam, MAYA, "can you look at mom's card statement for prices", [
        call("run_command", {"command": "check my bank"}), AIMessage(content="DONE")])
    assert [o.text for o in out] == ["Only the person who started this list can connect bank records."]


def test_price_researcher_subagent_looks_it_up_and_the_price_is_recorded(fam):
    fam.engine.price_searcher = FakeSearcher(MATCHES)
    out = say(fam, MOM, "is the sofa price right? check online", [
        call("search_inventory", {"query": "sofa"}, "a"),
        call("task", {"subagent_type": "price-researcher", "description": "blue velvet sofa, brand unknown"}, "b"),
        call("find_price_online", {"description": "blue velvet sofa"}, "c"),                      # inside the subagent
        AIMessage(content="About $1,399 at Acme Home; similar ones run $1,150 to $1,800."),       # subagent's summary
        call("apply_web_price", {"item_number": 1, "price": 1399, "low": 1150, "high": 1800,
                                 "retailer": "Acme Home", "url": "https://shop.example/sofa"}, "d"),
        AIMessage(content="A similar sofa is about $1,400 new at Acme Home, so I've updated it.")])
    assert out[-1].text.startswith("A similar sofa is about $1,400")
    assert fam.engine.price_searcher.asked == [("blue velvet sofa", None)]
    sofa = fam.store.get_item(fam.case_id, fam.ids["blue velvet sofa"])
    assert (sofa["price_low"], sofa["price_mid"], sofa["price_high"], sofa["price_source"]) == (1150, 1400, 1800, "web: Acme Home")
    assert fam.engine.handle(Inbound(MOM, MOM, "undo", now=NOW))[0].text == "Undone: that item is back to how it was."
    assert fam.store.get_item(fam.case_id, fam.ids["blue velvet sofa"])["price_source"] == "claude estimate"


def test_group_chat_only_when_addressed_and_failures_fall_back(fam):
    assert say(fam, MAYA, "is anyone else hungry", [AIMessage(content="should never be used")], group=True) == []
    out = say(fam, MAYA, "Ashline, what's the total?", [call("progress", {}), AIMessage(content="Nothing is confirmed yet.")], group=True)
    assert [o.text for o in out] == ["Nothing is confirmed yet."]
    out = say(fam, MAYA, "tell me a story", [])                                  # the model fails: the router answers
    assert out and "I'm not sure what you'd like me to do" in out[0].text
    fam.engine.assistant_model = None
    assert assistant.enabled(fam.engine) is False                               # no key, no model: off


def test_exact_commands_never_reach_the_agent(fam):
    out = say(fam, MAYA, "list", [AIMessage(content="should never be used")])
    assert out[0].text.startswith("Here is everything so far")


def test_find_and_apply_price(fam):
    found = find_price("blue velvet sofa", None, FakeSearcher(MATCHES))
    assert (found["found"], found["price"], found["low"], found["high"], found["retailer"]) == (True, 1399.0, 1150.0, 1800.0, "Acme Home")
    assert find_price("x", None, FakeSearcher([])) == {"found": False, "reason": "No suitable listing was found."}
    assert find_price("x")["found"] is False                                     # no key: says so, does not guess
    sofa = fam.ids["blue velvet sofa"]
    assert apply_price(fam.store, fam.case_id, sofa, found) == \
        "Recorded $1,400 for the blue velvet sofa from Acme Home, range $1,150 to $1,800."
    kept = json.loads(fam.store.get_item(fam.case_id, sofa)["price_ref"])
    refs = kept["online"]["matches"]
    assert refs[0]["url"] == "https://shop.example/sofa" and len(refs) == 3      # the links are the evidence
    assert kept["estimate"] == {"low": 540.0, "mid": 900, "high": 1350.0, "source": "claude estimate"}   # kept beside it
    fam.store.update_item(fam.case_id, sofa, price_source="set by family")
    assert "left as it is" in apply_price(fam.store, fam.case_id, sofa, found)   # the family's figure stands
    assert "incomplete" in apply_price(fam.store, fam.case_id, fam.ids["brass floor lamp"], {"price": 0})


def test_price_check_takes_the_most_valuable_first(fam):
    searcher, lines = FakeSearcher(MATCHES[:1]), []
    assert price_check(fam.case_id, fam.store, top=2, searcher=searcher, report=lines.append) == {"checked": 2, "found": 2}
    assert [a[0] for a in searcher.asked] == ["blue velvet sofa", "red stand mixer"]
    assert price_check(fam.case_id, fam.store, top=5, searcher=searcher)["checked"] == 1      # the rest; done ones skipped
    assert price_check(fam.case_id, fam.store)["checked"] == 0 and len(lines) == 2


def test_searcher_resumes_a_paused_turn_and_keeps_only_linked_prices():
    reply = ('Here is what I found.\n{"matches": [{"title": "Acme sofa", "price": "$1,399.00", "retailer": "Acme", '
             '"url": "https://shop.example/sofa"}, {"title": "no link", "price": 900, "retailer": "X", "url": ""}, '
             '{"title": "free", "price": 0, "retailer": "Y", "url": "https://y.example"}]}')

    class Client:
        def __init__(self):
            self.calls = []
            self.messages = self

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return NS(stop_reason="pause_turn", content=[NS(type="server_tool_use")])
            return NS(stop_reason="end_turn", content=[NS(type="text", text=reply)])

    client = Client()
    matches = ClaudeSearcher(client=client).search("blue velvet sofa", "Acme")
    assert matches == [{"title": "Acme sofa", "price": 1399.0, "retailer": "Acme", "url": "https://shop.example/sofa"}]
    assert client.calls[0]["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    assert [m["role"] for m in client.calls[1]["messages"]] == ["user", "assistant"]         # resumed, no "continue"
    assert "Brand: Acme" in client.calls[0]["messages"][0]["content"]
    assert parse_matches("nothing") == [] and parse_matches('{"matches": "x"}') == []
