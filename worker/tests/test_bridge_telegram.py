"""The Telegram bridge against a stand-in Bot API. Checks the bridge's own logic; the first live run
is on a real bot token."""
import io
import json

import pytest
from PIL import Image

from ashline.bridge_telegram import TelegramBridge, split_text
from ashline.chat import Engine, Outbound
from ashline.pricing import DefaultPricer
from ashline.store import Store

BOT_ID = 999


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


class Response:
    def __init__(self, body=None, content=b""):
        self.body, self.content, self.ok, self.status_code = body, content, True, 200

    def json(self):
        return self.body


class FakeTelegram:
    """Just enough of api.telegram.org: getMe, getUpdates, getFile, file download, and the send methods."""
    def __init__(self):
        self.updates, self.sent, self.next_id = [], [], 1

    def post(self, url, data=None, json=None, files=None, timeout=None):
        method, params = url.rsplit("/", 1)[1], (json or data or {})
        if method == "getMe":
            return Response({"ok": True, "result": {"id": BOT_ID, "is_bot": True, "username": "AshlineHomeBot"}})
        if method == "getUpdates":
            ready = [u for u in self.updates if u["update_id"] >= params.get("offset", 0)]
            return Response({"ok": True, "result": ready})
        if method == "getFile":
            return Response({"ok": True, "result": {"file_path": f"photos/{params['file_id']}.jpg"}})
        self.sent.append((method, params, sorted(files) if files else []))
        return Response({"ok": True, "result": {"message_id": len(self.sent)}})

    def get(self, url, timeout=None):
        return Response(content=jpeg())

    def message(self, text=None, chat=1, kind="private", user=1, name="Mom", photo=False, reply_to_bot=False, **extra):
        msg = {"message_id": self.next_id, "chat": {"id": chat, "type": kind},
               "from": {"id": user, "is_bot": False, "first_name": name}, **extra}
        if text is not None:
            msg["caption" if photo else "text"] = text
        if photo:
            msg["photo"] = [{"file_id": f"small{self.next_id}", "width": 90, "height": 60},
                            {"file_id": f"big{self.next_id}", "width": 1280, "height": 960}]
        if reply_to_bot:
            msg["reply_to_message"] = {"message_id": 1, "from": {"id": BOT_ID, "is_bot": True}}
        self.updates.append({"update_id": 100 + self.next_id, "message": msg})
        self.next_id += 1


def case_named(name):
    return next(c for c in Store().list_cases() if c.get("name") == name)


class Sees:
    def detect(self, path, room_hint=None, rooms=None):
        return [{"label": "blue velvet sofa", "category": "furniture", "confidence": 0.8, "bbox": [50, 200, 400, 420]}]


@pytest.fixture
def tg():
    api = FakeTelegram()
    engine = Engine(Store(), detector=Sees(), pricer=DefaultPricer(), batch_seconds=0)
    bridge = TelegramBridge(token="123:test", engine=engine, session=api)
    bridge.identify()
    return bridge, api


def texts(api, chat=None):
    return [p["text"] for m, p, _ in api.sent if m == "sendMessage" and (chat is None or str(p["chat_id"]) == str(chat))]


def test_onboarding_then_family_group_shares_one_list(tg):
    bridge, api = tg
    for text in ("/start", "The Rivera home", "skip", "living room"):
        api.message(text)
        bridge.poll()
    assert texts(api)[0].startswith("I'm Ashline.")
    invite = next(t for t in texts(api) if "make a new Telegram group" in t)
    assert "@AshlineHomeBot" in invite

    api.sent.clear()
    api.message(None, chat=-500, kind="group", user=1, group_chat_created=True)  # owner creates the family group
    bridge.poll()
    assert texts(api, -500) and texts(api, -500)[0].startswith("Hi everyone, I'm Ashline")
    assert Store().get_chat("tg:-500")["case_id"] == case_named("The Rivera home")["case_id"]

    api.message("from grandma's birthday", chat=-500, kind="group", user=2, name="Aunt Lena", photo=True)
    bridge.poll()
    bridge.send(bridge.engine.tick(owns=lambda c: c.startswith("tg:")))
    kinds = [m for m, _, _ in api.sent]
    assert "sendPhoto" in kinds                                                  # the annotated photo, in the group
    assert any("Blue velvet sofa" in t for t in texts(api, -500))
    lena = Store().get_member("tg:2")
    assert lena and lena["case_id"] == case_named("The Rivera home")["case_id"]


def test_group_chatter_is_ignored_but_mentions_and_replies_are_answered(tg):
    bridge, api = tg
    for text in ("hi", "The Rivera home", "skip", "living room"):
        api.message(text)
        bridge.poll()
    code = case_named("The Rivera home")["invite_code"]
    api.message(code, chat=-7, kind="supergroup")
    bridge.poll()
    api.sent.clear()
    api.message("dinner at 7?", chat=-7, kind="supergroup", user=3, name="Dad")
    bridge.poll()
    assert texts(api) == []                                                      # family chatter: silence
    api.message("@AshlineHomeBot how much so far", chat=-7, kind="supergroup", user=3, name="Dad")
    bridge.poll()
    assert texts(api) and "$" in texts(api)[-1]
    api.sent.clear()
    api.message("what's on the list", chat=-7, kind="supergroup", user=3, name="Dad", reply_to_bot=True)
    api.message("/list@SomeOtherBot", chat=-7, kind="supergroup", user=3)
    bridge.poll()
    assert len(texts(api)) >= 1                                                  # the reply got an answer...
    assert bridge.clean("/list@SomeOtherBot", {}, True) == ""                    # ...the other bot's command did not


def test_group_made_by_a_stranger_gets_pointed_to_setup(tg):
    bridge, api = tg
    api.message(None, chat=-9, kind="group", user=42, name="Stranger",
                new_chat_members=[{"id": BOT_ID, "is_bot": True, "first_name": "Ashline"}])
    bridge.poll()
    assert texts(api, -9)[0].startswith("Hi, I'm Ashline. To start, message me directly")
    assert Store().get_chat("tg:-9") is None


def test_code_in_a_group_still_links_it(tg):
    bridge, api = tg
    for text in ("hi", "The Rivera home", "skip", "living room"):
        api.message(text)
        bridge.poll()
    code = case_named("The Rivera home")["invite_code"]
    api.sent.clear()
    api.message(code, chat=-33, kind="group", user=7, name="Cousin")
    bridge.poll()
    assert any("joined" in t for t in texts(api, -33))


def test_offset_saved_so_restarts_do_not_repeat(tg):
    bridge, api = tg
    api.message("hi")
    bridge.poll()
    first = len(api.sent)
    again = TelegramBridge(token="123:test", engine=bridge.engine, session=api)
    again.identify()
    assert again.poll() == 0 and len(api.sent) == first


def test_supergroup_migration_keeps_the_list(tg):
    bridge, api = tg
    Store().set_case("c1", name="X")
    Store().set_chat("tg:-1", "c1", True)
    bridge.migrate({"chat": {"id": -1}, "migrate_to_chat_id": -100123})
    assert Store().get_chat("tg:-100123")["case_id"] == "c1"


def test_bridges_only_send_their_own_chats(tg):
    bridge, api = tg
    bridge.send([Outbound("space-imessage", "not mine"), Outbound("tg:5", "mine")])
    assert texts(api) == ["mine"]


def test_split_text_respects_the_limit():
    parts = split_text("line\n" * 2000, limit=100)
    assert all(len(p) <= 100 for p in parts) and "".join(parts).count("line") == 2000


# ------------------------------------------------------------ tappable answers
SAME = [("Same", "1"), ("Different", "2")]
ADD = [("Add it", "yes"), ("Leave it off", "no")]
MOVED_ON = "That question has moved on."


def press(api, data, message_id, chat=1, kind="private", user=1, name="Mom"):
    api.updates.append({"update_id": 100 + api.next_id, "callback_query": {
        "id": f"q{api.next_id}", "from": {"id": user, "is_bot": False, "first_name": name}, "data": data,
        "message": {"message_id": message_id, "chat": {"id": chat, "type": kind}}}})
    api.next_id += 1


def calls(api, method):
    return [p for m, p, _ in api.sent if m == method]


def keyboard(api):
    """The newest question sent: its message id and its buttons."""
    n, params = next((n, p) for n, (m, p, _) in reversed(list(enumerate(api.sent, 1)))
                     if m == "sendMessage" and "reply_markup" in p)
    return n, params["reply_markup"]["inline_keyboard"][0]


@pytest.fixture
def heard(tg, monkeypatch):
    """Every inbound the engine is handed."""
    bridge, api = tg
    seen, handle = [], bridge.engine.handle
    monkeypatch.setattr(bridge.engine, "handle", lambda inbound: seen.append(inbound) or handle(inbound))
    return seen


def test_a_question_gets_buttons_that_fit_telegrams_limit(tg):
    bridge, api = tg
    bridge.send([Outbound("tg:1", "Same sofa? 1 = same, 2 = different.", choices=SAME)])
    _, buttons = keyboard(api)
    assert [b["text"] for b in buttons] == ["Same", "Different"]
    assert all(0 < len(b["callback_data"].encode()) <= 64 for b in buttons)
    assert buttons[0]["callback_data"] != buttons[1]["callback_data"]
    assert texts(api) == ["Same sofa? 1 = same, 2 = different."]                  # the words stay for every channel


def test_buttons_go_under_the_text_after_a_photo(tg, tmp_path):
    bridge, api = tg
    image = tmp_path / "q.jpg"
    image.write_bytes(jpeg())
    bridge.send([Outbound("tg:1", "Same sofa? 1 = same, 2 = different.", image=str(image), choices=SAME)])
    assert [m for m, _, _ in api.sent] == ["sendPhoto", "sendMessage"]
    assert "reply_markup" not in api.sent[0][1] and "reply_markup" in api.sent[1][1]


def test_a_photo_without_text_carries_the_buttons_as_json(tg, tmp_path):
    bridge, api = tg
    image = tmp_path / "q.jpg"
    image.write_bytes(jpeg())
    bridge.send([Outbound("tg:1", "", image=str(image), choices=SAME)])
    method, params, _ = api.sent[0]
    assert method == "sendPhoto" and isinstance(params["reply_markup"], str)     # multipart: JSON text
    assert [b["text"] for b in json.loads(params["reply_markup"])["inline_keyboard"][0]] == ["Same", "Different"]


def test_a_message_without_choices_is_sent_as_before(tg):
    bridge, api = tg
    bridge.send([Outbound("tg:1", "Thanks, counted once.")])
    assert api.sent == [("sendMessage", {"chat_id": "1", "text": "Thanks, counted once.",
                                         "link_preview_options": {"is_disabled": True}}, [])]


def test_a_press_is_answered_clears_the_buttons_and_reaches_the_engine_as_the_reply(tg, heard):
    bridge, api = tg
    bridge.send([Outbound("tg:-40", "Same sofa? 1 = same, 2 = different.", choices=SAME)])
    message_id, buttons = keyboard(api)
    api.sent.clear()
    press(api, buttons[1]["callback_data"], message_id, chat=-40, kind="supergroup", user=2, name="Aunt Lena")
    bridge.poll()
    assert [m for m, _, _ in api.sent][:2] == ["answerCallbackQuery", "editMessageReplyMarkup"]
    assert calls(api, "answerCallbackQuery") == [{"callback_query_id": "q1"}]
    assert calls(api, "editMessageReplyMarkup") == [
        {"chat_id": "-40", "message_id": message_id, "reply_markup": {"inline_keyboard": []}}]
    assert len(heard) == 1
    inbound = heard[0]
    assert (inbound.chat_id, inbound.sender, inbound.sender_name, inbound.is_group) == ("tg:-40", "tg:2", "Aunt Lena", True)
    assert inbound.text == "Ashline 2"                                           # addressed to Ashline, as a mention is
    api.sent.clear()
    press(api, buttons[0]["callback_data"], message_id, chat=-40, kind="supergroup", user=2)
    bridge.poll()                                                                # already answered: moved on
    assert calls(api, "answerCallbackQuery") == [{"callback_query_id": "q2", "text": MOVED_ON}] and len(heard) == 1


def test_a_press_in_a_private_chat_is_the_bare_reply(tg, heard):
    bridge, api = tg
    bridge.send([Outbound("tg:1", "I heard a piano mentioned. Add it to the list? Reply yes or no.", choices=ADD)])
    message_id, buttons = keyboard(api)
    press(api, buttons[0]["callback_data"], message_id)
    bridge.poll()
    assert [(i.chat_id, i.sender, i.text, i.is_group) for i in heard] == [("tg:1", "tg:1", "yes", False)]


def test_a_new_question_takes_the_buttons_off_the_old_one(tg, heard):
    bridge, api = tg
    bridge.send([Outbound("tg:1", "Same sofa? 1 = same, 2 = different.", choices=SAME)])
    old_id, old = keyboard(api)
    bridge.send([Outbound("tg:5", "Add it? Reply yes or no.", choices=ADD)])
    assert calls(api, "editMessageReplyMarkup") == []                            # another chat's question stays live
    bridge.send([Outbound("tg:1", "Same lamp? 1 = same, 2 = different.", choices=SAME)])
    assert calls(api, "editMessageReplyMarkup") == [
        {"chat_id": "1", "message_id": old_id, "reply_markup": {"inline_keyboard": []}}]
    api.sent.clear()
    press(api, old[0]["callback_data"], old_id)
    bridge.poll()
    assert calls(api, "answerCallbackQuery") == [{"callback_query_id": "q1", "text": MOVED_ON}]
    assert heard == [] and calls(api, "editMessageReplyMarkup") == []


def test_an_unknown_button_is_answered_and_ignored(tg, heard):
    bridge, api = tg
    press(api, "nonsense", 77)
    bridge.poll()
    assert calls(api, "answerCallbackQuery") == [{"callback_query_id": "q1", "text": MOVED_ON}]
    assert heard == []


def test_live_questions_survive_a_restart(tg, heard):
    bridge, api = tg
    bridge.send([Outbound("tg:1", "Same sofa? 1 = same, 2 = different.", choices=SAME)])
    message_id, buttons = keyboard(api)
    again = TelegramBridge(token="123:test", engine=bridge.engine, session=api)
    press(api, buttons[0]["callback_data"], message_id)
    again.poll()
    assert [i.text for i in heard] == ["1"]


def test_a_failed_edit_or_answer_does_not_stop_the_bridge(tg, heard, monkeypatch):
    bridge, api = tg
    bridge.send([Outbound("tg:1", "Same sofa? 1 = same, 2 = different.", choices=SAME)])
    first_id, _ = keyboard(api)
    post = api.post

    def flaky(url, **kw):
        if url.endswith(("/editMessageReplyMarkup", "/answerCallbackQuery")):
            return Response({"ok": False, "description": "Bad Request: message is too old"})
        return post(url, **kw)

    monkeypatch.setattr(api, "post", flaky)
    bridge.send([Outbound("tg:1", "Same lamp? 1 = same, 2 = different.", choices=SAME)])   # the edit fails...
    second_id, second = keyboard(api)
    assert second_id != first_id                                                 # ...and the new question is live
    press(api, second[1]["callback_data"], second_id)
    api.message("how much so far")
    bridge.poll()                                                                # answering and editing fail too
    assert [i.text for i in heard] == ["2", "how much so far"]
