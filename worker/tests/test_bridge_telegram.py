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
    assert "make a new Telegram group" in texts(api)[-1] and "@AshlineHomeBot" in texts(api)[-1]

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
