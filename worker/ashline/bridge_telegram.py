"""Messaging bridge for Telegram, so a family can use Ashline inside their own group chat.

    ashline telegram

It long-polls Telegram's Bot API for new messages, hands each to the same conversation engine the
iMessage bridge uses, and sends the replies (and the annotated photos) back. No webhook and no public
URL are needed. It can run next to `ashline photon`: each bridge only sends to its own chats.

Setup, in Telegram:
    1. Message @BotFather, send /newbot, and copy the token it gives you.
    2. Send /setprivacy to @BotFather, pick the bot, choose Disable, so it sees photos posted in groups.
    3. In .env:  TELEGRAM_BOT_TOKEN=123456:AA...

How families use it: the owner messages the bot 1:1 and answers the few setup questions. Then they
create a new Telegram group for the family with the bot in it. The bot sees who created the group,
links it to that person's list and says hello. From then on every photo anyone posts in that group
goes into the one list. (Posting the HOME-xxxx code in a group links it too, as a fallback.)

Ids are prefixed "tg:" (chats "tg:<chat id>", people "tg:<user id>") so they can never collide with
iMessage phone numbers in the shared store.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from . import config
from .chat import TELEGRAM_PREFIX, Engine, Inbound, Outbound

API = "https://api.telegram.org"
POLL_TIMEOUT = 2                 # seconds Telegram holds a getUpdates call open; short so engine timers stay on time
TEXT_LIMIT = 4096                # Telegram's maximum message length
CAPTION_LIMIT = 1024
COMMANDS = {"start": "hi", "help": "help", "list": "list", "total": "how much so far", "review": "review",
            "export": "export", "undo": "undo", "pause": "pause"}


def owns(chat_id: str) -> bool:
    return chat_id.startswith(TELEGRAM_PREFIX)


def split_text(text: str, limit: int = TEXT_LIMIT) -> list[str]:
    """Telegram rejects messages over 4096 characters: break long ones at line ends."""
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            parts.append(line[:limit]); line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            parts.append(current); current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts or [""]


class TelegramBridge:
    def __init__(self, token: str | None = None, engine: Engine | None = None, session=None):
        self.token = token or config.env("TELEGRAM_BOT_TOKEN", required=True)
        self.http = session or requests.Session()
        self.engine = engine or Engine()
        self.state_path = config.data_dir() / "telegram_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("offset", 0)
        self.bot_id: int | None = None
        self.username: str = ""

    # ------------------------------------------------------------- Bot API
    def call(self, method: str, http_timeout: float = 30, files=None, **params):
        response = self.http.post(f"{API}/bot{self.token}/{method}", data=params if files else None,
                                  json=None if files else params, files=files, timeout=http_timeout)
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"{method}: {body.get('description') or response.status_code}")
        return body["result"]

    def identify(self) -> None:
        me = self.call("getMe")
        self.bot_id, self.username = me["id"], me.get("username") or ""
        if self.username:
            self.engine.telegram_handle = f"@{self.username}"

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state))

    # ------------------------------------------------------------- receive
    def poll(self) -> int:
        updates = self.call("getUpdates", http_timeout=POLL_TIMEOUT + 10, **{
            "offset": self.state["offset"], "allowed_updates": ["message"], "timeout": POLL_TIMEOUT})
        handled = 0
        for update in updates:
            self.state["offset"] = update["update_id"] + 1
            message = update.get("message")
            if message:
                self.migrate(message)
                self.send(self.joined(message))
                inbound = self.to_inbound(message)
                if inbound is not None:
                    self.send(self.engine.handle(inbound))
                    handled += 1
            self._save()                                  # never handle the same update twice after a restart
        return handled

    def migrate(self, message: dict) -> None:
        """Making the bot an admin can turn a group into a 'supergroup' with a new id. Keep the family's list."""
        new = message.get("migrate_to_chat_id")
        if not new:
            return
        old_id, new_id = f"{TELEGRAM_PREFIX}{message['chat']['id']}", f"{TELEGRAM_PREFIX}{new}"
        chat = self.engine.store.get_chat(old_id)
        if chat:
            self.engine.store.set_chat(new_id, chat["case_id"], True)

    def joined(self, message: dict) -> list[Outbound]:
        """The bot was put in a new group. If whoever made the group already has a home set up,
        link the group to that list straight away, so nobody has to post a code."""
        added = message.get("group_chat_created") or any(
            m.get("id") == self.bot_id for m in message.get("new_chat_members") or [])
        if not added or message["chat"].get("type") not in ("group", "supergroup"):
            return []
        chat_id = f"{TELEGRAM_PREFIX}{message['chat']['id']}"
        store = self.engine.store
        member = store.get_member(f"{TELEGRAM_PREFIX}{(message.get('from') or {}).get('id')}")
        case = store.get_case(member["case_id"]) if member and member.get("case_id") else None
        if not case:
            return [Outbound(chat_id, f"Hi, I'm Ashline. To start, message me directly ({self.engine.telegram_handle}) "
                                      "and I'll set up your home's list. If you already have a code, post it here.")]
        if store.get_chat(chat_id) and store.get_chat(chat_id)["case_id"] == case["case_id"]:
            return []
        store.set_chat(chat_id, case["case_id"], True)
        name = case.get("name") or "your home"
        return [Outbound(chat_id,
                         f"Hi everyone, I'm Ashline. I'm helping rebuild the list of everything in {name}. "
                         "Drop any photos from inside the house here, from any year, and I'll add what I find. "
                         "To ask me something, mention me or reply to one of my messages.")]

    def to_inbound(self, message: dict) -> Inbound | None:
        sender = message.get("from") or {}
        if sender.get("is_bot"):
            return None                                   # other bots, and our own messages
        chat = message["chat"]
        is_group = chat.get("type") in ("group", "supergroup")
        text = (message.get("text") or message.get("caption") or "").strip()
        text = self.clean(text, message, is_group)
        files, missing = [], 0
        file_id = None
        if message.get("photo"):
            file_id = max(message["photo"], key=lambda p: p.get("file_size") or p.get("width", 0) * p.get("height", 0))["file_id"]
        elif (message.get("document") or {}).get("mime_type", "").startswith("image/"):
            file_id = message["document"]["file_id"]       # a photo sent "as a file", full quality
        if file_id:
            data = self.download(file_id)
            if data:
                files.append((f"tg-{message['message_id']}.jpg", data))
            else:
                missing += 1
        if not text and not files and not missing:
            return None                                   # joins, stickers, pins
        name = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x) or sender.get("username")
        return Inbound(chat_id=f"{TELEGRAM_PREFIX}{chat['id']}", sender=f"{TELEGRAM_PREFIX}{sender.get('id')}",
                       text=text, attachments=files, is_group=is_group, sender_name=name, missing=missing)

    def clean(self, text: str, message: dict, is_group: bool) -> str:
        """Turn Telegram's ways of talking to a bot into the words the engine understands."""
        command = re.match(r"^/(\w+)(?:@(\w+))?\s*(.*)$", text, re.S)
        if command:
            name, target, rest = command.group(1).lower(), command.group(2), command.group(3)
            if target and self.username and target.lower() != self.username.lower():
                return ""                                 # a command for a different bot in the group
            text = (COMMANDS.get(name, name) + (" " + rest if rest else "")).strip()
            return f"Ashline {text}" if is_group else text
        if self.username:
            mention = re.compile(rf"@{re.escape(self.username)}\b[,:]?\s*", re.I)
            if mention.search(text):
                return "Ashline " + mention.sub("", text).strip()
        replied = (message.get("reply_to_message") or {}).get("from") or {}
        if is_group and self.bot_id and replied.get("id") == self.bot_id and text:
            return f"Ashline {text}"                      # a reply to one of the bot's messages is addressed to it
        return text

    def download(self, file_id: str, tries: int = 3) -> bytes | None:
        for attempt in range(1, tries + 1):
            try:
                path = self.call("getFile", file_id=file_id)["file_path"]
                response = self.http.get(f"{API}/file/bot{self.token}/{path}", timeout=60)
                if response.ok and response.content:
                    return response.content
            except Exception as exc:
                problem = exc
            else:
                problem = "empty file"
            if attempt < tries:
                time.sleep(attempt)
        print(f"telegram: could not download a photo after {tries} tries: {problem}")
        return None

    # ---------------------------------------------------------------- send
    def send(self, messages: list[Outbound]) -> None:
        for message in messages:
            if not owns(message.chat_id):
                continue                                  # the iMessage bridge sends those
            chat_id = message.chat_id[len(TELEGRAM_PREFIX):]
            try:
                if message.image:
                    with open(message.image, "rb") as f:
                        self.call("sendPhoto", http_timeout=60, files={"photo": (Path(message.image).name, f)}, chat_id=chat_id)
                for path in message.files:
                    with open(path, "rb") as f:
                        self.call("sendDocument", http_timeout=60, files={"document": (Path(path).name, f)}, chat_id=chat_id)
                if message.text:
                    for part in split_text(message.text):
                        self.call("sendMessage", chat_id=chat_id, text=part, link_preview_options={"is_disabled": True})
            except Exception as exc:
                print(f"telegram: could not send to {message.chat_id}: {exc}")

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        self.identify()
        print(f"Ashline is listening on Telegram as @{self.username}. Message the bot to start. Ctrl-C to stop.")
        while True:
            try:
                self.poll()
                self.send(self.engine.tick(owns=owns))
            except KeyboardInterrupt:
                return
            except Exception as exc:                      # a bad poll must not stop the bridge
                print(f"telegram: {exc}")
                time.sleep(2)
