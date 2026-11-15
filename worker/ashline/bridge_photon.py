"""Messaging bridge for Photon 2.0 (iMessage), in the worker itself.

    ashline photon

It polls the project for received messages, hands each to the conversation engine, and sends the
replies back into the same conversation. Polling needs no public webhook, so only the inventory
page has to be reachable from outside. Engine timers (photo digests, the daily review) run in the
same loop.

Needs `pip install photonhq-api` (the `photon` extra) and, in .env:
    PHOTON_API_KEY       a project API key (pho_sk_...)
    PHOTON_PROJECT_ID    pho_prj_...

Written from Photon's 2.0 documentation and the client's typed models. It has NOT been run
against a live Photon project: check a text round trip first, then a photo, then a group chat.
A shared line can only message people who texted it first, so everyone starts by texting the number.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone

from . import config
from .chat import TELEGRAM_PREFIX, Engine, Inbound, Outbound

POLL_SECONDS = 2.0
DOWNLOAD_TRIES = 5              # a photo's file can lag its message by a few seconds
DOWNLOAD_WAIT = 1.0             # seconds; grows with each try (1, 2, 3, 4)
IMAGE_TYPES = ("image/",)


def _field(obj, name, default=None):
    """Read a field from a client model, looking through its `.root` wrappers."""
    for _ in range(4):
        if obj is None:
            return default
        if hasattr(obj, name):
            value = getattr(obj, name)
            return default if type(value).__name__ == "MISSING" or value is None else value
        obj = getattr(obj, "root", None)
    return default


def unwrap(obj):
    while hasattr(obj, "root"):
        obj = obj.root
    return obj


def parts_of(content) -> list:
    """A message's content as a flat list of parts (a group message carries several)."""
    content = unwrap(content)
    if getattr(content, "type", None) == "group":
        return [unwrap(p) for p in content.parts]
    return [content]


def multipart_related(data: bytes, content_type: str, filename: str) -> tuple[bytes, str]:
    """Body for Photon's uploadAttachment: exactly two parts, JSON metadata then the file bytes.

    Per the OpenAPI spec, the JSON holds contentType, sizeBytes (must equal the file length) and an
    optional filename. Returns (body, the Content-Type header value with its boundary).
    """
    boundary = f"ashline-{uuid.uuid4().hex}"
    meta = json.dumps({"filename": filename, "contentType": content_type, "sizeBytes": len(data)})
    body = (f"--{boundary}\r\nContent-Type: application/json\r\n\r\n{meta}\r\n"
            f"--{boundary}\r\nContent-Type: {content_type}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/related; boundary={boundary}"


def _when(stamp) -> datetime:
    """Photon's createdAt (ISO text or a datetime) as an aware UTC datetime."""
    if isinstance(stamp, datetime):
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)                 # unreadable: treat as new rather than drop it
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


class PhotonBridge:
    def __init__(self, photon=None, engine: Engine | None = None, project: str | None = None):
        self.project = project or config.env("PHOTON_PROJECT_ID", required=True)
        if photon is None:
            from photon_api import Photon
            photon = Photon(headers={"Authorization": f"Bearer {config.env('PHOTON_API_KEY', required=True)}"})
        self.photon = photon
        self.engine = engine or Engine()
        self.state_path = config.data_dir() / "photon_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("since", (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(
            timespec="milliseconds").replace("+00:00", "Z"))
        self.state.setdefault("seen", [])
        self.state.setdefault("senders", {})          # conversation id -> the agent's id in it
        # Texts that arrived while this machine was off (or while another machine ran the bot) are not
        # answered on start-up: replaying hours of them floods every chat with stale replies.
        self.cutoff = datetime.now(timezone.utc) - timedelta(minutes=float(config.env("ASHLINE_BACKLOG_MINUTES", "5")))
        self.skipped = 0

    def _save(self) -> None:
        self.state["seen"] = self.state["seen"][-500:]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state))

    # ------------------------------------------------------------- receive
    def poll(self) -> int:
        """Handle every message received since the last poll. Returns how many."""
        from photon_api.rpc_generated import ListMessagesInput
        query = {"createdSince": self.state["since"], "direction": "received", "order": "asc", "pageSize": 100,
                 "platform": "imessage"}
        handled = 0
        while True:
            page = self.photon.projects.list_messages(
                ListMessagesInput.model_validate({"path": {"id": self.project}, "query": query}))
            for item in page.items:
                message = unwrap(item)
                if message.id in self.state["seen"] or getattr(message, "status", None) != "received":
                    continue
                self.state["seen"].append(message.id)
                self.state["since"] = max(self.state["since"], message.createdAt)
                if _when(message.createdAt) < self.cutoff:
                    self.skipped += 1
                    self._save()
                    continue
                inbound = self.to_inbound(message)
                if inbound is not None:
                    self.send(self.engine.handle(inbound))
                    handled += 1
                self._save()
            token = getattr(page, "nextPageToken", None)
            if not isinstance(token, str):
                break
            query["pageToken"] = token
        if self.skipped:
            print(f"ashline: skipped {self.skipped} text(s) that arrived while the bot was off here")
            self.skipped = 0
        return handled

    def to_inbound(self, message) -> Inbound | None:
        sender, space, agent = unwrap(message.sentBy), unwrap(message.space), unwrap(_field(message, "receivedBy"))
        if getattr(sender, "kind", None) != "human":
            return None                                  # the agent's own messages and system events
        if agent is not None and getattr(agent, "id", None):
            self.state["senders"][space.id] = agent.id
        text, files, missing = [], [], 0
        for part in parts_of(message.content):
            kind = getattr(part, "type", None)
            if kind == "text":
                text.append(part.text)
            elif kind == "attachment":
                data = self.download(part.attachmentId)
                if data:
                    files.append((part.attachmentId, data))
                else:
                    missing += 1
        if not text and not files and not missing:
            return None                                  # read receipts, typing updates
        return Inbound(chat_id=space.id, sender=sender.handle, text="\n".join(text).strip(), attachments=files,
                       is_group=getattr(space, "kind", "direct") != "direct", missing=missing)

    def download(self, attachment_id: str, tries: int = DOWNLOAD_TRIES) -> bytes | None:
        """An attachment's bytes. A photo can arrive a moment before its file is ready, so retry."""
        from photon_api.rpc_generated import DownloadAttachmentInput
        for attempt in range(1, tries + 1):
            try:
                data = self.photon.projects.download_attachment(DownloadAttachmentInput.model_validate(
                    {"path": {"id": self.project, "attachment_id": attachment_id}}))
                if data:
                    return data
                problem = "empty file"
            except Exception as exc:                     # not ready yet, a network blip, or not a file
                problem = exc
            if attempt < tries:
                time.sleep(DOWNLOAD_WAIT * attempt)
        print(f"photon: could not download attachment {attachment_id} after {tries} tries: {problem}")
        return None

    # ---------------------------------------------------------------- send
    def send(self, messages: list[Outbound]) -> None:
        for message in messages:
            if message.chat_id.startswith(TELEGRAM_PREFIX):
                continue                                 # the Telegram bridge sends those
            sender_id = self.state["senders"].get(message.chat_id)
            if not sender_id:
                print(f"photon: no known conversation for {message.chat_id}; message not sent")
                continue
            try:
                for path in ([message.image] if message.image else []) + list(message.files):
                    self._send(message.chat_id, sender_id, {"type": "attachment", "attachmentId": self.upload(path)})
                if message.text:
                    self._send(message.chat_id, sender_id, {"type": "text", "format": "plain", "text": message.text})
            except Exception as exc:
                print(f"photon: could not send to {message.chat_id}: {exc}")

    def _send(self, space_id: str, sender_id: str, content: dict) -> None:
        from photon_api.rpc_generated import SendMessageInput
        self.photon.projects.send_message(SendMessageInput.model_validate({
            "path": {"id": self.project},
            "headers": {"idempotency-key": f"ashline:{space_id}:{time.time_ns()}"},
            "body": {"spaceId": space_id, "senderId": sender_id, "content": content}}))

    def upload(self, path: str) -> str:
        from pathlib import Path

        from photon_api.rpc_generated import UploadAttachmentInput
        data = Path(path).read_bytes()
        kind = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "pdf": "application/pdf",
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "csv": "text/csv"}.get(Path(path).suffix.lstrip(".").lower(), "application/octet-stream")
        body, content_type = multipart_related(data, kind, Path(path).name)
        attachment = self.photon.projects.upload_attachment(UploadAttachmentInput.model_validate({
            "path": {"id": self.project}, "body": body,
            "headers": {"content-type": content_type, "content-length": str(len(body)),
                        "idempotency-key": f"ashline-upload:{Path(path).name}:{time.time_ns()}"}}))
        return unwrap(attachment).id

    # ---------------------------------------------------------------- loop
    def run(self, poll_seconds: float = POLL_SECONDS) -> None:
        print("Ashline is listening on Photon. Text the number to start. Ctrl-C to stop.")
        while True:
            try:
                self.poll()
                self.send(self.engine.tick(owns=lambda chat_id: not chat_id.startswith(TELEGRAM_PREFIX)))
            except KeyboardInterrupt:
                return
            except Exception as exc:                     # a bad poll must not stop the bridge
                print(f"photon: {exc}")
            time.sleep(poll_seconds)
