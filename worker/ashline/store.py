"""Case store (build spec 6.9) in SQLite.

This file creates the tables the home twin seed needs: cases, home, rooms,
photos, items and findings. Signed image URLs are never stored, only local
paths. The database lives under the git-ignored data directory.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SOURCES = ("photo", "voice", "scan", "aerial", "receipt")
ROLES = ("owner", "contributor")
ITEM_STATUS = ("suggested", "confirmed", "rejected")
FINDING_STATUS = ("open", "confirmed", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  case_id TEXT PRIMARY KEY, address TEXT, loss_date TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS home (
  case_id TEXT PRIMARY KEY, sid TEXT, lat REAL, lon REAL, parcel_wkt TEXT, homes_listing_id TEXT,
  scan_captured TEXT, capture_notes TEXT, aerial_collection TEXT, aerial_captured TEXT, listing TEXT);
CREATE TABLE IF NOT EXISTS rooms (
  case_id TEXT, room_id TEXT, label TEXT, name TEXT, area_m2 REAL, width REAL, depth REAL, height REAL,
  furnished INTEGER, source TEXT, PRIMARY KEY (case_id, room_id));
CREATE TABLE IF NOT EXISTS photos (
  case_id TEXT, photo_id TEXT, path TEXT, source TEXT, room_id TEXT, room_label TEXT, pano_id TEXT,
  face INTEGER, x REAL, y REAL, z REAL, captured TEXT, added_by TEXT, PRIMARY KEY (case_id, photo_id));
CREATE TABLE IF NOT EXISTS items (
  item_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, photo_id TEXT, label TEXT, category TEXT,
  brand_guess TEXT, room_id TEXT, room_label TEXT, bbox TEXT, confidence REAL, source TEXT,
  status TEXT DEFAULT 'suggested', cluster_id INTEGER, qty INTEGER DEFAULT 1, evidence_date TEXT,
  added_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS findings (
  finding_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, kind TEXT, label TEXT, detail TEXT,
  confidence REAL, source TEXT, evidence_date TEXT, status TEXT DEFAULT 'open');
CREATE INDEX IF NOT EXISTS items_case ON items (case_id, status);
CREATE TABLE IF NOT EXISTS members (
  member_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, phone TEXT, name TEXT, role TEXT,
  joined_at TEXT, paused INTEGER DEFAULT 0, state TEXT, last_seen TEXT, last_ask TEXT);
CREATE TABLE IF NOT EXISTS chats (
  chat_id TEXT PRIMARY KEY, case_id TEXT, is_group INTEGER DEFAULT 0, context TEXT);
CREATE TABLE IF NOT EXISTS questions (
  question_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, kind TEXT, payload TEXT, chat_id TEXT,
  asked_at TEXT, expires_at TEXT, status TEXT DEFAULT 'open', answer TEXT);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, member_id INTEGER, action TEXT, payload TEXT,
  at TEXT, undone INTEGER DEFAULT 0);
"""

# Columns added after the first version: {table: {column: declaration}}. Applied on open.
MIGRATIONS = {
    "home": {"listing": "TEXT"},
    "cases": {"name": "TEXT", "invite_code": "TEXT", "owner_token": "TEXT", "member_token": "TEXT",
              "coverage_limit": "REAL", "coverage_skipped": "INTEGER DEFAULT 0", "review_hour": "INTEGER DEFAULT 18",
              "last_review_at": "TEXT", "rooms_note": "TEXT", "reminders": "TEXT",
              "disaster": "TEXT", "insured": "TEXT", "room_causes": "TEXT"},
    "items": {"price_low": "REAL", "price_mid": "REAL", "price_high": "REAL", "price_source": "TEXT",
              "detail": "TEXT", "reviewed": "INTEGER DEFAULT 0", "price_ref": "TEXT", "cause": "TEXT"},
    "members": {"dm_chat": "TEXT"},
    "photos": {"number": "INTEGER", "batch_id": "TEXT", "taken": "TEXT", "status": "TEXT DEFAULT 'done'",
               "chat_id": "TEXT"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | None = None):
        if path is None:
            root = config.data_dir()
            root.mkdir(parents=True, exist_ok=True)
            path = str(root / "ashline.sqlite")
        # The LangChain agent runs its tools on worker threads; callers serialize their use of a Store.
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that databases created by an earlier version lack."""
        for table, columns in MIGRATIONS.items():
            have = {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}
            for column, declaration in columns.items():
                if column not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        self.db.commit()

    @contextmanager
    def tx(self):
        try:
            yield self.db
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _rows(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ---------------------------------------------------------------- case
    def upsert_case(self, case_id: str, address: str | None = None, loss_date: str | None = None) -> None:
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO cases (case_id, created_at) VALUES (?, ?)", (case_id, _now()))
            if address:
                db.execute("UPDATE cases SET address = ? WHERE case_id = ?", (address, case_id))
            if loss_date:
                db.execute("UPDATE cases SET loss_date = ? WHERE case_id = ?", (loss_date, case_id))

    def get_case(self, case_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM cases WHERE case_id = ?", case_id)
        return rows[0] if rows else None

    def delete_case(self, case_id: str) -> None:
        """The family's delete-case command: remove every row for the case."""
        with self.tx() as db:
            for table in ("cases", "home", "rooms", "photos", "items", "findings", "members", "chats",
                          "questions", "events"):
                db.execute(f"DELETE FROM {table} WHERE case_id = ?", (case_id,))

    def set_case(self, case_id: str, **fields) -> None:
        allowed = ("name", "invite_code", "owner_token", "member_token", "coverage_limit", "coverage_skipped",
                   "review_hour", "last_review_at", "loss_date", "address", "rooms_note", "reminders",
                   "disaster", "insured", "room_causes")
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO cases (case_id, created_at) VALUES (?, ?)", (case_id, _now()))
            for key, value in fields.items():
                if key in allowed:
                    db.execute(f"UPDATE cases SET {key} = ? WHERE case_id = ?", (value, case_id))

    def find_case(self, **where) -> dict | None:
        """Look a case up by invite_code, owner_token or member_token."""
        (key, value), = where.items()
        if key not in ("invite_code", "owner_token", "member_token"):
            raise ValueError(key)
        rows = self._rows(f"SELECT * FROM cases WHERE {key} = ?", value)
        return rows[0] if rows else None

    def list_cases(self) -> list[dict]:
        return self._rows("SELECT * FROM cases ORDER BY created_at")

    # ------------------------------------------------------------- members
    def add_member(self, case_id: str, phone: str, name: str | None, role: str = "contributor") -> dict:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        existing = self.get_member(phone, case_id)
        if existing:
            return existing
        with self.tx() as db:
            db.execute("INSERT INTO members (case_id, phone, name, role, joined_at, last_seen) VALUES (?,?,?,?,?,?)",
                       (case_id, phone, name, role, _now(), _now()))
        return self.get_member(phone, case_id)

    def get_member(self, phone: str, case_id: str | None = None) -> dict | None:
        """A person by phone number. Without a case, their most recent membership."""
        if case_id:
            rows = self._rows("SELECT * FROM members WHERE phone = ? AND case_id = ?", phone, case_id)
        else:
            rows = self._rows("SELECT * FROM members WHERE phone = ? ORDER BY member_id DESC", phone)
        if not rows:
            return None
        rows[0]["state"] = json.loads(rows[0]["state"]) if rows[0].get("state") else {}
        return rows[0]

    def get_members(self, case_id: str) -> list[dict]:
        rows = self._rows("SELECT * FROM members WHERE case_id = ? ORDER BY member_id", case_id)
        for row in rows:
            row["state"] = json.loads(row["state"]) if row.get("state") else {}
        return rows

    def set_member(self, member_id: int, **fields) -> None:
        allowed = ("name", "role", "paused", "state", "last_seen", "last_ask", "case_id", "dm_chat")
        with self.tx() as db:
            for key, value in fields.items():
                if key in allowed:
                    if key == "state":
                        value = json.dumps(value or {})
                    db.execute(f"UPDATE members SET {key} = ? WHERE member_id = ?", (value, member_id))

    # --------------------------------------------------------------- chats
    def set_chat(self, chat_id: str, case_id: str, is_group: bool = False) -> None:
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO chats (chat_id, case_id, is_group) VALUES (?,?,?)",
                       (chat_id, case_id, int(is_group)))
            db.execute("UPDATE chats SET case_id = ?, is_group = ? WHERE chat_id = ?", (case_id, int(is_group), chat_id))

    def get_chat(self, chat_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM chats WHERE chat_id = ?", chat_id)
        if not rows:
            return None
        rows[0]["context"] = json.loads(rows[0]["context"]) if rows[0].get("context") else {}
        return rows[0]

    def get_chats(self, case_id: str) -> list[dict]:
        return self._rows("SELECT * FROM chats WHERE case_id = ?", case_id)

    def set_chat_context(self, chat_id: str, context: dict) -> None:
        """Small per-chat memory: the numbered list last shown, the open burst of photos."""
        with self.tx() as db:
            db.execute("UPDATE chats SET context = ? WHERE chat_id = ?", (json.dumps(context), chat_id))

    # ----------------------------------------------------------- questions
    def add_question(self, case_id: str, kind: str, payload: dict, chat_id: str | None = None,
                     expires_at: str | None = None, asked_at: str | None = None) -> int:
        with self.tx() as db:
            cur = db.execute("INSERT INTO questions (case_id, kind, payload, chat_id, asked_at, expires_at)"
                             " VALUES (?,?,?,?,?,?)",
                             (case_id, kind, json.dumps(payload), chat_id, asked_at or _now(), expires_at))
            return cur.lastrowid

    def get_questions(self, case_id: str, status: str | None = "open", chat_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM questions WHERE case_id = ?", [case_id]
        if status:
            sql, args = sql + " AND status = ?", args + [status]
        if chat_id:
            sql, args = sql + " AND chat_id = ?", args + [chat_id]
        rows = self._rows(sql + " ORDER BY question_id", *args)
        for row in rows:
            row["payload"] = json.loads(row["payload"] or "{}")
        return rows

    def close_question(self, question_id: int, status: str, answer: str | None = None) -> None:
        with self.tx() as db:
            db.execute("UPDATE questions SET status = ?, answer = ? WHERE question_id = ?",
                       (status, answer, question_id))

    # -------------------------------------------------------------- events
    def log_event(self, case_id: str, action: str, payload: dict, member_id: int | None = None) -> int:
        """Record a change with enough detail to undo it."""
        with self.tx() as db:
            cur = db.execute("INSERT INTO events (case_id, member_id, action, payload, at) VALUES (?,?,?,?,?)",
                             (case_id, member_id, action, json.dumps(payload), _now()))
            return cur.lastrowid

    def last_event(self, case_id: str, member_id: int | None = None) -> dict | None:
        """The newest change not yet undone; with member_id, the newest one that person made."""
        mine = " AND member_id = ?" if member_id is not None else ""
        rows = self._rows(f"SELECT * FROM events WHERE case_id = ? AND undone = 0{mine} ORDER BY event_id DESC LIMIT 1",
                          case_id, *([member_id] if member_id is not None else []))
        if not rows:
            return None
        rows[0]["payload"] = json.loads(rows[0]["payload"] or "{}")
        return rows[0]

    def mark_undone(self, event_id: int) -> None:
        with self.tx() as db:
            db.execute("UPDATE events SET undone = 1 WHERE event_id = ?", (event_id,))

    # ---------------------------------------------------------------- home
    def set_home(self, case_id: str, **fields) -> None:
        allowed = ("sid", "lat", "lon", "parcel_wkt", "homes_listing_id", "scan_captured",
                   "capture_notes", "aerial_collection", "aerial_captured", "listing")
        fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if isinstance(fields.get("listing"), dict):
            fields["listing"] = json.dumps(fields["listing"])
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO home (case_id) VALUES (?)", (case_id,))
            for key, value in fields.items():
                db.execute(f"UPDATE home SET {key} = ? WHERE case_id = ?", (value, case_id))

    def get_home(self, case_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM home WHERE case_id = ?", case_id)
        if not rows:
            return None
        rows[0]["listing"] = json.loads(rows[0]["listing"]) if rows[0].get("listing") else None
        return rows[0]

    # --------------------------------------------------------------- rooms
    def set_rooms(self, case_id: str, rooms: list[dict], source: str = "scan") -> None:
        with self.tx() as db:
            db.execute("DELETE FROM rooms WHERE case_id = ? AND source = ?", (case_id, source))
            for r in rooms:
                furnished = None if r.get("furnished") is None else int(bool(r["furnished"]))
                db.execute("INSERT OR REPLACE INTO rooms VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (case_id, r["room_id"], r["label"], r.get("name") or r["label"], r.get("area_m2"), r.get("width"),
                            r.get("depth"), r.get("height"), furnished, source))

    def get_rooms(self, case_id: str) -> list[dict]:
        return self._rows("SELECT * FROM rooms WHERE case_id = ? ORDER BY area_m2 DESC", case_id)

    # -------------------------------------------------------------- photos
    def add_photos(self, case_id: str, photos: list[dict], added_by: str | None = None) -> None:
        with self.tx() as db:
            for p in photos:
                db.execute(
                    "INSERT OR REPLACE INTO photos (case_id, photo_id, path, source, room_id, room_label, pano_id,"
                    " face, x, y, z, captured, added_by, number, batch_id, taken, status, chat_id)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (case_id, p["photo_id"], p["path"], p.get("source", "photo"), p.get("room_id"),
                     p.get("room_label"), p.get("pano_id"), p.get("face"), p.get("x"), p.get("y"),
                     p.get("z"), p.get("captured"), p.get("added_by", added_by), p.get("number"),
                     p.get("batch_id"), p.get("taken"), p.get("status", "done"), p.get("chat_id")))

    def next_photo_number(self, case_id: str) -> int:
        """Family photos are numbered 1, 2, 3... so people can say 'show photo 12'."""
        row = self.db.execute("SELECT MAX(number) AS n FROM photos WHERE case_id = ?", (case_id,)).fetchone()
        return (row["n"] or 0) + 1

    def set_photo(self, case_id: str, photo_id: str, **fields) -> None:
        allowed = ("status", "room_id", "room_label", "taken", "batch_id")
        with self.tx() as db:
            for key, value in fields.items():
                if key in allowed:
                    db.execute(f"UPDATE photos SET {key} = ? WHERE case_id = ? AND photo_id = ?",
                               (value, case_id, photo_id))

    def get_photo(self, case_id: str, photo_id: str | None = None, number: int | None = None) -> dict | None:
        if photo_id is not None:
            rows = self._rows("SELECT * FROM photos WHERE case_id = ? AND photo_id = ?", case_id, photo_id)
        else:
            rows = self._rows("SELECT * FROM photos WHERE case_id = ? AND number = ?", case_id, number)
        return rows[0] if rows else None

    def get_photos(self, case_id: str, source: str | None = None) -> list[dict]:
        if source:
            return self._rows("SELECT * FROM photos WHERE case_id = ? AND source = ?", case_id, source)
        return self._rows("SELECT * FROM photos WHERE case_id = ?", case_id)

    # --------------------------------------------------------------- items
    def add_item(self, case_id: str, item: dict, source: str, photo: dict | None = None,
                 added_by: str | None = None) -> int:
        """Every new item starts as 'suggested'. Only the family can confirm it."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        photo = photo or {}
        with self.tx() as db:
            cur = db.execute(
                "INSERT INTO items (case_id, photo_id, label, category, brand_guess, room_id, room_label,"
                " bbox, confidence, source, status, qty, evidence_date, added_by, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,'suggested',?,?,?,?)",
                (case_id, photo.get("photo_id"), item["label"], item.get("category"), item.get("brand_guess"),
                 item.get("room_id", photo.get("room_id")), item.get("room_label", photo.get("room_label")),
                 json.dumps(item.get("bbox")), item.get("confidence"), source, item.get("qty", 1),
                 item.get("evidence_date", photo.get("captured")), added_by, _now()))
            return cur.lastrowid

    def clear_items(self, case_id: str, source: str) -> None:
        """Drop unconfirmed items of one source before re-seeding. Confirmed ones stay."""
        with self.tx() as db:
            db.execute("DELETE FROM items WHERE case_id = ? AND source = ? AND status = 'suggested'",
                       (case_id, source))

    def get_items(self, case_id: str, status: str | None = None, canonical_only: bool = False) -> list[dict]:
        sql, args = "SELECT * FROM items WHERE case_id = ?", [case_id]
        if status:
            sql, args = sql + " AND status = ?", args + [status]
        if canonical_only:
            sql += " AND (cluster_id IS NULL OR cluster_id = item_id)"
        rows = self._rows(sql + " ORDER BY item_id", *args)
        for row in rows:
            row["bbox"] = json.loads(row["bbox"]) if row["bbox"] else None
        return rows

    def set_item_status(self, case_id: str, item_id: int, status: str) -> None:
        if status not in ITEM_STATUS:
            raise ValueError(f"status must be one of {ITEM_STATUS}")
        with self.tx() as db:
            # A decision on the canonical item covers the duplicates merged into it.
            db.execute("UPDATE items SET status = ? WHERE case_id = ? AND (item_id = ? OR cluster_id = ?)",
                       (status, case_id, item_id, item_id))

    def get_item(self, case_id: str, item_id: int) -> dict | None:
        rows = self._rows("SELECT * FROM items WHERE case_id = ? AND item_id = ?", case_id, item_id)
        if not rows:
            return None
        rows[0]["bbox"] = json.loads(rows[0]["bbox"]) if rows[0]["bbox"] else None
        return rows[0]

    def update_item(self, case_id: str, item_id: int, **fields) -> None:
        allowed = ("label", "category", "brand_guess", "qty", "room_id", "room_label", "detail", "price_low",
                   "price_mid", "price_high", "price_source", "reviewed", "cluster_id", "status", "price_ref", "cause")
        with self.tx() as db:
            for key, value in fields.items():
                if key in allowed:
                    db.execute(f"UPDATE items SET {key} = ? WHERE case_id = ? AND item_id = ?",
                               (value, case_id, item_id))

    def get_sightings(self, case_id: str, item_id: int) -> list[dict]:
        """Every detection merged into an item, the item's own first: its evidence."""
        rows = self._rows("SELECT * FROM items WHERE case_id = ? AND (item_id = ? OR cluster_id = ?)"
                          " ORDER BY (item_id = ?) DESC, item_id", case_id, item_id, item_id, item_id)
        for row in rows:
            row["bbox"] = json.loads(row["bbox"]) if row["bbox"] else None
        return rows

    def set_cluster(self, case_id: str, canonical_id: int, member_ids: list[int], qty: int) -> None:
        with self.tx() as db:
            db.executemany("UPDATE items SET cluster_id = ? WHERE case_id = ? AND item_id = ?",
                           [(canonical_id, case_id, m) for m in member_ids])
            db.execute("UPDATE items SET qty = ? WHERE case_id = ? AND item_id = ?", (qty, case_id, canonical_id))

    # ------------------------------------------------------------ findings
    def set_findings(self, case_id: str, findings: list[dict], source: str = "aerial") -> None:
        """Replace open findings of a source. Answered ones (confirmed/rejected) are kept."""
        with self.tx() as db:
            db.execute("DELETE FROM findings WHERE case_id = ? AND source = ? AND status = 'open'", (case_id, source))
            answered = {(r["kind"], r["label"], r["detail"]) for r in db.execute(
                "SELECT kind, label, detail FROM findings WHERE case_id = ? AND source = ?", (case_id, source))}
            for f in findings:
                detail = json.dumps(f.get("detail") or {}, sort_keys=True)
                if (f["kind"], f["label"], detail) in answered:
                    continue
                db.execute("INSERT INTO findings (case_id, kind, label, detail, confidence, source, evidence_date)"
                           " VALUES (?,?,?,?,?,?,?)",
                           (case_id, f["kind"], f["label"], detail, f.get("confidence"), source, f.get("evidence_date")))

    def get_findings(self, case_id: str, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM findings WHERE case_id = ?", [case_id]
        if status:
            sql, args = sql + " AND status = ?", args + [status]
        rows = self._rows(sql + " ORDER BY finding_id", *args)
        for row in rows:
            row["detail"] = json.loads(row["detail"] or "{}")
        return rows

    def set_finding_status(self, case_id: str, finding_id: int, status: str) -> None:
        if status not in FINDING_STATUS:
            raise ValueError(f"status must be one of {FINDING_STATUS}")
        with self.tx() as db:
            db.execute("UPDATE findings SET status = ? WHERE case_id = ? AND finding_id = ?",
                       (status, case_id, finding_id))
