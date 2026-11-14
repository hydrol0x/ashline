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

SOURCES = ("photo", "voice", "scan", "aerial")
ITEM_STATUS = ("suggested", "confirmed", "rejected")
FINDING_STATUS = ("open", "confirmed", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  case_id TEXT PRIMARY KEY, address TEXT, loss_date TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS home (
  case_id TEXT PRIMARY KEY, sid TEXT, lat REAL, lon REAL, parcel_wkt TEXT, homes_listing_id TEXT,
  scan_captured TEXT, capture_notes TEXT, aerial_collection TEXT, aerial_captured TEXT);
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
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | None = None):
        if path is None:
            root = config.data_dir()
            root.mkdir(parents=True, exist_ok=True)
            path = str(root / "ashline.sqlite")
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

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
            for table in ("cases", "home", "rooms", "photos", "items", "findings"):
                db.execute(f"DELETE FROM {table} WHERE case_id = ?", (case_id,))

    # ---------------------------------------------------------------- home
    def set_home(self, case_id: str, **fields) -> None:
        allowed = ("sid", "lat", "lon", "parcel_wkt", "homes_listing_id", "scan_captured",
                   "capture_notes", "aerial_collection", "aerial_captured")
        fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
        with self.tx() as db:
            db.execute("INSERT OR IGNORE INTO home (case_id) VALUES (?)", (case_id,))
            for key, value in fields.items():
                db.execute(f"UPDATE home SET {key} = ? WHERE case_id = ?", (value, case_id))

    def get_home(self, case_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM home WHERE case_id = ?", case_id)
        return rows[0] if rows else None

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
                db.execute("INSERT OR REPLACE INTO photos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (case_id, p["photo_id"], p["path"], p.get("source", "photo"), p.get("room_id"),
                            p.get("room_label"), p.get("pano_id"), p.get("face"), p.get("x"), p.get("y"),
                            p.get("z"), p.get("captured"), added_by))

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
