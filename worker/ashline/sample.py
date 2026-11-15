"""Copy one home from another Ashline data folder into this one.

Used to put a ready-made home (photos already read, items placed on the plan) next to the
families' own homes: for a demo, or to move a single home between machines. Items get new
numbers here, so nothing collides with the homes already stored; the plan's sketch and the
merged repeats are pointed at the new numbers.
"""
from __future__ import annotations

import json
import secrets
import shutil
import sqlite3
from pathlib import Path

from . import config
from .store import Store

CASE_TABLES = ("home", "rooms", "photos", "findings")           # copied as they are, under the new case id


class ImportError_(RuntimeError):
    pass


def import_case(source_dir: str | Path, case_id: str, new_id: str | None = None, store: Store | None = None) -> dict:
    """Copy `case_id` from the data folder `source_dir` into this data folder as `new_id`.

    Returns {case_id, name, invite_code, owner_token, member_token, items, photos}. Raises
    ImportError_ when the source has no such home, or a home called `new_id` is already here.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    database = source_dir / "ashline.sqlite"
    if not database.is_file():
        raise ImportError_(f"No Ashline data in {source_dir} (ashline.sqlite is not there).")
    store = store or Store()
    new_id = new_id or case_id
    if store.get_case(new_id):
        raise ImportError_(f"A home with the id {new_id!r} is already here. Delete it first, or pass another --as id.")
    src = sqlite3.connect(database)
    src.row_factory = sqlite3.Row
    home = src.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if not home:
        raise ImportError_(f"No home {case_id!r} in {source_dir}.")

    def columns(db, table: str) -> list[str]:
        return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]

    target = config.case_dir(new_id)
    for name in ("photos", "aerial"):                              # the pictures; crops are made again on demand
        if (source_dir / case_id / name).is_dir():
            shutil.copytree(source_dir / case_id / name, target / name, dirs_exist_ok=True)
    for extra in ("sketch.json", "plan.json", "prices.json"):
        if (source_dir / case_id / extra).is_file():
            shutil.copyfile(source_dir / case_id / extra, target / extra)

    row = dict(home)
    # A code or link already used by a home here would open the wrong home: give this one fresh ones.
    for key, make in (("invite_code", lambda: f"HOME-{secrets.randbelow(9000) + 1000}"),
                      ("owner_token", lambda: secrets.token_urlsafe(24)), ("member_token", lambda: secrets.token_urlsafe(24))):
        while not row.get(key) or store.find_case(**{key: row[key]}):
            row[key] = make()
    row["case_id"] = new_id
    ids: dict[int, int] = {}
    with store.tx() as db:
        keep = [c for c in columns(db, "cases") if c in row]
        db.execute(f"INSERT INTO cases ({', '.join(keep)}) VALUES ({', '.join('?' for _ in keep)})", [row[c] for c in keep])
        for table in CASE_TABLES:
            here = columns(db, table)
            for r in src.execute(f"SELECT * FROM {table} WHERE case_id = ?", (case_id,)):
                r = dict(r)
                r["case_id"] = new_id
                r.pop("finding_id", None)                           # numbered afresh here
                if table == "photos" and r.get("path"):             # the picture now lives in this data folder
                    tail = str(r["path"]).replace("\\", "/").split(f"/{case_id}/", 1)[-1]
                    r["path"] = str(target / tail)
                    r["chat_id"] = None
                cols = [c for c in here if c in r]
                db.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                           [r[c] for c in cols])
        here = [c for c in columns(db, "items") if c != "item_id"]
        rows = [dict(r) for r in src.execute("SELECT * FROM items WHERE case_id = ? ORDER BY item_id", (case_id,))]
        for r in rows:
            old = r["item_id"]
            r["case_id"] = new_id
            cols = [c for c in here if c in r]
            ids[old] = db.execute(f"INSERT INTO items ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                                  [r[c] for c in cols]).lastrowid
        for r in rows:                                              # repeats point at the line they were merged into
            if r.get("cluster_id") is not None:
                db.execute("UPDATE items SET cluster_id = ? WHERE item_id = ?",
                           (ids.get(r["cluster_id"], ids[r["item_id"]]), ids[r["item_id"]]))
    src.close()

    sketch = target / "sketch.json"
    if sketch.is_file():                                            # the plan's dots, by the new numbers
        try:
            drawn = json.loads(sketch.read_text())
            drawn["items"] = [{**i, "id": ids[i["id"]]} for i in drawn.get("items") or [] if i.get("id") in ids]
            try:                                                    # so the sketch is not thought out of date
                from . import sketch_claude
                request = sketch_claude.sketch_request(new_id, store)
                if request:
                    drawn["digest"] = request["digest"]
            except Exception:
                pass
            sketch.write_text(json.dumps(drawn))
        except (ValueError, KeyError, TypeError):
            sketch.unlink()
    case = store.get_case(new_id)
    return {"case_id": new_id, "name": case.get("name"), "invite_code": case.get("invite_code"),
            "owner_token": case.get("owner_token"), "member_token": case.get("member_token"),
            "items": len(ids), "photos": len(store.get_photos(new_id))}
