"""CSV export (build spec 6.8) with the source and evidence columns.

`source` is photo | voice | scan | aerial. For scan and aerial items the
evidence column records the capture date, so an adjuster can see how old the
evidence is. Price columns stay empty until pricing (6.5) fills them in.
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import config
from .store import Store

COLUMNS = ["room", "category", "item", "qty", "brand_model", "age", "rcv_low", "rcv_mid", "rcv_high",
           "source", "evidence", "status", "photo_refs"]
DISCLAIMER = "Estimate for planning, not legal advice; confirm with your adjuster."


def evidence_text(source: str, date: str | None) -> str:
    when = f" captured {date}" if date else ""
    return {
        "scan": f"3D scan of the home{when}",
        "aerial": f"Aerial image{when}",
        "voice": "From memory (voice call), needs proof",
        "photo": f"Family photo{' taken ' + date if date else ''}",
    }.get(source, source)


def export_rows(case_id: str, store: Store | None = None, include_suggested: bool = False) -> list[dict]:
    store = store or Store()
    rows = []
    for item in store.get_items(case_id, canonical_only=True):
        if item["status"] == "rejected" or (item["status"] == "suggested" and not include_suggested):
            continue        # nothing the family has not confirmed goes into the inventory
        rows.append({
            "room": item["room_label"] or "", "category": item["category"] or "", "item": item["label"],
            "qty": item["qty"] or 1, "brand_model": item["brand_guess"] or "", "age": "",
            "rcv_low": "", "rcv_mid": "", "rcv_high": "",
            "source": item["source"], "evidence": evidence_text(item["source"], item["evidence_date"]),
            "status": item["status"], "photo_refs": item["photo_id"] or "",
        })
    rows.sort(key=lambda r: (r["room"], r["category"], r["item"]))
    return rows


def export_csv(case_id: str, store: Store | None = None, include_suggested: bool = False) -> Path:
    path = config.case_dir(case_id) / "inventory.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(export_rows(case_id, store, include_suggested))
        fh.write(f"# {DISCLAIMER}\n")
    return path
