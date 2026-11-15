"""Amazon order history: what the family paid for things they bought online.

Amazon lets a customer download their order history. "Request my data" gives a file named
Retail.OrderHistory.1.csv; the older order reports and some browser extensions give similar CSVs
with other column names. Each line of such a file is a product, its order, its date and what was
paid, which is the same kind of record as a card purchase: proof that the family owned the thing
and what it cost. So the lines are read into the shape of `bank.purchases()` and matched and
applied with the bank module's own code.

    read_orders(data)              the paid lines of an order-history CSV, as purchases
    looks_like_orders(name, data)  whether a file is one
    find_matches(case_id, ...)     which order line is which inventory item

The column names below come from published descriptions of these exports, not from a real file:
the parser has been tested on synthetic files only. Where the exports are known to differ,
several names are accepted for the same column. An amount is never guessed: a line with no
usable amount column is skipped, and so are cancelled or returned lines.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path

from .bank import match_purchases
from .pricing import price_sources

MERCHANT = "Amazon"
# Each kind of column under the names the exports are known to use, normalised (see `_key`).
ORDER_ID = ("order id", "order number", "order #")
ORDER_DATE = ("order date", "date", "purchase date")
PRODUCT = ("product name", "title", "item name", "product", "item", "description")
QUANTITY = ("quantity", "qty", "item quantity")
# What was paid for the whole line, in order of preference, then the price of one unit.
LINE_TOTAL = ("total owed", "item total", "line total")   # not a bare "total": that can be the whole order's
UNIT_PRICE = ("unit price", "purchase price per unit", "item price")
STATUS = ("order status", "status", "shipment status", "item status")
CURRENCY = ("currency", "currency code")
SKIPPED = ("cancel", "return", "refund")


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").replace("﻿", "").strip().strip("'\"").lower())


def _text(data: bytes | str) -> str:
    if isinstance(data, str):
        return data.lstrip("﻿")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")      # spreadsheet programs often save in this


def _rows(data: bytes | str) -> tuple[dict, list[list[str]]]:
    """{normalised column name: index} for the header, and the rows after it."""
    rows = [row for row in csv.reader(io.StringIO(_text(data))) if any(cell.strip() for cell in row)]
    if not rows:
        return {}, []
    header = {}
    for n, name in enumerate(rows[0]):
        header.setdefault(_key(name), n)
    return header, rows[1:]


def _column(header: dict, names: tuple[str, ...]) -> int | None:
    return next((header[name] for name in names if name in header), None)


def _cell(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    value = row[index].strip().strip("'").strip()
    return "" if value.lower() in ("not available", "n/a", "na", "-") else value


def _amount(value: str) -> float | None:
    """'$1,149.00' -> 1149.0. None for a blank or a value that is not a number."""
    cleaned = re.sub(r"[$,\s]|USD", "", value or "")
    if cleaned.startswith("(") and cleaned.endswith(")"):     # accounting style for a negative
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _date(value: str) -> str | None:
    """ISO timestamps ('2023-04-12T18:22:31Z', '2023-04-12 18:22:31 UTC') and MM/DD/YY(YY) as YYYY-MM-DD."""
    value = (value or "").strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", value):
        return value[:10]
    for form in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, form).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _quantity(value: str) -> int:
    try:
        return max(int(float(value)), 1)
    except ValueError:
        return 1


def read_orders(data: bytes | str) -> list[dict]:
    """The paid lines of an order-history CSV: [{id, date, amount, description, merchant, account, order}].

    `id` is the order id and the line's place within that order ("112-...#2"), so two lines of
    one order differ and the same file read twice gives the same ids. Newest first, like
    `bank.purchases()`.
    """
    header, rows = _rows(data)
    columns = {name: _column(header, names) for name, names in (
        ("order", ORDER_ID), ("date", ORDER_DATE), ("product", PRODUCT), ("qty", QUANTITY),
        ("total", LINE_TOTAL), ("unit", UNIT_PRICE), ("status", STATUS), ("currency", CURRENCY))}
    if columns["order"] is None or columns["product"] is None:
        return []
    lines_seen: dict[str, int] = {}
    out = []
    for row in rows:
        order = _cell(row, columns["order"])
        if not order:
            continue
        # Counted before any line is skipped, so a line keeps its id when another one is cancelled later.
        lines_seen[order] = lines_seen.get(order, 0) + 1
        product = _cell(row, columns["product"])
        status = _cell(row, columns["status"]).lower()
        currency = _cell(row, columns["currency"]).upper()
        if not product or any(word in status for word in SKIPPED) or currency not in ("", "USD"):
            continue
        qty = _quantity(_cell(row, columns["qty"]))
        amount = _amount(_cell(row, columns["total"]))
        if amount is None:
            unit = _amount(_cell(row, columns["unit"]))
            amount = round(unit * qty, 2) if unit is not None else None
        if amount is None or amount <= 0:
            continue
        out.append({"id": f"{order}#{lines_seen[order]}", "date": _date(_cell(row, columns["date"])),
                    "amount": round(amount, 2), "description": (f"{qty} x " if qty > 1 else "") + product,
                    "merchant": MERCHANT, "account": "", "order": order})
    out.sort(key=lambda p: p.get("date") or "", reverse=True)
    return out


def looks_like_orders(name: str, data: bytes) -> bool:
    """True for a CSV whose header names an order id and a product, so any other CSV is left alone."""
    if Path(name or "").suffix.lower() not in ("", ".csv", ".txt"):
        return False
    header, _ = _rows(data[:20000] if isinstance(data, bytes) else data)
    return _column(header, ORDER_ID) is not None and _column(header, PRODUCT) is not None


def find_matches(case_id: str, store, orders: list[dict]) -> list[dict]:
    """Order lines that match this case's inventory lines and are not already used as their price.

    The same exclusions as `bank.find_matches`: an item the family priced, or one that already
    has a paid price, is left alone; an order line that already is some item's price is not offered.
    """
    everything = [i for i in store.get_items(case_id, canonical_only=True) if i["status"] != "rejected"]
    paid = {i["item_id"]: price_sources(i).get("paid") for i in everything}
    used = {p.get("bank_id") for p in paid.values() if p}
    items = [i for i in everything if i.get("price_source") != "set by family" and not paid[i["item_id"]]]
    return match_purchases(items, [o for o in orders if o["id"] not in used])
