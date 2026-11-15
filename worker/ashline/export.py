"""Exports for the insurer (build spec 6.8, UX doc section 8).

- Spreadsheet (CSV and XLSX): one row per item with room, category, quantity, price range,
  evidence type, contributor and a reference to the evidence photo.
- PDF: cover summary, category totals, item table with thumbnails, and an appendix of the
  annotated evidence photos.

Grouped by category by default: under SB 495 insurers must accept category-grouped lists.
Only items the family confirmed are exported unless `include_suggested` is set. Memory-only
items are always labelled as such.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config, disaster
from .meter import DISCLAIMER, meter
from .rooms import tidy_room
from .pricing import in_use, money, price_sources
from .store import Store

COLUMNS = ["room", "category", "item", "qty", "brand_model", "age", "rcv_low", "rcv_mid", "rcv_high",
           "source", "evidence", "status", "photo_refs", "contributor", "evidence_link", "claim", "acv", "essential",
           "basis"]


def evidence_text(source: str, date: str | None) -> str:
    when = f" captured {date}" if date else ""
    return {
        "scan": f"3D scan of the home{when}",
        "aerial": f"Aerial image{when}",
        "voice": "From memory, needs proof",
        "photo": f"Family photo{' taken ' + date if date else ''}",
        "receipt": f"Receipt{' dated ' + date if date else ''}",
    }.get(source, source)


def purchase_text(item: dict) -> str:
    """'card purchase at IKEA on 2024-03-12 ($1,149.00)' for an item with a matched card purchase, else ''."""
    paid = price_sources(item).get("paid")
    if not paid:
        return ""
    try:
        amount = f" (${float(paid['amount']):,.2f})"
    except (KeyError, TypeError, ValueError):
        amount = ""
    proof = "order from" if paid.get("order") else "receipt from" if paid.get("receipt") else "card purchase at"
    return (f"{proof} {paid.get('merchant') or 'a store'}"
            + (f" {'on' if proof.startswith('card') else 'dated'} {paid['date']}" if paid.get("date") else "") + amount)


BASIS = {"family": "Set by the family", "paid": "Paid: card record or receipt", "online": "Online: current listings",
         "estimate": "Estimate: rough allowance", "": ""}


def _number(value) -> float | str:
    return round(float(value), 2) if value not in (None, "") else ""


def _evidence(item: dict) -> str:
    """What stands behind the line: how it was seen, plus the purchase record when there is one."""
    seen, bought = evidence_text(item["source"], item["evidence_date"]), purchase_text(item)
    if not bought:
        return seen
    if item["source"] == "voice":                           # the purchase is the proof a remembered item lacked
        return f"From memory, with a {bought}"
    return f"{seen}; {bought}"


def export_rows(case_id: str, store: Store | None = None, include_suggested: bool = False,
                group_by: str = "category") -> list[dict]:
    store = store or Store()
    case = store.get_case(case_id) or {}
    base = (config.env("ASHLINE_PUBLIC_URL") or "").rstrip("/")
    rows = []
    for item in store.get_items(case_id, canonical_only=True):
        if item["status"] == "rejected" or (item["status"] == "suggested" and not include_suggested):
            continue        # nothing the family has not confirmed goes into the inventory
        sightings = store.get_sightings(case_id, item["item_id"])
        people = sorted({("3D scan" if s["added_by"] == "scan" else s["added_by"]) for s in sightings if s.get("added_by")})
        link = ""
        if base and case.get("member_token") and item.get("photo_id") and item.get("bbox"):
            link = f"{base}/p/{case['member_token']}/sighting/{item['item_id']}.jpg"
        rows.append({
            "room": tidy_room(item["room_label"]) or "", "category": item["category"] or "", "item": item["label"],
            "qty": item["qty"] or 1, "brand_model": item["brand_guess"] or "", "age": "",
            "rcv_low": _number(item.get("price_low")), "rcv_mid": _number(item.get("price_mid")),
            "rcv_high": _number(item.get("price_high")), "basis": BASIS[in_use(item)],
            "source": item["source"], "evidence": _evidence(item),
            "status": item["status"],
            "photo_refs": ", ".join(dict.fromkeys(s["photo_id"] for s in sightings if s.get("photo_id"))),
            "contributor": ", ".join(people), "evidence_link": link, "_id": item["item_id"],
            **_claim_fields(case, item),
        })
    first, second = ("category", "room") if group_by == "category" else ("room", "category")
    order = {key: n for n, (key, _, _) in enumerate(disaster.claims(case))}
    rows.sort(key=lambda r: (order.get(r["_claim"], 0), r[first], r[second], r["item"].lower()))
    return rows


def _claim_fields(case: dict, item: dict) -> dict:
    key = disaster.claim_key(case, item)
    title, basis = next(((t, b) for k, t, b in disaster.claims(case) if k == key), ("Homeowners claim", "rcv"))
    each = disaster.acv(item.get("price_mid"), item.get("category")) if basis == "acv" else None
    return {"claim": title, "acv": _number(each), "essential": "yes" if disaster.is_essential(item) else "",
            "_claim": key, "_basis": basis}


def claim_groups(case: dict, rows: list[dict]) -> list[dict]:
    """The rows split by claim, with totals, in the order the claims are listed."""
    out = []
    for key, title, basis in disaster.claims(case):
        mine = [r for r in rows if r["_claim"] == key]
        qty = lambda r: int(r["qty"] or 1)
        out.append({"key": key, "title": title, "basis": basis, "rows": mine, "items": sum(qty(r) for r in mine),
                    "new": sum(float(r["rcv_mid"] or 0) * qty(r) for r in mine),
                    "acv": sum(float(r["acv"] or 0) * qty(r) for r in mine)})
    return out


def plain_cell(value):
    """Text a spreadsheet must not run. A family member names items in chat, and the insurer opens the file:
    a name starting like a formula is kept as text by putting an apostrophe in front."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def export_csv(case_id: str, store: Store | None = None, include_suggested: bool = False) -> Path:
    path = config.case_dir(case_id) / "inventory.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({c: plain_cell(row.get(c)) for c in COLUMNS}
                         for row in export_rows(case_id, store, include_suggested))
        fh.write(f"# {DISCLAIMER}\n")
    return path


def export_xlsx(case_id: str, store: Store | None = None, include_suggested: bool = False) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    store = store or Store()
    rows = export_rows(case_id, store, include_suggested)
    book = Workbook()
    sheet = book.active
    sheet.title = "Inventory"
    headers = ["Room", "Category", "Item", "Qty", "Brand / model", "Age", "Low", "Mid", "High", "Evidence type",
               "Evidence", "Status", "Photos", "Contributor", "Evidence link", "Claim", "Depreciated (est.)", "Essential",
               "Price basis"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([plain_cell(row[c]) for c in COLUMNS])
    for index, width in enumerate([22, 14, 44, 6, 18, 8, 10, 10, 10, 13, 38, 11, 22, 18, 40, 34, 16, 10, 24], 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2, min_col=7, max_col=9):
        for cell in row:
            cell.number_format = '"$"#,##0'
    for row in sheet.iter_rows(min_row=2, min_col=17, max_col=17):
        for cell in row:
            cell.number_format = '"$"#,##0'
    sheet.freeze_panes = "A2"
    sheet.append([])
    sheet.append([DISCLAIMER])

    totals = book.create_sheet("By category")
    totals.append(["Category", "Items", "Low", "Mid", "High"])
    for cell in totals[1]:
        cell.font = Font(bold=True)
    for name, t in category_totals(rows).items():
        totals.append([name, t["items"], t["low"], t["mid"], t["high"]])

    case = store.get_case(case_id) or {}
    groups = claim_groups(case, rows)
    if len(groups) > 1 or groups[0]["basis"] == "acv":
        summary = book.create_sheet("By claim", 1)
        summary.append(["Claim", "Items", "Replacement cost (new)", "Depreciated value (est.)"])
        for cell in summary[1]:
            cell.font = Font(bold=True)
        for g in groups:
            summary.append([g["title"], g["items"], g["new"], g["acv"] if g["basis"] == "acv" else None])
        summary.append([])
        summary.append([disaster.ACV_NOTE])
        summary.column_dimensions["A"].width = 44
        for col in "CD":
            summary.column_dimensions[col].width = 24
        for g in groups:                                   # one sheet per claim, ready to hand to that insurer
            if not g["rows"]:
                continue
            part = book.create_sheet(("Flood claim" if g["key"] == "flood" else "Wind and rain claim"
                                      if g["key"] == "wind" else "Claim")[:31])
            part.append(headers)
            for cell in part[1]:
                cell.font = Font(bold=True)
            for row in g["rows"]:
                part.append([plain_cell(row[c]) for c in COLUMNS])
            part.freeze_panes = "A2"
    if case.get("insured") == "no":
        essentials = book.create_sheet("Essentials (FEMA)")
        essentials.append(["Item", "Room", "Qty", "Replacement cost (mid)"])
        for cell in essentials[1]:
            cell.font = Font(bold=True)
        for row in rows:
            if row["essential"]:
                essentials.append([plain_cell(row["item"]), plain_cell(row["room"]), row["qty"], row["rcv_mid"]])
    path = config.case_dir(case_id) / "inventory.xlsx"
    book.save(path)
    return path


def category_totals(rows: list[dict]) -> dict[str, dict]:
    totals: dict[str, dict] = {}
    for row in rows:
        t = totals.setdefault(row["category"] or "other", {"items": 0, "low": 0.0, "mid": 0.0, "high": 0.0})
        qty = int(row["qty"] or 1)
        t["items"] += qty
        for key, column in (("low", "rcv_low"), ("mid", "rcv_mid"), ("high", "rcv_high")):
            t[key] += float(row[column] or 0) * qty
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]["mid"]))


def export_pdf(case_id: str, store: Store | None = None, include_suggested: bool = False,
               max_appendix_photos: int = 60) -> Path:
    """The PDF a family hands to the insurer."""
    import io

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from .annotate import annotate, crop_item

    store = store or Store()
    case = store.get_case(case_id) or {}
    rows = export_rows(case_id, store, include_suggested)
    m = meter(case_id, store)
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8.5, leading=11)
    note = ParagraphStyle("note", parent=small, textColor=colors.HexColor("#6C6863"))
    ink, line, red = colors.HexColor("#171615"), colors.HexColor("#D5DCDE"), colors.HexColor("#B3171D")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6C6863"))
        canvas.drawString(0.75 * inch, 0.5 * inch, DISCLAIMER)
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    from xml.sax.saxutils import escape

    def esc(value) -> str:
        """Paragraph text is markup: a stray & or < in a label would break the whole PDF."""
        return escape(str(value or ""))

    def picture(image, width, max_height=None) -> RLImage:
        buf = io.BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")                   # PNG screenshots with transparency, palette images
        image.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        height = width * image.height / image.width
        if max_height and height > max_height:              # a tall crop: fit it to the row instead
            width, height = width * max_height / height, max_height
        return RLImage(buf, width=width, height=height)

    title = ParagraphStyle("title", parent=styles["Title"], alignment=0, fontSize=22, leading=26, textColor=ink, spaceAfter=4)
    logo_path = Path(__file__).parent / "static" / "img" / "logo.png"
    story = []
    try:                                                # the logo is decoration; never fail the export for it
        mark = RLImage(str(logo_path), width=0.42 * inch, height=0.42 * inch * 304 / 360)
        brand = Table([[mark, Paragraph("<font name='Times-Bold' size='14'>Ashline</font>", small)]],
                      colWidths=[0.5 * inch, 2 * inch], hAlign="LEFT")
        brand.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story += [brand, Spacer(1, 0.18 * inch)]
    except Exception:
        pass
    story += [Paragraph(f"Contents inventory: {esc(case.get('name') or 'Home')}", title),
              Table([[""]], colWidths=[6.8 * inch], rowHeights=[2], hAlign="LEFT",
                    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), red)]))]
    things = sum(int(r['qty'] or 1) for r in rows)             # a line can stand for several of the same thing
    facts = [["Items listed", str(len(rows)) + (f" ({things} counting quantities)" if things != len(rows) else "")],
             ["Estimated replacement cost (mid)", money(sum(float(r['rcv_mid'] or 0) * int(r['qty'] or 1) for r in rows))],
             ["Range (low to high)", f"{money(m['low'])} to {money(m['high'])}"],
             ["Backed by a photo, scan or receipt", money(m["backed"])],
             ["From memory only", money(m["memory"])]]
    if case.get("address"):
        facts.insert(0, ["Address", Paragraph(esc(case["address"]), small)])
    if case.get("loss_date"):
        facts.insert(1 if case.get("address") else 0, ["Date of loss", case["loss_date"][:10]])
    groups = claim_groups(case, rows)
    split = len(groups) > 1 or groups[0]["basis"] == "acv"
    if case.get("disaster"):
        facts.insert(0, ["What happened", disaster.NAMES.get(case["disaster"], "disaster").capitalize()])
    if split:
        for g in groups:
            value = f"{g['items']} items · {money(g['new'])} new"
            if g["basis"] == "acv":
                value += f" · about {money(g['acv'])} depreciated"
            facts.append([g["title"], value])
    if m["limit"]:
        facts += [["Contents coverage limit", money(m["limit"])]]
        if m["advance"] is not None:
            facts += [["Paid without an inventory under SB 495", money(m["advance"])]]
        facts += [["Still to document", money(m["target"])]]
    table = Table(facts, colWidths=[3.6 * inch, 3.2 * inch])
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 10), ("LINEBELOW", (0, 0), (-1, -1), 0.4, line),
                               ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6C6863")),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    story += [Spacer(1, 0.15 * inch), table, Spacer(1, 0.2 * inch),
              Paragraph("Values are estimates of the cost to replace each item new today. Items marked "
                        "\"from memory\" have no photo or receipt behind them."
                        + (" " + disaster.ACV_NOTE if split and any(g["basis"] == "acv" for g in groups) else "")
                        + (" Under California SB 495 an insurer pays at least 60% of the contents limit, up to "
                           "$350,000, without an inventory after a declared-disaster total loss."
                           if disaster.uses_sb495(case) else ""), note),
              Spacer(1, 0.3 * inch), Paragraph("Totals by category", styles["Heading2"])]

    totals = [["Category", "Items", "Low", "Mid", "High"]] + [
        [name.title(), t["items"], money(t["low"]), money(t["mid"]), money(t["high"])]
        for name, t in category_totals(rows).items()]
    table = Table(totals, colWidths=[2.6 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch])
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    story += [table, PageBreak()]

    def thumb_for(row):
        item = store.get_item(case_id, row["_id"])
        photo = store.get_photo(case_id, item["photo_id"]) if item and item.get("photo_id") else None
        if photo and Path(photo["path"]).exists():
            try:
                return picture(crop_item(photo["path"], item["bbox"], max_side=220), 0.8 * inch, 0.8 * inch)
            except Exception:                               # one unreadable photo must not sink the PDF
                return ""
        return ""

    def items_table(subset, basis):
        acv_cols = basis == "acv"
        head = ["", "Item", "Room", "Qty"] + (["New", "Depr. (est.)"] if acv_cols else ["Low", "Mid", "High"]) + ["Evidence"]
        data = [head]
        for row in subset:
            brand = f" ({esc(row['brand_model'])})" if row["brand_model"] else ""
            label = f"{esc(row['item'])}{brand}<br/><font color='#6C6863'>{esc(row['category'])}</font>"
            cell = lambda v: money(v) if v != "" and v is not None else ""
            values = [cell(row["rcv_mid"]), cell(row["acv"])] if acv_cols else \
                     [cell(row["rcv_low"]), cell(row["rcv_mid"]), cell(row["rcv_high"])]
            data.append([thumb_for(row), Paragraph(label, small), Paragraph(esc(row["room"]), small), row["qty"], *values,
                         Paragraph(f"{esc(row['evidence'])}<br/><font color='#6C6863'>{esc(row['contributor'])}</font>", small)])
        widths = ([0.9, 2.1, 1.0, 0.4, 0.75, 0.75, 1.4] if acv_cols else [0.9, 2.1, 1.0, 0.4, 0.6, 0.6, 0.6, 1.3])
        t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
        last_money = 5 if acv_cols else 6
        t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, line),
                               ("ALIGN", (3, 0), (last_money, -1), "RIGHT"), ("TEXTCOLOR", (0, 0), (-1, -1), ink)]))
        return t

    if split:
        for n, g in enumerate(groups):
            if not g["rows"]:
                continue
            total = f"{g['items']} items · {money(g['new'])} new"
            if g["basis"] == "acv":
                total += f" · about {money(g['acv'])} depreciated"
            story += ([PageBreak()] if n and story and not isinstance(story[-1], PageBreak) else []) + [
                Paragraph(esc(g["title"]), styles["Heading2"]), Paragraph(total, note), Spacer(1, 0.08 * inch),
                items_table(g["rows"], g["basis"])]
    else:
        story += [Paragraph("Items", styles["Heading2"]), items_table(rows, groups[0]["basis"])]

    if case.get("insured") == "no":
        essentials = [r for r in rows if r["essential"]]
        if essentials:
            story += [PageBreak(), Paragraph("Essential items", styles["Heading2"]),
                      Paragraph("Appliances, furniture and a computer: the kinds of personal property FEMA's "
                                "Individuals and Households Program can help repair or replace for uninsured households.", note),
                      Spacer(1, 0.08 * inch), items_table(essentials, "rcv")]

    # Appendix: each evidence photo once, with the listed items numbered on it.
    by_photo: dict[str, list[dict]] = {}
    for row in rows:
        item = store.get_item(case_id, row["_id"])
        if item and item.get("photo_id") and item.get("bbox"):
            by_photo.setdefault(item["photo_id"], []).append(item)
    if by_photo:
        story += [PageBreak(), Paragraph("Evidence photos", styles["Heading2"]),
                  Paragraph("Original photos with the listed items marked. Nothing in a photo has been altered "
                            "other than the numbered outlines.", note), Spacer(1, 0.15 * inch)]
        for photo_id, items in list(by_photo.items())[:max_appendix_photos]:
            photo = store.get_photo(case_id, photo_id)
            if not photo or not Path(photo["path"]).exists():
                continue
            boxes = [{"n": n, "bbox": i["bbox"], "status": "listed"} for n, i in enumerate(items, 1)]
            when = photo.get("taken") or photo.get("captured") or ""
            who = "3D scan" if photo.get("added_by") == "scan" else (photo.get("added_by") or "")
            try:                                            # tall photos are fitted to the page height
                shot = picture(annotate(photo["path"], boxes, max_side=1100), 5.2 * inch, 6.6 * inch)
            except Exception:
                continue
            story += [shot,
                      Paragraph(esc(" · ".join(x for x in (photo_id, who, when) if x)), note),
                      Paragraph(esc(" · ".join(f"{n} {i['label']}" for n, i in enumerate(items, 1))), small),
                      Spacer(1, 0.25 * inch)]
        if len(by_photo) > max_appendix_photos:
            story.append(Paragraph(f"{len(by_photo) - max_appendix_photos} more evidence photos are available on request.", note))

    path = config.case_dir(case_id) / "inventory.pdf"
    SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                      title=f"Contents inventory: {case.get('name') or 'Home'}").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return path
