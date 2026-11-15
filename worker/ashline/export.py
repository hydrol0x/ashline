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
from pathlib import Path

from . import config
from .meter import DISCLAIMER, meter
from .rooms import tidy_room
from .pricing import money
from .store import Store

COLUMNS = ["room", "category", "item", "qty", "brand_model", "age", "rcv_low", "rcv_mid", "rcv_high",
           "source", "evidence", "status", "photo_refs", "contributor", "evidence_link"]


def evidence_text(source: str, date: str | None) -> str:
    when = f" captured {date}" if date else ""
    return {
        "scan": f"3D scan of the home{when}",
        "aerial": f"Aerial image{when}",
        "voice": "From memory (voice call), needs proof",
        "photo": f"Family photo{' taken ' + date if date else ''}",
        "receipt": f"Receipt{' dated ' + date if date else ''}",
    }.get(source, source)


def _number(value) -> float | str:
    return round(float(value), 2) if value not in (None, "") else ""


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
            "rcv_high": _number(item.get("price_high")),
            "source": item["source"], "evidence": evidence_text(item["source"], item["evidence_date"]),
            "status": item["status"],
            "photo_refs": ", ".join(dict.fromkeys(s["photo_id"] for s in sightings if s.get("photo_id"))),
            "contributor": ", ".join(people), "evidence_link": link, "_id": item["item_id"],
        })
    first, second = ("category", "room") if group_by == "category" else ("room", "category")
    rows.sort(key=lambda r: (r[first], r[second], r["item"].lower()))
    return rows


def export_csv(case_id: str, store: Store | None = None, include_suggested: bool = False) -> Path:
    path = config.case_dir(case_id) / "inventory.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(export_rows(case_id, store, include_suggested))
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
               "Evidence", "Status", "Photos", "Contributor", "Evidence link"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([row[c] for c in COLUMNS])
    for index, width in enumerate([22, 14, 44, 6, 18, 8, 10, 10, 10, 13, 38, 11, 22, 18, 40], 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2, min_col=7, max_col=9):
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
    note = ParagraphStyle("note", parent=small, textColor=colors.HexColor("#6B655C"))
    ink, line = colors.HexColor("#2A2826"), colors.HexColor("#D9D2C7")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6B655C"))
        canvas.drawString(0.75 * inch, 0.5 * inch, DISCLAIMER)
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def picture(image, width, max_height=None) -> RLImage:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        height = width * image.height / image.width
        if max_height and height > max_height:              # a tall crop: fit it to the row instead
            width, height = width * max_height / height, max_height
        return RLImage(buf, width=width, height=height)

    story = [Paragraph(f"Contents inventory: {case.get('name') or 'Home'}", styles["Title"])]
    facts = [["Items listed", str(sum(int(r['qty'] or 1) for r in rows))],
             ["Estimated replacement cost (mid)", money(sum(float(r['rcv_mid'] or 0) * int(r['qty'] or 1) for r in rows))],
             ["Range (low to high)", f"{money(m['low'])} to {money(m['high'])}"],
             ["Backed by a photo, scan or receipt", money(m["backed"])],
             ["From memory only", money(m["memory"])]]
    if case.get("address"):
        facts.insert(0, ["Address", case["address"]])
    if case.get("loss_date"):
        facts.insert(1 if case.get("address") else 0, ["Date of loss", case["loss_date"][:10]])
    if m["limit"]:
        facts += [["Contents coverage limit", money(m["limit"])],
                  ["Paid without an inventory under SB 495", money(m["advance"])],
                  ["Still to document", money(m["target"])]]
    table = Table(facts, colWidths=[3.6 * inch, 3.2 * inch])
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 10), ("LINEBELOW", (0, 0), (-1, -1), 0.4, line),
                               ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6B655C")),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    story += [Spacer(1, 0.15 * inch), table, Spacer(1, 0.2 * inch),
              Paragraph("Values are estimates of the cost to replace each item new today. Items marked "
                        "\"from memory\" have no photo or receipt behind them. Under California SB 495 an insurer "
                        "pays at least 60% of the contents limit, up to $350,000, without an inventory after a "
                        "declared-disaster total loss.", note),
              Spacer(1, 0.3 * inch), Paragraph("Totals by category", styles["Heading2"])]

    totals = [["Category", "Items", "Low", "Mid", "High"]] + [
        [name.title(), t["items"], money(t["low"]), money(t["mid"]), money(t["high"])]
        for name, t in category_totals(rows).items()]
    table = Table(totals, colWidths=[2.6 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch])
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    story += [table, PageBreak(), Paragraph("Items", styles["Heading2"])]

    data = [["", "Item", "Room", "Qty", "Low", "Mid", "High", "Evidence"]]
    for row in rows:
        item = store.get_item(case_id, row["_id"])
        photo = store.get_photo(case_id, item["photo_id"]) if item and item.get("photo_id") else None
        thumb = ""
        if photo and Path(photo["path"]).exists():
            try:
                thumb = picture(crop_item(photo["path"], item["bbox"], max_side=220), 0.8 * inch, 0.8 * inch)
            except OSError:
                thumb = ""
        brand = f" ({row['brand_model']})" if row["brand_model"] else ""
        data.append([thumb, Paragraph(f"{row['item']}{brand}<br/><font color='#6B655C'>{row['category']}</font>", small),
                     Paragraph(row["room"], small), row["qty"],
                     money(row["rcv_low"]) if row["rcv_low"] != "" else "", money(row["rcv_mid"]) if row["rcv_mid"] != "" else "",
                     money(row["rcv_high"]) if row["rcv_high"] != "" else "",
                     Paragraph(f"{row['evidence']}<br/><font color='#6B655C'>{row['contributor']}</font>", small)])
    table = Table(data, colWidths=[0.9 * inch, 2.1 * inch, 1.0 * inch, 0.4 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch, 1.3 * inch],
                  repeatRows=1)
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, line),
                               ("ALIGN", (3, 0), (6, -1), "RIGHT"), ("TEXTCOLOR", (0, 0), (-1, -1), ink)]))
    story.append(table)

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
            story += [picture(annotate(photo["path"], boxes, max_side=1100), 5.2 * inch),
                      Paragraph(" · ".join(x for x in (photo_id, who, when) if x), note),
                      Paragraph(" · ".join(f"{n} {i['label']}" for n, i in enumerate(items, 1)), small),
                      Spacer(1, 0.25 * inch)]
        if len(by_photo) > max_appendix_photos:
            story.append(Paragraph(f"{len(by_photo) - max_appendix_photos} more evidence photos are available on request.", note))

    path = config.case_dir(case_id) / "inventory.pdf"
    SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                      title=f"Contents inventory: {case.get('name') or 'Home'}").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return path
