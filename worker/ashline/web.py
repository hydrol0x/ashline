"""The inventory page and the worker's HTTP endpoints (UX doc section 8, build spec 6.1).

    uvicorn ashline.web:app --port 8000

- `GET /p/<token>`: the family's private page. The link is long and unguessable; there is no
  login. The owner's link shows export and remove; a member's link does not.
- The page is rendered from the case store on every request, so it is always live.
- `POST /inbound`: a message from the messaging bridge; the replies are in the response.
- `POST /tick`: run what is due (photo digests, the daily review). With ASHLINE_BRIDGE_URL set
  the worker ticks by itself and posts the messages to the bridge's /outbound.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import ipaddress
import json
import math
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config, disaster
from .annotate import annotate, crop_item
from .meter import DISCLAIMER, meter
from .pricing import in_use, load_calibration, money, money_range, price_sources
from .rooms import tidy_room
from .store import Store

TEMPLATES = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                        autoescape=select_autoescape(["html"]))
MEMORY_SOURCES = ("voice",)
SOURCE_NAMES = {"photo": "family photo", "scan": "3D scan", "aerial": "aerial image", "voice": "from memory",
                "receipt": "receipt"}


def _ago(stamp: str | None, now: datetime | None = None) -> str:
    if not stamp:
        return "just now"
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return "recently"
    seconds = ((now or datetime.now(timezone.utc)) - then).total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def parse_price(value) -> float | None:
    """'$450', '1,200', '1.2k', '$1.5 million' -> a number of dollars; None if there is no number."""
    text = str(value or "").lower().replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:(k|thousand|million|mm|m)(?![a-z]))?", text)
    if not match:
        return None
    unit = match.group(2) or ""
    return float(match.group(1)) * (1000 if unit in ("k", "thousand") else 1_000_000 if unit else 1)


def read_amount(value, low: float, high: float) -> float | None:
    """A typed dollar amount from low to high, or None. A minus sign is refused, not dropped."""
    text = str(value if value is not None else "")
    if re.search(r"-\s*\$?\s*\d", text):
        return None
    amount = parse_price(text)
    if amount is None or not math.isfinite(amount) or not low <= amount <= high:
        return None
    return amount


async def json_body(request: Request) -> dict:
    """The request's JSON object; a 400 when the body is not one."""
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Something went wrong sending that. Try again.")
    return body


def _title(label: str) -> str:
    return label[:1].upper() + label[1:] if label else ""


def item_view(item: dict, sightings: list[dict], case: dict | None = None) -> dict:
    """What a card shows for one inventory line."""
    sources = [s["source"] for s in sightings]
    photo_like = sum(1 for s in sources if s in ("photo", "scan", "aerial"))
    qty = int(item.get("qty") or 1)
    low, high, mid = item.get("price_low"), item.get("price_high"), item.get("price_mid")
    price = money_range(low, high) if mid else ""
    contributors = sorted({("3D scan" if s["added_by"] == "scan" else s["added_by"])
                           for s in sightings if s.get("added_by")})
    paid = price_sources(item).get("paid") or {}
    bank = int(bool(paid.get("amount")))                     # a card purchase or a receipt is proof, whichever price is in use
    return {
        "id": item["item_id"],
        "title": _title(item["label"]),
        "room": tidy_room(item.get("room_label")) or "Unsorted",
        "category": item.get("category") or "other",
        "detail": item.get("detail") or "",
        "brand": item.get("brand_guess") or "",
        "qty": qty,
        "price": (f"{qty} × {price}" if qty > 1 and price else price),
        "value": float(mid or 0) * qty,
        "status": item["status"],
        "memory_only": bool(sources) and all(s in MEMORY_SOURCES for s in sources),
        "has_photo": any(s.get("photo_id") and s.get("bbox") for s in sightings),
        "badges": {"photo": photo_like, "memory": sources.count("voice"), "receipt": sources.count("receipt") + bank},
        "contributors": contributors,
        "created": item.get("created_at") or "",
        "needs_proof": photo_like == 0 and sources.count("receipt") + bank == 0,
        **claim_view(item, case or {}, qty),
        "price_from": BASIS_WORDS.get(in_use(item), "") if mid else "",     # which price the card shows
    }


def claim_view(item: dict, case: dict, qty: int) -> dict:
    """Which claim an item belongs to, and its depreciated value where that claim pays it."""
    cause = disaster.item_cause(case, item)
    key = disaster.claim_key(case, item)
    basis = next((b for k, _, b in disaster.claims(case) if k == key), "rcv")
    each = disaster.acv(item.get("price_mid"), item.get("category")) if basis == "acv" else None
    return {"cause": cause, "claim": key, "basis": basis,
            "acv": (each * qty) if each else None, "acv_text": money(each * qty) if each else "",
            "essential": disaster.is_essential(item)}


def claim_totals(case: dict, items: list[dict]) -> list[dict]:
    """One row per claim this disaster means, from the confirmed items."""
    rows = []
    for key, title, basis in disaster.claims(case):
        mine = [i for i in items if i["claim"] == key and i["status"] == "confirmed"]
        rows.append({"key": key, "title": title, "basis": basis, "items": len(mine),
                     "new": sum(i["value"] for i in mine), "acv": sum(i["acv"] or 0 for i in mine)})
    return rows



BASIS_WORDS = {"family": "yours", "paid": "paid", "online": "online", "estimate": "estimate"}


def price_stack(item: dict) -> list[dict]:
    """Every price the item has, strongest first: [{kind, title, detail, amount, in_use, links}].

    What the family typed in, what they paid (a card purchase), what it sells for online today,
    and the rough estimate. The one in use is the item's price; the others are shown beside it.
    """
    kept, using, qty = price_sources(item), in_use(item), int(item.get("qty") or 1)
    rows = []
    if using == "family":
        rows.append({"kind": "family", "title": "Your price", "detail": "Set by the family",
                     "amount": money(item.get("price_mid")), "links": []})
    paid = kept.get("paid")
    if paid and paid.get("amount"):
        proof = ("Order from" if paid.get("order") else "Receipt from" if paid.get("receipt") else "Card purchase at")
        where = f"{proof} {paid.get('merchant') or 'a store'}" + (f", {paid['date']}" if paid.get("date") else "")
        rows.append({"kind": "paid", "title": "What you paid", "detail": where + (f" · for all {qty}" if qty > 1 else ""),
                     "amount": f"${float(paid['amount']):,.2f}", "links": []})
    online = kept.get("online")
    if online and online.get("mid"):
        matches = [m for m in online.get("matches") or [] if isinstance(m, dict) and m.get("price")]
        links = []
        for match in matches[:4]:
            url = str(match.get("url") or "")
            links.append({"text": f"{money(match['price'])} at {match.get('retailer') or 'a retailer'}"
                                  + (f": {str(match['title'])[:70]}" if match.get("title") else ""),
                          "url": url if url.startswith(("https://", "http://")) else ""})
        count = len(matches) or 1
        rows.append({"kind": "online", "title": "Found online",
                     "detail": f"{count} listing{'s' if count != 1 else ''} for the same kind of thing",
                     "amount": money_range(online.get("low"), online.get("high")), "links": links})
    guess = kept.get("estimate") or ({"low": item.get("price_low"), "mid": item.get("price_mid"),
                                      "high": item.get("price_high"), "source": item.get("price_source")}
                                     if using == "estimate" else None)
    if guess and guess.get("mid"):
        rough = guess.get("source") == "category default"
        rows.append({"kind": "estimate", "title": "Estimate",
                     "detail": "A rough figure for this kind of item" if rough else "A rough allowance for buying it new today",
                     "amount": money_range(guess.get("low"), guess.get("high")), "links": []})
    for row in rows:
        row["in_use"] = row["kind"] == using
    return rows


def scan_link(store: Store, case_id: str, item: dict) -> str | None:
    """Where to walk to this thing in the home's 3D scan, when it was seen in one."""
    from . import layout
    if item.get("source") != "scan" or not item.get("photo_id"):
        return None
    sweep = layout.sweep_of(store.get_photo(case_id, item["photo_id"]))
    return layout.tour_url((store.get_home(case_id) or {}).get("sid"), sweep) if sweep else None


def shop_search(item: dict) -> list[dict]:
    """Search links for the same or a similar thing: [{text, url}]. A search, not a listing we found."""
    from urllib.parse import quote_plus
    words = " ".join(filter(None, [item.get("brand_guess"), re.sub(r"\(.*?\)|,.*$", "", item.get("label") or "")])).strip()
    if not words:
        return []
    query = quote_plus(words[:120])
    return [{"text": "Search Amazon for this", "url": f"https://www.amazon.com/s?k={query}"},
            {"text": "Search Google Shopping", "url": f"https://www.google.com/search?tbm=shop&q={query}"}]


def price_evidence(item: dict) -> list[dict]:
    """Where a looked-up price came from: [{text, url}], for the price in use and the others kept."""
    out = []
    for row in price_stack(item):
        if row["kind"] == "paid":
            out.append({"text": f"{row['amount']} · {row['detail']}", "url": ""})
        out += row["links"]
    return out[:5]


def mini_plan(case_id: str, store: Store, rooms: dict[str, int]) -> dict | None:
    """The room outlines of the home's plan, drawn small on the list page. None when there are no rooms.

    A room is tappable (has a name) only when cards carry that room name."""
    from . import layout
    try:
        plan = layout.build(case_id, store)
    except Exception as exc:                                  # the list must still render without a plan
        print(f"ashline: no small plan ({exc})")
        return None
    if not plan.get("rooms"):
        return None
    pad = max(plan["width"], plan["height"]) * 0.03
    return {"kind": plan["kind"],
            "box": f"{-pad:.2f} {-pad:.2f} {plan['width'] + 2 * pad:.2f} {plan['height'] + 2 * pad:.2f}",
            "rooms": [{"id": r["id"], "name": r["name"] if r["name"] in rooms else "",
                       "points": " ".join(f"{x:g},{y:g}" for x, y in r["points"])} for r in plan["rooms"]]}


def page_data(case_id: str, store: Store, role: str) -> dict:
    case = store.get_case(case_id) or {}
    everything = store.get_items(case_id)
    by_canonical: dict[int, list[dict]] = {}
    for row in everything:
        by_canonical.setdefault(row["cluster_id"] or row["item_id"], []).append(row)
    items, removed = [], []
    for row in everything:
        if (row["cluster_id"] or row["item_id"]) != row["item_id"]:
            continue
        view = item_view(row, by_canonical[row["item_id"]], case)
        view["shot_from"] = {"scan": "3D scan", "photo": "Family photo", "aerial": "Aerial image"}.get(
            row.get("source"), "") if view["has_photo"] else ""            # where the photo on the card came from
        paid = price_sources(row).get("paid") or {}
        view["proof"] = (("Card record" if paid.get("bank_id") or (row.get("price_source") or "").startswith("bank record")
                          and not (paid.get("receipt") or paid.get("order")) else "Receipt")
                         if view["badges"]["receipt"] else "")
        (removed if row["status"] == "rejected" else items).append(view)
    items.sort(key=lambda i: -i["value"])

    rooms: dict[str, int] = {}
    waiting: dict[str, int] = {}
    for item in items:
        rooms[item["room"]] = rooms.get(item["room"], 0) + 1
        if item["status"] == "suggested":
            waiting[item["room"]] = waiting.get(item["room"], 0) + 1
    m = meter(case_id, store)
    scale = max(m["target"] or 0, m["documented"] + m["pending"], 1)
    members = store.get_members(case_id)
    confirmed = [i for i in items if i["status"] == "confirmed"]
    proven = sum(1 for i in confirmed if not i["memory_only"] or i["badges"]["receipt"])
    latest = max([i["created"] for i in items] + [case.get("created_at") or ""], default="")
    return {
        "case": case,
        "name": case.get("name") or "Your home",
        "role": role,
        "items": items,
        "removed": removed if role == "owner" else [],
        "disaster": case.get("disaster") or disaster.FIRE,
        "claims": claim_totals(case, items),
        "show_essentials": case.get("insured") == "no",
        "essentials": sum(1 for i in items if i["essential"]),
        "confirmed": sum(1 for i in items if i["status"] == "confirmed"),
        "rooms": sorted(rooms.items(), key=lambda kv: -kv[1]),
        "meter": m,
        "money": money,
        "bar": {"backed": 100 * m["backed"] / scale, "memory": 100 * m["memory"] / scale,
                "pending": 100 * m["pending"] / scale},
        "updated": _ago(latest),
        "members": len(members),
        "member_names": [mb["name"].strip() for mb in members if (mb.get("name") or "").strip()],
        "room_rows": [{"name": r, "count": n, "waiting": waiting.get(r, 0)}
                      for r, n in sorted(rooms.items(), key=lambda kv: -kv[1])],
        "waiting": sum(waiting.values()),
        "counts": {"backed": proven, "memory": len(confirmed) - proven, "pending": sum(waiting.values())},
        "plan": mini_plan(case_id, store, rooms),
        "calibrated": load_calibration()["calibrated"],
        "disclaimer": DISCLAIMER,
    }


def create_app(store_factory=Store, lookup=None) -> FastAPI:
    app = FastAPI(title="Ashline", docs_url=None, redoc_url=None)

    def resolve(token: str) -> tuple[Store, dict, str]:
        """The case a page link belongs to, and whether it is the owner's link."""
        store = store_factory()
        for key, role in (("owner_token", "owner"), ("member_token", "member")):
            case = store.find_case(**{key: token})
            if case:
                return store, case, role
        raise HTTPException(status_code=404, detail="This link is not valid.")

    def owned_item(store: Store, case: dict, item_id: int) -> dict:
        try:
            item = store.get_item(case["case_id"], item_id)
        except OverflowError:                             # an id too big for the database is not an item
            item = None
        if not item:
            raise HTTPException(status_code=404)
        return item

    @app.get("/", response_class=HTMLResponse)
    def landing():
        number = (config.env("ASHLINE_NUMBER") or "").strip()
        digits = re.sub(r"[^\d+]", "", number)
        shown = number
        match = re.fullmatch(r"\+?1?(\d{3})(\d{3})(\d{4})", digits)
        if match:
            shown = f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
        return TEMPLATES.get_template("landing.html").render(number=digits, number_display=shown)

    def check_view(address: str) -> dict:
        """What /check shows for an address: only plain facts, never a scan id, listing id or coordinate."""
        from .costar.index import home_lookup
        from .costar.vexcel import in_geofence
        try:
            res = (lookup or home_lookup)(address=address)
        except Exception:                            # the index could not be fetched or read
            return {"state": "error"}
        if res.get("needs_unit"):
            street, _, rest = address.partition(",")  # the unit goes on the street line, where the parser looks
            units = [{"unit": u, "address": f"{street.strip()} Unit {u}" + (f",{rest}" if rest else "")}
                     for u in res.get("units") or []]
            return {"state": "unit", "units": units}
        if not res.get("found"):
            return {"state": "none"}
        captured = ""
        match = re.match(r"(\d{4})-(\d{2})", str(res.get("original_created") or ""))
        if match and 1 <= int(match.group(2)) <= 12:
            captured = datetime(int(match.group(1)), int(match.group(2)), 1).strftime("%B %Y")
        lat, lon = res.get("lat"), res.get("lon")
        aerial = None if lat is None or lon is None else bool(in_geofence(lat, lon))
        return {"state": "found", "captured": captured, "rooms": res.get("room_count"),
                "floors": res.get("floor_count"), "aerial": aerial}

    @app.get("/check", response_class=HTMLResponse)
    def check(address: str = ""):
        """Is there a 3D scan of this home from before? A form, and the answer on the same page."""
        address = address.strip()[:200]
        number = re.sub(r"[^\d+]", "", (config.env("ASHLINE_NUMBER") or "").strip())
        result = check_view(address) if address else None
        return TEMPLATES.get_template("check.html").render(address=address, result=result, number=number)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/p/{token}", response_class=HTMLResponse)
    def page(token: str):
        store, case, role = resolve(token)
        data = page_data(case["case_id"], store, role)
        return TEMPLATES.get_template("inventory.html").render(token=token, **data)

    @app.get("/p/{token}/plan", response_class=HTMLResponse)
    def plan(token: str):
        """A 2D plan of the home with a dot where each item was recorded."""
        from . import layout
        store, case, _ = resolve(token)
        return TEMPLATES.get_template("plan.html").render(
            token=token, name=case.get("name") or "Your home", layout=layout.build(case["case_id"], store))

    @app.get("/p/{token}/plan.json")
    def plan_json(token: str):
        from . import layout
        store, case, _ = resolve(token)
        return layout.build(case["case_id"], store)

    @app.get("/p/{token}/crop/{item_id}.jpg")
    def crop(token: str, item_id: int):
        store, case, _ = resolve(token)
        item = owned_item(store, case, item_id)
        photo = store.get_photo(case["case_id"], item["photo_id"]) if item.get("photo_id") else None
        if not photo:
            raise HTTPException(status_code=404)
        key = hashlib.sha1(f"{photo['path']}|{item['bbox']}".encode()).hexdigest()[:12]
        cached = config.case_dir(case["case_id"]) / "crops" / f"{key}.jpg"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            crop_item(photo["path"], item["bbox"]).save(cached, quality=85)
        return Response(cached.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/p/{token}/sighting/{item_id}.jpg")
    def sighting(token: str, item_id: int):
        """The full original photo with this one item's box drawn on it."""
        store, case, _ = resolve(token)
        item = owned_item(store, case, item_id)
        photo = store.get_photo(case["case_id"], item["photo_id"]) if item.get("photo_id") else None
        if not photo:
            raise HTTPException(status_code=404)
        buf = io.BytesIO()
        annotate(photo["path"], [{"n": 1, "bbox": item["bbox"], "status": "new"}], max_side=1400).save(
            buf, format="JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

    @app.get("/p/{token}/item/{item_id}.json")
    def item_detail(token: str, item_id: int):
        store, case, role = resolve(token)
        item = owned_item(store, case, item_id)
        sightings = store.get_sightings(case["case_id"], item_id)
        view = item_view(item, sightings, case)
        calibrated = load_calibration()["calibrated"]
        basis = {"set by family": "Price set by the family.",
                 "category default": "A rough figure for this kind of item. Set the price if you know it."}.get(
            item.get("price_source") or "", "Estimate of buying it new today, at a typical mid-range price.")
        source = item.get("price_source") or ""
        evidence = price_evidence(item)
        if source.startswith("bank record"):
            basis = "What the family paid, from their card purchases."
        elif source.startswith("receipt:"):
            basis = "What the family paid, from a receipt or order record they sent."
        elif source.startswith("web:"):
            basis = "Price of the closest match found online today."
        elif item.get("price_mid") and source != "set by family":
            basis += (" The range covers 90% of similar items in our test set." if calibrated
                      else " The range is a rough allowance for uncertainty.")
        where = None                                      # its room and spot on the plan, for the small locator
        try:
            from . import layout
            pin = next((p for p in layout.build(case["case_id"], store)["pins"] if p["id"] == item_id), None)
            if pin and (pin.get("room_id") is not None or pin.get("x") is not None):
                where = {"room_id": pin.get("room_id"), "x": pin.get("x"), "y": pin.get("y")}
        except Exception as exc:                          # the sheet must open without a plan
            print(f"ashline: no plan spot for item {item_id} ({exc})")
        return JSONResponse({
            "where": where,
            **view, "label": item["label"], "price_basis": basis, "price_evidence": evidence,
            "shop_search": shop_search(item), "tour": scan_link(store, case["case_id"], item),
            "price_sources": price_stack(item),
            "price_mid": item.get("price_mid"), "editable": role == "owner",
            "sightings": [{
                "id": s["item_id"], "source": SOURCE_NAMES.get(s["source"], s["source"]),
                "by": "3D scan" if s.get("added_by") == "scan" else s.get("added_by"),
                "date": s.get("evidence_date"), "label": s["label"],
                "image": f"/p/{token}/sighting/{s['item_id']}.jpg" if s.get("photo_id") and s.get("bbox") else None,
            } for s in sightings],
        })

    @app.post("/p/{token}/item/{item_id}")
    async def item_action(token: str, item_id: int, request: Request):
        store, case, role = resolve(token)
        if role != "owner":
            raise HTTPException(status_code=403, detail="Only the case owner can change items.")
        item = owned_item(store, case, item_id)
        body = await json_body(request)
        action, case_id = body.get("action"), case["case_id"]
        before = {k: item.get(k) for k in ("status", "label", "qty", "price_low", "price_mid", "price_high",
                                           "price_source", "price_ref", "cause")}
        if action in ("confirm", "remove", "restore"):
            status = {"confirm": "confirmed", "remove": "rejected", "restore": "suggested"}[action]
            store.set_item_status(case_id, item_id, status)
        elif action == "cause" and body.get("value") in ("flood", "wind"):
            store.update_item(case_id, item_id, cause=body["value"])
        elif action == "rename" and str(body.get("value") or "").strip():
            store.update_item(case_id, item_id, label=str(body["value"]).strip()[:120])
        elif action == "qty":
            typed = str(body.get("value") if body.get("value") is not None else "").strip()
            if not re.fullmatch(r"\d{1,3}", typed) or not 1 <= int(typed) <= 999:
                raise HTTPException(status_code=400, detail="Give a whole number from 1 to 999.")
            store.update_item(case_id, item_id, qty=int(typed))
        elif action == "price":
            value = read_amount(body.get("value"), 1, 10_000_000)
            if value is None:
                raise HTTPException(status_code=400, detail="Give a price from $1 to $10 million.")
            store.update_item(case_id, item_id, price_low=value, price_mid=value, price_high=value,
                              price_source="set by family")
        else:
            raise HTTPException(status_code=400, detail="Unknown action.")
        store.log_event(case_id, f"item.{action}", {"item_id": item_id, "before": before})
        return {"ok": True}

    @app.post("/p/{token}/coverage")
    async def coverage(token: str, request: Request):
        store, case, role = resolve(token)
        if role != "owner":
            raise HTTPException(status_code=403, detail="Only the case owner can set this.")
        body = await json_body(request)
        if body.get("skip"):
            store.set_case(case["case_id"], coverage_skipped=1)
        else:
            limit = read_amount(body.get("limit"), 1, 999_999_999)
            if limit is None:
                raise HTTPException(status_code=400, detail="We could not read that. Try a number like 120000 or 120k.")
            loss_date = str(body.get("loss_date") or "")[:10]
            if loss_date:
                try:
                    date.fromisoformat(loss_date)
                except ValueError:
                    raise HTTPException(status_code=400, detail="We could not read that date.")
            store.set_case(case["case_id"], coverage_limit=limit, coverage_skipped=0)
            if loss_date:
                store.set_case(case["case_id"], loss_date=loss_date)
        return {"ok": True}

    def preview_context(token: str, store: Store, case: dict) -> dict:
        from . import export as exporter
        m = meter(case["case_id"], store)
        rows = exporter.export_rows(case["case_id"], store)
        total = {"items": sum(int(r["qty"] or 1) for r in rows),
                 "low": sum(float(r["rcv_low"] or 0) * int(r["qty"] or 1) for r in rows),
                 "mid": sum(float(r["rcv_mid"] or 0) * int(r["qty"] or 1) for r in rows),
                 "high": sum(float(r["rcv_high"] or 0) * int(r["qty"] or 1) for r in rows)}
        groups = exporter.claim_groups(case, rows)
        return {"token": token, "name": case.get("name") or "Your home", "rows": rows,
                "claim_groups": groups if (len(groups) > 1 or groups[0]["basis"] == "acv") else [],
                "groups": exporter.category_totals(rows), "total": total, "pending": m.get("pending_items") or 0,
                "money": money, "money_range": money_range, "disclaimer": DISCLAIMER}

    @app.get("/p/{token}/view/{kind}", response_class=HTMLResponse)
    def preview(token: str, kind: str):
        store, case, role = resolve(token)
        if role != "owner":
            raise HTTPException(status_code=403, detail="Only the case owner can see the exports.")
        if kind not in ("sheet", "pdf"):
            raise HTTPException(status_code=404)
        return TEMPLATES.get_template(f"preview_{kind}.html").render(**preview_context(token, store, case))

    static_root = (Path(__file__).parent / "static").resolve()
    static_types = {".mjs": "text/javascript", ".js": "text/javascript", ".css": "text/css", ".png": "image/png",
                    ".jpg": "image/jpeg", ".woff2": "font/woff2", ".svg": "image/svg+xml"}

    # A short fingerprint of the static files. Every asset URL carries ?v=<it>, so a new deploy never
    # gets stuck behind a copy Cloudflare or a phone cached earlier (including a cached 404).
    digest = hashlib.sha1()
    for f in sorted(static_root.rglob("*")):
        if f.is_file() and f.suffix in static_types:
            digest.update(f.name.encode()); digest.update(f.read_bytes())
    asset_v = digest.hexdigest()[:10]
    TEMPLATES.globals["v"] = asset_v

    @app.get("/static/{path:path}")
    def static(path: str):
        target = (static_root / path).resolve()
        if static_root not in target.parents or not target.is_file() or target.suffix not in static_types:
            return Response('{"detail":"Not Found"}', status_code=404, media_type="application/json",
                            headers={"Cache-Control": "no-store"})
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        if target.suffix == ".css":                  # pass the fingerprint on to fonts and images the CSS loads
            css = re.sub(r"url\((/static/[^)?]+)\)", lambda m: f"url({m.group(1)}?v={asset_v})", target.read_text())
            return Response(css, media_type="text/css", headers=headers)
        return FileResponse(target, media_type=static_types[target.suffix], headers=headers)

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(static_root / "img" / "favicon.png", media_type="image/png")

    @app.get("/p/{token}/export.{kind}")
    def export(token: str, kind: str, inline: int = 0):
        store, case, role = resolve(token)
        if role != "owner":
            raise HTTPException(status_code=403, detail="Only the case owner can export.")
        from . import export as exporter
        # A header carries plain ASCII only: a home named in another script still downloads, under its real name.
        real = (case.get("name") or "inventory").strip() or "inventory"
        name = re.sub(r"[^a-z0-9]+", "-", real.lower()).strip("-") or "inventory"

        def named(kind_: str, how: str = "attachment") -> dict:
            from urllib.parse import quote
            return {"Content-Disposition": f"{how}; filename=\"{name}.{kind_}\"; filename*=UTF-8''{quote(real)}.{kind_}"}
        if kind == "csv":
            return Response(exporter.export_csv(case["case_id"], store).read_bytes(), media_type="text/csv",
                            headers=named("csv"))
        if kind == "xlsx":
            return Response(exporter.export_xlsx(case["case_id"], store).read_bytes(),
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            headers=named("xlsx"))
        if kind == "pdf":
            return Response(exporter.export_pdf(case["case_id"], store).read_bytes(), media_type="application/pdf",
                            headers=named("pdf", "inline" if inline else "attachment"))
        raise HTTPException(status_code=404)

    # ------------------------------------------------------------ messaging bridge (build spec 6.1)
    def engine():
        from .chat import Engine
        return Engine(store_factory())

    def check_secret(request: Request) -> None:
        # These two routes speak for any phone number, and a reply can carry a family's private link.
        # With a secret set, every call must carry it. With none, only this machine may call: anything
        # that came through a tunnel or from another machine is refused.
        secret = config.env("ASHLINE_INBOUND_SECRET")
        if secret:
            if not hmac.compare_digest(request.headers.get("x-ashline-secret") or "", secret):
                raise HTTPException(status_code=401, detail="Wrong or missing X-Ashline-Secret.")
            return
        forwarded = any(request.headers.get(h) for h in ("x-forwarded-for", "x-forwarded-host", "forwarded", "x-real-ip",
                                                           "cf-connecting-ip"))
        try:
            remote = not ipaddress.ip_address(request.client.host if request.client else "").is_loopback
        except ValueError:
            remote = False                                  # not a network address: a caller inside this process
        if forwarded or remote:
            raise HTTPException(status_code=401, detail="Set ASHLINE_INBOUND_SECRET to use this from another machine.")

    @app.post("/inbound")
    async def inbound(request: Request):
        """A message from the bridge. Replies come back in the response body.

        Body: {chat_id, sender_phone, sender_name?, text?, is_group?, attachments?: [{filename?, bytes_b64}]}.
        `chat_id` is the group chat's id, or the sender's phone for a 1:1 chat.
        """
        check_secret(request)
        body = await request.json()
        sender = str(body.get("sender_phone") or "").strip()
        if not sender:
            raise HTTPException(status_code=400, detail="sender_phone is required.")
        from .chat import Inbound
        files = []
        for n, attachment in enumerate(body.get("attachments") or []):
            try:
                files.append((attachment.get("filename") or f"attachment-{n}",
                              base64.b64decode(attachment.get("bytes_b64") or "")))
            except (ValueError, TypeError):
                continue
        message = Inbound(chat_id=str(body.get("chat_id") or body.get("case_id") or sender), sender=sender,
                          text=str(body.get("text") or ""), attachments=files, is_group=bool(body.get("is_group")),
                          sender_name=body.get("sender_name"))
        replies = await asyncio.to_thread(lambda: engine().handle(message))
        return {"messages": [wire(m) for m in replies]}

    @app.post("/tick")
    async def tick(request: Request):
        """Run what is due (photo digests, the daily review) and return the messages to send."""
        check_secret(request)
        replies = await asyncio.to_thread(lambda: engine().tick())
        return {"messages": [wire(m) for m in replies]}

    return app


def wire(message) -> dict:
    """An outbound message as JSON for the bridge: {to, text, image_b64?, files?}."""
    out = {"to": message.chat_id, "text": message.text}
    if message.image and Path(message.image).exists():
        out["image_b64"] = base64.b64encode(Path(message.image).read_bytes()).decode()
    files = [{"name": Path(f).name, "bytes_b64": base64.b64encode(Path(f).read_bytes()).decode()}
             for f in message.files if Path(f).exists()]
    if files:
        out["files"] = files
    return out


async def _ticker(app: FastAPI) -> None:
    """With a bridge configured, push what `tick` produces to its /outbound endpoint."""
    import requests
    bridge = (config.env("ASHLINE_BRIDGE_URL") or "").rstrip("/")
    every = float(config.env("ASHLINE_TICK_SECONDS") or 10)
    while True:
        await asyncio.sleep(every)
        try:
            from .chat import Engine
            for message in await asyncio.to_thread(lambda: Engine().tick()):
                await asyncio.to_thread(lambda m=message: requests.post(f"{bridge}/outbound", json=wire(m), timeout=60))
        except Exception as exc:                              # keep ticking: one bad case must not stop the rest
            print(f"ashline ticker: {exc}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_ticker(app)) if config.env("ASHLINE_BRIDGE_URL") else None
    yield
    if task:
        task.cancel()


app = create_app()
app.router.lifespan_context = _lifespan
