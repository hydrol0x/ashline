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
import io
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config
from .annotate import annotate, crop_item
from .meter import DISCLAIMER, meter
from .pricing import load_calibration, money, money_range
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
    """'$450', '1,200', '1.2k', '450 dollars' -> a number of dollars; None if there is no number."""
    text = str(value or "").lower().replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(k)?", text)
    if not match:
        return None
    return float(match.group(1)) * (1000 if match.group(2) else 1)


def _title(label: str) -> str:
    return label[:1].upper() + label[1:] if label else ""


def item_view(item: dict, sightings: list[dict]) -> dict:
    """What a card shows for one inventory line."""
    sources = [s["source"] for s in sightings]
    photo_like = sum(1 for s in sources if s in ("photo", "scan", "aerial"))
    qty = int(item.get("qty") or 1)
    low, high, mid = item.get("price_low"), item.get("price_high"), item.get("price_mid")
    price = money_range(low, high) if mid else ""
    contributors = sorted({("3D scan" if s["added_by"] == "scan" else s["added_by"])
                           for s in sightings if s.get("added_by")})
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
        "badges": {"photo": photo_like, "memory": sources.count("voice"), "receipt": sources.count("receipt")},
        "contributors": contributors,
        "created": item.get("created_at") or "",
        "needs_proof": photo_like == 0 and sources.count("receipt") == 0,
    }


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
        (removed if row["status"] == "rejected" else items).append(item_view(row, by_canonical[row["item_id"]]))
    items.sort(key=lambda i: -i["value"])

    rooms: dict[str, int] = {}
    for item in items:
        rooms[item["room"]] = rooms.get(item["room"], 0) + 1
    m = meter(case_id, store)
    scale = max(m["target"] or 0, m["documented"] + m["pending"], 1)
    members = store.get_members(case_id)
    latest = max([i["created"] for i in items] + [case.get("created_at") or ""], default="")
    return {
        "case": case,
        "name": case.get("name") or "Your home",
        "role": role,
        "items": items,
        "removed": removed if role == "owner" else [],
        "confirmed": sum(1 for i in items if i["status"] == "confirmed"),
        "rooms": sorted(rooms.items(), key=lambda kv: -kv[1]),
        "meter": m,
        "money": money,
        "bar": {"backed": 100 * m["backed"] / scale, "memory": 100 * m["memory"] / scale,
                "pending": 100 * m["pending"] / scale},
        "updated": _ago(latest),
        "members": len(members),
        "calibrated": load_calibration()["calibrated"],
        "disclaimer": DISCLAIMER,
    }


def create_app(store_factory=Store) -> FastAPI:
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
        item = store.get_item(case["case_id"], item_id)
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

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/p/{token}", response_class=HTMLResponse)
    def page(token: str):
        store, case, role = resolve(token)
        data = page_data(case["case_id"], store, role)
        return TEMPLATES.get_template("inventory.html").render(token=token, **data)

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
        view = item_view(item, sightings)
        calibrated = load_calibration()["calibrated"]
        basis = {"set by family": "Price set by the family.",
                 "category default": "A rough figure for this kind of item. Set the price if you know it."}.get(
            item.get("price_source") or "", "Estimate of buying it new today, at a typical mid-range price.")
        if item.get("price_mid") and item.get("price_source") != "set by family":
            basis += (" The range covers 90% of similar items in our test set." if calibrated
                      else " The range is a rough allowance for uncertainty.")
        return JSONResponse({
            **view, "label": item["label"], "price_basis": basis,
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
        body = await request.json()
        action, case_id = body.get("action"), case["case_id"]
        before = {k: item.get(k) for k in ("status", "label", "qty", "price_low", "price_mid", "price_high", "price_source")}
        if action in ("confirm", "remove", "restore"):
            status = {"confirm": "confirmed", "remove": "rejected", "restore": "suggested"}[action]
            store.set_item_status(case_id, item_id, status)
        elif action == "rename" and str(body.get("value") or "").strip():
            store.update_item(case_id, item_id, label=str(body["value"]).strip()[:120])
        elif action == "qty":
            digits = re.sub(r"[^\d]", "", str(body.get("value") or ""))
            if not digits:
                raise HTTPException(status_code=400, detail="Give a number of items.")
            store.update_item(case_id, item_id, qty=max(1, min(999, int(digits))))
        elif action == "price":
            value = parse_price(body.get("value"))
            if not value or value <= 0:
                raise HTTPException(status_code=400, detail="Give a price above zero.")
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
        body = await request.json()
        if body.get("skip"):
            store.set_case(case["case_id"], coverage_skipped=1)
        else:
            limit = float(str(body.get("limit") or "0").replace(",", "").replace("$", "") or 0)
            if limit <= 0:
                raise HTTPException(status_code=400, detail="Give the contents coverage limit in dollars.")
            store.set_case(case["case_id"], coverage_limit=limit, coverage_skipped=0)
            if body.get("loss_date"):
                store.set_case(case["case_id"], loss_date=str(body["loss_date"])[:10])
        return {"ok": True}

    @app.get("/p/{token}/export.{kind}")
    def export(token: str, kind: str):
        store, case, role = resolve(token)
        if role != "owner":
            raise HTTPException(status_code=403, detail="Only the case owner can export.")
        from . import export as exporter
        name = (case.get("name") or "inventory").replace(" ", "-").lower()
        if kind == "csv":
            return Response(exporter.export_csv(case["case_id"], store).read_bytes(), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
        if kind == "xlsx":
            return Response(exporter.export_xlsx(case["case_id"], store).read_bytes(),
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})
        if kind == "pdf":
            return Response(exporter.export_pdf(case["case_id"], store).read_bytes(), media_type="application/pdf",
                            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})
        raise HTTPException(status_code=404)

    # ------------------------------------------------------------ messaging bridge (build spec 6.1)
    def engine():
        from .chat import Engine
        return Engine(store_factory())

    def check_secret(request: Request) -> None:
        secret = config.env("ASHLINE_INBOUND_SECRET")
        if secret and request.headers.get("x-ashline-secret") != secret:
            raise HTTPException(status_code=401, detail="Wrong or missing X-Ashline-Secret.")

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
