"""Looking a price up online (an agent searches retailers for the item or a similar one).

Claude's web search tool does the searching and returns the closest matches with price,
retailer and link. A found price replaces the model's estimate for that item: the closest
match becomes the mid value, the spread of the matches becomes the range, and the links are
kept as evidence. It is used for the most valuable items and on request, not for every line:
each lookup is a search and takes a few seconds.

Retailer sites are not scraped. Nothing here inflates a value: when matches disagree the
closest match is used, and used or refurbished listings are excluded by the prompt.
"""
from __future__ import annotations

import json
import re
import threading

from . import config
from .pricing import kept_price, money, nice
from .store import Store

SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}
MAX_RESUMES = 3

PROMPT = """Find what this household item costs to buy NEW today from US retailers, for a home contents
insurance inventory.

Item: {description}
Brand: {brand}

Search for it. Prefer the same brand and model. If the exact product is not sold any more or no brand is
known, use the closest similar items of ordinary mid-range quality; that is fine and expected. Exclude
used, refurbished, rental and wholesale-lot listings.

Then reply with ONLY a JSON object, no other text:
{{"matches": [{{"title": "<product name as listed>", "price": <price in US dollars, a number>,
"retailer": "<store name>", "url": "<link to the listing>"}}]}}
Give up to 4 matches, the closest match first. If you found nothing suitable, reply {{"matches": []}}.
Never invent a price or a link: every match must come from a search result."""


WITH_PHOTO = """

The photo above shows the item as it stood in the home. Use it to judge the style, size, material
and quality, and prefer listings that look like the same kind of thing. If a brand or model is
readable in the photo, search for that."""


class ClaudeSearcher:
    def __init__(self, model: str | None = None, client=None):
        config.load_env()
        self.model = model or config.env("ASHLINE_SEARCH_MODEL") or config.env("ASHLINE_CLAUDE_MODEL", "claude-opus-5-5")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def search(self, description: str, brand: str | None = None, image=None) -> list[dict]:
        """Closest matches first: [{title, price, retailer, url}]. Empty when nothing was found.

        `image` is an optional picture of the item itself (a PIL image): the model then looks for
        listings that look like it, not merely ones with the same name.
        """
        text = PROMPT.format(description=description, brand=brand or "unknown")
        content = text
        if image is not None:
            import base64
            import io
            small = image.convert("RGB")
            small.thumbnail((768, 768))
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=85)
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(buf.getvalue()).decode()}},
                       {"type": "text", "text": text + WITH_PHOTO}]
        messages = [{"role": "user", "content": content}]
        response = None
        for _ in range(MAX_RESUMES + 1):
            response = self.client.messages.create(model=self.model, max_tokens=16000, tools=[SEARCH_TOOL],
                                                   messages=messages)
            if response.stop_reason != "pause_turn":
                break
            # A long search paused: send the turn back unchanged and the server resumes it.
            messages = [messages[0], {"role": "assistant", "content": response.content}]
        if response is None or response.stop_reason in ("refusal", "max_tokens", "pause_turn"):
            return []
        text = "".join(getattr(block, "text", "") for block in response.content if getattr(block, "type", "") == "text")
        return parse_matches(text)


def parse_matches(text: str) -> list[dict]:
    """The matches in a reply, keeping only complete, plausible ones."""
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        return []
    try:
        raw = json.loads(found.group(0)).get("matches", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    matches = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            price = float(str(entry.get("price", "")).replace("$", "").replace(",", ""))
        except ValueError:
            continue
        url = str(entry.get("url") or "")
        if price <= 0 or not url.startswith(("http://", "https://")):
            continue                                    # no link, no evidence
        matches.append({"title": str(entry.get("title") or "")[:160], "price": price,
                        "retailer": str(entry.get("retailer") or "")[:60] or "online retailer", "url": url[:500]})
    return matches[:4]


def get_searcher():
    """A searcher when an Anthropic key is set, else None."""
    if (config.env("ASHLINE_PRICE_SEARCH") or "").lower() == "off" or not config.env("ANTHROPIC_API_KEY"):
        return None
    return ClaudeSearcher()


def find_price(description: str, brand: str | None = None, searcher=None, image=None) -> dict:
    """{found, price, low, high, retailer, url, matches}. `price` is the closest match."""
    searcher = searcher or get_searcher()
    if searcher is None:
        return {"found": False, "reason": "Online price lookup is not set up (it needs ANTHROPIC_API_KEY)."}
    try:
        matches = searcher.search(description, brand, image=image) if image is not None else searcher.search(description, brand)
    except Exception as exc:
        return {"found": False, "reason": f"The search failed: {exc}"}
    if not matches:
        return {"found": False, "reason": "No suitable listing was found."}
    prices = [m["price"] for m in matches]
    return {"found": True, "price": matches[0]["price"], "low": min(prices), "high": max(prices),
            "retailer": matches[0]["retailer"], "url": matches[0]["url"], "matches": matches}


def apply_price(store: Store, case_id: str, item_id: int, found: dict, member_id: int | None = None,
                log: bool = True) -> str:
    """Put a price found online on an item, keeping the links as evidence. Can be undone, unless
    `log` is False (a lookup nobody asked for must not become the thing "undo" reverses)."""
    item = store.get_item(case_id, item_id)
    if not item:
        return "That item is not on the list."
    if item.get("price_source") == "set by family":
        return "The family set this price themselves, so it was left as it is."
    if (item.get("price_source") or "").startswith("bank record"):
        return "This price is what the family paid, from their card purchases, so it was left as it is."
    try:
        price = float(found["price"])
        low, high = float(found.get("low") or price), float(found.get("high") or price)
    except (KeyError, TypeError, ValueError):
        return "No price was recorded: the lookup result was incomplete."
    if price <= 0:
        return "No price was recorded: the lookup result was incomplete."
    low, high = min(low, price), max(high, price)
    retailer = str(found.get("retailer") or "online retailer")[:60]
    refs = found.get("matches") or [{"title": item["label"], "price": price, "retailer": retailer,
                                     "url": str(found.get("url") or "")}]
    if log:
        store.log_event(case_id, "item.price", {"item_id": item_id, "before": {k: item.get(k) for k in (
            "status", "label", "qty", "price_low", "price_mid", "price_high", "price_source")}}, member_id)
    store.update_item(case_id, item_id, price_low=nice(low), price_mid=nice(price), price_high=nice(high),
                      price_source=f"web: {retailer}", price_ref=json.dumps(refs[:4]))
    span = f", range {money(nice(low))} to {money(nice(high))}" if nice(high) > nice(low) else ""
    return f"Recorded {money(nice(price))} for the {item['label']} from {retailer}{span}."


def price_check(case_id: str, store: Store | None = None, top: int = 20, searcher=None, report=None) -> dict:
    """Look up the `top` most valuable items of a case online. Returns {checked, found, skipped}."""
    store = store or Store()
    searcher = searcher or get_searcher()
    if searcher is None:
        return {"checked": 0, "found": 0, "reason": "Online price lookup needs ANTHROPIC_API_KEY."}
    items = [i for i in store.get_items(case_id, canonical_only=True)
             if i["status"] != "rejected" and not kept_price(i)]
    items.sort(key=lambda i: -((i.get("price_mid") or 0) * (i.get("qty") or 1)))
    checked = found = 0
    for item in items[:max(0, top)]:
        checked += 1
        result = find_price(item["label"], item.get("brand_guess"), searcher, image=item_picture(store, case_id, item))
        if result.get("found"):
            found += 1
            line = apply_price(store, case_id, item["item_id"], result)
        else:
            line = f"No listing found for the {item['label']}; the estimate stays."
        if report:
            report(line)
    return {"checked": checked, "found": found}


# ------------------------------------------------- without being asked
AUTO_TOP = 6                    # items looked up after each batch of new items, most valuable first
AUTO_MIN = 150.0                # only items estimated at this much or more (for the whole quantity)
AUTO_TRUST = 3.0                # an unasked lookup is used only within this factor of the estimate, either way
_news_lock = threading.Lock()


def item_picture(store: Store, case_id: str, item: dict):
    """The crop of the photo that shows the item, or None when there is no photo of it."""
    try:
        bbox = json.loads(item["bbox"]) if isinstance(item.get("bbox"), str) else item.get("bbox")
        photo = store.get_photo(case_id, item["photo_id"]) if item.get("photo_id") else None
        if not photo or not bbox:
            return None
        from .annotate import crop_item
        return crop_item(photo["path"], bbox)
    except Exception:
        return None


def auto_enabled() -> bool:
    return (config.env("ASHLINE_AUTO_PRICE") or "on").lower() not in ("off", "0", "no", "false")


def auto_price(case_id: str, store: Store, item_ids: list[int] | None = None, searcher=None,
               top: int | None = None, minimum: float | None = None, chat_id: str | None = None) -> list[dict]:
    """Look up the bigger new items online without being asked, and record what was found.

    Considers `item_ids` (or every line), leaves out prices the family set, paid or already
    looked up, and anything estimated under `minimum`; takes the `top` most valuable. Lookups
    run a few at a time. A found price far from the estimate (more than three times, or under a
    third) is not used: with nobody watching, that is more likely the wrong product than a
    better number. Each result is also kept as news for the chat to mention once.
    Returns [{item_id, label, price, retailer, before}].
    """
    searcher = searcher or get_searcher()
    if searcher is None:
        return []
    top = int(config.env("ASHLINE_AUTO_PRICE_TOP") or AUTO_TOP) if top is None else top
    minimum = float(config.env("ASHLINE_AUTO_PRICE_MIN") or AUTO_MIN) if minimum is None else minimum
    wanted = set(item_ids) if item_ids is not None else None
    items = [i for i in store.get_items(case_id, canonical_only=True)
             if i["status"] != "rejected" and not kept_price(i) and (wanted is None or i["item_id"] in wanted)
             and (i.get("price_mid") or 0) * (i.get("qty") or 1) >= minimum]
    items.sort(key=lambda i: -((i.get("price_mid") or 0) * (i.get("qty") or 1)))
    items = items[:max(0, top)]
    if not items:
        return []
    pictures = {i["item_id"]: item_picture(store, case_id, i) for i in items}

    def look(item: dict) -> dict:
        return find_price(item["label"], item.get("brand_guess"), searcher, image=pictures[item["item_id"]])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        found = list(pool.map(look, items))
    results = []
    for item, result in zip(items, found):
        if not result.get("found"):
            continue
        estimate = float(item.get("price_mid") or 0)
        if estimate and not estimate / AUTO_TRUST <= float(result["price"]) <= estimate * AUTO_TRUST:
            continue        # far from the estimate: likely the wrong product, so nobody's price changes unasked
        line = apply_price(store, case_id, item["item_id"], result, log=False)
        if line.startswith("Recorded"):
            results.append({"item_id": item["item_id"], "label": item["label"], "price": nice(result["price"]),
                            "retailer": result.get("retailer") or "an online retailer", "before": item.get("price_mid"),
                            "chat_id": chat_id})
    if results:
        add_news(case_id, results)
    return results


def _news_path(case_id: str):
    return config.case_dir(case_id) / "price_news.json"


def add_news(case_id: str, results: list[dict]) -> None:
    with _news_lock:
        path = _news_path(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        waiting = json.loads(path.read_text()) if path.exists() else []
        path.write_text(json.dumps(waiting + results))


def take_news(case_id: str) -> list[dict]:
    """Looked-up prices not yet mentioned in the chat. Each is returned once."""
    with _news_lock:
        path = _news_path(case_id)
        if not path.exists():
            return []
        try:
            waiting = json.loads(path.read_text())
        except ValueError:
            waiting = []
        path.unlink()
        return waiting
