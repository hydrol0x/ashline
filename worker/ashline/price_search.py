"""Looking a price up online (an agent searches retailers for the item or a similar one).

A lookup runs in four stages (`ClaudeSearcher.lookup`):
  LOOK    from the photo crop and the label, the model writes a short spec of the thing: a
          shopper's product name, material, size class, style, quality tier, any readable brand,
          one thing or a set, and 2 or 3 queries a shopper would type (structured output).
  SEARCH  Claude's web search tool runs those queries and lists up to 8 candidate listings
          with title, price, retailer, link and how many units each sells.
  VERIFY  a separate call, which sees the crop, the spec and the candidates but not the search,
          marks each listing keep or drop: same kind of thing and size, new, a US retailer,
          and a price for one unit (a set is divided by its count).
  PRICE   from the kept listings only: the median is the mid (the lower middle one for an even
          count), the lowest and highest kept are the range.
If a stage fails the older single call (`ClaudeSearcher.search`) is used instead.

A found price replaces the model's estimate for that item, with the links kept as evidence.
Retailer sites are not scraped. Nothing here inflates a value: when listings disagree the
lower middle one is used.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from . import config
from .pricing import in_use, kept_price, money, nice, record_price
from .store import Store

SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}     # the single-call lookup
STAGED_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
MAX_RESUMES = 3
MAX_CANDIDATES = 8
EFFORT = "medium"

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

    # ---- the lookup in stages
    def _create(self, **request):
        from .detect_claude import FALLBACK_BETA, FALLBACK_MODELS
        if self.model.startswith(FALLBACK_MODELS):
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **request)
        return self.client.messages.create(**request)

    def _structured(self, content, schema: dict) -> dict:
        response = self._create(model=self.model, max_tokens=16000,
                                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
                                messages=[{"role": "user", "content": content}])
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"stopped early ({response.stop_reason})")
        return json.loads(_text(response))

    def look(self, description: str, brand: str | None = None, image=None) -> dict:
        """Stage a: the spec of the thing, as a shopper would search for it."""
        what = "the photo above and the label" if image is not None else "the label"
        text = LOOK_PROMPT.format(description=description, brand=brand or "unknown", what=what)
        content = [_photo_block(image), {"type": "text", "text": text}] if image is not None else text
        return clean_spec(self._structured(content, SPEC_SCHEMA), description)

    def gather(self, spec: dict) -> list[dict]:
        """Stage b: candidate listings from the web search, each one from a site the search returned."""
        queries = "\n".join(f"- {q}" for q in spec["queries"])
        messages = [{"role": "user", "content": SEARCH_PROMPT.format(spec=spec_line(spec), queries=queries)}]
        response, hosts = None, set()
        for _ in range(MAX_RESUMES + 1):
            response = self._create(model=self.model, max_tokens=16000, tools=[STAGED_SEARCH_TOOL],
                                    output_config={"effort": EFFORT}, messages=messages)
            hosts |= result_hosts(response.content)
            if response.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": response.content}]
        if response is None or response.stop_reason in ("refusal", "max_tokens", "pause_turn"):
            raise RuntimeError(f"search stopped ({getattr(response, 'stop_reason', None)})")
        found = parse_candidates(_text(response))
        if hosts:                                   # a link to a site the search never returned was made up
            found = [c for c in found if _host(c["url"]) in hosts]
        return found

    def verify(self, spec: dict, candidates: list[dict], image=None) -> list[dict]:
        """Stage c: an independent check of each candidate. Returns one verdict per candidate
        {keep, reason, units}; a candidate the check said nothing about is dropped."""
        listed = "\n".join(f"{n}. {c['title']} | ${c['price']:,.2f} | {c['retailer']} | {c['url']} | "
                           f"listed as {c['units']} unit{'s' if c['units'] != 1 else ''}"
                           for n, c in enumerate(candidates))
        what = "The photo above shows the item as it stood in the home." if image is not None else ""
        text = VERIFY_PROMPT.format(what=what, spec=spec_line(spec), candidates=listed)
        content = [_photo_block(image), {"type": "text", "text": text}] if image is not None else text
        raw = self._structured(content, VERIFY_SCHEMA).get("verdicts") or []
        verdicts = [{"keep": False, "reason": "not checked", "units": c["units"]} for c in candidates]
        for v in raw if isinstance(raw, list) else []:
            try:
                n = int(v["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= n < len(candidates):
                try:
                    units = max(1, int(v.get("units") or candidates[n]["units"]))
                except (TypeError, ValueError):
                    units = candidates[n]["units"]
                verdicts[n] = {"keep": v.get("keep") is True, "reason": str(v.get("reason") or "")[:160], "units": units}
        return verdicts

    def lookup(self, description: str, brand: str | None = None, image=None) -> dict | None:
        """The lookup in stages: {matches, spec, queries, checked, dropped}, `matches` being the kept
        listings priced per unit. None when a stage failed (the caller then uses `search`)."""
        try:
            spec = self.look(description, brand, image)
            if not spec["enough_to_price"]:
                return judge(spec, [], [])              # the part that sets the price cannot be seen: no search
            candidates = self.gather(spec)
            verdicts = self.verify(spec, candidates, image) if candidates else []
        except Exception:
            return None
        return judge(spec, candidates, verdicts)


def judge(spec: dict, candidates: list[dict], verdicts: list[dict]) -> dict:
    """The kept listings priced for one unit, with what was looked at."""
    kept = []
    for c, v in zip(candidates, verdicts):
        if not v.get("keep"):
            continue
        units = max(1, int(v.get("units") or 1))
        match = {"title": c["title"], "price": round(c["price"] / units, 2), "retailer": c["retailer"], "url": c["url"]}
        if units > 1:
            match["per_unit"] = f"a set of {units} for ${c['price']:,.0f}, divided by {units}"
        kept.append(match)
    return {"matches": kept, "spec": {k: v for k, v in spec.items() if k != "queries"}, "queries": spec["queries"],
            "checked": len(candidates), "dropped": len(candidates) - len(kept)}


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


# ------------------------------------------------- the lookup in stages
TIERS = ("budget", "mid", "premium")

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"}, "size_class": {"type": "string"}, "material": {"type": "string"},
        "color": {"type": "string"}, "style": {"type": "string"},
        "quality_tier": {"type": "string", "enum": list(TIERS)},
        "brand": {"type": "string"}, "model": {"type": "string"}, "brand_read": {"type": "boolean"},
        "is_set": {"type": "boolean"}, "set_count": {"type": "integer"},
        "cannot_see": {"type": "string"}, "enough_to_price": {"type": "boolean"},
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "size_class", "material", "color", "style", "quality_tier", "brand", "model", "brand_read",
                 "is_set", "set_count", "cannot_see", "enough_to_price", "queries"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "keep": {"type": "boolean"}, "reason": {"type": "string"},
                       "units": {"type": "integer"}},
        "required": ["index", "keep", "reason", "units"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False,
}

LOOK_PROMPT = """You are helping price a household item for a home contents insurance inventory.

Item as listed: {description}
Brand noted: {brand}

Using {what}, write what a shopper would need to buy the same thing new. Trust the photo over the
label: if the label says "sectional" and the photo shows a 3-seat sofa, it is a 3-seat sofa.
- kind: the item type as a product name ("dining chair", "3-seat sofa", "coffee table"), not a description
- size_class: the size class with an estimate ("5x7 rug", "1.1 cu ft", "queen", "80-inch 3-seat",
  "55 inch"), or "" if size does not matter for this thing
- material, color, style: short, "" when you cannot tell
- quality_tier: budget, mid or premium, judged from the build and finish you can see; mid when unsure
- brand and model: only what you can actually read in the photo, or the brand noted above. Never guess
  one from the look. brand_read is true only when the brand was read or noted, not guessed.
- is_set and set_count: true with the count only when the line itself is a set (a set of 6 glasses);
  one chair is not a set. set_count is 1 when it is not a set.
- cannot_see: what you cannot see that matters for the price (a bed frame hidden by bedding), or ""
- enough_to_price: false when the part that sets the price cannot be seen, so a search would only guess
- queries: 2 or 3 searches a shopper would type, short: type + size + material + color, for example
  "black metal frame dining chair wood seat" or "gray shag area rug 5x7". No long feature lists. Add the
  brand only when it was read."""

SEARCH_PROMPT = """Find what this household item costs to buy NEW today from US retailers.

What it is: {spec}

Search with these queries (adjust them if they find nothing useful):
{queries}

Return up to 8 listings for new items from US retailers. For each give the title as listed, the price
as listed now in US dollars (not the "was" price), the retailer, the link, and `units`: the pack size,
how many items the listing sells (1 for a single item, 2 for a set of 2 chairs). Copy prices and links
exactly from the results; never invent one. Skip used, refurbished, rental, parts and accessories.

Then reply with ONLY a JSON object, no other text:
{{"candidates": [{{"title": "...", "price": <number>, "retailer": "...", "url": "...", "units": <number>}}]}}
If you found nothing suitable, reply {{"candidates": []}}."""

VERIFY_PROMPT = """You are checking listings found online for a household item in a home contents
insurance inventory. {what}

The item: {spec}

Listings found:
{candidates}

For each listing decide keep or drop, with a short reason. Keep only listings that are the same item
type and size class and the same quality tier, new, in stock, and sold in US dollars by a store that
sells in the US. Drop:
- a different shape or size class (round when the item is rectangular, a sectional for a 3-seat sofa)
- parts or accessories sold alone (a glass carafe without the coffee maker)
- used, refurbished, open box, rental or out of stock
- designer or premium pieces when the item is budget or mid
- stores pricing in another currency
Give `units`, the pack size: how many items the listing sells (1 for a single item, 2 for a set of 2
chairs), so the price can be divided per item. If the item itself is a set, one whole set counts as 1.
Give one verdict per listing, using its index."""


def _photo_block(image) -> dict:
    import base64
    import io
    small = image.convert("RGB")
    small.thumbnail((768, 768))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def _text(response) -> str:
    return "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")


def spec_line(spec: dict) -> str:
    """The spec as one line a person or a model can read."""
    parts = [spec.get("brand"), spec.get("model"), spec.get("kind")]
    extra = [spec.get("size_class"), spec.get("material"), spec.get("color"), spec.get("style"),
             f"{spec.get('quality_tier') or 'mid'} quality"]
    if spec.get("is_set") and int(spec.get("set_count") or 1) > 1:
        extra.append(f"a set of {int(spec['set_count'])}")
    return " ".join(p for p in parts if p) + " (" + ", ".join(e for e in extra if e) + ")"


def clean_spec(raw: dict, description: str) -> dict:
    """A spec with every field present and sensible. Raises ValueError when it is unusable."""
    if not isinstance(raw, dict):
        raise ValueError("no spec")
    spec = {k: str(raw.get(k) or "").strip()[:80]
            for k in ("kind", "size_class", "material", "color", "style", "brand", "model", "cannot_see")}
    spec["kind"] = spec["kind"] or description[:80]
    spec["quality_tier"] = raw.get("quality_tier") if raw.get("quality_tier") in TIERS else "mid"
    spec["brand_read"] = bool(raw.get("brand_read")) and bool(spec["brand"])
    spec["enough_to_price"] = raw.get("enough_to_price") is not False
    spec["is_set"] = bool(raw.get("is_set"))
    try:
        spec["set_count"] = max(1, int(raw.get("set_count") or 1))
    except (TypeError, ValueError):
        spec["set_count"] = 1
    queries = [str(q).strip()[:120] for q in raw.get("queries") or [] if str(q).strip()]
    if not queries:
        raise ValueError("no queries")
    spec["queries"] = queries[:3]
    return spec


def parse_candidates(text: str) -> list[dict]:
    """The candidate listings in a search reply: complete ones with a price and a link."""
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        return []
    try:
        raw = json.loads(found.group(0)).get("candidates", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            price = float(str(entry.get("price", "")).replace("$", "").replace(",", ""))
            units = max(1, int(float(entry.get("units") or 1)))
        except (TypeError, ValueError):
            continue
        url = str(entry.get("url") or "")
        if price <= 0 or not url.startswith(("http://", "https://")):
            continue                                    # no link, no evidence
        out.append({"title": str(entry.get("title") or "")[:160], "price": price, "units": units,
                    "retailer": str(entry.get("retailer") or "")[:60] or "online retailer", "url": url[:500]})
    return out[:MAX_CANDIDATES]


def _host(url: str) -> str:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def result_hosts(content) -> set[str]:
    """The sites the web search actually returned, read from its result blocks."""
    hosts = set()
    for block in content or []:
        if getattr(block, "type", "") != "web_search_tool_result":
            continue
        results = getattr(block, "content", None)
        for r in results if isinstance(results, list) else []:     # an error comes back as an object, not a list
            url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else None)
            if url:
                hosts.add(_host(url))
    return hosts


def lower_median(values: list[float]) -> float:
    """The middle value; for an even count the lower of the two middle ones (never raise a value)."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def priced(kept: list[dict]) -> dict:
    """{price, low, high, matches} from the kept listings, closest to the mid first."""
    mid = lower_median([m["price"] for m in kept])
    ordered = sorted(kept, key=lambda m: (abs(m["price"] - mid), m["price"]))
    return {"price": mid, "low": min(m["price"] for m in kept), "high": max(m["price"] for m in kept),
            "matches": ordered}


def get_searcher():
    """A searcher when an Anthropic key is set, else None."""
    if (config.env("ASHLINE_PRICE_SEARCH") or "").lower() == "off" or not config.env("ANTHROPIC_API_KEY"):
        return None
    return ClaudeSearcher()


def find_price(description: str, brand: str | None = None, searcher=None, image=None) -> dict:
    """{found, price, low, high, retailer, url, matches}.

    With a searcher that looks in stages (`lookup`), `price` is the median of the listings the
    check kept, and the result also carries `spec`, `queries`, `checked`, `dropped` and
    `verified: True`. Otherwise, or when a stage failed, `price` is the closest match of one search.
    """
    searcher = searcher or get_searcher()
    if searcher is None:
        return {"found": False, "reason": "Online price lookup is not set up (it needs ANTHROPIC_API_KEY)."}
    staged = None
    if callable(getattr(searcher, "lookup", None)):
        try:
            staged = searcher.lookup(description, brand, image=image)
        except Exception:
            staged = None                               # the single search below
    if staged is not None:
        seen = {k: staged.get(k) for k in ("spec", "queries", "checked", "dropped")}
        kept, spec = staged.get("matches") or [], staged.get("spec") or {}
        read = bool(spec.get("brand_read") and spec.get("model"))
        if len(kept) < 2 and not (kept and read):
            # One listing is not enough evidence, unless it is for the brand and model read in the photo.
            if spec.get("enough_to_price") is False:
                reason = "Not enough of it can be seen to price it" + (f" ({spec['cannot_see']})." if spec.get("cannot_see") else ".")
            elif staged.get("checked"):
                reason = (f"Not enough evidence: {len(kept)} of the {staged['checked']} listings looked at "
                          f"{'was' if len(kept) == 1 else 'were'} the same kind of thing.")
            else:
                reason = "No suitable listing was found."
            return {"found": False, "reason": reason, **seen}
        best = priced(staged["matches"])
        return {"found": True, "price": best["price"], "low": best["low"], "high": best["high"],
                "retailer": best["matches"][0]["retailer"], "url": best["matches"][0]["url"],
                "matches": best["matches"], "verified": True, **seen}
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
    """Keep a price found online for an item, with the links as evidence. It becomes the item's
    price unless the family set one or paid one (those are stronger; the found price is still
    kept and shown beside them). Can be undone, unless `log` is False (a lookup nobody asked for
    must not become the thing "undo" reverses)."""
    item = store.get_item(case_id, item_id)
    if not item:
        return "That item is not on the list."
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
            "status", "label", "qty", "price_low", "price_mid", "price_high", "price_source", "price_ref")}}, member_id)
    online = {"low": nice(low), "mid": nice(price), "high": nice(high), "retailer": retailer,
              "matches": refs[:MAX_CANDIDATES if found.get("verified") else 4]}
    for key in ("spec", "queries", "checked", "dropped"):
        if found.get(key) is not None:
            online[key] = found[key]
    record_price(store, case_id, item_id, "online", online)
    span = f", range {money(nice(low))} to {money(nice(high))}" if nice(high) > nice(low) else ""
    said = f"{money(nice(price))} for the {item['label']} from {retailer}{span}"
    stronger = in_use(item)
    if stronger == "family":
        return f"Found {said}. The family set this price themselves, so it was left as it is; the listing is kept beside it."
    if stronger == "paid":
        return f"Found {said}. The price stays what the family paid; the listing is kept beside it."
    return f"Recorded {said}."


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
AUTO_TRUST = 3.0                # a lone listing is used unasked only within this factor of the estimate, either way
AGREE = 2.0                     # two checked listings within this factor of each other are trusted whatever the estimate


def fixed_figure(item: dict) -> bool:
    """True when the item's estimate is the fixed figure for its whole category, which says
    nothing about this particular thing."""
    return (item.get("price_source") or "").startswith("category default")


def trusted(result: dict, item: dict) -> bool:
    """May a price found without anyone asking replace the item's estimate?

    Two or more checked listings that agree with each other (the two nearest the mid, within 2x)
    are used whatever the estimate says. Otherwise the 3x rule against the estimate holds, except
    that a fixed category figure is no yardstick at all."""
    matches = result.get("matches") or []
    if result.get("verified") and len(matches) >= 2:
        a, b = float(matches[0]["price"]), float(matches[1]["price"])
        if min(a, b) > 0 and max(a, b) / min(a, b) <= AGREE:
            return True
    if fixed_figure(item):
        return True
    estimate = float(item.get("price_mid") or 0)
    return not estimate or estimate / AUTO_TRUST <= float(result["price"]) <= estimate * AUTO_TRUST


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
    run a few at a time. A found price is used when `trusted` says so: two checked listings that
    agree, or a lone one near the estimate (any, when the estimate is a fixed category figure).
    Each result is also kept as news for the chat to mention once.
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
        if not trusted(result, item):
            continue        # likely the wrong product, so nobody's price changes unasked
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
    """Keep results for the chat to mention. Each batch is its own file, written whole and then
    renamed into place, so a reader in another process never sees half of one."""
    folder = _news_path(case_id).parent / "price_news"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{os.getpid()}-{threading.get_ident()}"
    part = folder / f"{name}.part"
    part.write_text(json.dumps(results))
    os.replace(part, folder / f"{name}.json")


def take_news(case_id: str) -> list[dict]:
    """Looked-up prices not yet mentioned in the chat. Each is returned once: a batch is claimed
    by renaming it, which only one process can do."""
    folder = _news_path(case_id).parent / "price_news"
    waiting = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        mine = path.with_suffix(f".taken-{os.getpid()}")
        try:
            os.replace(path, mine)
        except OSError:
            continue                                        # another process took it first
        try:
            waiting += json.loads(mine.read_text())
        except ValueError:
            pass
        mine.unlink(missing_ok=True)
    return waiting
