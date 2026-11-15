"""Replacement-cost estimates with ranges (build spec 6.5).

Point estimate: what the item costs to buy new today (replacement cost value),
a typical mid-market figure, from Claude or, with no key, a rough category
default. Ashline never suggests inflating values.

Range: split conformal prediction. `eval/calibrate.py` measures, on items
with known prices, how far estimates miss in log space and writes the 90th
percentile `q` to eval/calibration.json; the range is [mid * e^-q, mid * e^q].
Until that file exists a default `q` is used and the range is labelled
uncalibrated, so no coverage claim is made that has not been measured.
"""
from __future__ import annotations

import json
import math
import re

from . import config
from .store import Store

DEFAULT_Q = 0.45                 # about -36% / +57%; a placeholder until calibration is run
CHUNK = 60

# Rough new-replacement mid values by category, used only when no model is available.
CATEGORY_DEFAULT = {"furniture": 450, "electronics": 350, "appliances": 400, "kitchen": 45, "decor": 60,
                    "lighting": 80, "bedding": 70, "clothing": 45, "tools": 90, "sports": 120, "toys": 30,
                    "outdoor": 200, "office": 120, "other": 50}

PROMPT = """You are estimating replacement costs for a home contents insurance inventory.

For each item below give `mid`: what one unit costs to buy NEW today in US dollars, at a typical
mid-market retailer. This is replacement cost value, not resale value and not the original price.
- Use the brand when one is given; otherwise assume an ordinary mid-range product, not a premium one.
- Never inflate. When unsure between two prices, choose the lower.
- A line with a quantity is priced per unit.
- A line that describes a group ("pile of clothing", "assorted toiletries") is priced as that
  whole group.
Each line is: id | category | label | brand (or -).

{lines}"""

PRICES_SCHEMA = {
    "type": "object",
    "properties": {"prices": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "mid": {"type": "number"}},
        "required": ["id", "mid"], "additionalProperties": False}}},
    "required": ["prices"], "additionalProperties": False,
}


class PricingError(RuntimeError):
    pass


def load_calibration() -> dict:
    """{'q': float, 'by_category': {...}, 'calibrated': bool, ...} from eval/calibration.json."""
    path = config.REPO_ROOT / "eval" / "calibration.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if float(data.get("q", 0)) > 0:
                return {**data, "calibrated": True}
        except (ValueError, TypeError):
            pass
    return {"q": DEFAULT_Q, "by_category": {}, "calibrated": False}


def nice(value: float) -> float:
    """Round to a figure a person would say: $45, $380, $1,250, $4,500."""
    if value <= 0:
        return 0.0
    step = 5 if value < 100 else 10 if value < 500 else 50 if value < 2000 else 100 if value < 10000 else 500
    return float(max(step, round(value / step) * step))


def price_range(mid: float, category: str | None = None, calibration: dict | None = None) -> tuple[float, float]:
    calibration = calibration or load_calibration()
    q = float((calibration.get("by_category") or {}).get(category or "", calibration["q"]))
    return nice(mid * math.exp(-q)), nice(mid * math.exp(q))


def money(value: float | None) -> str:
    """$45, $8,400, $96k, $1.2M."""
    if value is None:
        return "?"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M".replace(".0M", "M")
    if value >= 10_000:
        return f"${value / 1000:.0f}k"
    return f"${value:,.0f}"


def money_range(low: float | None, high: float | None) -> str:
    """$400–$750, $4–7k."""
    if low is None or high is None:
        return "?"
    if round(low) == round(high):
        return money(low)
    if low >= 1000 and high >= 1000:
        def k(v):
            return f"{v / 1000:.1f}".rstrip("0").rstrip(".")
        return f"${k(low)}–{k(high)}k"
    return f"{money(low)}–{money(high)}"


# Typical new prices for common household things, budget to mid-market, used only when no model is
# available. Matched on words in the label; the longest phrase wins ("office chair" before "chair").
# Checked against real listings in November 2026 for the ones a price study looked up (coffee maker
# about $40, nightstand about $51, dining chair about $80, microwave about $90, coffee table about
# $111); the rest are set on the same scale, lower when unsure.
TYPICAL_PRICE = {
    # seating
    "dining chair": 90, "kitchen chair": 80, "chair": 80, "armchair": 300, "accent chair": 200, "recliner": 450,
    "office chair": 150, "desk chair": 150, "gaming chair": 180, "bar stool": 60, "stool": 45, "bench": 120,
    "ottoman": 90, "sofa": 800, "couch": 800, "sofa bed": 600, "futon": 250, "sectional": 1200, "loveseat": 550,
    # tables and storage
    "coffee table": 120, "side table": 60, "end table": 60, "console table": 120, "dining table": 400,
    "kitchen table": 300, "desk": 150, "tv stand": 150, "bookshelf": 100, "bookcase": 100, "shelf": 40,
    "dresser": 300, "chest of drawers": 250, "nightstand": 60, "bedside table": 60, "wardrobe": 300,
    "cabinet": 150, "shoe rack": 35,
    # beds and bedding
    "bed": 350, "bed frame": 300, "headboard": 150, "mattress": 500, "crib": 200, "duvet cover": 45,
    "duvet": 60, "comforter": 60, "quilt": 60, "pillow": 20, "sheet set": 35, "blanket": 30,
    # decor and lighting
    "lamp": 40, "table lamp": 40, "desk lamp": 30, "floor lamp": 70, "ceiling light": 80, "rug": 75,
    "area rug": 75, "runner rug": 40, "mirror": 70, "curtains": 35, "painting": 80, "picture frame": 20,
    "plant pot": 20, "clock": 30, "vase": 25,
    # screens and electronics
    "tv": 400, "television": 400, "monitor": 180, "laptop": 700, "desktop computer": 800, "computer": 700,
    "tablet": 350, "printer": 130, "speaker": 90, "soundbar": 200, "headphones": 80, "game console": 450,
    "router": 90, "camera": 450, "smart speaker": 50,
    # appliances
    "refrigerator": 1000, "fridge": 1000, "mini fridge": 180, "freezer": 400, "washer": 650,
    "washing machine": 650, "dryer": 600, "dishwasher": 550, "range": 700, "stove": 700, "oven": 700,
    "cooktop": 400, "range hood": 200, "microwave": 100, "vacuum": 150, "robot vacuum": 250,
    "air conditioner": 300, "space heater": 50, "fan": 40, "ceiling fan": 120, "air purifier": 120,
    "humidifier": 45, "dehumidifier": 200, "water heater": 700,
    # kitchen
    "toaster": 30, "toaster oven": 80, "kettle": 30, "electric kettle": 30, "coffee maker": 40,
    "espresso machine": 200, "blender": 50, "stand mixer": 300, "food processor": 80, "air fryer": 80,
    "slow cooker": 45, "instant pot": 90, "pressure cooker": 90, "rice cooker": 50, "pots and pans set": 100,
    "cookware set": 100, "frying pan": 30, "knife set": 60, "dinnerware set": 60, "cutlery set": 40,
    # other
    "bicycle": 450, "bike": 450, "treadmill": 700, "exercise bike": 350, "grill": 300, "lawn mower": 350,
    "suitcase": 90, "guitar": 250, "piano": 3000, "sewing machine": 200, "stroller": 200, "car seat": 180,
}


SMALL_THINGS = {"remote control": 20, "extension cord": 15, "power strip": 20, "laptop sleeve": 25, "table": 200,
                "table top": 200, "seat cushion": 25, "chair cushion": 25, "phone case": 20, "tv remote": 20,
                "microwave oven": 100, "vacuum cleaner": 150, "sectional sofa": 1200, "toaster oven": 60,
                "sofa bed": 600, "bed frame": 350, "bunk bed": 400}
# A label ending in one of these names a part or an accessory, not the thing itself.
ACCESSORY = ("sleeve", "case", "cover", "remote", "control", "cord", "cable", "charger", "cushion", "mat", "holder",
             "bag", "box", "pouch", "tray", "stand", "panel", "sign", "strap", "shelf", "rack", "hose")
_SPLIT = re.compile(r"\s(?:with|on|in|for|of|from|under|beside|next to|against)\s|[,(;]")


def typical_price(label: str | None) -> float | None:
    """The table's figure for the thing a label names, or None.

    The thing is the noun the label's first phrase ends on: "bed" in "bed with gray upholstered
    headboard", "dining chair" in "wood-back dining chair with black metal frame". A table phrase
    found only elsewhere ("laptop" in "teal laptop sleeve", "tv" in "black TV remote control") is a
    part or an accessory of something dearer and is not used. When the first phrase names nothing
    the table knows, a bracketed guess such as "(smart speaker or clock)" may still be read.
    """
    text = (label or "").lower()
    table = {**TYPICAL_PRICE, **SMALL_THINGS}

    def clean(part: str) -> str:
        return " " + re.sub(r"[^a-z0-9]+", " ", part).strip() + " "

    def ends_on(words: str) -> tuple[str, float] | None:
        best = None
        for phrase, price in table.items():
            if words.endswith(f" {phrase} ") or words.endswith(f" {phrase}s "):
                if best is None or len(phrase) > len(best[0]):
                    best = (phrase, price)
        return best

    text = re.sub(r"^\s*(?:a |the |\d+ )?(?:set|pair|group|couple|pile|stack|row|collection|lot)s? of (?:\d+ )?", "", text)
    head = clean(_SPLIT.split(text, 1)[0])
    found = ends_on(head)
    if found:
        return float(found[1])
    if head.strip().split(" ")[-1:] and head.strip().split(" ")[-1].rstrip("s") in ACCESSORY:
        return None
    for guess in re.findall(r"\(([^)]*)\)", text):                  # "(smart speaker or small clock)"
        for part in re.split(r"\sor\s|,|/", guess):
            found = ends_on(clean(part))
            if found:
                return float(found[1])
    return None


class DefaultPricer:
    """No model: a typical price for the kind of thing when the label names one, else a rough
    figure per category, each clearly labelled as such."""
    source = "category default"
    typical = "typical price"

    def prices(self, items: list[dict]) -> dict[int, float]:
        return {i["item_id"]: typical_price(i.get("label")) or float(CATEGORY_DEFAULT.get(i.get("category") or "other", 50))
                for i in items}

    def source_for(self, item: dict) -> str:
        return self.typical if typical_price(item.get("label")) else self.source


class ClaudePricer:
    source = "claude estimate"

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        config.load_env()
        self.model = model or config.env("ASHLINE_CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = effort or config.env("ASHLINE_CLAUDE_EFFORT", "medium")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def prices(self, items: list[dict]) -> dict[int, float]:
        out: dict[int, float] = {}
        for start in range(0, len(items), CHUNK):
            chunk = items[start:start + CHUNK]
            lines = "\n".join(f"{n} | {i.get('category') or 'other'} | {i['label']} | {i.get('brand_guess') or '-'}"
                              for n, i in enumerate(chunk))
            for row in self._ask(PROMPT.format(lines=lines)):
                try:
                    n, mid = int(row["id"]), float(row["mid"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= n < len(chunk) and mid > 0:
                    out[chunk[n]["item_id"]] = mid
        return out

    def _ask(self, prompt: str) -> list[dict]:
        import anthropic
        from .detect_claude import FALLBACK_BETA, FALLBACK_MODELS
        request = dict(model=self.model, max_tokens=16000,
                       output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": PRICES_SCHEMA}},
                       messages=[{"role": "user", "content": prompt}])
        try:
            if self.model.startswith(FALLBACK_MODELS):
                response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **request)
            else:
                response = self.client.messages.create(**request)
        except anthropic.APIStatusError as exc:
            raise PricingError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise PricingError("Could not reach the Anthropic API.") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise PricingError(f"Claude stopped early ({response.stop_reason}).")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            rows = json.loads(text)["prices"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PricingError(f"Unexpected reply from Claude: {text[:200]}") from exc
        return rows if isinstance(rows, list) else []


def get_pricer():
    """`ASHLINE_PRICER` = claude | defaults. Default: claude when an Anthropic key is set."""
    kind = (config.env("ASHLINE_PRICER") or ("claude" if config.env("ANTHROPIC_API_KEY") else "defaults")).lower()
    if kind == "claude":
        return ClaudePricer()
    if kind == "defaults":
        return DefaultPricer()
    raise PricingError(f"Unknown ASHLINE_PRICER {kind!r}: use claude or defaults.")


ROUGH = ("category default", "typical price")      # figures made without a model


def _cache_key(item: dict) -> str:
    return "|".join(str(item.get(k) or "").strip().lower() for k in ("label", "category", "brand_guess"))


# ------------------------------------------------- every price an item has
# An item can have up to three prices at once, kept in `price_ref`:
#   paid      what the family paid: a card purchase or a receipt {amount, each, merchant, date, description, bank_id | receipt}
#   online    the closest listings found for it today           {low, mid, high, retailer, matches: [{title, price, retailer, url}]}
#   estimate  the model's (or the category's) rough allowance   {low, mid, high, source}
# The strongest one is the item's price (price_low / price_mid / price_high / price_source). A price
# the family typed in beats all three and is not kept here: it is the item's price itself.
STRENGTH = ("paid", "online", "estimate")


def price_sources(item: dict) -> dict:
    """The prices kept for an item, by kind. Reads the older list form too."""
    try:
        kept = json.loads(item.get("price_ref") or "null")
    except (TypeError, ValueError):
        kept = None
    if isinstance(kept, dict):
        return {k: v for k, v in kept.items() if k in STRENGTH and isinstance(v, dict)}
    out: dict = {}
    source = item.get("price_source") or ""
    if isinstance(kept, list) and kept and isinstance(kept[0], dict):
        if source.startswith("bank record"):
            ref = kept[0]
            out["paid"] = {"amount": ref.get("price"), "each": item.get("price_mid"), "merchant": ref.get("retailer"),
                           "date": ref.get("date"), "description": ref.get("title"), "bank_id": ref.get("bank_id")}
        else:
            out["online"] = {"low": item.get("price_low"), "mid": item.get("price_mid"), "high": item.get("price_high"),
                             "retailer": kept[0].get("retailer"), "matches": kept}
    return out


def in_use(item: dict) -> str:
    """Which price is the item's price now: family, paid, online, estimate, or '' when it has none."""
    source = item.get("price_source") or ""
    if source == "set by family":
        return "family"
    if source.startswith(("bank record", "receipt:")):
        return "paid"
    if source.startswith("web:"):
        return "online"
    return "estimate" if item.get("price_mid") else ""


def record_price(store, case_id: str, item_id: int, kind: str, data: dict | None) -> dict | None:
    """Keep one more price for an item (or drop it with data=None) and put the strongest in use.

    The estimate in use when the first better price arrives is kept alongside it, so the page can
    show all of them. A price the family set stays the item's price whatever else is recorded.
    """
    item = store.get_item(case_id, item_id)
    if not item:
        return None
    kept = price_sources(item)
    if "estimate" not in kept and in_use(item) == "estimate":
        kept["estimate"] = {"low": item.get("price_low"), "mid": item.get("price_mid"), "high": item.get("price_high"),
                            "source": item.get("price_source")}
    if data is None:
        kept.pop(kind, None)
    else:
        kept[kind] = data
    fields = {"price_ref": json.dumps(kept) if kept else None}
    if in_use(item) != "family":
        if "paid" in kept:
            each = kept["paid"].get("each")
            proof = "receipt" if kept["paid"].get("receipt") or kept["paid"].get("order") else "bank record"
            fields.update(price_low=each, price_mid=each, price_high=each,
                          price_source=f"{proof}: {str(kept['paid'].get('merchant') or 'purchase')[:60]}")
        elif "online" in kept:
            web = kept["online"]
            fields.update(price_low=web.get("low"), price_mid=web.get("mid"), price_high=web.get("high"),
                          price_source=f"web: {str(web.get('retailer') or 'online retailer')[:60]}")
        elif "estimate" in kept:
            guess = kept["estimate"]
            fields.update(price_low=guess.get("low"), price_mid=guess.get("mid"), price_high=guess.get("high"),
                          price_source=guess.get("source"))
    store.update_item(case_id, item_id, **fields)
    return store.get_item(case_id, item_id)


def kept_price(item: dict) -> bool:
    """True when the item's price is better than an estimate and must not be replaced by one:
    the family set it, it is what they paid (a bank record), or it was found online."""
    source = item.get("price_source") or ""
    return source == "set by family" or source.startswith(("bank record", "receipt:", "web:"))


def price_items(case_id: str, store: Store | None = None, pricer=None, only_missing: bool = True) -> dict:
    """Price the inventory lines of a case. Returns {priced, source, calibrated}.

    A price the family set, a card purchase or a price found online is never overwritten. Estimates are cached per case
    by label, so re-seeding a case does not pay for the same item twice. If the
    model call fails, category defaults are used and the reason is reported.
    """
    store = store or Store()
    # A rough figure (no model was available) is not a price worth keeping: it is worked out again,
    # so a home priced before a key was set, or before the table of typical prices, gets better numbers.
    items = [i for i in store.get_items(case_id, canonical_only=True) if i["status"] != "rejected"
             and not kept_price(i) and not (only_missing and i.get("price_mid") and i.get("price_source") not in ROUGH)]
    if not items:
        return {"priced": 0, "source": None, "calibrated": load_calibration()["calibrated"]}

    cache_path = config.case_dir(case_id) / "prices.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [i for i in items if cache.get(_cache_key(i), {}).get("source") in (None, *ROUGH)]
    note = None
    if todo:
        pricer = pricer or get_pricer()
        default = DefaultPricer()
        try:
            fresh, source = pricer.prices(todo), pricer.source
        except Exception as exc:                 # never lose the inventory over pricing
            pricer = default
            fresh, source = default.prices(todo), default.source
            note = f"category defaults used (pricing failed: {exc})"
        source_for = getattr(pricer, "source_for", None)
        missing = [i for i in todo if i["item_id"] not in fresh]
        fallback = default.prices(missing)
        for item in todo:
            mid = fresh.get(item["item_id"])
            cache[_cache_key(item)] = ({"mid": mid, "source": source_for(item) if source_for else source} if mid else
                                       {"mid": fallback[item["item_id"]], "source": default.source_for(item)})
        cache_path.write_text(json.dumps(cache, indent=1))

    calibration = load_calibration()
    sources = set()
    for item in items:
        entry = cache[_cache_key(item)]
        mid = nice(entry["mid"])
        low, high = price_range(mid, item.get("category"), calibration)
        store.update_item(case_id, item["item_id"], price_low=low, price_mid=mid, price_high=high,
                          price_source=entry["source"])
        sources.add(entry["source"])
    return {"priced": len(items), "source": note or ", ".join(sorted(sources)), "calibrated": calibration["calibrated"]}
