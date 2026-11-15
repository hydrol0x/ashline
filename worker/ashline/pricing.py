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


class DefaultPricer:
    """No model: a rough figure per category, clearly labelled as such."""
    source = "category default"

    def prices(self, items: list[dict]) -> dict[int, float]:
        return {i["item_id"]: float(CATEGORY_DEFAULT.get(i.get("category") or "other", 50)) for i in items}


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


def _cache_key(item: dict) -> str:
    return "|".join(str(item.get(k) or "").strip().lower() for k in ("label", "category", "brand_guess"))


# ------------------------------------------------- every price an item has
# An item can have up to three prices at once, kept in `price_ref`:
#   paid      what the family paid, from a card purchase        {amount, each, merchant, date, description, bank_id}
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
    if source.startswith("bank record"):
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
            fields.update(price_low=each, price_mid=each, price_high=each,
                          price_source=f"bank record: {str(kept['paid'].get('merchant') or 'card purchase')[:60]}")
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
    return source == "set by family" or source.startswith(("bank record", "web:"))


def price_items(case_id: str, store: Store | None = None, pricer=None, only_missing: bool = True) -> dict:
    """Price the inventory lines of a case. Returns {priced, source, calibrated}.

    A price the family set, a card purchase or a price found online is never overwritten. Estimates are cached per case
    by label, so re-seeding a case does not pay for the same item twice. If the
    model call fails, category defaults are used and the reason is reported.
    """
    store = store or Store()
    items = [i for i in store.get_items(case_id, canonical_only=True) if i["status"] != "rejected"
             and not kept_price(i) and not (only_missing and i.get("price_mid"))]
    if not items:
        return {"priced": 0, "source": None, "calibrated": load_calibration()["calibrated"]}

    cache_path = config.case_dir(case_id) / "prices.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [i for i in items if _cache_key(i) not in cache]
    note = None
    if todo:
        pricer = pricer or get_pricer()
        try:
            fresh, source = pricer.prices(todo), pricer.source
        except Exception as exc:                 # never lose the inventory over pricing
            fresh, source = DefaultPricer().prices(todo), DefaultPricer.source
            note = f"category defaults used (pricing failed: {exc})"
        missing = [i for i in todo if i["item_id"] not in fresh]
        fallback = DefaultPricer().prices(missing)
        for item in todo:
            mid = fresh.get(item["item_id"])
            cache[_cache_key(item)] = ({"mid": mid, "source": source} if mid else
                                       {"mid": fallback[item["item_id"]], "source": DefaultPricer.source})
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
