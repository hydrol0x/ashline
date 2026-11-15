"""The meter: how much of the recoverable money the family has documented (build spec 6.8).

California SB 495 (effective Jan 1, 2026): after a declared-disaster total loss the insurer pays
at least 60% of the contents limit, capped at $350,000, without an inventory. The rest still
needs documentation, and the policyholder gets at least 100 days to provide proof of loss.

Estimate for planning, not legal advice; confirm with your adjuster.
"""
from __future__ import annotations

from datetime import date, timedelta

from .pricing import money
from .store import Store

ADVANCE_SHARE = 0.60
ADVANCE_CAP = 350_000
PROOF_OF_LOSS_DAYS = 100
DISCLAIMER = "Estimates for planning; not legal or insurance advice. Confirm values with your adjuster."
MEMORY_SOURCES = ("voice",)          # from memory only: adjusters are more likely to question these


def sb495(coverage_limit: float) -> dict:
    """What is paid without a list, and what still needs documentation."""
    advance = min(ADVANCE_SHARE * coverage_limit, ADVANCE_CAP)
    return {"limit": coverage_limit, "advance": advance, "remaining": max(0.0, coverage_limit - advance)}


def _value(item: dict, key: str = "price_mid") -> float:
    return float(item.get(key) or 0) * int(item.get("qty") or 1)


def meter(case_id: str, store: Store | None = None, today: date | None = None) -> dict:
    """Totals for the meter card. Mid estimates; suggestions are 'pending', never 'documented'."""
    store = store or Store()
    case = store.get_case(case_id) or {}
    items = store.get_items(case_id, canonical_only=True)
    confirmed = [i for i in items if i["status"] == "confirmed"]
    suggested = [i for i in items if i["status"] == "suggested"]
    def proven(item: dict) -> bool:                      # a photo, scan or receipt, or a matching card purchase
        return item["source"] not in MEMORY_SOURCES or (item.get("price_source") or "").startswith("bank record")
    backed = [i for i in confirmed if proven(i)]
    memory = [i for i in confirmed if not proven(i)]

    out = {
        "items": len(confirmed),
        "pending_items": len(suggested),
        "backed": sum(_value(i) for i in backed),            # photo, scan or receipt behind it
        "memory": sum(_value(i) for i in memory),            # from memory only
        "pending": sum(_value(i) for i in suggested),        # not yet confirmed by the family
        "low": sum(_value(i, "price_low") for i in confirmed),
        "high": sum(_value(i, "price_high") for i in confirmed),
        "limit": case.get("coverage_limit"), "advance": None, "target": None, "percent": None,
        "due": None, "days_left": None,
    }
    out["documented"] = out["backed"] + out["memory"]
    if case.get("coverage_limit"):
        rule = sb495(float(case["coverage_limit"]))
        out.update(advance=rule["advance"], target=rule["remaining"])
        if rule["remaining"] > 0:
            out["percent"] = min(100, round(100 * out["documented"] / rule["remaining"]))
    if case.get("loss_date"):
        try:
            due = date.fromisoformat(case["loss_date"][:10]) + timedelta(days=PROOF_OF_LOSS_DAYS)
            left = (due - (today or date.today())).days
            if left >= 0:                                    # a past date is for the adjuster, not a countdown
                out.update(due=due.isoformat(), days_left=left)
        except ValueError:
            pass
    return out


def meter_line(m: dict) -> str:
    """One line for chat: 'You're at $96k of $160k documented. 12 more items are waiting for your OK.'"""
    if m["target"]:
        text = f"You're at {money(m['documented'])} of {money(m['target'])} documented."
    else:
        text = f"{m['items']} items documented so far, about {money(m['documented'])}."
    if m["pending_items"]:
        text += f"\n{m['pending_items']} more item{'s' if m['pending_items'] != 1 else ''} " \
                f"({money(m['pending'])}) {'are' if m['pending_items'] != 1 else 'is'} waiting for an OK."
    if m["days_left"] is not None:
        text += f"\nProof of loss is due in {m['days_left']} days."
    return text
