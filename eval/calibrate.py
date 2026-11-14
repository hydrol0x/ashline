"""Measure how wide a price range has to be (build spec 6.5 and 7).

    python eval/calibrate.py eval/known_prices.csv

`known_prices.csv` is a list you collect by hand from retailer pages: columns
`label,category,brand,price` (the price to buy the item new today). About 100 to
200 rows across the usual categories is enough to start.

Split conformal prediction: half the rows are used to find `q`, the 90th
percentile of |log(true) - log(estimate)|; the other half are used to check
how often the true price falls inside [estimate * e^-q, estimate * e^q].
The result is written to eval/calibration.json, which `ashline.pricing` reads.
Until this has been run, Ashline labels its ranges as a rough allowance.

This script prices the items with the configured pricer (Claude when
ANTHROPIC_API_KEY is set), so it makes paid API calls.
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

from ashline.pricing import get_pricer

TARGET = 0.90


def main(path: str, seed: int = 7) -> None:
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if float(r.get("price") or 0) > 0]
    if len(rows) < 40:
        sys.exit(f"{len(rows)} usable rows: collect at least 40 (100 to 200 is better).")
    items = [{"item_id": n, "label": r["label"], "category": r.get("category") or "other",
              "brand_guess": r.get("brand") or None} for n, r in enumerate(rows)]
    estimates = get_pricer().prices(items)
    scored = [(abs(math.log(float(rows[n]["price"])) - math.log(estimates[n])), rows[n].get("category") or "other")
              for n in range(len(rows)) if estimates.get(n)]
    random.Random(seed).shuffle(scored)
    half = len(scored) // 2
    calibrate, test = scored[:half], scored[half:]
    residuals = sorted(r for r, _ in calibrate)
    # The conformal quantile: the ceil((n + 1) * 0.9)-th smallest residual.
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * TARGET))
    q = residuals[rank - 1]
    coverage = sum(1 for r, _ in test if r <= q) / len(test)
    result = {"q": round(q, 4), "by_category": {}, "n_calibration": len(calibrate), "n_test": len(test),
              "coverage": round(coverage, 3), "target": TARGET,
              "range": f"x{math.exp(-q):.2f} to x{math.exp(q):.2f} of the estimate"}
    out = Path(__file__).parent / "calibration.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"\nWrote {out}. On the held-out half, {coverage:.0%} of true prices fell inside the range "
          f"(target {TARGET:.0%}). With {len(test)} test items this figure is approximate.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
