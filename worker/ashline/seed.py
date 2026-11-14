"""The home twin seed: address -> scan + aerial -> first evidence batch of a case.

`inside` and `outside` are the two halves the agent runs as subagents. Both
only ever write suggestions: scan items are 'suggested', aerial hits are open
findings that become questions. Nothing is confirmed without the family.
"""
from __future__ import annotations

from . import config, detect as detect_mod
from .costar import matterport, vexcel
from .costar.index import home_lookup
from .dedupe import cluster_scan_items
from .gaps import gap_questions
from .pipeline import detect_cached
from .store import Store


def inside(case_id: str, sid: str, captured: str | None = None, run_detection: bool = True,
           max_faces: int = 40, store: Store | None = None, detector=None, client=None) -> dict:
    """Scan rooms and photos -> rooms table, photos, suggested items. Returns a short summary."""
    store = store or Store()
    rooms = matterport.scan_rooms(sid, client)
    store.set_rooms(case_id, rooms["rooms"], source="scan")
    store.set_home(case_id, sid=sid, scan_captured=captured, capture_notes=rooms["capture_notes"])

    shots = matterport.scan_photos(sid, case_id, max_faces=max_faces, captured=captured, client=client)
    store.add_photos(case_id, shots["photos"], added_by="scan")

    suggested = clusters = 0
    if run_detection and shots["photos"]:
        store.clear_items(case_id, "scan")
        for photo in shots["photos"]:
            found = detect_cached(case_id, photo, detector, get_detector=detect_mod.get_detector)
            for item in found:
                store.add_item(case_id, item, source="scan", photo=photo, added_by="scan")
        items = [i for i in store.get_items(case_id, status="suggested") if i["source"] == "scan"]
        by_id = {p["photo_id"]: p for p in shots["photos"]}
        for cluster in cluster_scan_items(items, by_id):
            store.set_cluster(case_id, cluster["canonical_id"], cluster["member_ids"], cluster["qty"])
        suggested = len(items)
        clusters = len(store.get_items(case_id, status="suggested", canonical_only=True))

    return {
        "sid": sid,
        "rooms": [{"label": r["name"], "area_m2": r["area_m2"], "furnished": r["furnished"]} for r in rooms["rooms"]],
        "capture_notes": rooms["capture_notes"],
        "captured": captured,
        "faces": len(shots["photos"]),
        "panos_used": shots["panos_used"], "panos_total": shots["panos_total"],
        "detections": suggested,
        "suggested_items": clusters,          # after merging repeats across panoramas
        "message": matterport.scan_confirmation_message(rooms["rooms"], captured, rooms["capture_notes"]),
    }


def aerial_findings(aerial: dict) -> list[dict]:
    """Turn an aerial_parcel result into open findings: things to ask about, not items."""
    date = aerial.get("capture_date")
    findings = []
    for s in aerial.get("structures") or []:
        if not s["primary"]:
            findings.append({"kind": "structure", "label": "outbuilding", "evidence_date": date,
                             "detail": {"index": s["index"], "footprint_sqft": s["footprint_sqft"],
                                        "roof_material": s["roof_material"]}})
    seen_labels = set()
    for d in aerial.get("detections") or []:
        if d["on_structure"] is not None:
            continue                    # a "shed" on a mapped outbuilding is already a structure question
        seen_labels.add(d["label"])
        findings.append({"kind": "yard", "label": d["label"], "confidence": d["confidence"],
                         "evidence_date": date, "detail": {"n": d["n"], "bbox_wkt": d["bbox_wkt"], "by": "detection"}})
    for name, value in (aerial.get("attributes") or {}).items():
        if name in ("vehicles", "solar panels") or name in seen_labels or not value:
            continue
        findings.append({"kind": "yard", "label": name, "evidence_date": date, "detail": {"by": "parcel attribute"}})
    return findings


def outside(case_id: str, lat: float, lon: float, loss_date: str | None = None,
            store: Store | None = None, client=None) -> dict:
    """Aerial parcel -> home record and open findings. Returns a short summary."""
    store = store or Store()
    aerial = vexcel.aerial_parcel(lat, lon, case_id, before=loss_date, client=client)
    if not aerial.get("covered"):
        return {"covered": False, "reason": aerial.get("reason")}
    findings = aerial_findings(aerial)
    store.set_findings(case_id, findings, source="aerial")
    store.set_home(case_id, lat=lat, lon=lon, parcel_wkt=aerial["parcel_wkt"],
                   aerial_collection=aerial.get("collection"), aerial_captured=aerial.get("capture_date"))
    return {
        "covered": True,
        "parcel_found": aerial["parcel_found"],
        "capture_date": aerial.get("capture_date"),
        "after_loss": aerial.get("after_loss"),
        "flights": len(aerial.get("collections") or []),
        "structures": len(aerial.get("structures") or []),
        "outbuildings": sum(1 for f in findings if f["kind"] == "structure"),
        "yard_findings": sorted({f["label"] for f in findings if f["kind"] == "yard"}),
        "annotated_path": aerial.get("annotated_path"),
    }


def seed_home_twin(case_id: str, address: str | None = None, lat: float | None = None,
                   lon: float | None = None, loss_date: str | None = None, run_detection: bool = True,
                   max_faces: int = 40, store: Store | None = None, sid: str | None = None) -> dict:
    """Run the whole seed for a case. Safe to call again: API results are cached per case.

    The home is found by `address`, by `lat`/`lon`, or directly by scan id `sid`.
    """
    store = store or Store()
    loss_date = loss_date or config.env("ASHLINE_LOSS_DATE")
    found = home_lookup(address, lat, lon, sid=sid)
    store.upsert_case(case_id, address=address or found.get("address"), loss_date=loss_date)
    result: dict = {"case_id": case_id, "lookup": found}
    if found["found"]:
        store.set_home(case_id, sid=found["sid"], lat=found["lat"], lon=found["lon"],
                       homes_listing_id=found["homes_listing_id"])
        result["inside"] = inside(case_id, found["sid"], found["original_created"], run_detection,
                                  max_faces, store)
        lat, lon = found["lat"], found["lon"]
    if lat is not None and lon is not None:
        result["outside"] = outside(case_id, lat, lon, loss_date, store)
    result["questions"] = gap_questions(case_id, store, limit=5)
    return result
