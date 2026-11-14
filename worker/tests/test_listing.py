"""listing_record against a made-up kit. The real kit's format is unverified, so
both shapes that were described (CSV and JSONL) are exercised. The test kit is
not encrypted: the standard library cannot write encrypted zips."""
import json
import zipfile

import pytest

from ashline.costar.listing import kit_info, listing_record


@pytest.fixture(params=["csv", "jsonl"])
def kit(request, tmp_path, monkeypatch):
    path = tmp_path / "kit.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("index.md", "# fake kit")
        if request.param == "csv":
            z.writestr("residential-listings.csv", "listing_id,beds,baths,sqft\n9001,3,2,1500\n9002,2,1,900\n")
        else:
            z.writestr("data/residential-listings.jsonl",
                       "\n".join(json.dumps(r) for r in [{"ListingId": 9001, "beds": 3, "sqft": 1500},
                                                         {"ListingId": 9002, "beds": 2, "sqft": 900}]))
        z.writestr("commercial-property/1.json", "{}")
    monkeypatch.setenv("COSTAR_KIT_ZIP", str(path))
    monkeypatch.setenv("COSTAR_ZIP_PASSWORD", "not-a-real-password")
    monkeypatch.delenv("COSTAR_KIT_URL", raising=False)
    return request.param


def test_listing_join(kit):
    hit = listing_record("9001")
    assert hit["found"] and str(hit["listing"]["beds"]) == "3" and "residential" in hit["file"]
    assert listing_record("424242")["found"] is False
    info = kit_info()
    assert info["configured"] and len(info["listing_files"]) == 1 and "index.md" in info["files"]


def test_not_configured(monkeypatch):
    from ashline import config
    for name in ("COSTAR_KIT_ZIP", "COSTAR_KIT_URL", "COSTAR_ZIP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [])
    assert listing_record("9001") == {"found": False, "reason": "CoStar kit not configured."}
    assert kit_info()["configured"] is False
