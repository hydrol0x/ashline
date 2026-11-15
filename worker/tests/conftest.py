"""Shared fixtures. Every address, id and coordinate here is made up: real
CoStar, Matterport and Vexcel data must never be committed."""
import csv

import pytest

from ashline.costar.index import HomeIndex

COLUMNS = ["sid", "name", "segment", "lat", "lon", "latlon_source", "address", "city", "state",
           "postal_code", "internal_id", "costar_property_id", "homes_listing_id", "splat",
           "drone_photos", "pano_count", "room_count", "floor_count", "created", "original_created",
           "defurnished", "ai_summary"]

ROWS = [
    # sid, segment, lat, lon, address, city, state, zip, listing, panos, rooms, floors, captured
    ("fakeA000001", "residential", 34.0500, -118.4000, "12 Example Ave, Los Angeles, CA, 90001, US", "Los Angeles", "CA", "90001", "9001", 40, 14, 1, "2026-03-04"),
    ("fakeA000002", "residential", 34.0500, -118.4000, "12 Example Ave, Los Angeles, CA, 90001, US", "Los Angeles", "CA", "90001", "9002", 44, 15, 1, "2024-06-01"),
    ("fakeB000001", "residential", 34.0600, -118.4100, "12 Example St, Los Angeles, CA, 90002, US", "Los Angeles", "CA", "90002", "9003", 30, 12, 2, "2025-05-05"),
    ("fakeC000001", "residential", 34.0700, -118.4200, "500 N Sample Blvd Unit 7, 7, Los Angeles, CA, 90003, US", "Los Angeles", "CA", "90003", "9004", 20, 8, 1, "2025-08-08"),
    ("fakeC000002", "residential", 34.0700, -118.4200, "500 N Sample Blvd, Unit 9, Los Angeles, CA, 90003, US", "Los Angeles", "CA", "90003", "9005", 22, 9, 1, "2025-09-09"),
    ("fakeD000001", "residential", 34.0800, -118.4300, "77 Mockingbird Point Ln, , , , US", "", "", "", "", 35, 13, 1, "2026-01-01"),
    ("fakeE000001", "residential", 29.7600, -95.3700, "9 S Placeholder Dr, Houston, Texas, 77001, US", "Houston", "Texas", "77001", "9006", 50, 18, 2, "2026-02-02"),
    ("fakeF000001", "commercial", 34.0500, -118.4000, "12 Example Ave, Los Angeles, CA, 90001, US", "Los Angeles", "CA", "90001", "", 90, 40, 3, "2026-04-04"),
    ("fakeG000001", "residential", "", "", "3 Nolatlon Ct, Los Angeles, CA, 90004, US", "Los Angeles", "CA", "90004", "9007", 31, 12, 1, "2026-05-05"),
]


def _row(t):
    sid, segment, lat, lon, address, city, state, postal, listing, panos, rooms, floors, captured = t
    row = dict.fromkeys(COLUMNS, "")
    row.update(sid=sid, name=address.split(",")[0], segment=segment, lat=str(lat), lon=str(lon),
               address=address, city=city, state=state, postal_code=postal, homes_listing_id=listing,
               pano_count=str(panos), room_count=str(rooms), floor_count=str(floors),
               created="2026-11-09", original_created=captured)
    return row


@pytest.fixture
def index_csv(tmp_path):
    path = tmp_path / "models_index.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_row(t) for t in ROWS)
    return path


@pytest.fixture
def index(index_csv):
    return HomeIndex.from_csv(index_csv)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    # A developer's .env and shell hold real keys and the real CoStar kit. Tests must never read
    # them: no paid API call, no real data. Tests that need a setting set it themselves.
    from ashline import config
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [])
    for name in ("MATTERPORT_API_KEY", "MATTERPORT_API_TOKEN", "VEXCEL_API_KEY", "COSTAR_KIT_URL",
                 "COSTAR_KIT_ZIP", "COSTAR_ZIP_PASSWORD", "ANTHROPIC_API_KEY", "ASHLINE_VLM_API_KEY",
                 "ASHLINE_DETECTOR", "ASHLINE_DEMO_SID", "ASHLINE_DEMO_ADDRESS", "ASHLINE_LOSS_DATE",
                 "ASHLINE_INDEX_CSV", "ASHLINE_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASHLINE_DATA_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("ASHLINE_MERGER", "rules")


@pytest.fixture(autouse=True)
def _full_setup_questions(monkeypatch, request):
    """The demo turns the address and what-happened questions off; the tests cover the full setup."""
    if "demo_setup" not in request.node.name:
        monkeypatch.setenv("ASHLINE_ASK_ADDRESS", "on")
        monkeypatch.setenv("ASHLINE_ASK_WHAT_HAPPENED", "on")
