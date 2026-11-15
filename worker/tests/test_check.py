"""The public "check an address" page. The lookup is always injected: no real index is downloaded."""
import re
from html import unescape
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ashline.costar.index import home_lookup
from ashline.web import create_app

FOUND = {"found": True, "match": "address", "sid": "fakeZ999999", "address": "12 Example Ave, Los Angeles, CA 90001",
         "lat": 34.0512, "lon": -118.4021, "homes_listing_id": "88812", "pano_count": 40, "room_count": 14,
         "floor_count": 2, "original_created": "2026-03-04", "distance_m": None, "other_scans_at_address": 0}


def client_for(result=None, lookup=None):
    calls = []

    def fake(address):
        calls.append(address)
        return dict(result)

    return TestClient(create_app(lookup=lookup or fake)), calls


def get(client, address):
    return client.get("/check", params={"address": address}).text


def test_empty_form_runs_no_lookup():
    client, calls = client_for(FOUND)
    html = client.get("/check").text
    assert "Is there a record of your home from before?" in html
    assert 'name="address"' in html and ">Check</button>" in html
    assert "on file" not in html and calls == []


def test_found_with_date_inside_aerial_coverage(monkeypatch):
    monkeypatch.setenv("ASHLINE_NUMBER", "+1 (555) 010-0000")
    client, calls = client_for(FOUND)
    html = get(client, "12 Example Ave, Los Angeles, CA 90001")
    assert calls == ["12 Example Ave, Los Angeles, CA 90001"]
    assert "A 3D scan of this home is on file." in html
    assert "Captured in March 2026" in html
    assert "Covers 14 rooms on 2 floors" in html
    assert "Aerial photos cover this area" in html
    assert "Nothing goes on the list until you say yes." in html
    assert 'href="sms:+15550100000&body=12%20Example%20Ave%2C%20Los%20Angeles%2C%20CA%2090001"' in html


def test_found_without_date_outside_aerial_coverage():
    result = dict(FOUND, original_created=None, lat=40.71, lon=-74.0, room_count=1, floor_count=None)
    client, _ = client_for(result)
    html = get(client, "12 Example Ave, Somewhere, NY 10001")
    assert "A 3D scan of this home is on file." in html
    assert "Captured" not in html
    assert "Covers 1 room</li>" in html and "floor" not in html
    assert "No aerial photos of this area" in html
    assert "sms:" not in html                                   # no number configured


def test_no_internal_identifier_or_coordinate_in_html():
    client, _ = client_for(FOUND)
    html = get(client, "12 Example Ave, Los Angeles, CA 90001")
    for secret in ("fakeZ999999", "88812", "34.05", "-118.40", "34.0512", "118.4021", "pano"):
        assert secret not in html


def test_needs_unit_chips_rerun_with_unit(index):
    client, _ = client_for(lookup=lambda address: home_lookup(address=address, index=index))
    html = get(client, "500 N Sample Blvd, Los Angeles, CA 90003")
    assert "Which unit is yours?" in html
    links = re.findall(r'<a class="chip" href="([^"]+)">Unit (\w+)</a>', html)
    assert [label for _, label in links] == ["7", "9"]
    addresses = [parse_qs(urlparse(unescape(href)).query)["address"][0] for href, _ in links]
    assert addresses == ["500 N Sample Blvd Unit 7, Los Angeles, CA 90003", "500 N Sample Blvd Unit 9, Los Angeles, CA 90003"]
    again = client.get(unescape(links[1][0])).text
    assert "A 3D scan of this home is on file." in again and "Captured in September 2025" in again
    assert "fakeC" not in html + again and "Unit 7" not in again      # no neighbour, no scan id


def test_not_found_is_kind():
    client, _ = client_for({"found": False, "reason": "No 3D scan on file for this home."})
    html = get(client, "1 Nowhere Rd, Testville, CA 90000")
    assert "There is no 3D scan of this home on file." in html
    assert "That is normal" in html and "own photos and videos" in html
    assert "No 3D scan on file for this home." not in html          # the tool's reason is not shown


def test_lookup_failure_does_not_crash():
    def broken(address):
        raise OSError("index unavailable")
    client, _ = client_for(lookup=broken)
    html = get(client, "12 Example Ave, Los Angeles, CA 90001")
    assert "could not check right now" in html


def test_address_is_escaped():
    client, _ = client_for({"found": False, "reason": "none"})
    html = get(client, '<script>alert(1)</script> 12 Example Ave')
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; 12 Example Ave" in html


def test_landing_links_to_check():
    client, _ = client_for(FOUND)
    assert 'href="/check"' in client.get("/").text
