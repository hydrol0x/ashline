import pytest

from ashline.costar.index import home_lookup, parse_address, split_street


def test_exact_address_returns_newest_scan(index):
    res = home_lookup("12 Example Ave, Los Angeles, CA, 90001, US", index=index)
    assert res["found"] and res["match"] == "address"
    assert res["sid"] == "fakeA000001"            # 2026 capture beats the 2024 one
    assert res["other_scans_at_address"] == 1
    assert res["original_created"] == "2026-03-04"
    assert res["homes_listing_id"] == "9001"
    assert (res["pano_count"], res["room_count"], res["floor_count"]) == (40, 14, 1)
    assert res["lat"] == pytest.approx(34.05) and res["lon"] == pytest.approx(-118.4)


@pytest.mark.parametrize("text", [
    "12 example avenue, los angeles, ca 90001",
    "12 Example Ave",
    "12 EXAMPLE AVE 90001",
    "12 example avenue los angeles",
    "12 Example Ave., Brentwood, CA",            # a neighbourhood name the index does not use
])
def test_the_way_people_type_it(index, text):
    assert home_lookup(text, index=index)["sid"] == "fakeA000001"


def test_suffix_and_zip_tell_streets_apart(index):
    assert home_lookup("12 Example St", index=index)["sid"] == "fakeB000001"
    assert home_lookup("12 Example Ave, Los Angeles, CA 90002", index=index)["found"] is False
    assert home_lookup("12 Example", index=index)["found"] is True      # suffix left out


def test_direction_must_not_contradict(index):
    assert home_lookup("9 S Placeholder Dr, Houston", index=index)["sid"] == "fakeE000001"
    assert home_lookup("9 Placeholder Drive, Houston TX", index=index)["sid"] == "fakeE000001"
    assert home_lookup("9 N Placeholder Dr, Houston", index=index)["found"] is False


def test_units(index):
    assert home_lookup("500 N Sample Blvd #9, Los Angeles", index=index)["sid"] == "fakeC000002"
    assert home_lookup("500 N Sample Blvd Apt 7", index=index)["sid"] == "fakeC000001"
    assert home_lookup("500 N Sample Blvd Unit 8", index=index)["found"] is False
    ambiguous = home_lookup("500 N Sample Blvd, Los Angeles", index=index)
    assert ambiguous["found"] is False and ambiguous["needs_unit"] and ambiguous["units"] == ["7", "9"]


def test_row_with_empty_city_state_zip(index):
    assert home_lookup("77 Mockingbird Point Lane", index=index)["sid"] == "fakeD000001"
    # "Point" is part of the street name here, not a suffix to ignore.
    assert home_lookup("77 Mockingbird Ln", index=index)["found"] is False


def test_typo_needs_matching_zip(index):
    assert home_lookup("12 Exampel Ave, Los Angeles, CA 90001", index=index)["sid"] == "fakeA000001"
    assert home_lookup("12 Exampel Ave", index=index)["found"] is False


def test_no_scan_is_a_clean_answer(index):
    for text in ["1 Nowhere Ln, Pacific Palisades, CA 90272", "hello", "Los Angeles"]:
        res = home_lookup(text, index=index)
        assert res["found"] is False and "photos" in res["reason"]
    assert home_lookup("", index=index)["found"] is False
    assert home_lookup(index=index)["found"] is False


def test_commercial_rows_are_ignored(index):
    assert all(r["segment"] == "residential" for r in index.rows)


def test_nearest_within_30_m(index):
    near = home_lookup(lat=34.0601, lon=-118.4100, index=index)          # about 11 m north
    assert near["found"] and near["match"] == "nearest" and near["sid"] == "fakeB000001"
    assert 5 < near["distance_m"] < 20
    assert home_lookup(lat=34.0610, lon=-118.4100, index=index)["found"] is False   # about 110 m


def test_address_miss_falls_back_to_lat_lon(index):
    res = home_lookup("1 Unknown Rd", lat=34.0600, lon=-118.4100, index=index)
    assert res["sid"] == "fakeB000001" and res["match"] == "nearest"


def test_parse_address_parts():
    p = parse_address("505 South Demo Avenue Unit 109, 109, Los Angeles, CA, 90049, US")
    assert (p.number, p.unit, p.postal_code) == ("505", "109", "90049")
    assert p.street == ("s", "demo", "ave") and p.rest == frozenset({"los", "angeles", "ca"})
    assert split_street(p.street) == ("s", ("demo",), "ave", "")
    assert split_street(("s", "blvd")) == ("", ("s",), "blvd", "")
