"""Amazon order-history files read as purchases and matched to inventory lines.

Every file here is synthetic: written from the documented column names, not copied from a real export.
"""
import json

import pytest

from ashline import bank, cli, orders
from test_chat import MOM, Family, photo, texts

# The "Request my data" layout (Retail.OrderHistory.1.csv), a subset of its columns, with a byte-order mark.
REQUEST_MY_DATA = (
    "﻿\"Website\",\"Order ID\",\"Order Date\",\"Currency\",\"Unit Price\",\"Unit Price Tax\",\"Total Owed\","
    "\"ASIN\",\"Quantity\",\"Order Status\",\"Product Name\"\n"
    "\"Amazon.com\",\"112-0000001-0000001\",\"2024-03-12T18:22:31Z\",\"USD\",\"1065.00\",\"84.00\",\"$1,149.00\","
    "\"B000000001\",\"1\",\"Closed\",\"Blue Velvet Sofa, 3-Seat, Mid-Century\"\n"
    "\"Amazon.com\",\"112-0000001-0000001\",\"2024-03-12T18:22:31Z\",\"USD\",\"89.00\",\"7.00\",\"96.00\","
    "\"B000000002\",\"1\",\"Cancelled\",\"Brass Floor Lamp\"\n"
    "\"Amazon.com\",\"112-0000002-0000002\",\"2023-11-25T09:01:02.000Z\",\"USD\",\"87.50\",\"7.00\",\"189.00\","
    "\"B000000003\",\"2\",\"Closed\",\"Brass Floor Lamp with Marble Base\"\n"
    "\"Amazon.com\",\"112-0000003-0000003\",\"2022-01-02T10:00:00Z\",\"USD\",\"9.99\",\"0.80\",\"10.79\","
    "\"B000000004\",\"1\",\"Closed\",\"USB-C Cable, 6 ft\"\n"
    "\"Amazon.com\",\"112-0000004-0000004\",\"2022-02-02T10:00:00Z\",\"USD\",\"0\",\"0\",\"0.00\","
    "\"B000000005\",\"1\",\"Closed\",\"Free sample\"\n"
    "\"Amazon.com\",\"112-0000005-0000005\",\"2022-03-02T10:00:00Z\",\"USD\",\"5.00\",\"0\",\"5.00\","
    "\"B000000006\",\"1\",\"Closed\",\"\"\n"
).encode("utf-8")

# The older order-report layout: MM/DD/YY dates, "Title", a price per unit and an item total.
OLD_REPORT = (
    "Order Date,Order ID,Title,Category,Purchase Price Per Unit,Quantity,Order Status,Item Total\n"
    "03/12/24,111-0000009-0000009,Oak Coffee Table,Furniture,$389.00,1,Shipped,\"$1,420.50\"\n"
    "04/01/24,111-0000010-0000010,Throw Pillow Covers,Home,$12.50,4,Shipped,\n"
    "05/01/24,111-0000011-0000011,Desk Lamp,Lighting,$30.00,1,Returned,$32.10\n"
)


def test_the_request_my_data_file_is_read_as_purchases():
    rows = orders.read_orders(REQUEST_MY_DATA)
    assert rows[0] == {"id": "112-0000001-0000001#1", "date": "2024-03-12", "amount": 1149.0,
                       "description": "Blue Velvet Sofa, 3-Seat, Mid-Century", "merchant": "Amazon", "account": "",
                       "order": "112-0000001-0000001"}
    assert [r["id"] for r in rows] == ["112-0000001-0000001#1", "112-0000002-0000002#1", "112-0000003-0000003#1"]
    lamp = rows[1]                                  # the whole line, for both lamps, with tax
    assert lamp["amount"] == 189.0 and lamp["description"] == "2 x Brass Floor Lamp with Marble Base"
    assert lamp["date"] == "2023-11-25"
    # Cancelled, zero and nameless lines are left out; the same file read as text gives the same lines.
    assert orders.read_orders(REQUEST_MY_DATA.decode("utf-8")) == rows


def test_an_older_report_layout_is_read_too_and_an_amount_is_never_guessed():
    rows = orders.read_orders(OLD_REPORT)
    assert [(r["date"], r["amount"], r["description"]) for r in rows] == [
        ("2024-04-01", 50.0, "4 x Throw Pillow Covers"),             # no item total: price per unit times quantity
        ("2024-03-12", 1420.5, "Oak Coffee Table")]                  # the item total wins over the unit price
    without_amounts = "Order ID,Title,Quantity\n111-1,Oak Coffee Table,1\n"
    assert orders.read_orders(without_amounts) == []


def test_only_an_order_history_is_taken_for_one():
    assert orders.looks_like_orders("Retail.OrderHistory.1.csv", REQUEST_MY_DATA)
    assert orders.looks_like_orders("orders.csv", OLD_REPORT.encode())
    assert not orders.looks_like_orders("budget.csv", b"Date,Description,Amount\n2024-01-01,Rent,1200\n")
    assert not orders.looks_like_orders("photo.jpg", REQUEST_MY_DATA)
    assert not orders.looks_like_orders("empty.csv", b"")
    assert orders.read_orders(b"Date,Description,Amount\n2024-01-01,Rent,1200\n") == []


def test_order_lines_are_matched_applied_once_and_undone():
    f = Family().setup()
    f.say(MOM, photos=[photo(640)])
    f.wait()
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    lamp = next(i for i in f.store.get_items(f.case_id) if "lamp" in i["label"])
    estimate = sofa["price_mid"]

    matches = orders.find_matches(f.case_id, f.store, orders.read_orders(REQUEST_MY_DATA))
    assert {m["item"]["item_id"]: m["purchase"]["id"] for m in matches} == {
        sofa["item_id"]: "112-0000001-0000001#1", lamp["item_id"]: "112-0000002-0000002#1"}   # not the cable
    assert bank.describe(next(m for m in matches if m["item"]["item_id"] == sofa["item_id"])).startswith(
        "blue velvet sofa: $1,149 at Amazon on 2024-03-12")

    assert bank.apply_matches(f.store, f.case_id, matches, member_id=None) == 2
    after = f.store.get_item(f.case_id, sofa["item_id"])
    assert after["price_mid"] == 1149.0 and after["price_source"] == "receipt: Amazon"
    assert json.loads(after["price_ref"])["paid"]["bank_id"] == "112-0000001-0000001#1"
    event = f.store.last_event(f.case_id)
    assert event["action"] == "bank" and len(event["payload"]["before"]) == 2

    assert orders.find_matches(f.case_id, f.store, orders.read_orders(REQUEST_MY_DATA)) == []     # nothing new

    assert texts(f.say(MOM, "undo")) == ["Undone: those prices are back to the estimates."]
    back = f.store.get_item(f.case_id, sofa["item_id"])
    assert back["price_mid"] == estimate and back["price_source"] != "bank record: Amazon"


def test_a_price_the_family_set_is_left_alone():
    f = Family().setup()
    f.say(MOM, photos=[photo(640)])
    f.wait()
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    f.store.update_item(f.case_id, sofa["item_id"], price_mid=1000, price_source="set by family")
    matches = orders.find_matches(f.case_id, f.store, orders.read_orders(REQUEST_MY_DATA))
    assert sofa["item_id"] not in {m["item"]["item_id"] for m in matches}


def test_the_orders_command_reads_matches_and_applies(tmp_path, capsys):
    f = Family().setup()
    f.say(MOM, photos=[photo(640)])
    f.wait()
    path = tmp_path / "Retail.OrderHistory.1.csv"
    path.write_bytes(REQUEST_MY_DATA)
    cli.main(["orders", "--case", f.case_id, str(path)])
    out = capsys.readouterr().out
    assert out.startswith("Read 3 paid order line(s). 2 match items on the list:") and "--apply" in out
    assert not any((i.get("price_source") or "").startswith("bank") for i in f.store.get_items(f.case_id))

    cli.main(["orders", "--case", f.case_id, str(path), "--apply"])
    assert "Prices updated on 2 item(s)." in capsys.readouterr().out
    cli.main(["orders", "--case", f.case_id, str(path)])
    assert "0 match items on the list." in capsys.readouterr().out

    other = tmp_path / "budget.csv"
    other.write_text("Date,Description,Amount\n2024-01-01,Rent,1200\n")
    with pytest.raises(SystemExit):
        cli.main(["orders", "--case", f.case_id, str(other)])
    assert "does not look like an order history" in capsys.readouterr().err
