"""Things a stranger, or an odd name, must not be able to break: the files an insurer opens, the
download header, the bridge routes and the command line."""
import io

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ashline import cli, export
from ashline.store import Store
from ashline.web import create_app


@pytest.fixture
def home():
    store = Store()
    store.set_case("fam1", name="The Example home", owner_token="owner-token", member_token="member-token")
    store.add_member("fam1", "+15550001", "Mom", "owner")
    item = store.add_item("fam1", {"label": '=HYPERLINK("http://example.invalid","click")', "category": "decor",
                                   "room_label": "+1 room"}, source="voice", added_by="@Mom")
    store.update_item("fam1", item, price_low=60, price_mid=100, price_high=150, price_source="claude estimate")
    store.set_item_status("fam1", item, "confirmed")
    return TestClient(create_app()), store


def test_a_name_that_looks_like_a_formula_stays_text(home):
    client, _ = home
    assert export.plain_cell("=1+1") == "'=1+1" and export.plain_cell("-2") == "'-2" and export.plain_cell("@x") == "'@x"
    assert export.plain_cell("sofa") == "sofa" and export.plain_cell(3.5) == 3.5 and export.plain_cell(None) is None
    text = client.get("/p/owner-token/export.csv").text
    assert "'=HYPERLINK" in text and "\n+1 room" not in text and ",=HYPERLINK" not in text
    from openpyxl import load_workbook
    sheet = load_workbook(io.BytesIO(client.get("/p/owner-token/export.xlsx").content))["Inventory"]
    cells = {cell.value: cell for cell in sheet[2]}
    name = next(value for value in cells if isinstance(value, str) and "HYPERLINK" in value)
    assert name.startswith("'=") and cells[name].data_type == "s"                # text, not a live formula
    assert sheet["A2"].value == "'+1 room"


def test_every_spreadsheet_column_has_a_heading(home):
    client, _ = home
    from openpyxl import load_workbook
    sheet = load_workbook(io.BytesIO(client.get("/p/owner-token/export.xlsx").content))["Inventory"]
    headings = [cell.value for cell in sheet[1]]
    assert len(headings) == len(export.COLUMNS) and all(headings) and headings[-1] == "Price basis"


@pytest.mark.parametrize("name", ["Nguyễn home 日本", 'La casa de José "Pepe"', "   ", "The Example home"])
def test_a_home_named_in_any_script_still_downloads(home, name):
    client, store = home
    store.set_case("fam1", name=name)
    for kind in ("csv", "xlsx", "pdf"):
        got = client.get(f"/p/owner-token/export.{kind}")
        assert got.status_code == 200
        header = got.headers["content-disposition"]
        header.encode("latin-1")                                                  # what a header can carry
        plain = header.split('filename="')[1].split('"')[0]
        assert plain.endswith("." + kind) and plain.isascii() and " " not in plain and len(plain) > len(kind) + 1
        assert "filename*=UTF-8''" in header


def test_the_bridge_routes_answer_this_machine_only_without_a_secret(home, monkeypatch):
    client, _ = home
    monkeypatch.delenv("ASHLINE_INBOUND_SECRET", raising=False)
    said = {"sender_phone": "+15550001", "chat_id": "+15550001", "text": "list"}
    assert client.post("/inbound", json=said).status_code == 200                # a caller in this process
    through_a_tunnel = client.post("/inbound", json=said, headers={"X-Forwarded-For": "203.0.113.9"})
    assert through_a_tunnel.status_code == 401 and "owner-token" not in through_a_tunnel.text
    assert client.post("/tick", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 401
    from_elsewhere = TestClient(create_app(), client=("203.0.113.9", 50000))
    assert from_elsewhere.post("/inbound", json=said).status_code == 401
    assert TestClient(create_app(), client=("127.0.0.1", 50000)).post("/inbound", json=said).status_code == 200
    monkeypatch.setenv("ASHLINE_INBOUND_SECRET", "s3cret")                      # with a secret, it is the only key
    assert from_elsewhere.post("/inbound", json=said, headers={"X-Ashline-Secret": "s3cret"}).status_code == 200
    assert from_elsewhere.post("/inbound", json=said, headers={"X-Ashline-Secret": "wrong"}).status_code == 401
    assert client.post("/inbound", json=said).status_code == 401


def test_the_command_line_says_so_plainly(home, capsys, tmp_path):
    for words in (["export", "--case", "nope"], ["questions", "--case", "nope"], ["delete", "--case", "nope"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(words)
        assert stopped.value.code == 1 and "No case 'nope'" in capsys.readouterr().err
    assert Store().get_case("nope") is None
    (tmp_path / "a-file").write_text("not a folder")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["plan", "--case", "fam1", "--png", str(tmp_path / "a-file" / "plan.png")])
    assert stopped.value.code == 1 and "Cannot write" in capsys.readouterr().err
