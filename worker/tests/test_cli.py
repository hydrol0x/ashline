import json

import pytest

from ashline import cli, config, seed
from ashline.costar import index as index_mod

from fakes import FakeMatterport, FakeVexcel

REAL_MATTERPORT_CLIENT = seed.matterport.MatterportClient


@pytest.fixture
def offline(index, monkeypatch, tmp_path):
    monkeypatch.setattr(index_mod, "_INDEX", index)
    monkeypatch.setattr(seed.matterport, "MatterportClient", FakeMatterport)
    monkeypatch.setattr(seed.vexcel, "VexcelClient", FakeVexcel)
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [tmp_path / ".env"])
    for name in ("ASHLINE_DEMO_SID", "ASHLINE_DEMO_ADDRESS", "ASHLINE_LOSS_DATE"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / ".env"


def test_seed_without_a_home_says_what_to_set(offline, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["seed", "--case", "demo"])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2 and "ASHLINE_DEMO_SID" in err and str(offline) in err


def test_seed_by_scan_id_from_dotenv(offline, capsys):
    offline.write_text('ASHLINE_DEMO_SID=fakeA000001   # the stand-in home\nASHLINE_LOSS_DATE=2025-01-07\n')
    cli.main(["seed", "--case", "demo", "--no-detect"])
    out = json.loads(capsys.readouterr().out)
    assert out["lookup"]["match"] == "sid" and out["inside"]["faces"] == 36
    assert out["outside"]["capture_date"] == "2024-08-13"


def test_seed_by_flag_and_quoted_address_with_hash(offline, capsys, monkeypatch):
    cli.main(["seed", "--case", "a", "--sid", "fakeB000001", "--no-detect"])
    assert json.loads(capsys.readouterr().out)["lookup"]["sid"] == "fakeB000001"
    offline.write_text('ASHLINE_DEMO_ADDRESS="500 N Sample Blvd #9, Los Angeles"\n')
    cli.main(["seed", "--case", "b", "--no-detect"])
    assert json.loads(capsys.readouterr().out)["lookup"]["sid"] == "fakeC000002"


def test_missing_key_is_one_line_not_a_traceback(offline, capsys, monkeypatch, index):
    monkeypatch.setattr(seed.matterport, "MatterportClient", REAL_MATTERPORT_CLIENT)
    monkeypatch.delenv("MATTERPORT_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["seed", "--case", "demo", "--sid", "fakeA000001", "--no-detect"])
    assert "MATTERPORT_API_KEY is not set" in capsys.readouterr().err
