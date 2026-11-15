"""The Claude room merge against a stand-in client: no API call is made."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from ashline import seed
from ashline.dedupe import cluster_scan_items
from ashline.merge_claude import GROUPS_SCHEMA, ClaudeGrouper, MergeError, room_lines


class FakeMessages:
    def __init__(self, replies, stop_reason="end_turn"):
        self.replies, self.stop_reason, self.calls = list(replies), stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def fake_client(*replies, stop_reason="end_turn"):
    beta = FakeMessages(replies, stop_reason)
    return SimpleNamespace(messages=FakeMessages([]), beta=SimpleNamespace(messages=beta)), beta


def det(item_id, pano, face, label, room="r1", category="furniture"):
    return {"item_id": item_id, "photo_id": f"{pano}-f{face}", "label": label, "bbox": [500, 900, 800, 1500],
            "category": category, "room_id": room, "room_label": "Living Room", "confidence": 0.8}


def faces(*panos):
    return {f"{p}-f{f}": {"pano_id": p, "face": f, "width": 2048} for p in panos for f in (1, 2, 3, 4)}


# Three viewpoints describe the same four stools three ways; word matching leaves them apart.
ITEMS = [
    det(1, "p1", 1, "cream swivel bar stool with walnut frame"),
    det(2, "p1", 2, "cream swivel bar stool with walnut frame"),
    det(3, "p1", 3, "cream bar stool, partly visible"),
    det(4, "p2", 1, "swivel counter seat with cream cushion and wooden legs"),
    det(5, "p2", 2, "swivel counter seat with cream cushion and wooden legs"),
    det(6, "p3", 1, "mid-century wooden stool with cane back"),
    det(7, "p1", 4, "Moomin plush toy", category="toys"),
    det(8, "p2", 3, "Totoro plush toy", category="toys"),
]
PHOTOS = faces("p1", "p2", "p3")
REPLY = json.dumps({"groups": [{"ids": [0, 1, 2, 3, 4, 5], "best_id": 0}, {"ids": [6], "best_id": 6},
                               {"ids": [7], "best_id": 7}]})


def test_rules_alone_leave_the_stools_apart():
    stool_lines = [c for c in cluster_scan_items(ITEMS, PHOTOS) if c["key"] in ("stool", "seat")]
    assert len(stool_lines) > 1


def test_claude_groups_them_and_the_count_stays_what_one_panorama_saw():
    client, beta = fake_client(REPLY)
    clusters = cluster_scan_items(ITEMS, PHOTOS, ClaudeGrouper(client=client))
    by_canon = {c["canonical_id"]: c for c in clusters}
    assert len(clusters) == 3
    assert sorted(by_canon[1]["member_ids"]) == [1, 2, 3, 4, 5, 6]
    assert by_canon[1]["qty"] == 3                 # panorama p1 saw three; the merge cannot raise that
    assert by_canon[7]["qty"] == 1 and by_canon[8]["qty"] == 1          # two different toys stay two lines

    call = beta.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": GROUPS_SCHEMA}
    prompt = call["messages"][0]["content"]
    assert isinstance(prompt, str)                 # text only: no image is sent
    assert "(Living Room), from 3 photos or panoramas" in prompt
    assert "0 | A | furniture | cream swivel bar stool with walnut frame" in prompt
    assert "7 | B | toys | Totoro plush toy" in prompt


def test_a_careless_reply_cannot_lose_or_duplicate_items():
    sloppy = json.dumps({"groups": [{"ids": [0, 1, 99], "best_id": 42},      # unknown id, best_id not in group
                                    {"ids": [1, 2], "best_id": 2},            # id 1 repeated
                                    {"ids": [], "best_id": 0}]})              # empty group; ids 3..7 never mentioned
    client, _ = fake_client(sloppy)
    clusters = cluster_scan_items(ITEMS, PHOTOS, ClaudeGrouper(client=client))
    members = sorted(m for c in clusters for m in c["member_ids"])
    assert members == [1, 2, 3, 4, 5, 6, 7, 8]                               # each detection exactly once
    assert [sorted(c["member_ids"]) for c in clusters][:2] == [[1, 2], [3]]
    assert all(c["canonical_id"] in c["member_ids"] for c in clusters)


def test_answers_are_cached_per_case():
    client, beta = fake_client(REPLY)
    first = cluster_scan_items(ITEMS, PHOTOS, ClaudeGrouper("fam1", client=client))
    again = cluster_scan_items(ITEMS, PHOTOS, ClaudeGrouper("fam1", client=client))    # no second reply queued
    assert len(beta.calls) == 1 and again == first
    cluster_scan_items(ITEMS[:2], PHOTOS, ClaudeGrouper("fam1", client=fake_client(
        json.dumps({"groups": [{"ids": [0, 1], "best_id": 0}]}))[0]))                   # different list: asked again


def test_single_object_rooms_are_not_sent():
    client, beta = fake_client()
    assert len(cluster_scan_items(ITEMS[:1], PHOTOS, ClaudeGrouper(client=client))) == 1
    assert beta.calls == []


def test_bad_replies_raise_and_the_seed_falls_back_to_rules():
    for client, _ in (fake_client("not json"), fake_client('{"groups": 5}'), fake_client("", stop_reason="refusal")):
        with pytest.raises(MergeError):
            cluster_scan_items(ITEMS, PHOTOS, ClaudeGrouper(client=client))
    client, _ = fake_client("not json")
    clusters, how = seed.merge_scan_items("fam1", ITEMS, PHOTOS, ClaudeGrouper(client=client))
    assert how.startswith("rules (Claude merge failed") and sorted(m for c in clusters for m in c["member_ids"]) == list(range(1, 9))
    clusters, how = seed.merge_scan_items("fam1", ITEMS, PHOTOS, ClaudeGrouper(client=fake_client(REPLY)[0]))
    assert how == "claude" and len(clusters) == 3


def test_merger_choice(monkeypatch, tmp_path):
    from ashline import config
    monkeypatch.setattr(config, "_dotenv_paths", lambda: [tmp_path / ".env"])
    monkeypatch.delenv("ASHLINE_MERGER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert seed.merger_kind() == "rules"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert seed.merger_kind() == "claude"
    monkeypatch.setenv("ASHLINE_MERGER", "rules")
    assert seed.merger_kind() == "rules"
    assert seed.merge_scan_items("fam1", ITEMS, PHOTOS)[1] == "rules"


def test_room_lines_letter_panoramas():
    objs = [{"pano": "x", "label": "a", "category": None}, {"pano": "y", "label": "b", "category": "decor"},
            {"pano": "x", "label": "c", "category": "toys"}]
    text, count = room_lines(objs)
    assert text.splitlines() == ["0 | A | other | a", "1 | B | decor | b", "2 | A | toys | c"] and count == 2
