"""Cross-panorama de-duplication (build spec 6.4) decided by Claude, per room.

Detections of one room, written from several panoramas with no knowledge of
each other, describe the same object in different words. Matching words cannot
tell that "navy three-seat sofa" and "dark blue fabric couch" are one sofa
while two plush toys are two things; a model reading the list can.

One text-only request per room, no images. The reply only groups ids: it
writes no labels and no quantities. The line's wording is always one of the
original detections, and its quantity is computed in `dedupe.py` from what a
single panorama showed, so this step can merge lines but cannot invent items
or inflate counts. Answers are cached per case.
"""
from __future__ import annotations

import hashlib
import json

import anthropic

from . import config
from .detect_claude import DEFAULT_MODEL, FALLBACK_BETA, FALLBACK_MODELS

GROUPS_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "integer"}},
                    "best_id": {"type": "integer",
                                "description": "The id in this group whose label names the line best"},
                },
                "required": ["ids", "best_id"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}

PROMPT = """You are tidying a home contents inventory for an insurance claim.

Below are objects detected in one room of a home ({room}), from {panoramas} 360-degree panoramas taken at
different spots in that room. Each line is: id | panorama | category | label.

The same physical object is usually detected from more than one panorama, and each detection was
written without knowledge of the others, so the wording differs: "navy three-seat sofa with
wooden legs" and "dark blue fabric couch" can be the same sofa.

Group the ids so that each group is one line of the inventory:
- Put together detections of the same physical object seen from different panoramas.
- Also put together identical matching objects that belong on one line, such as a set of matching
  dining chairs or a pair of matching nightstands, from the same or different panoramas.
- Two detections from the same panorama are always two different physical objects. Group them only
  if they are a matching set.
- An object described as a reflection in a mirror is the same object as the real one in the room.
- Do not group things that are merely the same type: two different plush toys, a TV and its
  remote, a spray can and a trash can stay separate. When unsure, keep them separate; a person
  reviews every line.

For each group give `ids` and `best_id`, the id whose label names the line best (a complete view,
the most specific wording). Every id must appear in exactly one group; an object with no match is
a group of one.

{lines}"""


class MergeError(RuntimeError):
    pass


def room_lines(objects: list[dict]) -> tuple[str, int]:
    """The id | panorama | category | label table for a room, and how many panoramas it spans."""
    letters: dict = {}
    rows = []
    for index, obj in enumerate(objects):
        letter = letters.setdefault(obj["pano"], chr(ord("A") + len(letters) % 26) * (1 + len(letters) // 26))
        rows.append(f"{index} | {letter} | {obj.get('category') or 'other'} | {obj['label']}")
    return "\n".join(rows), len(letters)


class ClaudeGrouper:
    """Callable for `cluster_scan_items(..., grouper=...)`."""

    def __init__(self, case_id: str | None = None, model: str | None = None, effort: str | None = None,
                 client: anthropic.Anthropic | None = None):
        config.load_env()
        self.case_id = case_id
        self.model = model or config.env("ASHLINE_CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = effort or config.env("ASHLINE_CLAUDE_EFFORT", "medium")
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:                       # only built when a room is not cached
            self._client = anthropic.Anthropic()
        return self._client

    def _cache_path(self, prompt: str):
        if not self.case_id:
            return None
        folder = config.case_dir(self.case_id) / "merge"
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return folder / f"{digest}.json"

    def __call__(self, room_label: str | None, objects: list[dict]) -> list[tuple[list[int], int | None]]:
        if len(objects) < 2:
            return [([i], None) for i in range(len(objects))]
        lines, panoramas = room_lines(objects)
        prompt = PROMPT.format(room=room_label or "room", panoramas=panoramas, lines=lines)
        cache = self._cache_path(prompt)
        if cache is not None and cache.exists():
            groups = json.loads(cache.read_text())["groups"]
        else:
            groups = self._ask(prompt)
            if cache is not None:
                cache.write_text(json.dumps({"room": room_label, "prompt": prompt, "groups": groups}, indent=1))
        return [([i for i in g.get("ids", []) if isinstance(i, int)], g.get("best_id")) for g in groups
                if isinstance(g, dict)]

    def _ask(self, prompt: str) -> list[dict]:
        request = dict(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": GROUPS_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            if self.model.startswith(FALLBACK_MODELS):
                response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **request)
            else:
                response = self.client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise MergeError("Anthropic rejected the key. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.RateLimitError as exc:
            raise MergeError("Anthropic rate limit hit after retries.") from exc
        except anthropic.APIStatusError as exc:
            raise MergeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise MergeError("Could not reach the Anthropic API.") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise MergeError(f"Claude stopped early ({response.stop_reason}).")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            groups = json.loads(text)["groups"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MergeError(f"Unexpected reply from Claude: {text[:200]}") from exc
        if not isinstance(groups, list):
            raise MergeError("Unexpected reply from Claude: groups is not a list.")
        return groups
