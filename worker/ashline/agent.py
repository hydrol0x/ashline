"""Ashline orchestrator on LangChain deepagents, with the home twin seed.

deepagents features used:
- Subagents: `inside` (scan rooms and photos) and `outside` (parcel and yard)
  each do their work and hand a short summary back to the orchestrator.
- Planning: the built-in to-do tool holds the per-case plan.
- Virtual filesystem: the orchestrator writes one inventory note per room.
- Human-in-the-loop: `interrupt_on` gates the tools that confirm or reject
  something, so they wait for a family reply. Nothing is added unconfirmed.
"""
from __future__ import annotations

from . import config, seed
from .costar.index import home_lookup as _home_lookup
from .gaps import gap_questions as _gap_questions
from .store import Store

ORCHESTRATOR_PROMPT = """You are Ashline. You help a family that lost their home in a wildfire rebuild
the contents inventory their insurer asks for. You talk over text messages.

Tone: gentle, short messages, never pushy. The family can pause at any time.
Hard rule: everything you find is a suggestion. Never confirm, add or merge an item unless a
family member said so. A listing scan can show staging or a previous owner's furniture.

When a case starts and you have the address:
1. Call home_lookup. If there is no scan, say so plainly and carry on with family photos.
   If it needs a unit number, ask for it.
2. If there is a scan, delegate the inside work to the `inside` subagent and the outside work to
   the `outside` subagent (use the task tool; give each the case_id and what it needs). With no
   scan but a lat/lon, still run `outside`.
3. Send the family ONE message: what the scan shows and when it was captured, any photographer
   notes, and ask whether to use it as a starting point.
4. Use gap_questions to pick what to ask next, one question at a time.
Keep a to-do list for the case and write one inventory note per room under /inventory/."""

INSIDE_PROMPT = """You do the inside work for one case: the 3D scan of the home.
Call seed_inside once with the case_id, scan id and capture date you were given. Then reply with a
short summary: rooms found, how many photos were processed, how many suggested items, and the
confirmation message for the family. Do not list every item. Everything is a suggestion."""

OUTSIDE_PROMPT = """You do the outside work for one case: the aerial view of the lot.
Call seed_outside once with the case_id, lat, lon and loss date you were given. Then reply with a
short summary: capture date of the aerial image (say so if it is from after the loss), how many
buildings are on the lot, and which yard things were seen. These are questions for the family,
not items."""


# ------------------------------------------------------------------- tools
def home_lookup(address: str = "", lat: float | None = None, lon: float | None = None) -> dict:
    """Find the 3D scan of a home by address text or lat/lon. Returns scan id, lat/lon and capture date, or found=False."""
    return _home_lookup(address or None, lat, lon)


def seed_inside(case_id: str, sid: str, captured: str = "") -> dict:
    """Load the scan's rooms and panorama photos into the case and run item detection. Items are stored as suggested."""
    seed.listing(case_id, sid)              # the sale listing behind the scan, when the CoStar kit is set up
    return seed.inside(case_id, sid, captured or None)


def seed_outside(case_id: str, lat: float, lon: float, loss_date: str = "") -> dict:
    """Load the parcel, buildings and yard findings from the aerial image taken before loss_date (YYYY-MM-DD)."""
    return seed.outside(case_id, lat, lon, loss_date or config.env("ASHLINE_LOSS_DATE"))


def home_listing(case_id: str) -> dict:
    """What the sale listing says about the home: beds, baths, size, year built, garage, appliances."""
    home = Store().get_home(case_id) or {}
    return home.get("listing") or {"found": False}


def gap_questions(case_id: str, limit: int = 3) -> list[dict]:
    """The next questions worth asking the family, most valuable first."""
    return _gap_questions(case_id, limit=limit)


def list_suggestions(case_id: str, room: str = "") -> list[dict]:
    """Suggested (unconfirmed) items of a case, optionally for one room label."""
    items = Store().get_items(case_id, status="suggested", canonical_only=True)
    if room:
        items = [i for i in items if room.lower() in (i["room_label"] or "").lower()]
    return [{"item_id": i["item_id"], "label": i["label"], "qty": i["qty"], "room": i["room_label"],
             "source": i["source"]} for i in items]


def confirm_item(case_id: str, item_id: int) -> str:
    """Mark an item as confirmed. Only after a family member said it was theirs."""
    Store().set_item_status(case_id, item_id, "confirmed")
    return f"item {item_id} confirmed"


def reject_item(case_id: str, item_id: int) -> str:
    """Mark an item as not theirs. Only after a family member said so."""
    Store().set_item_status(case_id, item_id, "rejected")
    return f"item {item_id} rejected"


def answer_finding(case_id: str, finding_id: int, theirs: bool) -> str:
    """Record the family's answer about an aerial finding (an outbuilding or a yard item)."""
    Store().set_finding_status(case_id, finding_id, "confirmed" if theirs else "rejected")
    return f"finding {finding_id} {'confirmed' if theirs else 'rejected'}"


ORCHESTRATOR_TOOLS = [home_lookup, home_listing, gap_questions, list_suggestions, confirm_item, reject_item,
                      answer_finding]
GATED_TOOLS = {"confirm_item": True, "reject_item": True, "answer_finding": True}

SUBAGENTS = [
    {"name": "inside", "description": "Inside of the home: loads the 3D scan's rooms and panorama photos "
     "and detects items. Give it case_id, scan id (sid) and capture date.",
     "system_prompt": INSIDE_PROMPT, "tools": [seed_inside]},
    {"name": "outside", "description": "Outside of the home: parcel, buildings and yard from aerial "
     "imagery. Give it case_id, lat, lon and the loss date.",
     "system_prompt": OUTSIDE_PROMPT, "tools": [seed_outside]},
]


def _model(model=None):
    if model is not None and not isinstance(model, str):
        return model
    name = model or config.env("ASHLINE_AGENT_MODEL", required=True)
    base_url = config.env("ASHLINE_AGENT_BASE_URL")
    if base_url and name.startswith("openai:"):       # an OpenAI-compatible gateway such as OpenRouter
        from langchain.chat_models import init_chat_model
        return init_chat_model(name, base_url=base_url, api_key=config.env("ASHLINE_AGENT_API_KEY", required=True),
                               use_responses_api=False)
    return name


def build_agent(model=None, checkpointer=None):
    """The Ashline deep agent. Pass a checkpointer to use the confirm/reject interrupts."""
    from deepagents import create_deep_agent

    chat = _model(model)
    return create_deep_agent(
        model=chat,
        tools=ORCHESTRATOR_TOOLS,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[{**spec, "model": chat} for spec in SUBAGENTS],
        interrupt_on=GATED_TOOLS if checkpointer is not None else None,
        checkpointer=checkpointer,
        name="ashline",
    )
