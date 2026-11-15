"""Natural-language chat: a LangChain deep agent that talks with the family about their case.

Photos, exact commands and answers to Ashline's own questions stay on the fixed path in
`chat.py`. Everything else a person writes ("what did we find in the kitchen?", "we also had a
piano in the den", "is the sofa price right?") comes here. The agent answers from the case
store through tools, and can act only in ways the family could already act:

- it reads the inventory, the meter, the home's facts and the open questions;
- it adds a remembered item as a SUGGESTION, never as a confirmed item;
- it runs one of Ashline's existing commands on the person's behalf, through the same code and
  the same checks as if they had typed it (so deleting still asks for confirmation, and only
  the owner can export).

Built with deepagents: the planning tool and virtual files are available to it, and a
`price-researcher` subagent looks prices up online when asked.
"""
from __future__ import annotations

import functools
import threading
from datetime import datetime

from . import config
from .gaps import gap_questions
from .meter import meter, meter_line
from .pricing import money_range, price_items

HISTORY_TURNS = 8

SYSTEM_PROMPT = """You are Ashline. You help a family that lost their home in a wildfire rebuild the contents
inventory their insurer asks for. You are talking with {who} by text message.

How to write:
- Short: one to three sentences, plain text, no markdown, no bullet symbols, no emoji.
- Gentle and plain. No cheerful filler, no exclamation marks. These are things from a home that is gone.
- Never pressure. If they want to stop, say that is fine.

What you may say:
- Facts about the inventory, prices and the home come ONLY from your tools. Never invent an item, a price,
  a count or a date. If a tool has no answer, say you don't have that yet.
- Prices are estimates of buying the item new today. Never suggest raising a value.
- You are not a lawyer or an adjuster. For questions about coverage or the claim, say what Ashline tracks
  and suggest they confirm with their adjuster.

What you may do:
- Use search_inventory, inventory_overview, progress, about_the_home and questions_to_ask to answer.
- When they tell you about something they owned that is not on the list, use remember_item. It becomes a
  suggestion marked "from memory"; tell them it waits for {owner} to say OK.
- When they ask for something Ashline has a command for, use run_command with the exact command. The person
  receives the command's own reply directly, so afterwards answer with just: DONE
- To check a price online, hand the item to the price-researcher subagent with the task tool, then use
  apply_web_price with what it found.
- You cannot place phone calls, edit photos, or confirm items yourself. {owner} confirms by replying OK
  to a review.

{situation}"""

RESEARCHER_PROMPT = """You find what one household item costs to buy new today, for an insurance inventory.
Call find_price_online once with the item's description and brand. Then reply in one or two sentences
with: the typical price, the range you saw, and the retailer of the closest match. If nothing close was
found, say so. Never guess a price that the search did not return."""


def enabled(engine) -> bool:
    """On when a chat model is available (set on the engine, or an Anthropic key with deepagents installed)."""
    if getattr(engine, "assistant_model", None) is not None:
        return True
    if (config.env("ASHLINE_ASSISTANT") or "").lower() == "off" or not config.env("ANTHROPIC_API_KEY"):
        return False
    try:
        import deepagents  # noqa: F401
        import langchain_anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _model(engine):
    model = getattr(engine, "assistant_model", None)
    return model if model is not None else (config.env("ASHLINE_CHAT_MODEL") or config.env("ASHLINE_AGENT_MODEL")
                                            or "anthropic:claude-opus-5-5")


def build_tools(engine, inbound, member: dict, case: dict, now: datetime, outbox: list) -> tuple[list, list]:
    """The agent's tools for this one message, bound to the case and the person writing.

    Returns (tools for the main agent, tools for the price-researcher subagent).
    """
    store, case_id = engine.store, case["case_id"]

    def _lines(room: str = "", query: str = "") -> list[dict]:
        words = [w for w in query.lower().split() if len(w) > 1]
        out = []
        for item in store.get_items(case_id, canonical_only=True):
            if item["status"] == "rejected":
                continue
            if room and room.lower() not in (item.get("room_label") or "").lower():
                continue
            if words and not all(w in item["label"].lower() or w in (item.get("category") or "") for w in words):
                continue
            out.append(item)
        out.sort(key=lambda i: -((i.get("price_mid") or 0) * (i.get("qty") or 1)))
        return out

    def _evidence(item: dict) -> str:
        sightings = store.get_sightings(case_id, item["item_id"])
        kinds = {s["source"] for s in sightings}
        if kinds <= {"voice"}:
            return "from memory, no photo"
        return f"{len(sightings)} photo{'s' if len(sightings) != 1 else ''}" + (" (3D scan)" if "scan" in kinds else "")

    def search_inventory(query: str = "", room: str = "", limit: int = 12) -> list[dict]:
        """Items on the list, most valuable first. `query` matches words in the item's name or category;
        `room` matches the room name. Each result has a `number`: the person can use it afterwards in
        commands such as "remove 2" or "price 2 300"."""
        found = _lines(room, query)[:max(1, min(int(limit), 25))]
        context = engine._context(inbound.chat_id)
        context["list"] = {str(n): i["item_id"] for n, i in enumerate(found, 1)}
        engine._save(inbound.chat_id, context)
        return [{"number": n, "item": i["label"], "room": i.get("room_label") or "unsorted", "quantity": i.get("qty") or 1,
                 "estimate": money_range(i.get("price_low"), i.get("price_high")) if i.get("price_mid") else "not priced",
                 "price_from": i.get("price_source") or "", "status": i["status"], "evidence": _evidence(i)}
                for n, i in enumerate(found, 1)]

    def inventory_overview() -> dict:
        """How many items are on the list per room, how many are confirmed, and the totals."""
        rooms: dict[str, dict] = {}
        for item in _lines():
            room = rooms.setdefault(item.get("room_label") or "unsorted", {"items": 0, "confirmed": 0, "value": 0.0})
            room["items"] += 1
            room["confirmed"] += item["status"] == "confirmed"
            room["value"] += (item.get("price_mid") or 0) * (item.get("qty") or 1)
        m = meter(case_id, store)
        return {"rooms": {name: {**r, "value": round(r["value"])} for name, r in rooms.items()},
                "confirmed_items": m["items"], "waiting_for_ok": m["pending_items"],
                "documented_dollars": round(m["documented"]), "waiting_dollars": round(m["pending"]),
                "family_members": [x.get("name") or "someone" for x in store.get_members(case_id)]}

    def progress() -> str:
        """One sentence on how much is documented so far, against what the insurer still needs if known."""
        return meter_line(meter(case_id, store))

    def about_the_home() -> dict:
        """What Ashline knows about the home itself: the rooms, the 3D scan, the aerial photo and the
        sale listing (beds, baths, year built, what the listing mentions)."""
        home = store.get_home(case_id) or {}
        listing = home.get("listing") or {}
        return {"name": case.get("name"), "rooms": [r.get("name") or r["label"] for r in store.get_rooms(case_id)],
                "scan_captured": home.get("scan_captured"), "aerial_captured": home.get("aerial_captured"),
                "date_of_loss": case.get("loss_date"),
                "listing": {k: listing.get(k) for k in ("beds", "baths", "year_built", "garage_spaces", "list_date")
                            if listing.get(k)} or None,
                "listing_mentions": sorted(k.replace("_", " ") for k, v in (listing.get("has") or {}).items() if v),
                "other_buildings_on_the_lot": sum(1 for f in store.get_findings(case_id) if f["kind"] == "structure")}

    def questions_to_ask(limit: int = 3) -> list[str]:
        """The gaps most worth asking this family about next, most valuable first."""
        return [q["text"] for q in gap_questions(case_id, store, limit=max(1, min(int(limit), 5)))]

    def remember_item(description: str, room: str = "", quantity: int = 1) -> str:
        """Add something the person says they owned that is not on the list. It is stored as a
        suggestion marked "from memory" and waits for the owner's OK. `description` is a short noun
        phrase such as "upright piano" or "hand tools, assorted"."""
        label = " ".join(description.split())[:120]
        if not label:
            return "Nothing was added: the description was empty."
        qty = max(1, min(int(quantity or 1), 99))
        item_id = store.add_item(case_id, {"label": label, "category": "other", "qty": qty, "room_label": room.strip() or None,
                                           "detail": "From memory"}, source="voice", added_by=member.get("name"))
        store.set_cluster(case_id, item_id, [item_id], qty)
        price_items(case_id, store, engine.pricer)
        store.log_event(case_id, "add", {"item_id": item_id}, member["member_id"])
        item = store.get_item(case_id, item_id)
        estimate = money_range(item.get("price_low"), item.get("price_high")) if item.get("price_mid") else "not priced yet"
        return (f"Added as a suggestion marked \"from memory\": {qty} x {label} (estimate {estimate}). "
                "It is not confirmed until the owner says OK. A photo of it would make it stronger.")

    def run_command(command: str) -> str:
        """Run one of Ashline's commands for the person, exactly as if they had typed it. Commands:
        list · how much so far · review · ok · undo · export · pause · help · call me · delete my case ·
        show photo N · show the ROOM · remove N · split N · price N AMOUNT · N is a NEW NAME
        (N is a number from the last list they were shown or from search_inventory)."""
        from .intent import valid_command
        exact = valid_command(command)
        if not exact:
            return f"{command!r} is not one of Ashline's commands, so nothing was done."
        fresh_member = store.get_member(inbound.sender, case_id) or member
        replies = (engine._answer_pending(inbound, fresh_member, case, exact)
                   or engine._command(inbound, fresh_member, case, exact, now))
        if not replies:
            return f"{exact!r} did nothing right now (there may be nothing for it to act on)."
        outbox.extend(replies)
        return "The person has received: " + " | ".join(r.text for r in replies)[:900]

    def apply_web_price(item_number: int, price: float, low: float, high: float, retailer: str, url: str = "") -> str:
        """Record a price found online for an item (by its `number` from search_inventory). `price` is
        the closest match, `low` and `high` the range seen, `retailer` where it was found."""
        from .price_search import apply_price
        item_id = engine._context(inbound.chat_id).get("list", {}).get(str(item_number))
        if not item_id:
            return f"There is no item number {item_number} in the last list."
        return apply_price(store, case_id, item_id, {"price": price, "low": low, "high": high,
                                                     "retailer": retailer, "url": url}, member["member_id"])

    def find_price_online(description: str, brand: str = "") -> dict:
        """Search retailers for what an item costs new today. Returns the matches with price, retailer
        and link, or found=False."""
        from .price_search import find_price
        return find_price(description, brand or None, searcher=getattr(engine, "price_searcher", None))

    # The agent may call tools in parallel, on worker threads. They share one store, so run one at a time.
    lock = threading.Lock()

    def one_at_a_time(tool):
        @functools.wraps(tool)
        def locked(*args, **kwargs):
            with lock:
                return tool(*args, **kwargs)
        return locked

    main = [one_at_a_time(t) for t in (search_inventory, inventory_overview, progress, about_the_home,
                                       questions_to_ask, remember_item, run_command, apply_web_price)]
    return main, [one_at_a_time(find_price_online)]


def _text(message) -> str:
    """The text of a model message, whether its content is a string or a list of blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def converse(engine, inbound, member: dict, case: dict, text: str, now: datetime):
    """Answer one free-form message. Returns a list of Outbound, or None to fall back to the router."""
    if not text.strip() or not enabled(engine):
        return None
    try:
        from deepagents import create_deep_agent
        from langchain_core.messages import AIMessage, HumanMessage

        from .chat import Outbound
    except ImportError:
        return None

    outbox: list = []
    tools, research_tools = build_tools(engine, inbound, member, case, now, outbox)
    owner = engine._owner(case["case_id"]) or {}
    is_owner = member["role"] == "owner"
    open_questions = engine.store.get_questions(case["case_id"], chat_id=inbound.chat_id)
    situation = " ".join(filter(None, [
        f"The home is called {case.get('name')}." if case.get("name") else "",
        f"{member.get('name') or 'This person'} is {'the owner of the case' if is_owner else 'a relative helping'}.",
        "This is the family group chat, so keep it especially brief." if inbound.is_group else "",
        "Ashline has an open question to this chat (answer 1 or 2, or yes or no)." if open_questions else "",
    ]))
    prompt = SYSTEM_PROMPT.format(who=member.get("name") or "a family member",
                                  owner="you" if is_owner else (owner.get("name") or "the owner"), situation=situation)
    model = _model(engine)
    try:
        agent = create_deep_agent(
            model=model, tools=tools, system_prompt=prompt, name="ashline-chat",
            subagents=[{"name": "price-researcher", "model": model, "tools": research_tools,
                        "description": "Looks up what one item costs to buy new today from online retailers. "
                                       "Give it the item's description and brand.",
                        "system_prompt": RESEARCHER_PROMPT}])
        context = engine._context(inbound.chat_id)
        history = [(HumanMessage if turn["role"] == "user" else AIMessage)(turn["text"])
                   for turn in context.get("talk", [])[-HISTORY_TURNS:]]
        result = agent.invoke({"messages": history + [HumanMessage(text)]}, {"recursion_limit": 40})
        reply = _text(result["messages"][-1])
    except Exception as exc:                              # no network, bad key, model error: use the router
        print(f"ashline: the chat agent could not answer ({exc})")
        return outbox or None

    if reply.strip(" .").upper() == "DONE" and outbox:
        reply = ""
    context = engine._context(inbound.chat_id)            # tools may have changed it
    said = reply or " ".join(o.text for o in outbox)[:300]
    context["talk"] = (context.get("talk", []) + [{"role": "user", "text": text[:500]},
                                                  {"role": "assistant", "text": said[:500]}])[-HISTORY_TURNS:]
    engine._save(inbound.chat_id, context)
    out = list(outbox)
    if reply:
        out.append(Outbound(inbound.chat_id, reply[:1200]))
    return out or None
