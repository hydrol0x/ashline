"""The conversation: what Ashline says, and when it stays quiet (UX doc sections 2 to 9).

The engine is independent of the channel. A bridge (Photon, or the local simulator) hands it an
`Inbound` message and sends whatever `Outbound` messages come back. `tick()` is called on a timer
for the things that happen later: the digest after a burst of photos, the daily review, questions
that expire.

Principles the code follows:
- Quiet by default. In a group chat Ashline answers photos, messages addressed to it, replies to
  its own pending question and command words. Anything else gets silence.
- Everything is a suggestion until the owner says OK.
- It never quotes the family's conversation back; it only ever surfaces items.
"""
from __future__ import annotations

import io
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageOps

from . import config, detect as detect_mod, disaster
from .annotate import annotate, legend, side_by_side
from .costar.matterport import _month_year
from .dedupe import item_key, merge_into_inventory
from .gaps import gap_questions
from .meter import meter, meter_line
from .pipeline import detect_cached
from .pricing import money, money_range, price_items
from .rooms import tidy_room
from .store import Store

BATCH_SECONDS = 60                      # wait this long after the last photo of a burst
ANNOTATE_UP_TO = 5                       # bursts this small get every photo back annotated
QUESTION_HOURS = 4                      # an unanswered "same item?" expires after this
CODE = re.compile(r"\b(HOME-\d{4})\b", re.I)
NEW_ITEMS = (r"(?:(?:no,?\s*)?(?:they|these|those|it|this|that)(?:'re| are| were| was|'s| is)?\s+)?(?:new|"
             r"not (?:previously |already )?(?:added|listed|on (?:the|my) list)|(?:were|was)n't (?:previously |already )?(?:added|listed)|"
             r"not the same(?: (?:ones?|things?|items?))?|different(?: (?:ones?|things?|items?))?)")
ADDRESS_CMD = re.compile(r"(?:(?:the |our |my )?(?:home |house |street )?address(?: is|:)?|we lived at|it was at|it's at)\s+(?P<addr>\d.+)", re.I)
ADDRESS = re.compile(r"\d{1,6}\s+(?:[NSEW]\.?\s+)?[\w .'-]{2,40}\s(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|"
                     r"ct|court|pl|place|ter|terrace|pkwy|parkway|hwy|highway|cir|circle|trl|trail)\b\.?(?:[ ,].*)?", re.I)
WHAT_HAPPENED = "One more question: what happened to the home? A fire, a flood, a hurricane or storm, or something else?"
INSURED_Q = "Do you have insurance that covers it? Yes, no or not sure is fine."
TELEGRAM_PREFIX = "tg:"                            # chat and sender ids that came from the Telegram bridge
ADDRESSED = re.compile(r"^\s*@?ashline\b[,:]?\s*", re.I)
NOT_SURE = re.compile(r"(?:idk|i ?(?:don'?t|dont|do not) know|don'?t know|dont know|dunno|not sure|no idea|no clue|"
                      r"skip|pass|\?+|)", re.I)       # "I don't know" to a setup question: skip it
MEMORY_QUESTIONS = 6                    # a text "call" asks at most this many questions
MEMORY_COMMANDS = ("help", "menu", "commands", "list", "export", "review", "undo", "map", "receipt", "call me",
                   "how much", "how much so far", "total", "check my bank", "show me the list", "what can you do?")
REVIEW_AT_ONCE = 30                     # more than this waiting: the review goes one room at a time
DIRECTED_ASK_BELOW = 5                  # a room with fewer items than this is worth asking about
DEADLINE_DAYS = (60, 30, 10)            # proof-of-loss reminders, and nothing else

INTRO = ("I'm Ashline. I help families rebuild the list of belongings insurance asks for after a fire, flood or storm. "
         "It takes a few minutes at a time, and you can stop whenever. What should I call your home?")
HELP = ("Send me photos from inside the home, from any year, and I'll list what's in them.\n\n"
        "You can also ask, in your own words:\n"
        "• list: the full inventory\n"
        "• how much so far\n"
        "• show photo 3\n"
        "• map: where things were in the home\n"
        "• check my bank: use what you paid as the price\n"
        "• send a receipt or order page with the word receipt\n"
        "• call me: remember things no photo shows\n"
        "• export: spreadsheet and PDF\n"
        "• undo · pause")
# Household things worth noticing when the family mentions them to each other.
HEARD_WORDS = ("piano", "guitar", "violin", "drum kit", "tv", "television", "sofa", "couch", "dining table", "table",
               "desk", "bed", "dresser", "bike", "bicycle", "treadmill", "freezer", "washer", "dryer", "sewing machine",
               "computer", "laptop", "camera", "jewelry", "watch", "china", "silverware", "quilt", "rug", "painting",
               "record player", "records", "tools", "table saw", "lawn mower", "grill", "kayak", "surfboard", "skis",
               "golf clubs", "wine", "books", "bookshelf", "armchair", "recliner", "crib", "stroller", "telescope")
OWNERSHIP = re.compile(r"\b(remember|we had|had a|had an|there was|don't forget|what about|grandma's|grandpa's|"
                       r"mom's|dad's|our|my|her|his)\b", re.I)


@dataclass
class Inbound:
    chat_id: str
    sender: str                                   # phone number
    text: str = ""
    attachments: list = field(default_factory=list)   # (filename, bytes) pairs
    is_group: bool = False
    sender_name: str | None = None
    now: datetime | None = None
    missing: int = 0                              # attachments the bridge could not download


@dataclass
class Outbound:
    chat_id: str
    text: str
    image: str | None = None                      # path of an image to attach
    files: list = field(default_factory=list)     # paths of files to attach
    choices: list = field(default_factory=list)   # (label, reply_text) pairs; a bridge may show them as buttons


def _now(value: datetime | None = None) -> datetime:
    return value or datetime.now(timezone.utc)


def _join(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1] if parts else ""


def parse_money(text: str) -> float | None:
    """'$400,000', '400k', '1.2 million' -> dollars."""
    match = re.search(r"\$?\s*([\d][\d,]*\.?\d*)\s*(k|thousand|m|million)?", text.lower())
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return value * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(match.group(2) or "", 1)


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def parse_date(text: str) -> str | None:
    """'January 2025', 'Jan 7, 2025', '1/7/2025', '2025-01-07' -> 'YYYY-MM-DD'. A month alone means its first day."""
    low = text.lower()
    match = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", low)
    if match:
        year, month, day = (int(g) for g in match.groups())
    else:
        match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\b", low)
        if match:
            month, day, year = (int(g) for g in match.groups())
        else:
            month = next((n for n, name in enumerate(MONTHS, 1) if re.search(rf"\b{name[:3]}[a-z]*\b", low)), None)
            year = re.search(r"\b(20\d\d)\b", low)
            if not (month and year):
                return None
            day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b(?!\d)", re.sub(r"20\d\d", "", low))
            year, day = int(year.group(1)), int(day.group(1)) if day else 1
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def split_rooms(text: str) -> list[str]:
    """'living room, kitchen, 3 bedrooms and a garage' -> Living room, Kitchen, Bedroom, Bedroom 2, Bedroom 3, Garage."""
    rooms = []
    for part in re.split(r",|;|\band\b|\n", text.lower()):
        part = re.sub(r"^(?:it|we|there|they)?\s*(?:only|just)?\s*(?:has|had|have|was|were|is|are|got)\s+", "",
                      part.strip())                                 # "it only has basement" -> "basement"
        part = re.sub(r"^(?:only|just)\s+", "", part)
        part = re.sub(r"\b(a|an|the|one|plus|our|my)\b", " ", part).strip(" .")
        part = re.sub(r"\s+", " ", part)
        if not part:
            continue
        match = re.match(r"(\d+|two|three|four|five|six)\s+(.+)", part)
        count = 1
        if match:
            count = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}.get(match.group(1)) or int(match.group(1))
            part = match.group(2)
        part = re.sub(r"s$", "", part.strip()) if count > 1 else part.strip()
        name = part[:1].upper() + part[1:]
        for n in range(1, min(count, 12) + 1):
            rooms.append(name if n == 1 else f"{name} {n}")
    return rooms[:30]


class Engine:
    def __init__(self, store: Store | None = None, detector=None, pricer=None, grouper=None,
                 public_url: str | None = None, number: str | None = None, batch_seconds: int | None = None,
                 interpreter=None):
        self.store = store or Store()
        self.detector, self.pricer, self.grouper = detector, pricer, grouper
        self._prints: dict[str, tuple] = {}
        # Turns a message that matches no exact command into one (ashline.intent). None: built on
        # first use when an Anthropic key is set; False: exact commands only.
        self._interpreter = interpreter                  # photo path -> fingerprint, to spot repeats
        self.public_url = (public_url or config.env("ASHLINE_PUBLIC_URL") or "").rstrip("/")
        self.number = number or config.env("ASHLINE_NUMBER") or "this number"
        handle = (config.env("TELEGRAM_BOT_USERNAME") or "").lstrip("@")
        self.telegram_handle = f"@{handle}" if handle else "the Ashline bot"        # the Telegram bridge fills this in
        self.batch_seconds = batch_seconds if batch_seconds is not None else int(
            config.env("ASHLINE_BATCH_SECONDS") or BATCH_SECONDS)
        self.video_frames = int(config.env("ASHLINE_VIDEO_FRAMES") or 14)     # stills kept from one video
        self.bank = None                                    # a bank client; built from NESSIE_API_KEY when first needed
        self.receipt_reader = None                          # reads receipts and order pages; built from the key when first needed
        self.searcher = None                                # looks prices up online; built from the key when first needed
        self.background = True                              # False runs the after-work (price lookups) inline, for tests
        self._running: set = set()                          # (job, case) pairs with a background thread at work

    # ------------------------------------------------------------ helpers
    def link(self, case: dict, member: dict) -> str:
        token = case["owner_token"] if member["role"] == "owner" else case["member_token"]
        return f"{self.public_url}/p/{token}" if self.public_url else f"/p/{token}"

    def _context(self, chat_id: str) -> dict:
        chat = self.store.get_chat(chat_id)
        return dict(chat["context"]) if chat else {}

    def _save(self, chat_id: str, context: dict) -> None:
        self.store.set_chat_context(chat_id, context)

    def _owner(self, case_id: str) -> dict | None:
        return next((m for m in self.store.get_members(case_id) if m["role"] == "owner"), None)

    def _room_names(self, case_id: str) -> list[str]:
        return [r.get("name") or r["label"] for r in self.store.get_rooms(case_id)]

    def _display(self, item: dict) -> str:
        price = money_range(item.get("price_low"), item.get("price_high")) if item.get("price_mid") else ""
        qty = f"{item['qty']} × " if (item.get("qty") or 1) > 1 else ""
        return f"{qty}{item['label']}" + (f" · {price}" if price else "")

    # ---------------------------------------------------------- entry point
    def _member_for(self, inbound: Inbound) -> dict | None:
        """Who is writing, in the home this chat belongs to.

        The chat decides, not the person: someone can belong to two homes (their own, and a relative's
        group they joined). A message in a group goes to that group's home; a 1:1 message goes to the
        home that 1:1 chat is linked to. Only a chat that isn't linked yet falls back to the person's
        most recent membership."""
        chat = self.store.get_chat(inbound.chat_id)
        if chat and chat.get("case_id"):
            mine = self.store.get_member(inbound.sender, chat["case_id"])
            if mine or inbound.is_group:
                return mine                                # None in a group: they're added below as new
        return self.store.get_member(inbound.sender)

    def handle(self, inbound: Inbound) -> list[Outbound]:
        now = _now(inbound.now)
        text = (inbound.text or "").strip()
        member = self._member_for(inbound)

        code = CODE.search(text)
        if code:
            return self._join_case(inbound, code.group(1).upper(), member)

        attempt = None if member or inbound.is_group else re.search(r"\bhome\s*[-#:]?\s*(\d+)\b", text, re.I)
        if attempt:                                       # a mistyped join code: never start a new home for it
            if len(attempt.group(1)) == 4 and self.store.find_case(invite_code=f"HOME-{attempt.group(1)}"):
                return self._join_case(inbound, f"HOME-{attempt.group(1)}", member)
            return [Outbound(inbound.chat_id, "That code doesn't look quite right. Codes look like HOME-1234. "
                                              "Could you check it with the person who sent it to you and send it again?")]

        if member is None:
            chat = self.store.get_chat(inbound.chat_id)
            if inbound.is_group and chat:                 # someone new in the family's inventory chat
                member = self.store.add_member(chat["case_id"], inbound.sender, inbound.sender_name)
            elif inbound.is_group:
                return []                                 # a group Ashline was not introduced to: say nothing
            else:
                return self._start_onboarding(inbound)

        case = self.store.get_case(member["case_id"])
        if case is None:                                  # their case was deleted
            self.store.set_member(member["member_id"], case_id="")
            return self._start_onboarding(inbound) if not inbound.is_group else []
        if not self.store.get_chat(inbound.chat_id):
            self.store.set_chat(inbound.chat_id, case["case_id"], inbound.is_group)
        self.store.set_member(member["member_id"], last_seen=now.isoformat(), paused=0)   # writing again ends a pause
        if not inbound.is_group and member.get("dm_chat") != inbound.chat_id:
            self.store.set_member(member["member_id"], dm_chat=inbound.chat_id)          # where to reach them 1:1

        step = member["state"].get("step")
        if step in ("home_name", "address", "use_scan", "loss_date", "rooms") and not inbound.is_group:
            return self._continue_onboarding(inbound, member, case, text)
        if step in ("disaster", "flooded", "insured") and not inbound.is_group and not (inbound.attachments or inbound.missing):
            answered = self._answer_setup(inbound, member, case, text)
            if answered is not None:
                return answered
            self.store.set_member(member["member_id"], state={})          # not an answer: drop the question
            member = self.store.get_member(inbound.sender)

        if inbound.attachments or inbound.missing:
            from . import receipts
            if receipts.wanted(inbound.text, inbound.attachments):      # "receipt" in the message, or a PDF
                return self._receipts(inbound, member, case, now)
            return self._intake(inbound, member, case, now)

        if inbound.is_group:
            addressed = bool(ADDRESSED.match(text))
            text = ADDRESSED.sub("", text).strip()
            reply = self._answer_pending(inbound, member, case, text) or self._command(inbound, member, case, text, now)
            if reply is not None:
                return reply
            if addressed:
                from . import assistant
                return (assistant.converse(self, inbound, member, case, text, now)
                        or self._interpret(inbound, member, case, text, now) or [Outbound(inbound.chat_id, HELP)])
            self._listen(inbound.chat_id, text)           # unsure = silent
            return []

        reply = (self._answer_pending(inbound, member, case, text)
                 or self._command(inbound, member, case, text, now)
                 or self._memory_answer(inbound, member, case, text))
        if reply is not None:
            return reply
        if re.fullmatch(r"(?:ok(?:ay)?|k|cool|nice|great|thanks?|thank you|ty|got it|perfect|awesome|sounds good|"
                        r"👍|❤️|🙏)[ .!]*", text.strip().lower()):
            return [Outbound(inbound.chat_id, "Anytime. Send more photos whenever you have them.")]
        from . import assistant                             # the LangChain agent, when a model is available
        return (assistant.converse(self, inbound, member, case, text, now)
                or self._interpret(inbound, member, case, text, now)
                or [Outbound(inbound.chat_id, "I'm not sure what you'd like me to do with that.\n\n" + HELP)])

    def _interpret(self, inbound: Inbound, member: dict, case: dict, text: str, now: datetime):
        """A message no exact command matched: let the model pick a real command or a short reply."""
        if self._interpreter is None:
            from .intent import get_interpreter
            self._interpreter = get_interpreter() or False
        if not self._interpreter or not text.strip():
            return None
        case_id = case["case_id"]
        open_qs = self.store.get_questions(case_id, chat_id=inbound.chat_id)
        numbered = self._context(inbound.chat_id).get("list") or {}
        context = "; ".join(filter(None, [
            f"Ashline's open question: {'same item?' if open_qs[0]['kind'] == 'same' else 'add an item mentioned?'}"
            if open_qs else "no open question",
            f"Ashline's last numbered list had items 1 to {len(numbered)}" if numbered else "",
            "sender owns the case" if member["role"] == "owner" else "sender is a relative",
        ]))
        result = self._interpreter(text, context)
        command = result.get("command")
        if command:
            reply = (self._answer_pending(inbound, member, case, command)
                     or self._command(inbound, member, case, command, now))
            if reply is not None:
                return reply
        if result.get("reply"):
            return [Outbound(inbound.chat_id, result["reply"])]
        return None

    # ----------------------------------------------------------- onboarding
    def _start_onboarding(self, inbound: Inbound) -> list[Outbound]:
        case_id = "case-" + secrets.token_hex(4)
        while True:
            code = f"HOME-{secrets.randbelow(9000) + 1000}"
            if not self.store.find_case(invite_code=code):
                break
        self.store.set_case(case_id, invite_code=code, owner_token=secrets.token_urlsafe(24),
                            member_token=secrets.token_urlsafe(24))
        member = self.store.add_member(case_id, inbound.sender, inbound.sender_name, role="owner")
        self.store.set_member(member["member_id"], state={"step": "home_name"}, dm_chat=inbound.chat_id)
        self.store.set_chat(inbound.chat_id, case_id, False)
        return [Outbound(inbound.chat_id, INTRO)]

    def _continue_onboarding(self, inbound: Inbound, member: dict, case: dict, text: str) -> list[Outbound]:
        skipped = bool(NOT_SURE.fullmatch(text.strip(" .!")))
        if member["state"]["step"] == "home_name":
            if skipped:
                text = ""
            name = re.sub(r"^(call it|it's|its)\s+", "", text, flags=re.I).strip(" .\"'") or "Our home"
            if not re.match(r"(the|our|my)\b", name, re.I):
                name = f"The {name}"
            self.store.set_case(case["case_id"], name=name[:60])
            self.store.set_member(member["member_id"], state={"step": "address"})
            return [Outbound(inbound.chat_id, "What was the address? I'll check whether there's a 3D scan or a sale "
                                              "listing of the home to start from. You can say skip.")]

        if member["state"]["step"] == "address":
            found = {"found": False}
            if text.lower().strip(" .") not in ("skip", "no", "later", "not now") and not skipped:
                self.store.set_case(case["case_id"], address=text[:200])
                try:
                    found = self.lookup(text)
                except Exception:                         # the scan index is a bonus: never block onboarding on it
                    found = {"found": False}
            if found.get("found"):
                self.store.set_member(member["member_id"], state={"step": "use_scan", "scan": {
                    k: found.get(k) for k in ("sid", "lat", "lon", "homes_listing_id", "original_created")}})
                when = _month_year(found.get("original_created"))
                size = f" with {found['room_count']} rooms" if found.get("room_count") else ""
                return [Outbound(inbound.chat_id,
                                 f"I found a 3D scan of this address{' from ' + when if when else ''}{size}. "
                                 "Use it as a starting point? It may show staging or a previous owner's things, so "
                                 "nothing is added until you confirm it.")]
            self.store.set_member(member["member_id"], state={"step": "rooms"})
            lead = "I don't have a scan of that address, which is normal. " if found.get("reason") and "skip" not in text.lower() \
                and text.lower().strip(" .") not in ("no", "later", "not now") else ""
            return [Outbound(inbound.chat_id, lead + "Which rooms did it have? A rough list is fine.")]

        if member["state"]["step"] == "use_scan":
            scan = member["state"].get("scan") or {}
            later = member["state"].get("later")
            if text.lower().strip(" .!") in ("yes", "y", "ok", "okay", "sure", "yes please", "use it"):
                # The aerial photo has to be from before the fire, so this one date is asked now.
                self.store.set_member(member["member_id"], state={"step": "loss_date", "scan": scan, "later": later})
                return [Outbound(inbound.chat_id, "When did it happen? A month and year is enough. It lets me use an "
                                                  "aerial photo of the home from before it. You can say skip.")]
            if later:
                self.store.set_member(member["member_id"], state={})
                return [Outbound(inbound.chat_id, "That's fine, we'll leave it.")]
            self.store.set_member(member["member_id"], state={"step": "rooms"})
            return [Outbound(inbound.chat_id, "That's fine, we'll leave it. Which rooms did it have? A rough list is fine.")]

        if member["state"]["step"] == "loss_date":
            scan = member["state"].get("scan") or {}
            when = parse_date(text)
            if when:
                self.store.set_case(case["case_id"], loss_date=when)
            self.store.set_member(member["member_id"], state={})
            self.store.set_home(case["case_id"], sid=scan.get("sid"), lat=scan.get("lat"), lon=scan.get("lon"),
                                homes_listing_id=scan.get("homes_listing_id"))
            context = self._context(inbound.chat_id)
            context["seed"] = {"scan": scan, "member_id": member["member_id"]}
            self._save(inbound.chat_id, context)
            if member["state"].get("later"):                # setup was already done: no invite again
                self.store.set_member(member["member_id"], state={})
                return [Outbound(inbound.chat_id, "Thank you. I'll go through the scan now; it takes a few minutes "
                                                  "and I'll text you when it's done.")]
            self.store.set_member(member["member_id"], state={"step": "disaster"})
            return [Outbound(inbound.chat_id, "Thank you. I'll go through the scan now; it takes a few minutes and "
                                              "I'll text you when it's done.")] + self._invite_messages(inbound, case) \
                + [Outbound(inbound.chat_id, WHAT_HAPPENED)]

        rooms = [] if skipped else split_rooms(text)
        if rooms and not self.store.get_rooms(case["case_id"]):
            self.store.set_rooms(case["case_id"], [{"room_id": f"fam-{n}", "label": re.sub(r" \d+$", "", r), "name": r}
                                                   for n, r in enumerate(rooms, 1)], source="family")
        if not skipped:
            self.store.set_case(case["case_id"], rooms_note=text[:300])
        self.store.set_member(member["member_id"], state={"step": "disaster"})
        out = self._invite_messages(inbound, case)
        out[0].text = "Thank you. " + out[0].text
        return out + [Outbound(inbound.chat_id, WHAT_HAPPENED)]

    # ------------------------------------------- what happened, and which claim(s) it means
    def _answer_setup(self, inbound: Inbound, member: dict, case: dict, text: str) -> list[Outbound] | None:
        """The optional questions at the end of setup. A message that isn't an answer returns None and
        the question is dropped, so a family that just starts sending photos or asking things is never stuck."""
        step, chat_id = member["state"].get("step"), inbound.chat_id
        if step == "disaster":
            kind = disaster.parse_disaster(text)
            if not kind and NOT_SURE.fullmatch(text.strip(" .!")):
                self.store.set_member(member["member_id"], state={})
                return [Outbound(chat_id, "That's fine, we can leave it for now. "
                                          "Send photos from inside the home whenever you're ready.")]
            if not kind:
                return None
            self.store.set_case(case["case_id"], disaster=kind)
            if kind == disaster.STORM:
                self.store.set_member(member["member_id"], state={"step": "flooded"})
                return [Outbound(chat_id, "I'm sorry. After a hurricane or storm there are usually two claims: flood "
                                          "insurance for rising water, and homeowners insurance for wind and rain. "
                                          "I'll keep them apart for you.\n\nDid any rooms flood? Name them, like "
                                          "\"basement and kitchen\", or say none or all.")]
            self.store.set_member(member["member_id"], state={"step": "insured"})
            lead = {disaster.FLOOD: "I'm sorry. Flood insurance pays what things were worth, not the price new, so "
                                    "I'll show both for every item.",
                    disaster.FIRE: "I'm sorry."}.get(kind, "I'm sorry, that sounds awful.")
            return [Outbound(chat_id, f"{lead}\n\n{INSURED_Q}")]
        if step == "flooded":
            rooms = [r.get("name") or r["label"] for r in self.store.get_rooms(case["case_id"])]
            causes = disaster.parse_flooded(text, rooms)
            if causes is None:
                return None
            self.store.set_case(case["case_id"], room_causes=json.dumps(causes))
            self.store.set_member(member["member_id"], state={"step": "insured"})
            return [Outbound(chat_id, causes_line(causes) + f"\n\n{INSURED_Q}")]
        if step == "insured":
            answer = disaster.parse_insured(text)
            if not answer:
                return None
            self.store.set_case(case["case_id"], insured=answer)
            self.store.set_member(member["member_id"], state={})
            if answer == "no":
                return [Outbound(chat_id, "Okay. Without insurance, FEMA can help replace essentials like appliances, "
                                          "beds and other furniture, and a computer. I'll mark those on your list so "
                                          "they're easy to show.\n\nSend photos from inside the home whenever you're ready.")]
            if answer == "unsure":
                return [Outbound(chat_id, "That's fine. The list works either way, and you can check the policy later."
                                          "\n\nSend photos from inside the home whenever you're ready.")]
            return [Outbound(chat_id, "Good. I'll keep the list in the form your adjuster expects.\n\n"
                                      "Send photos from inside the home whenever you're ready.")]
        return None

    def _set_address(self, inbound: Inbound, member: dict, case: dict, address: str) -> list[Outbound]:
        """The home's address, given after setup: save it and look for a 3D scan or listing, as setup does."""
        chat_id = inbound.chat_id
        self.store.set_case(case["case_id"], address=address[:200])
        try:
            found = self.lookup(address)
        except Exception:                                   # the scan index is a bonus: never fail on it
            found = {"found": False}
        if found.get("found"):
            self.store.set_member(member["member_id"], state={"step": "use_scan", "later": True, "scan": {
                k: found.get(k) for k in ("sid", "lat", "lon", "homes_listing_id", "original_created")}})
            when = _month_year(found.get("original_created"))
            size = f" with {found['room_count']} rooms" if found.get("room_count") else ""
            return [Outbound(chat_id, f"Saved the address. I found a 3D scan of it{' from ' + when if when else ''}{size}. "
                                      "Use it as a starting point? It may show staging or a previous owner's things, "
                                      "so nothing is added until you confirm it.")]
        note = ""
        if disaster.uses_sb495(self.store.get_case(case["case_id"])) and (case.get("disaster") or "fire") != "fire":
            note = (" Since it's in California, SB 495 applies: after a declared disaster the insurer pays part of "
                    "the contents limit without an itemized list, and I'll count that in.")
        return [Outbound(chat_id, f"Saved the address: {address}. I don't have a 3D scan or listing for it, which is "
                                  f"normal, so we'll keep building the list from your photos.{note}")]

    def _set_cause(self, case: dict, chat_id: str, text: str) -> list[Outbound] | None:
        """'the kitchen flooded' / 'bedroom was wind damage' / 'it was a hurricane', any time later."""
        low = text.lower().strip(" .!")
        said = re.fullmatch(r"(?:it was|it's|this was|that was|we had|there was)\s+(?:a |an )?(.+)", low)
        if not said and not case.get("disaster") and re.fullmatch(           # a bare "fire" to the unanswered question
                r"(?:a |an |the )?(?:house |forest |brush |grass |camp )?(?:fire|wildfire|flood|flooding|hurricane|storm|"
                r"tropical storm|tornado|typhoon|derecho|nor'?easter|earthquake|landslide|mudslide|explosion|smoke)", low):
            said = re.fullmatch(r"(.+)", low)
        if said and disaster.parse_disaster(said.group(1)) and not re.search(r"\b(room|kitchen|bed|bath)", low):
            kind = disaster.parse_disaster(said.group(1))
            self.store.set_case(case["case_id"], disaster=kind)
            note = (" I'll keep flood and wind damage apart. Tell me which rooms flooded, like \"the basement flooded\"."
                    if kind == disaster.STORM else "")
            return [Outbound(chat_id, f"Got it, a {disaster.NAMES[kind]}.{note}")]
        if (case.get("disaster") or disaster.FIRE) != disaster.STORM:
            return None
        flooded = re.fullmatch(r"(?:the |our |my )?(.+?)\s+(?:flooded|was flooded|got flooded|had flooding|was underwater|"
                               r"was water damage|is flood damage|was flood damage)", low)
        windy = re.fullmatch(r"(?:the |our |my )?(.+?)\s+(?:was|is|had)\s+(?:wind|rain|wind and rain|roof|storm)(?: damage)?", low)
        match, cause = (flooded, "flood") if flooded else (windy, "wind") if windy else (None, None)
        if not match:
            return None
        room = tidy_room(match.group(1))
        causes = disaster.room_causes(case)
        causes[room.lower()] = cause
        self.store.set_case(case["case_id"], room_causes=json.dumps(causes))
        where = "the flood claim" if cause == "flood" else "the wind and rain claim"
        return [Outbound(chat_id, f"Got it. Everything in the {room.lower()} goes on {where}.")]

    def _invite_messages(self, inbound: Inbound, case: dict) -> list[Outbound]:
        case = self.store.get_case(case["case_id"])
        short = re.sub(r"^The ", "", case["name"])
        if inbound.chat_id.startswith(TELEGRAM_PREFIX):
            return [Outbound(inbound.chat_id,
                             f"You're set up. Now make a new Telegram group for the family and add me ({self.telegram_handle}) "
                             "when you create it. I'll link the group to this list on my own, and every photo anyone "
                             "drops in it goes into the same list. You can keep sending me photos here too.\n\n"
                             f"If I ever don't link a group, post {case['invite_code']} in it.")]
        whose = "our home's" if re.fullmatch(r"(?:our|my) home", short, re.I) else f"the {short}"      # the default name
        forward = (f"We're using Ashline to rebuild {whose} inventory. Text {case['invite_code']} to "
                   f"{self.number}, then send any photos from inside the house, from any year.")
        return [
            Outbound(inbound.chat_id,
                     f"Your code is {case['invite_code']}. Anyone in the family who has photos can text "
                     f"{case['invite_code']} to this number to join; just forward them the next message. "
                     "Everything they send goes into the same list. You can also send me photos yourself, any time."),
            Outbound(inbound.chat_id, forward),
        ]

    def lookup(self, address: str) -> dict:
        """Is there a 3D scan of this address? (Separate so tests and offline use can replace it.)"""
        from .costar.index import home_lookup
        return home_lookup(address)

    def _seed_from_scan(self, case: dict, chat_id: str, job: dict, now: datetime) -> list[Outbound]:
        """Run the home twin seed the owner agreed to, then say what it found. Nothing is confirmed."""
        from . import seed
        case_id, scan = case["case_id"], job["scan"]
        try:
            seed.seed_home_twin(case_id, sid=scan["sid"], loss_date=case.get("loss_date"), store=self.store)
        except Exception as exc:
            return [Outbound(chat_id, f"I couldn't go through the scan just now ({exc}). You can still send photos; "
                                      "I'll work from those.")]
        priced = price_items(case_id, self.store, self.pricer)
        self._research_later(case_id, chat_id, None)
        waiting = self.store.get_items(case_id, status="suggested", canonical_only=True)
        total = sum((i.get("price_mid") or 0) * (i.get("qty") or 1) for i in waiting)
        rooms = len({i["room_label"] for i in waiting if i.get("room_label")})
        member = next((m for m in self.store.get_members(case_id) if m["member_id"] == job["member_id"]), None)
        text = (f"I've gone through the scan: {len(waiting)} possible things in {rooms} room{'s' if rooms != 1 else ''}"
                + (f", about {money(total)}" if priced["priced"] and total else "")
                + ". None of it is on the list until you say so. Say review to go through it a room at a time"
                + (f", or see it all: {self.link(case, member)}" if member else "") + ".")
        out = [Outbound(chat_id, text)]
        questions = [q for q in gap_questions(case_id, self.store) if q["kind"] in ("structure", "listing")]
        if questions:
            out.append(Outbound(chat_id, questions[0]["text"]))
        return out

    def _join_case(self, inbound: Inbound, code: str, member: dict | None) -> list[Outbound]:
        case = self.store.find_case(invite_code=code)
        if not case:
            return [] if inbound.is_group else [Outbound(
                inbound.chat_id, "I don't recognise that code. Check it with the person who sent it to you.")]
        existing = self.store.get_member(inbound.sender, case["case_id"])
        if not existing:
            joined = self.store.add_member(case["case_id"], inbound.sender, inbound.sender_name)
            if not inbound.is_group:
                self.store.set_member(joined["member_id"], dm_chat=inbound.chat_id)
        elif member and member["case_id"] != case["case_id"]:
            pass                                           # also in another case: the code switches them over below
        if (not existing and member and member["case_id"] != case["case_id"] and member["role"] == "owner"
                and member["state"].get("step") in ("home_name", "address") and not inbound.is_group
                and not self.store.get_items(member["case_id"]) and len(self.store.get_members(member["case_id"])) == 1):
            self._delete(member["case_id"])                # a "hello?" before the code started an empty home: drop it
        self.store.set_chat(inbound.chat_id, case["case_id"], inbound.is_group)
        name = case.get("name") or "the home"
        name = name[:1].lower() + name[1:] if name.startswith("The ") else name
        if existing:
            return [Outbound(inbound.chat_id, f"You're already part of {name}. Send photos whenever you like.")]
        return [Outbound(inbound.chat_id,
                         f"You've joined {name}. Send any photos from inside the house: holidays, birthdays, "
                         "video calls, real estate listings, anything.")]

    # --------------------------------------------------------- photo intake
    def _intake(self, inbound: Inbound, member: dict, case: dict, now: datetime) -> list[Outbound]:
        case_id = case["case_id"]
        context = self._context(inbound.chat_id)
        burst = context.get("burst") or {"member_id": member["member_id"], "photos": [], "acked": False}
        folder = config.case_dir(case_id) / "photos"
        folder.mkdir(parents=True, exist_ok=True)
        saved = 0
        caption_room = self._caption_room(case_id, inbound.text)
        repeats, bad = [], 0
        known = self._fingerprints(case_id)
        attachments, clips, video_note = self._expand_videos(case_id, inbound.attachments)
        for name, data in attachments:
            photo = normalize_photo(data)
            if photo is None:
                bad += looks_like_image(data)               # a broken photo is mentioned; a PDF or note is not
                continue
            print_ = (photo_fingerprint(photo["image"]), photo["image"].size)
            twin = next((n for n, (f, size) in known if size == print_[1] and bin(f ^ print_[0]).count("1") <= 3), None)
            if twin is not None:                            # the same photo again: nothing new in it
                repeats.append(twin)
                continue
            number = self.store.next_photo_number(case_id)
            photo_id = f"fam-{number}"
            path = folder / f"{photo_id}.jpg"
            photo["image"].save(path, quality=92)
            self.store.add_photos(case_id, [{
                "photo_id": photo_id, "path": str(path), "source": "photo", "number": number,
                "taken": photo["taken"], "captured": photo["taken"], "status": "queued", "chat_id": inbound.chat_id,
                "room_label": caption_room, "batch_id": clips.get(name),
                "added_by": member.get("name") or inbound.sender_name or "family"}])
            burst["photos"].append(photo_id)
            known.append((number, print_))
            self._prints[str(path)] = print_
            saved += 1
        unreadable = inbound.missing + bad
        if unreadable:
            note = Outbound(inbound.chat_id, "I couldn't open one of the photos. Could you send it again?"
                            if unreadable == 1 else
                            f"I couldn't open {unreadable} of the photos. Could you send them again?")
        if not saved:
            if video_note:
                return [Outbound(inbound.chat_id, video_note)]
            if unreadable:
                return [note]
            if repeats and clips:
                return [Outbound(inbound.chat_id, "I already have that video, so nothing new was added.")]
            if repeats:
                which = ", ".join(f"photo {n}" for n in sorted(set(repeats)) if n)
                return [Outbound(inbound.chat_id, f"I already have that one ({which}), so nothing new was added."
                                 if which else "I already have that one, so nothing new was added.")]
            return []
        burst["last_at"] = now.isoformat()
        out = []
        if not burst["acked"]:                              # once per burst
            burst["acked"] = True
            out.append(Outbound(inbound.chat_id, "Got the video, looking through it." if clips else
                                "Got it, looking through them." if len(burst["photos"]) > 1 or saved > 1
                                else "Got it, looking at it."))
            if len(burst["photos"]) >= 500:
                out[-1].text = ("That's a lot, thank you. I'll work through them and send a summary in about "
                                "20 minutes.")
        context["burst"] = burst
        self._save(inbound.chat_id, context)
        return out + ([note] if unreadable else []) + ([Outbound(inbound.chat_id, video_note)] if video_note else [])

    def _plan(self, case: dict, chat_id: str, member: dict) -> list[Outbound]:
        """The plan of the home as a picture, with the link to the page where it can be explored."""
        from . import layout
        drawn = layout.build(case["case_id"], self.store)
        placed = len(drawn["pins"]) - drawn["unplaced"]
        if not drawn["rooms"] or not placed:
            return [Outbound(chat_id, "There's nothing to put on a plan yet. Send photos or a video walking through "
                                      "the home and I'll draw one.")]
        image = layout.render_png(drawn, config.case_dir(case["case_id"]) / "plan.png")
        kind = "from the 3D scan" if drawn["kind"] == "scan" else "sketched from your photos, so it is approximate"
        return [Outbound(chat_id, f"Here is the home with {placed} item{'s' if placed != 1 else ''} where "
                                  f"{'they were' if placed != 1 else 'it was'} recorded ({kind}). Tap around it here: "
                                  f"{self.link(case, member)}/plan", image=str(image))]

    def _receipts(self, inbound: Inbound, member: dict, case: dict, now: datetime) -> list[Outbound]:
        """Receipts, invoices or order pages: read them and offer what was paid as the price of matching items."""
        from . import bank, receipts
        chat_id, case_id = inbound.chat_id, case["case_id"]
        if member["role"] != "owner":
            owner = self._owner(case_id)
            return [Outbound(chat_id, f"Thank you. Only {(owner or {}).get('name') or 'the person who started this list'} "
                                      "can add receipts, since they change prices. Could you send it to them?")]
        reader = self.receipt_reader or receipts.get_reader()
        if reader is None:
            return [Outbound(chat_id, "I can't read receipts yet on this setup. You can still tell me a price: "
                                      "for example \"price 3 $450\".")]
        folder = config.case_dir(case_id) / "receipts"
        folder.mkdir(parents=True, exist_ok=True)
        bought, shops, unread = [], [], 0
        for name, data in inbound.attachments:
            number = len(list(folder.glob("rcpt-*"))) + 1
            receipt_id = f"rcpt-{number}"
            (folder / f"{receipt_id}{'.pdf' if receipts.is_pdf(data) else '.jpg'}").write_bytes(data)
            try:
                reading = reader.read(data)
            except receipts.ReceiptError as exc:
                print(f"ashline: receipt not read ({exc})")
                unread += 1
                continue
            if not reading["is_receipt"] or not reading["lines"]:
                unread += 1
                continue
            (folder / f"{receipt_id}.json").write_text(json.dumps(reading, indent=1))
            bought += receipts.as_purchases(reading, receipt_id)
            shops.append(reading.get("merchant") or "a store")
        if not bought:
            return [Outbound(chat_id, "I couldn't read that as a receipt or an order. If it is a photo of the home, send it "
                                      "again without the word receipt." if unread else "I didn't get a file with that.")]

        everything = [i for i in self.store.get_items(case_id, canonical_only=True) if i["status"] != "rejected"]
        from .pricing import price_sources
        open_items = [i for i in everything if i.get("price_source") != "set by family" and not price_sources(i).get("paid")]
        matches = sorted(bank.match_purchases(open_items, bought), key=lambda m: -m["purchase"]["amount"])[:8]
        matched = {m["purchase"]["id"] for m in matches}
        priced = [i for i in everything if i not in open_items]                 # already have a price the family gave or paid
        known = bank.match_purchases(priced, [p for p in bought if p["id"] not in matched])
        matched |= {m["purchase"]["id"] for m in known}
        extra = [p for p in bought if p["id"] not in matched and p["amount"] >= 20][:8]      # things not on the list yet
        context = self._context(chat_id)
        context.pop("bank", None)
        context.pop("receipt_adds", None)
        if matches:
            context["bank"] = [{"item_id": m["item"]["item_id"], "purchase": m["purchase"]} for m in matches]
        if extra:
            context["receipt_adds"] = extra
        self._save(chat_id, context)

        count = len(bought)
        text = (f"I read the receipt from {shops[0]}" if len(shops) == 1 else f"I read {len(shops)} receipts") \
            + f": {count} thing{'s' if count != 1 else ''} bought."
        if matches:
            lines = "\n".join(f"{n}. {_cap(bank.describe(m))}" for n, m in enumerate(matches, 1))
            text += (f"\n\n{'These match' if len(matches) != 1 else 'This matches'} things on your list:\n{lines}\n\n"
                     f"Should I use what you paid as the price for {'these' if len(matches) != 1 else 'it'}? "
                     "A receipt is also proof for the insurer. Reply yes or no.")
        if known:
            names = "; ".join(_cap(m["item"]["label"]) for m in known)
            text += f"\n\nAlready on your list with a price: {names}. I left {'those' if len(known) != 1 else 'it'} as {'they are' if len(known) != 1 else 'it is'}."
        if extra:
            names = "; ".join(f"{_cap(p['description'])} ({money(p['amount'])})" for p in extra)
            text += (f"\n\n{'Also on it, not' if matches else 'Not'} on your list yet: {names}. "
                     f"Reply \"add\" to put {'them' if len(extra) != 1 else 'it'} on the list.")
        if not matches and not extra and not known:
            text += " None of it matches something on your list, and nothing on it looks like a belonging to add."
        if unread:
            text += f"\n\nI couldn't read {unread} of the files."
        return [Outbound(chat_id, text, choices=[("Use these prices", "yes"), ("Keep as is", "no")] if matches else [])]

    def _receipt_add(self, case: dict, chat_id: str, member: dict, lines: list) -> list[Outbound]:
        """Put receipt lines that are not on the list yet onto it, as suggestions priced at what was paid."""
        from .pricing import record_price
        case_id, added = case["case_id"], []
        for line in lines:
            label = re.sub(r"^\d+ x ", "", line["description"])
            qty = int(line.get("quantity") or 1)
            item_id = self.store.add_item(case_id, {"label": label, "category": detect_mod.guess_category(label), "qty": qty,
                                                    "evidence_date": line.get("date")},
                                          source="receipt", added_by=member.get("name"))
            self.store.set_cluster(case_id, item_id, [item_id], qty)
            record_price(self.store, case_id, item_id, "paid", {
                "amount": line["amount"], "each": round(line["amount"] / qty, 2), "merchant": line.get("merchant") or "a store",
                "date": line.get("date"), "description": line["description"], "bank_id": line["id"], "receipt": line.get("receipt")})
            added.append(item_id)
        self.store.log_event(case_id, "receipt_add", {"items": added}, member["member_id"])
        context = self._context(chat_id)
        context.pop("receipt_adds", None)
        self._save(chat_id, context)
        count = len(added)
        return [Outbound(chat_id, f"Added {count} thing{'s' if count != 1 else ''} from the receipt as "
                                  f"{'suggestions' if count != 1 else 'a suggestion'}, priced at what you paid. Tell me which room "
                                  f"{'they were' if count != 1 else 'it was'} in when you review. Say undo to go back.")]

    def _bank(self, case: dict, chat_id: str, member: dict) -> list[Outbound]:
        """Look in the family's card purchases for what they paid for things on the list."""
        from . import bank
        if member["role"] != "owner":
            return [Outbound(chat_id, "Only the person who started this list can connect bank records.")]
        try:
            matches = bank.find_matches(case["case_id"], self.store, self.bank)
        except config.MissingKey:
            return [Outbound(chat_id, "Bank records aren't connected yet, so I can't look at purchases.")]
        except bank.BankError as exc:
            return [Outbound(chat_id, f"I couldn't read the bank records just now. {exc}")]
        if not matches:
            return [Outbound(chat_id, "I looked through the card purchases and found none that clearly match "
                                      "something on the list.")]
        matches = sorted(matches, key=lambda m: -m["purchase"]["amount"])[:8]       # the biggest purchases first
        context = self._context(chat_id)
        context["bank"] = [{"item_id": m["item"]["item_id"], "purchase": m["purchase"]} for m in matches]
        self._save(chat_id, context)
        lines = "\n".join(f"{n}. {_cap(bank.describe(m))}" for n, m in enumerate(matches, 1))
        count = len(matches)
        found = (f"I found {count} card purchases that match things on your list" if count != 1
                 else "I found 1 card purchase that matches something on your list")
        return [Outbound(chat_id, f"{found}:\n{lines}\n\nShould I use what you paid as the price for "
                                  f"{'these' if count != 1 else 'it'}? A purchase record is also proof for the "
                                  "insurer. Reply yes or no.", choices=[("Use these prices", "yes"), ("Keep as is", "no")])]

    def _bank_apply(self, case: dict, chat_id: str, member: dict, pending: list) -> list[Outbound]:
        from . import bank
        case_id = case["case_id"]
        matches = [{"item": item, "purchase": row["purchase"]} for row in pending
                   if (item := self.store.get_item(case_id, row["item_id"]))]
        done = bank.apply_matches(self.store, case_id, matches, member["member_id"])
        context = self._context(chat_id)
        context.pop("bank", None)
        self._save(chat_id, context)
        proof = "a receipt" if any(row["purchase"].get("receipt") for row in pending) else "a bank record"
        more = context.get("receipt_adds")
        waiting = sum(1 for row in pending if (item := self.store.get_item(case_id, row["item_id"])) and item["status"] == "suggested")
        progress = (f"{'They' if done != 1 else 'It'} still {'wait' if done != 1 else 'waits'} for your OK in the review before "
                    f"{'they count' if done != 1 else 'it counts'} on the list." if waiting
                    else meter_line(meter(case_id, self.store)).splitlines()[0])
        return [Outbound(chat_id, f"Done. {done} item{'s now carry' if done != 1 else ' now carries'} the price you "
                                  f"paid, marked as {proof}. {progress} "
                                  "Say undo to go back." + (" Reply \"add\" to put the other things on the receipt on the list." if more else ""))]

    def _research_later(self, case_id: str, chat_id: str, item_ids: list[int] | None) -> None:
        """Have the price agent look the bigger new items up online, without anyone asking.

        It runs in the background, since each lookup is a web search; what it finds is mentioned
        in the chat by the next tick. Needs an Anthropic key, and ASHLINE_AUTO_PRICE not off.
        """
        from . import price_search
        if not price_search.auto_enabled():
            return
        searcher = self.searcher or price_search.get_searcher()
        if searcher is None:
            return

        def work(store):
            try:
                price_search.auto_price(case_id, store, item_ids, searcher, chat_id=chat_id)
            except Exception as exc:                        # the estimates stay
                print(f"ashline: automatic price lookup failed ({exc})")
        if not self.background:
            return work(self.store)
        self._in_background(("prices", case_id), work)

    def _in_background(self, job: tuple, work) -> None:
        """Run `work(store)` on a thread with its own database connection. One job of a kind per
        case at a time: a second burst of photos does not start a second, overlapping run."""
        if job in self._running:
            return
        self._running.add(job)
        import threading

        def run():
            store = type(self.store)(self.store.path)
            try:
                work(store)
            finally:
                store.db.close()
                self._running.discard(job)
        threading.Thread(target=run, daemon=True).start()

    def _price_news(self, case: dict, owns=None) -> list[Outbound]:
        """Tell the family, once, which prices the agent firmed up online."""
        from . import price_search
        news = price_search.take_news(case["case_id"])
        if not news:
            return []
        owner = self._owner(case["case_id"])
        by_chat: dict[str, list[dict]] = {}
        later = []
        for row in news:
            chat = row.get("chat_id") or (owner or {}).get("dm_chat") or (owner or {}).get("phone")
            if chat and owns and not owns(chat):
                later.append(row)                           # another bridge's chat: left for that bridge to say
            elif chat:
                by_chat.setdefault(chat, []).append(row)
        if later:
            price_search.add_news(case["case_id"], later)
        out = []
        for chat, rows in by_chat.items():
            lines = []
            for row in rows[:8]:
                was = f" (I had guessed about {money(row['before'])})" if row.get("before") else ""
                lines.append(f"• {_cap(row['label'])}: {money(row['price'])} at {row['retailer']}{was}")
            count = len(rows)
            out.append(Outbound(chat, f"I looked up {count} of the bigger item{'s' if count != 1 else ''} online to "
                                      "firm up the price" + ("s" if count != 1 else "") + ":\n" + "\n".join(lines)
                                      + "\n\nEach item's page shows the listing the price came from."))
        return out

    def _sketch_later(self, case_id: str) -> None:
        """Redraw the plan sketch in the background after new photos, when a model is available."""
        from . import layout, sketch_claude
        if not sketch_claude.available() or (layout.load_plan(case_id) or {}).get("rooms"):
            return
        def work(store):
            try:
                sketch_claude.refresh(case_id, store)
            except Exception as exc:                        # the plan falls back to the simple sketch
                print(f"ashline: plan sketch not drawn ({exc})")
        self._in_background(("sketch", case_id), work)

    def _expand_videos(self, case_id: str, attachments: list) -> tuple[list, dict, str | None]:
        """Videos become stills that go through the photo pipeline.

        Returns (attachments with each video replaced by its stills, {still name: clip id}, a note
        to send when a video could not be read).
        """
        from . import video
        out, clips, note = [], {}, None
        for name, data in attachments:
            if not video.is_video(name, data):
                out.append((name, data))
                continue
            try:
                frames = video.frames_as_attachments(name, data, self.video_frames)
            except video.VideoError as exc:
                note = f"I couldn't open the video. {exc}" if "ffmpeg" in str(exc) else \
                    "I couldn't open the video. Could you send it again, or send a few photos instead?"
                continue
            # Unique per video: photo numbers only go up, and the count covers several videos in one message.
            clip = f"clip-{self.store.next_photo_number(case_id) + len(clips)}"
            for frame_name, frame in frames:
                frame_name = f"{clip}-{frame_name}"         # two videos may carry the same file name
                clips[frame_name] = clip
                out.append((frame_name, frame))
        return out, clips, note

    def tick(self, now: datetime | None = None, owns=None) -> list[Outbound]:
        """Things that happen later. Call every few seconds.
        owns(chat_id) -> bool limits the work to one bridge's chats, so the iMessage and Telegram bridges
        can run side by side without taking (and losing) each other's digests and reminders."""
        now = _now(now)
        owns = owns or (lambda chat_id: True)
        out: list[Outbound] = []
        for case in self.store.list_cases():
            for chat in self.store.get_chats(case["case_id"]):
                if not owns(chat["chat_id"]):
                    continue
                context = self._context(chat["chat_id"])
                job = context.pop("seed", None)
                if job:                                       # the owner agreed to start from the scan
                    self._save(chat["chat_id"], context)
                    out += self._seed_from_scan(case, chat["chat_id"], job, now)
                    context = self._context(chat["chat_id"])
                burst = context.get("burst")
                if burst and now - datetime.fromisoformat(burst["last_at"]) >= timedelta(seconds=self.batch_seconds):
                    context.pop("burst")
                    self._save(chat["chat_id"], context)
                    out += self._digest(case, chat["chat_id"], burst, now)
            for question in self.store.get_questions(case["case_id"]):
                if question.get("expires_at") and question["expires_at"] <= now.isoformat():
                    self.store.close_question(question["question_id"], "expired")     # both stay as suggestions
            owner = self._owner(case["case_id"])
            if owner and owns(owner.get("dm_chat") or owner["phone"]):   # the owner's own platform sends these
                out += self._daily_review(case, now)
                out += self._reminders(case, now)
            out += self._price_news(case, owns)
        return out

    def _reminders(self, case: dict, now: datetime) -> list[Outbound]:
        """Proof-of-loss reminders at 60, 30 and 10 days, and at most one nudge a week after silence."""
        import json
        owner = self._owner(case["case_id"])
        if not owner or owner.get("paused") or not owner.get("dm_chat") or now.hour != int(case.get("review_hour") or 18):
            return []
        sent = json.loads(case.get("reminders") or "{}")
        m = meter(case["case_id"], self.store, today=now.date())
        text = None
        if m["days_left"] in DEADLINE_DAYS and m["days_left"] not in sent.get("deadline", []):
            sent.setdefault("deadline", []).append(m["days_left"])
            text = f"A gentle reminder: proof of loss is due in {m['days_left']} days. " + meter_line(m).split("\nProof of loss")[0]
        else:
            quiet_since = max(owner.get("last_seen") or "", sent.get("last") or "")       # never two in one week
            if quiet_since and now - datetime.fromisoformat(quiet_since) >= timedelta(days=7) and m["target"]:
                text = (f"You're at {money(m['documented'])} of {money(m['target'])}. Even a few more photos would help, "
                        "whenever you feel up to it.")
        if not text:
            return []
        sent["last"] = now.isoformat()
        self.store.set_case(case["case_id"], reminders=json.dumps(sent))
        return [Outbound(owner["dm_chat"], text)]

    @staticmethod
    def _default_grouper(case_id: str):
        """Claude decides which detections across photos are one object, when a key is set.
        ASHLINE_MERGER=rules turns it off (word-matching rules only)."""
        kind = (config.env("ASHLINE_MERGER") or ("claude" if config.env("ANTHROPIC_API_KEY") else "rules")).lower()
        if kind != "claude":
            return None
        from .merge_claude import ClaudeGrouper
        return ClaudeGrouper(case_id)

    def _fingerprints(self, case_id: str) -> list[tuple]:
        """(photo number, (fingerprint, size)) of the family photos already in the case."""
        out = []
        for p in self.store.get_photos(case_id, source="photo"):
            path = p.get("path")
            if path not in self._prints:
                try:
                    with Image.open(path) as image:
                        self._prints[path] = (photo_fingerprint(image), image.size)
                except (OSError, TypeError):
                    continue
            out.append((p.get("number"), self._prints[path]))
        return out

    def _caption_room(self, case_id: str, text: str) -> str | None:
        """The room a photo's caption names ('kitchen', 'this is the garage'), if any.

        Matches one of the home's rooms first; a short caption that names no known room is used as
        the room name itself, so 'dorm room' or 'Maya's bedroom' still keeps photos apart.
        """
        caption = (tidy_room(text) or "").lower()
        if not caption:
            return None
        # Rooms of the home, then rooms earlier photos were given, so "basement" and "Basement" are one room.
        known = list(self._room_names(case_id))
        known += [p["room_label"] for p in self.store.get_photos(case_id, source="photo") if p.get("room_label")]
        for name in sorted(dict.fromkeys(known), key=len, reverse=True):
            if re.search(rf"\b{re.escape(name.lower())}\b", caption):
                return name
        if len(caption.split()) <= 3 and not re.search(r"\d", caption):
            return caption[:1].upper() + caption[1:]
        return None

    def _digest(self, case: dict, chat_id: str, burst: dict, now: datetime) -> list[Outbound]:
        """Read a burst of photos and reply once. A photo that can't be read is kept and retried with the
        next burst; it never stops the others."""
        self._unread = 0
        out = self._read_burst(case, chat_id, burst, now)
        if self._unread:
            n = self._unread
            note = (f"I couldn't read {'one of the photos' if n == 1 else f'{n} of the photos'} just now. "
                    f"I've kept {'it' if n == 1 else 'them'} and will try again with the next photos you send.")
            out = out + [Outbound(chat_id, note)] if out else [Outbound(chat_id, note)]
        return out

    def _read_burst(self, case: dict, chat_id: str, burst: dict, now: datetime) -> list[Outbound]:
        case_id = case["case_id"]
        photos = [self.store.get_photo(case_id, pid) for pid in burst["photos"]]
        photos = [p for p in photos if p]
        earlier = [p for p in self.store.get_photos(case_id)        # photos that failed last time get another go
                   if p.get("status") == "failed" and p["photo_id"] not in burst["photos"]]
        photos += earlier[:5]
        rooms = self._room_names(case_id)
        sender = next((m for m in self.store.get_members(case_id) if m["member_id"] == burst["member_id"]), None)
        who = (sender or {}).get("name") or "you"
        get_detector = (lambda: self.detector) if self.detector else detect_mod.get_detector

        new_ids, found = [], 0
        for photo in photos:
            try:
                items = detect_cached(case_id, photo, None, get_detector=lambda: _RoomAware(get_detector(), rooms))
            except detect_mod.DetectionError as exc:
                print(f"ashline: could not read photo {photo['photo_id']}: {exc}")
                self.store.set_photo(case_id, photo["photo_id"], status="failed")
                self._unread += 1
                continue
            # A room the family named with the photo wins over the detector's guess, so a chair in the
            # bedroom is never merged with the one in the kitchen.
            separate = any("photo_room" in i for i in items)        # the detector says where the photo was taken
            room = photo.get("room_label") or next(
                (i[key] for key in (("photo_room",) if separate else ("room",)) for i in items if i.get(key)), None)
            self.store.set_photo(case_id, photo["photo_id"], status="done", room_label=room)
            photo = {**photo, "room_label": room}
            for item in items:
                # An item seen through a doorway is filed under the room it stands in, not the photo's,
                # so it merges with the sightings taken inside that room.
                elsewhere = item.get("photo_room") and item.get("room") and item["room"] != item["photo_room"]
                new_ids.append(self.store.add_item(case_id, {**item, "room_label": item["room"] if elsewhere else room},
                                                   source="photo", photo=photo, added_by=photo.get("added_by")))
            found += len(items)

        if not found:
            if self._unread:
                return []                                  # the note about unread photos says it all
            return [Outbound(chat_id, "Couldn't make out items in this one, that's fine." if len(photos) == 1
                             else "Couldn't make out items in those, that's fine.")]

        try:
            merged = merge_into_inventory(self.store, case_id, new_ids, self.grouper or self._default_grouper(case_id))
        except Exception as exc:                           # the model merger failed: word rules instead
            print(f"ashline: merge by model failed ({exc}); using rules")
            merged = merge_into_inventory(self.store, case_id, new_ids, None)
        price_items(case_id, self.store, self.pricer)
        self._sketch_later(case_id)
        self._research_later(case_id, chat_id, merged["new_lines"])
        lines = [self.store.get_item(case_id, i) for i in merged["new_lines"]]
        lines.sort(key=lambda i: -((i.get("price_mid") or 0) * (i.get("qty") or 1)))

        out: list[Outbound] = []
        shots: list[Outbound] = []
        context = self._context(chat_id)
        if len(photos) == 1:                                # one photo on its own: the annotated version
            out.append(self._annotated(case_id, chat_id, photos[0], context,
                                       extra=self._counts_line(len(lines), merged["merged"])))
        else:
            years = sorted({p["taken"][:4] for p in photos if p.get("taken")})
            span = f" ({years[0]}–{years[-1]})" if len(years) > 1 else f" ({years[0]})" if years else ""
            text = (f"Got {len(photos)} photos from {who}{span}.\n"
                    f"{found} items found · {len(lines)} new" + (f" · {merged['merged']} already listed" if merged["merged"] else ""))
            top = [f"• {_cap(self._display(i))}" for i in lines[:3] if i.get("price_mid")]
            if top:
                text += "\n\nBiggest finds:\n" + "\n".join(top)
            out.append(Outbound(chat_id, text))
            if len(photos) <= ANNOTATE_UP_TO:                # a few photos: each comes back annotated too,
                for p in photos:                             # after the summary and any question
                    shot = self._annotated(case_id, chat_id, p, context)
                    if not shot.text.startswith("I couldn't"):
                        shot.text = (f"Photo {p['number']}\n" if p.get("number") else "") + shot.text
                        shots.append(shot)

        heard = context.pop("heard", [])
        if heard:                                           # never a quote, only the item
            thing = heard[0]
            self.store.add_question(case_id, "add", {"label": thing}, chat_id,
                                    (now + timedelta(hours=QUESTION_HOURS)).isoformat(), now.isoformat())
            out.append(Outbound(chat_id, f"I heard a {thing} mentioned. Add it to the list? Reply yes or no.",
                                choices=[("Add it", "yes"), ("Leave it off", "no")]))
        elif merged["maybe"]:
            out += self._ask_same(case_id, chat_id, merged["maybe"], now)

        owner = self._owner(case_id)
        link_for = sender or owner
        if link_for and len(photos) > 1:
            out[0].text += f"\n\nFull list: {self.link(case, link_for)}"
        self._save(chat_id, context)
        # After a digest of several photos only: a single annotated photo is left to stand on its own.
        ask = self._directed_ask(case_id, chat_id, sender, now) if len(out) == 1 and len(photos) > 1 else None
        return out + ([ask] if ask else []) + shots

    def _directed_ask(self, case_id: str, chat_id: str, member: dict | None, now: datetime) -> Outbound | None:
        """One specific request to the person who just sent photos. At most once a day per person."""
        if not member or (member.get("last_ask") or "")[:10] == now.isoformat()[:10]:
            return None
        counts: dict[str, int] = {}
        for item in self.store.get_items(case_id, canonical_only=True):
            if item["status"] != "rejected" and item.get("room_label"):
                counts[item["room_label"].lower()] = counts.get(item["room_label"].lower(), 0) + 1
        thin = [(counts.get((r.get("name") or r["label"]).lower(), 0), r.get("name") or r["label"])
                for r in self.store.get_rooms(case_id)
                if not re.search(r"bath|hall|closet|other", (r.get("name") or r["label"]).lower())]
        thin = [t for t in thin if t[0] < DIRECTED_ASK_BELOW]
        if not thin or not counts:
            return None
        have, room = min(thin)
        self.store.set_member(member["member_id"], last_ask=now.isoformat())
        name = f"{member['name']}, " if member.get("name") else ""
        lead = f"the {room.lower()} only has {have} item{'s' if have != 1 else ''} so far" if have else \
            f"there's nothing from the {room.lower()} yet"
        return Outbound(chat_id, f"{name}{lead}. Any birthday or holiday photos taken there?")

    @staticmethod
    def _counts_line(new: int, merged: int) -> str:
        text = f"{new} new" if new else "nothing new"
        return text + (f" · {merged} already listed" if merged else "")

    def _make_new(self, case_id: str, chat_id: str, member: dict, number: str | None) -> list[Outbound]:
        """The family says items Ashline matched to the list are really new: give the ones in the
        last photo their own lines again. Undo puts them back."""
        context = self._context(chat_id)
        in_photo = context.get("photo_items", {})
        targets = [in_photo[number]] if number and number in in_photo else ([] if number else list(in_photo.values()))
        moved, was = [], {}
        for item_id in targets:
            item = self.store.get_item(case_id, item_id)
            if item and item.get("cluster_id") not in (None, item_id):
                was[item_id] = item["cluster_id"]
                self.store.update_item(case_id, item_id, cluster_id=item_id)
                self.store.set_item_status(case_id, item_id, "suggested")
                moved.append(item_id)
        if not moved:
            return [Outbound(chat_id, f"Number {number} is already its own item." if number else
                             "Everything in that photo is already its own item, so there's nothing to change.")]
        self.store.log_event(case_id, "unmatch", {"was": was}, member["member_id"])   # undo puts each back
        price_items(case_id, self.store, self.pricer)
        for n, item_id in in_photo.items():
            if item_id in moved:
                context.setdefault("list", {})[n] = item_id
        self._save(chat_id, context)
        names = [self.store.get_item(case_id, i)["label"] for i in moved]
        what = names[0] if len(names) == 1 else f"{len(names)} items ({', '.join(dict.fromkeys(names))})"
        return [Outbound(chat_id, f"Got it, those are new. Added {what} to the list, waiting for your OK. "
                                  "Say undo if that was a mistake.")]

    def _annotated(self, case_id: str, chat_id: str, photo: dict, context: dict, extra: str = "") -> Outbound:
        """The original photo with numbered boxes, and a legend in the text."""
        shown, boxes, entries, in_photo, listed = {}, [], [], {}, 0
        detections = [i for i in self.store.get_items(case_id) if i["photo_id"] == photo["photo_id"]
                      and i["status"] != "rejected" and i.get("bbox")]
        new_canon = {i["item_id"] for i in detections if i["cluster_id"] in (None, i["item_id"])}
        open_maybe = {q["payload"].get("b") for q in self.store.get_questions(case_id) if q["kind"] == "same"}
        for n, item in enumerate(detections, 1):
            canonical = item["cluster_id"] or item["item_id"]
            status = "maybe" if canonical in open_maybe else "new" if item["item_id"] in new_canon else "listed"
            boxes.append({"n": n, "bbox": item["bbox"], "status": status})
            line = self.store.get_item(case_id, canonical) or item
            entries.append({"n": n, "label": self._display(line) if status != "listed" else f"{line['label']} · already listed"})
            shown[str(n)] = canonical
            in_photo[str(n)] = item["item_id"]
            listed += status == "listed"
        out_path = config.case_dir(case_id) / "annotated" / f"{photo['photo_id']}.jpg"
        annotate(photo["path"], boxes, out_path)
        context["list"] = shown
        context["photo_items"] = in_photo                 # which detection in this photo each number is
        text = legend(entries) if entries else "I couldn't make out items in this one."
        if extra and entries:
            text = f"Found {len(entries)} item{'s' if len(entries) != 1 else ''} · {extra}\n\n{text}"
        if listed:
            text += ('\n\nNot the same things? Reply "new" and I\'ll list them separately.' if listed > 1 else
                     '\n\nNot the same thing? Reply "new" and I\'ll list it separately.')
        return Outbound(chat_id, text, image=str(out_path))

    def _ask_same(self, case_id: str, chat_id: str, pairs: list, now: datetime) -> list[Outbound]:
        a, b = (self.store.get_item(case_id, i) for i in pairs[0])
        photo_a, photo_b = (self.store.get_photo(case_id, i["photo_id"]) for i in (a, b))
        if not (photo_a and photo_b):
            return []
        qid = self.store.add_question(case_id, "same", {"a": a["item_id"], "b": b["item_id"]}, chat_id,
                                      (now + timedelta(hours=QUESTION_HOURS)).isoformat(), now.isoformat())
        image = side_by_side(photo_a["path"], a["bbox"], photo_b["path"], b["bbox"],
                             config.case_dir(case_id) / "questions" / f"same-{qid}.jpg")
        noun = item_key(b["label"]) or "item"
        more = f" ({len(pairs) - 1} more might be duplicates too.)" if len(pairs) > 1 else ""
        return [Outbound(chat_id, f"Same {noun}? 1 = same, 2 = different.{more}", image=str(image),
                         choices=[("Same", "1"), ("Different", "2")])]

    # ------------------------------------------------------- pending answers
    def _answer_paid(self, chat_id: str, member: dict, case: dict, low: str):
        """A reply to "use what you paid as the price?" or "add the other things on the receipt?".

        Asked most recently, so it is answered first: otherwise a "yes" meant for the prices
        would be taken by an older open question, such as "same item?".
        """
        if member["role"] != "owner":
            return None
        context = self._context(chat_id)
        adds, pending = context.get("receipt_adds"), context.get("bank")
        if adds and low in ("add", "add them", "add it", "yes add", "add those", "add all", "add to list", "add to the list",
                            "add them to the list", "add them to list", "add it to the list", "add it to list"):
            return self._receipt_add(case, chat_id, member, adds)
        if pending:
            yes, no = yes_or_no(low)
            if yes or low in ("use them", "use those", "use bank prices", "ok", "okay"):
                return self._bank_apply(case, chat_id, member, pending)
            context.pop("bank", None)                       # anything else: the offer lapses
            self._save(chat_id, context)
            if no or low in ("skip", "leave them", "not now"):
                return [Outbound(chat_id, "Okay, the prices stay as they are.")]
        return None

    def _answer_pending(self, inbound: Inbound, member: dict, case: dict, text: str):
        case_id, low = case["case_id"], text.lower().strip(" .!")
        paid = self._answer_paid(inbound.chat_id, member, case, low)
        if paid is not None:
            return paid
        state = member["state"]
        if state.get("step") == "coverage" and member["role"] == "owner" and not inbound.is_group:
            self.store.set_member(member["member_id"], state={k: v for k, v in state.items() if k != "step"})
            if low in ("skip", "no", "not now", "later"):
                self.store.set_case(case_id, coverage_skipped=1)
                return [Outbound(inbound.chat_id, "That's fine. " + meter_line(meter(case_id, self.store)))]
            limit = parse_money(text)
            if limit and limit >= 1000:
                self.store.set_case(case_id, coverage_limit=limit, coverage_skipped=0)
                return [Outbound(inbound.chat_id, meter_line(meter(case_id, self.store)))]
            return None
        if state.get("step") == "delete" and not inbound.is_group:
            self.store.set_member(member["member_id"], state={})
            if low in ("yes", "delete", "yes delete", "yes, delete") or yes_or_no(low)[0] or re.search(
                    r"\b(yes|yeah|yep|confirm)\b.*\bdelete\b|\bdelete\b.*\b(it|everything|my case)\b", low):
                self._delete(case_id)
                return [Outbound(inbound.chat_id, "Everything for this case has been deleted: photos, items and messages.")]
            return [Outbound(inbound.chat_id, "Okay, nothing was deleted.")]

        questions = [q for q in self.store.get_questions(case_id, chat_id=inbound.chat_id)]
        if not questions:
            return None
        yes, no = yes_or_no(low)
        if not (yes or no):
            return None
        kinds = {q["kind"] for q in questions}
        if len(questions) > 1 and low in ("yes", "no", "y", "n", "ok", "okay") and len(kinds) > 1:
            names = [("the " + (item_key((self.store.get_item(case_id, q["payload"]["b"]) or {}).get("label", "")) or "item"))
                     if q["kind"] == "same" else f"the {q['payload'].get('label', 'item')}" for q in questions[:2]]
            return [Outbound(inbound.chat_id, f"{'Yes' if yes else 'No'} to {names[0]} or {names[1]}?")]
        question = questions[0]
        self.store.close_question(question["question_id"], "answered", "yes" if yes else "no")
        if question["kind"] == "same":
            a, b = question["payload"]["a"], question["payload"]["b"]
            if yes:
                moved = [s["item_id"] for s in self.store.get_sightings(case_id, b)]
                for item_id in moved:
                    self.store.update_item(case_id, item_id, cluster_id=a)
                self.store.log_event(case_id, "merge", {"into": a, "from": b, "moved": moved}, member["member_id"])
                return [Outbound(inbound.chat_id, "Thanks, counted once.")]
            return [Outbound(inbound.chat_id, "Thanks, kept as two.")]
        if question["kind"] == "add":
            if yes:
                item_id = self.store.add_item(case_id, {"label": question["payload"]["label"], "category": "other"},
                                              source="voice", added_by=member.get("name"))
                self.store.set_cluster(case_id, item_id, [item_id], 1)
                price_items(case_id, self.store, self.pricer)
                self.store.log_event(case_id, "add", {"item_id": item_id}, member["member_id"])
                return [Outbound(inbound.chat_id, f"Added the {question['payload']['label']}, marked \"from memory\". "
                                                  "A photo of it would make it stronger.")]
            return [Outbound(inbound.chat_id, "Okay, left it off.")]
        return None

    # -------------------------------------------------------------- commands
    def _command(self, inbound: Inbound, member: dict, case: dict, text: str, now: datetime):
        case_id, low = case["case_id"], text.lower().strip(" .!?")
        chat_id, owner = inbound.chat_id, member["role"] == "owner"

        if low in ("help", "?", "what can you do"):
            return [Outbound(chat_id, HELP)]
        cause = self._set_cause(case, chat_id, text)
        if cause is not None:
            return cause
        said = ADDRESS_CMD.fullmatch(text.strip())
        address = said.group("addr") if said else (text.strip() if ADDRESS.fullmatch(text.strip()) else None)
        if address and owner:
            return self._set_address(inbound, member, case, address.strip())
        if low in ("pause", "not today", "stop", "not now") and not member["state"].get("memory"):
            self.store.set_member(member["member_id"], paused=1)
            return [Outbound(chat_id, "Okay. I'll stay quiet until you write again.")]
        if low in ("list", "show me everything", "show everything", "the list", "link") or re.fullmatch(
                r"(?:show (?:me )?|see |send |what'?s on |what is on )?(?:the |my |our )?(?:whole |full )?list", low):
            return [Outbound(chat_id, f"Here is everything so far: {self.link(case, member)}")]
        if re.fullmatch(r"(check |look (at|in|through) |search )?(my |the |our )?(bank|card|credit card|purchases?|statements?)"
                        r"( records?| statements?| purchases?)?", low):
            return self._bank(case, chat_id, member)
        if re.fullmatch(r"(show (me )?)?(the )?(map|plan|floor ?plan|layout)( of the (home|house))?", low):
            return self._plan(case, chat_id, member)
        if re.fullmatch(r"(how much( so far)?|total|progress|how are we doing|where are we)", low):
            m = meter(case_id, self.store)
            out = [Outbound(chat_id, meter_line(m))]
            sent = json.loads(case.get("reminders") or "{}")      # per-home notes; "coverage_asked": asked once already
            if owner and not inbound.is_group and not m["limit"] and not case.get("coverage_skipped") \
                    and not sent.get("coverage_asked"):
                self.store.set_case(case_id, reminders=json.dumps({**sent, "coverage_asked": True}))
                self.store.set_member(member["member_id"], state={**member["state"], "step": "coverage"})
                out.append(Outbound(chat_id, "If you tell me the contents coverage limit on the policy, I can show how "
                                             "much is left to document. Or say skip."))
            return out
        match = re.fullmatch(r"show (?:me )?(?:photo|picture|pic)?\s*#?(\d+)", low)
        if match:
            photo = self.store.get_photo(case_id, number=int(match.group(1)))
            if not photo:
                return [Outbound(chat_id, f"I don't have a photo {match.group(1)}.")]
            context = self._context(chat_id)
            reply = self._annotated(case_id, chat_id, photo, context)
            self._save(chat_id, context)
            return [reply]
        match = re.fullmatch(r"show (?:me )?(?:the )?(.+?)(?: ones| photos| pictures)?", low)
        if match and match.group(1) not in ("everything", "me everything"):
            wanted = match.group(1)
            photos = [p for p in self.store.get_photos(case_id, source="photo")
                      if wanted in (p.get("room_label") or "").lower()][:3]
            if not photos:
                return [Outbound(chat_id, f"I don't have photos of the {wanted} yet.")]
            context, out = self._context(chat_id), []
            for photo in photos:
                out.append(self._annotated(case_id, chat_id, photo, context))
            self._save(chat_id, context)
            return out
        if low in ("review", "what's new", "whats new"):
            return self._review(case, chat_id, member, now) or [Outbound(chat_id, "Nothing new is waiting for your OK.")]
        if low in ("ok", "okay", "ok add all", "add all", "looks good", "yes") and owner:
            ids = [i for i in self._context(chat_id).get("review", [])]
            if not ids:
                return None
            changed = [i for i in ids if (self.store.get_item(case_id, i) or {}).get("status") == "suggested"]
            for item_id in changed:
                self.store.set_item_status(case_id, item_id, "confirmed")
            self.store.log_event(case_id, "confirm", {"items": changed}, member["member_id"])
            context = self._context(chat_id)
            context.pop("review", None)
            self._save(chat_id, context)
            return [Outbound(chat_id, f"Added {len(changed)} item{'s' if len(changed) != 1 else ''}. "
                             + meter_line(meter(case_id, self.store)))]
        match = (re.fullmatch(r"(?:remove|delete|drop)\s+#?(\d+)", low)
                 or re.fullmatch(r"#?(\d+)\s+(?:isn't|isnt|is not|wasn't|wasnt|was not)\s+(?:ours|mine|hers|his)", low))
        if match:
            item_id = self._context(chat_id).get("list", {}).get(match.group(1))
            item = self.store.get_item(case_id, item_id) if item_id else None
            if not item:
                return [Outbound(chat_id, f"I don't have a number {match.group(1)} in the last list I sent.")]
            if not owner and item["status"] == "confirmed":
                boss = (self._owner(case_id) or {}).get("name") or "the owner"
                return [Outbound(chat_id, f"That one is already confirmed, so I'll leave it for {boss} to decide.")]
            self.store.log_event(case_id, "remove", {"item_id": item_id, "status": item["status"]}, member["member_id"])
            self.store.set_item_status(case_id, item_id, "rejected")
            return [Outbound(chat_id, f"Removed the {item['label']}. Say undo if that was a mistake.")]
        match = (re.fullmatch(r"(?:price\s+)?#?(\d+)\s+(?:was|is|cost|costs|at|for)?\s*(\$\s*[\d][\d,.]*\s*k?|[\d][\d,.]*\s*(?:k|dollars))", low)
                 or re.fullmatch(r"price\s+#?(\d+)\s+(\$?\s*[\d][\d,.]*\s*k?)", low))
        if match:
            item_id = self._context(chat_id).get("list", {}).get(match.group(1))
            item, value = (self.store.get_item(case_id, item_id) if item_id else None), parse_money(match.group(2))
            if item and value:
                self.store.log_event(case_id, "item.price", {"item_id": item_id, "before": {k: item.get(k) for k in (
                    "status", "label", "qty", "price_low", "price_mid", "price_high", "price_source")}}, member["member_id"])
                self.store.update_item(case_id, item_id, price_low=value, price_mid=value, price_high=value,
                                       price_source="set by family")
                return [Outbound(chat_id, f"Set the {item['label']} to {money(value)}, marked as a price set by the family.")]
        match = (re.fullmatch(r"#?(\d+)\s+(?:is|was|are)\s+(?:new|different|not the same|not (?:already )?(?:listed|added))", low.strip(" .!"))
                 or re.fullmatch(NEW_ITEMS, low.strip(" .!")))
        if match:
            return self._make_new(case_id, chat_id, member, match.group(1) if match.groups() else None)
        match = re.fullmatch(r"(?:split|separate)\s+#?(\d+)", low)
        if match:
            item_id = self._context(chat_id).get("list", {}).get(match.group(1))
            others = [s for s in self.store.get_sightings(case_id, item_id) if s["item_id"] != item_id] if item_id else []
            if not others:
                return [Outbound(chat_id, f"Number {match.group(1)} is only one item, so there's nothing to split.")]
            for sighting in others:
                self.store.update_item(case_id, sighting["item_id"], cluster_id=sighting["item_id"])
            self.store.log_event(case_id, "split", {"into": item_id, "moved": [s["item_id"] for s in others]},
                                 member["member_id"])
            return [Outbound(chat_id, f"Split into {len(others) + 1} separate items. Say undo if that was a mistake.")]
        moved = (re.fullmatch(r"#?(\d+)\s+(?:was|is|were|are)\s+in\s+(?:the\s+|our\s+|my\s+)?(.+)", low.strip(" .!"))
                 or re.fullmatch(r"move\s+#?(\d+)\s+(?:to|into)\s+(?:the\s+|our\s+|my\s+)?(.+)", low.strip(" .!")))
        counted = (re.fullmatch(r"#?(\d+)\s*x\s*(\d+)", low.strip(" .!"))
                   or re.fullmatch(r"(?:qty|quantity)\s+#?(\d+)\s+(\d+)", low.strip(" .!"))
                   or (lambda m: m and (m[0], m[2], m[1]))(
                       re.fullmatch(r"there (?:were|are|was) (\d+) of (?:number |#)?(\d+)", low.strip(" .!"))))
        if moved or counted:
            number, value = (moved or counted)[1], (moved or counted)[2]
            item_id = self._context(chat_id).get("list", {}).get(number)
            item = self.store.get_item(case_id, item_id) if item_id else None
            if not item:
                return [Outbound(chat_id, f"I don't have a number {number} in the last list I sent.")]
            if moved:
                room = next((r for r in self.store.get_rooms(case_id) if (r.get("label") or "").lower() == value
                             or (r.get("name") or "").lower() == value), None)
                label = (room or {}).get("label") or value[:1].upper() + value[1:60]
                self.store.log_event(case_id, "room", {"item_id": item_id, "room_id": item.get("room_id"),
                                                       "room_label": item.get("room_label")}, member["member_id"])
                self.store.update_item(case_id, item_id, room_label=label, room_id=(room or {}).get("room_id"))
                return [Outbound(chat_id, f"Moved the {item['label']} to the {label.lower()}. Say undo if that was a mistake.")]
            qty = int(value)
            if not 1 <= qty <= 999:
                return [Outbound(chat_id, "That number looks too big. How many were there?")]
            self.store.log_event(case_id, "qty", {"item_id": item_id, "qty": item.get("qty")}, member["member_id"])
            self.store.update_item(case_id, item_id, qty=qty)
            return [Outbound(chat_id, f"Got it: {qty} of the {item['label']}. Say undo if that was a mistake.")]
        match = re.fullmatch(r"#?(\d+)\s+(?:is|was)\s+(?:an?\s+|the\s+)?(.+)", text.strip(" ."), flags=re.I)
        if match:
            item_id = self._context(chat_id).get("list", {}).get(match.group(1))
            item = self.store.get_item(case_id, item_id) if item_id else None
            if item:
                from .pricing import price_sources
                fields = ("price_low", "price_mid", "price_high", "price_source", "price_ref")
                self.store.log_event(case_id, "rename", {"item_id": item_id, "label": item["label"],
                                                         "prices": {k: item.get(k) for k in fields}}, member["member_id"])
                keep = item.get("price_source") == "set by family" or price_sources(item).get("paid")
                self.store.update_item(case_id, item_id, label=match.group(2).strip()[:120],
                                       **({} if keep else dict(price_mid=None, price_low=None, price_high=None, price_source=None)))
                price_items(case_id, self.store, self.pricer)
                fresh = self.store.get_item(case_id, item_id)
                return [Outbound(chat_id, f"Updated: {self._display(fresh)}.")]
        if low == "undo":
            done = self._undo(case_id, None if owner else member["member_id"])
            if not done and not owner and self.store.last_event(case_id):
                return [Outbound(chat_id, "There's nothing of yours to undo. You can undo the changes you made yourself, "
                                          "and the person who started the list can undo anything.")]
            return [Outbound(chat_id, done or "There's nothing to undo.")]
        if low in ("export", "send the list", "send me the list", "download"):
            if not owner:
                boss = (self._owner(case_id) or {}).get("name") or "the owner"
                return [Outbound(chat_id, f"Only {boss} can export. Here is the list: {self.link(case, member)}")]
            from . import export as exporter
            m = meter(case_id, self.store)
            if not m["items"]:
                return [Outbound(chat_id, "Nothing is confirmed yet, so there's nothing to export. Say review to see "
                                          "what's waiting for your OK.")]
            files = [str(exporter.export_pdf(case_id, self.store)), str(exporter.export_xlsx(case_id, self.store))]
            return [Outbound(chat_id, f"Here are the PDF for the insurer and the spreadsheet: {m['items']} items, about "
                                      f"{money(m['documented'])}. Estimates for planning, not legal or insurance "
                                      "advice; confirm values with your adjuster.", files=files)]
        if low in ("call me", "call", "let's talk", "lets talk"):
            return self._memory_start(inbound, member, case)
        if (low in ("delete my case", "delete everything", "delete all")
                or re.fullmatch(r"(?:can you |could you |please |pls )*(?:delete|erase|wipe|remove)\s+"
                                r"(?:my |the |this |our )?(?:case|data|list|everything|all of it)(?: please| pls)?", low)) \
                and owner and not inbound.is_group:
            self.store.set_member(member["member_id"], state={"step": "delete"})
            return [Outbound(chat_id, "This removes every photo, item and message for this case, for everyone. "
                                      "Reply yes to delete, or anything else to keep it.")]
        return None

    def _undo(self, case_id: str, member_id: int | None = None) -> str | None:
        event = self.store.last_event(case_id, member_id)
        if not event:
            return None
        payload, action = event["payload"], event["action"]
        if action == "confirm":
            for item_id in payload["items"]:
                self.store.set_item_status(case_id, item_id, "suggested")
            text = f"Undone: those {len(payload['items'])} items are back to waiting for an OK."
        elif action == "remove":
            self.store.set_item_status(case_id, payload["item_id"], payload["status"])
            text = "Undone: it's back on the list."
        elif action == "rename":
            from .pricing import price_sources
            now_item, prices = self.store.get_item(case_id, payload["item_id"]) or {}, payload.get("prices", {})
            if price_sources(now_item).get("paid") != price_sources(prices).get("paid"):
                prices = {}                                 # a price paid since the rename stays
            self.store.update_item(case_id, payload["item_id"], label=payload["label"], **prices)
            text = "Undone: the name is back to what it was."
        elif action == "room":
            self.store.update_item(case_id, payload["item_id"], room_id=payload["room_id"], room_label=payload["room_label"])
            text = "Undone: it's back in the room it was in."
        elif action == "qty":
            self.store.update_item(case_id, payload["item_id"], qty=payload["qty"])
            text = "Undone: the number is back to what it was."
        elif action == "merge":
            for item_id in payload["moved"]:
                self.store.update_item(case_id, item_id, cluster_id=payload["from"])
            text = "Undone: those are two separate items again."
        elif action == "add":
            self.store.set_item_status(case_id, payload["item_id"], "rejected")
            text = "Undone: I've taken that off the list."
        elif action == "unmatch":
            for item_id, cluster in payload["was"].items():
                self.store.update_item(case_id, int(item_id), cluster_id=cluster)
            text = "Undone: those are back on the lines they were matched to."
        elif action == "split":
            for item_id in payload["moved"]:
                self.store.update_item(case_id, item_id, cluster_id=payload["into"])
            text = "Undone: those are one item again."
        elif action == "memory_add":                        # things someone remembered in a memory chat
            for item_id in payload["items"]:
                self.store.set_item_status(case_id, item_id, "rejected")
            text = f"Undone: I've taken {'those' if len(payload['items']) != 1 else 'that'} off the list."
        elif action == "receipt_add":                       # things added from a receipt
            for item_id in payload["items"]:
                self.store.set_item_status(case_id, item_id, "rejected")
            text = "Undone: I've taken those off the list."
        elif action == "bank":                              # prices taken from card purchases
            for row in payload["before"]:
                self.store.update_item(case_id, row["item_id"], **{k: row[k] for k in (
                    "price_low", "price_mid", "price_high", "price_source", "price_ref")})
            text = "Undone: those prices are back to the estimates."
        elif action.startswith("item."):                  # a change made on the inventory page
            before = payload["before"]
            prices = ("price_low", "price_mid", "price_high", "price_source", "price_ref") if action == "item.price" else ()
            self.store.update_item(case_id, payload["item_id"], **{k: before[k] for k in (
                "label", "qty", "cause") + prices if k in before})
            self.store.set_item_status(case_id, payload["item_id"], before["status"])
            text = "Undone: that item is back to how it was."
        else:
            return None
        self.store.mark_undone(event["event_id"])
        return text

    def _delete(self, case_id: str) -> None:
        import shutil
        shutil.rmtree(config.case_dir(case_id), ignore_errors=True)
        self.store.delete_case(case_id)

    # ---------------------------------------------------------------- review
    def _review(self, case: dict, chat_id: str, member: dict, now: datetime) -> list[Outbound] | None:
        """The owner's short review of what is waiting: 'Reply OK to add all, or tell me what's wrong.'"""
        case_id = case["case_id"]
        waiting = [i for i in self.store.get_items(case_id, status="suggested", canonical_only=True)]
        fresh = [i for i in waiting if not i.get("reviewed")] or waiting
        if not fresh:
            return None
        room_note = ""
        if len(fresh) > REVIEW_AT_ONCE:                     # too many for one message: one room at a time
            by_room: dict[str, list[dict]] = {}
            for item in fresh:
                by_room.setdefault(item.get("room_label") or "Unsorted", []).append(item)
            room, fresh = max(by_room.items(), key=lambda kv: sum((i.get("price_mid") or 0) * (i.get("qty") or 1)
                                                                  for i in kv[1]))
            left = len(by_room) - 1
            room_note = f"{room}: " + (f"({left} more room{'s' if left != 1 else ''} after this.) " if left else "")
        fresh.sort(key=lambda i: -((i.get("price_mid") or 0) * (i.get("qty") or 1)))
        people = sorted({("the 3D scan" if i["added_by"] == "scan" else i["added_by"]) for i in fresh if i.get("added_by")})
        total = sum((i.get("price_mid") or 0) * (i.get("qty") or 1) for i in fresh)
        context = self._context(chat_id)
        context["review"] = [i["item_id"] for i in fresh]
        context["list"] = {str(n): i["item_id"] for n, i in enumerate(fresh, 1)}
        self._save(chat_id, context)
        for item in fresh:
            self.store.update_item(case_id, item["item_id"], reviewed=1)
        self.store.set_case(case_id, last_review_at=now.isoformat())
        top = "\n".join(f"{n}. {_cap(self._display(i))}" for n, i in enumerate(fresh[:5], 1))
        more = f"\n…and {len(fresh) - 5} more" if len(fresh) > 5 else ""
        text = (room_note + f"{len(fresh)} new item{'s' if len(fresh) != 1 else ''}"
                + (f" from {_join(people)}" if people else "") + (f", about {money(total)}" if total else "")
                + f".\n\n{top}{more}\n\nReply OK to add them all, or tell me what's wrong (\"remove {min(4, len(fresh))}\").\n"
                + f"See them: {self.link(case, member)}")
        return [Outbound(chat_id, text)]

    def _daily_review(self, case: dict, now: datetime) -> list[Outbound]:
        owner = self._owner(case["case_id"])
        if not owner or owner.get("paused") or now.hour != int(case.get("review_hour") or 18):
            return []
        if (case.get("last_review_at") or "")[:10] == now.isoformat()[:10]:
            return []                                       # once a day; an unanswered one rolls into tomorrow's
        chat_id = owner.get("dm_chat") or owner["phone"]
        if not self.store.get_chat(chat_id):
            self.store.set_chat(chat_id, case["case_id"], False)
        waiting = self.store.get_items(case["case_id"], status="suggested", canonical_only=True)
        if waiting and all(i.get("reviewed") for i in waiting):    # nothing new since the last review
            last = case.get("last_review_at") or ""
            if last and datetime.fromisoformat(last) > now - timedelta(days=7):
                return []                                   # the same list again would only nag
            self.store.set_case(case["case_id"], last_review_at=now.isoformat())
            count = len(waiting)
            return [Outbound(chat_id, f"{count} thing{'s are' if count != 1 else ' is'} still waiting for your OK. "
                                      "No rush. Say review whenever you're ready.")]
        return self._review(case, chat_id, owner, now) or []

    # ------------------------------------------------- memory "call" by text
    def _memory_start(self, inbound: Inbound, member: dict, case: dict) -> list[Outbound]:
        queue = [q for q in gap_questions(case["case_id"], self.store) if q["kind"] != "yard"][:MEMORY_QUESTIONS]
        intro = "I can't call from this number yet, so let's do it here, one question at a time. Say stop whenever."
        if not queue:
            return [Outbound(inbound.chat_id, intro + " What's one thing you remember that no photo would show?")] \
                if self._begin_memory(member, [{"text": "", "kind": "open"}]) else []
        self._begin_memory(member, queue)
        return [Outbound(inbound.chat_id, intro), Outbound(inbound.chat_id, queue[0]["text"])]

    def _begin_memory(self, member: dict, queue: list[dict]) -> bool:
        self.store.set_member(member["member_id"], state={"memory": {
            "queue": [q["text"] for q in queue], "at": 0, "added": []}})
        return True

    def _memory_answer(self, inbound: Inbound, member: dict, case: dict, text: str):
        memory = member["state"].get("memory")
        if not memory or inbound.is_group:
            return None
        case_id, low = case["case_id"], text.lower().strip(" .!")
        ended = low in ("stop", "that's enough", "thats enough", "enough", "done", "i don't want to talk about that",
                        "not today", "no more", "pause", "quit", "exit", "end")
        import difflib
        if not ended and (low in MEMORY_COMMANDS or difflib.get_close_matches(low, MEMORY_COMMANDS, n=1, cutoff=0.75)):
            self.store.set_member(member["member_id"], state={"memory": memory})
            return [Outbound(inbound.chat_id, "We're still in our memory chat. Say stop to end it, then I can help "
                                              "with that. Or tell me about the question: " + memory["queue"][memory["at"]])]
        if not ended and low not in ("skip", "nothing", "no", "i don't remember", "not sure", "yes", "ok", "okay", "k",
                                     "yeah", "yep", "nope", "idk", "don't know", "dont know", "thanks", "thank you"):
            added = self._add_remembered(case_id, member, text)
            memory["added"] += added
            if added:
                self.store.log_event(case_id, "memory_add", {"items": added}, member["member_id"])
        memory["at"] += 1
        if ended or memory["at"] >= len(memory["queue"]):
            self.store.set_member(member["member_id"], state={})
            if not memory["added"]:
                return [Outbound(inbound.chat_id, "That's fine. Thank you for trying. We can pick it up another day.")]
            price_items(case_id, self.store, self.pricer)
            items = [i for i in (self.store.get_item(case_id, n) for n in memory["added"]) if i and i["status"] != "rejected"]
            if not items:
                return [Outbound(inbound.chat_id, "That's fine. Thank you for trying. We can pick it up another day.")]
            total = sum((i.get("price_mid") or 0) * (i.get("qty") or 1) for i in items)
            context = self._context(inbound.chat_id)
            context["review"] = [i["item_id"] for i in items]
            context["list"] = {str(n): i["item_id"] for n, i in enumerate(items, 1)}
            self._save(inbound.chat_id, context)
            names = " · ".join(f"{n} {i['label']}" for n, i in enumerate(items[:6], 1))
            return [Outbound(inbound.chat_id,
                             f"That was really helpful. From our chat: {len(items)} item{'s' if len(items) != 1 else ''}, "
                             f"about {money(total)}. They're marked \"from memory\". If anyone has photos of them, "
                             f"they'd make these stronger. {names}. Reply OK to add them, or tell me what's wrong.")]
        self.store.set_member(member["member_id"], state={"memory": memory})
        return [Outbound(inbound.chat_id, memory["queue"][memory["at"]])]

    def _add_remembered(self, case_id: str, member: dict, text: str) -> list[int]:
        """Add what someone remembers as suggestions marked "from memory". "a piano worth 3000" keeps the price they gave."""
        added = []
        for label, qty in extract_things(text):
            price = None
            match = re.fullmatch(r"(.+?),?\s+(?:that was |that's |which was )?(?:worth|cost|costs|costing|valued at|about|"
                                 r"around|for|at)\s+(?:about |around )?(\$?\s*[\d][\d,.]*\s*k?(?:\s*dollars)?)", label)
            if match and parse_money(match.group(2)):
                label, price = match.group(1).strip(), parse_money(match.group(2))
            if not label:
                continue
            item_id = self.store.add_item(case_id, {"label": label, "category": "other", "qty": qty,
                                                    "detail": "From memory"}, source="voice",
                                          added_by=member.get("name"))
            self.store.set_cluster(case_id, item_id, [item_id], qty)
            if price:
                each = round(price / qty, 2)
                self.store.update_item(case_id, item_id, price_low=each, price_mid=each, price_high=each,
                                       price_source="set by family")
            added.append(item_id)
        return added

    # ------------------------------------------------------------ listening
    def _listen(self, chat_id: str, text: str) -> None:
        """Family talking to each other. Note a belonging if one is clearly mentioned; say nothing."""
        low = text.lower()
        if not OWNERSHIP.search(low):
            return
        thing = next((w for w in sorted(HEARD_WORDS, key=len, reverse=True) if re.search(rf"\b{re.escape(w)}\b", low)), None)
        if not thing:
            return
        context = self._context(chat_id)
        heard = context.get("heard", [])
        if thing not in heard:
            context["heard"] = heard + [thing]              # the item only, never the sentence
            self._save(chat_id, context)


class _RoomAware:
    """Passes the home's room names to detectors that can use them."""

    def __init__(self, detector, rooms: list[str]):
        self.detector, self.rooms = detector, rooms

    def detect(self, path, room_hint=None):
        try:
            return self.detector.detect(path, room_hint, rooms=self.rooms)
        except TypeError:                                   # a detector without room support
            return self.detector.detect(path, room_hint)


YES_WORDS = {"1", "same", "yes", "y", "yep", "yeah", "ok", "okay", "yup", "sure"}
NO_WORDS = {"2", "different", "no", "n", "nope", "nah", "separate"}
FILLER_WORDS = {"it's", "its", "it", "the", "is", "a", "they're", "theyre", "are", "please", "add", "one", "ones"}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def yes_or_no(low: str) -> tuple[bool, bool]:
    """A short reply to a yes/no or 'same item?' question: '1', 'same', '1 same', '2, different', 'not the same'."""
    tokens = re.findall(r"[a-z0-9']+", low)
    if not tokens or len(tokens) > 4:
        return False, False
    if "not" in tokens:
        return False, "same" in tokens and all(t in FILLER_WORDS | {"not", "same"} for t in tokens)
    yes = all(t in YES_WORDS | FILLER_WORDS for t in tokens) and any(t in YES_WORDS for t in tokens)
    no = all(t in NO_WORDS | FILLER_WORDS for t in tokens) and any(t in NO_WORDS for t in tokens)
    return yes, no


def causes_line(causes: dict) -> str:
    flooded = [r for r, c in causes.items() if c == "flood"]
    if not flooded:
        return "Got it, no flooding. Everything goes on the wind and rain claim."
    if all(c == "flood" for c in causes.values()):
        return "Got it, everything flooded. It all goes on the flood claim."
    names = ", ".join(r.capitalize() for r in flooded)
    return f"Got it: {names} flooded, so those go on the flood claim. The other rooms go on the wind and rain claim."


def looks_like_image(data: bytes) -> bool:
    """JPEG, PNG, GIF, WebP or HEIC by its first bytes."""
    head = data[:16] if data else b""
    from .video import IMAGE_BRANDS
    return (head.startswith((b"\xff\xd8", b"\x89PNG", b"GIF8")) or head[8:12] == b"WEBP"
            or (head[4:8] == b"ftyp" and head[8:12] in IMAGE_BRANDS))      # HEIC, not a video or a voice memo


def photo_fingerprint(image) -> int:
    """A 64-bit difference hash: the same photo sent twice gets (nearly) the same number."""
    small = image.convert("L").resize((9, 8))
    px = list(small.tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def normalize_photo(data: bytes) -> dict | None:
    """Bytes as received -> upright RGB image and the date it was taken (build spec 6.2)."""
    try:
        try:
            import pillow_heif                              # iPhone photos arrive as HEIC
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        image = Image.open(io.BytesIO(data))
        taken = None
        try:
            exif = image.getexif()
            raw = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)       # DateTimeOriginal, DateTime
            if raw:
                taken = str(raw)[:10].replace(":", "-")
        except Exception:
            taken = None
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((4096, 4096))
        return {"image": image, "taken": taken}
    except Exception:
        return None


QUANTITY = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "couple": 2,
            "a couple of": 2, "pair of": 2, "few": 3, "a few": 3}


def extract_things(text: str) -> list[tuple[str, int]]:
    """'a table saw, two bikes and a bunch of tools' -> [('table saw', 1), ('bikes', 2), ('tools, assorted', 1)].

    A plain splitter: good enough for a text conversation, and it never invents an item. Vague
    answers become one group item rather than many lines.
    """
    text = re.sub(r"^(yes|yeah|yep|oh|um|well|there was|there were|we had|i had|i think|also|and)\b[,\s]*", "",
                  text.strip(), flags=re.I)
    things = []
    for part in re.split(r",|;|\band\b|\bplus\b|\n", text):
        part = part.strip(" .!").lower()
        part = re.sub(r"^(there was|there were|we had|i had|also|maybe|probably|i think)\s+", "", part)
        if not part or len(part) > 80 or part in ("yes", "no", "that's it", "thats it"):
            continue
        qty = 1
        match = re.match(r"(\d+|a couple of|a few|pair of|couple|few|two|three|four|five|six|one|an|a)\s+(.+)", part)
        if match:
            qty = QUANTITY.get(match.group(1)) or int(match.group(1))
            part = match.group(2)
        group = re.match(r"(?:a\s+)?(?:bunch|lot|lots|pile|box|boxes|set|collection|some|all kinds)\s+of\s+(.+)", part) \
            or re.match(r"(?:some|assorted|various|all the|all our|my|our)\s+(.+)", part)
        if group:
            things.append((f"{group.group(1).strip()}, assorted", 1))
        elif part:
            things.append((part, min(qty, 50)))
    return things[:12]
