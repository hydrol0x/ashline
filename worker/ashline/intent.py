"""Understanding messages that are not one of Ashline's exact commands.

People mistype, ramble and phrase things their own way ("can u send me the list pls",
"the second one is a different chair", "how much have we got"). When a message matches no
command, Claude maps it to ONE of Ashline's real commands, written in its exact form, or to a
short reply. The chat engine then runs that command through the same code as a typed one, so
the model can choose an action but never invent one. Deleting the case is offered, but it only
starts Ashline's own confirmation question; nothing is deleted until the person replies yes.
"""
from __future__ import annotations

import json

from . import config

DEFAULT_MODEL = "claude-haiku-5-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": ["string", "null"],
                    "description": "One command from the list, in its exact form, or null"},
        "reply": {"type": ["string", "null"],
                  "description": "A short, warm reply when no command fits, else null"},
    },
    "required": ["command", "reply"],
    "additionalProperties": False,
}

PROMPT = """You are the message router for Ashline, a text-message assistant that helps a family rebuild the list of
belongings their insurer needs after their home burned down. Be gentle; these people have lost their home.

A family member sent this message, and it did not match any exact command:
<message>{message}</message>

Context: {context}

Map the message to at most ONE of these commands, written exactly in this form (N is a number from Ashline's
last numbered list, X is text the person gave):
- 1                  (answer "same item" to Ashline's open same-item question; also "yes" to a yes/no question)
- 2                  (answer "different item"; also "no" to a yes/no question)
- list               (send the link to the full inventory page)
- how much so far    (the running total)
- map                (a plan of the home showing where each item was)
- check my bank      (look in the family's card purchases for what they paid for items on the list)
- show photo N       (send photo N back with its numbered boxes)
- show X             (photos of a room, e.g. show kitchen)
- review             (go through new items waiting for approval)
- ok                 (approve all items waiting for review)
- remove N           (take item N off the list)
- split N            (item N is really several different objects)
- new                (the items Ashline just called "already listed" are NOT on the list yet; list them as new.
                      Use this for "these weren't added", "not the same", "those are different", "they're new")
- N is new           (only item N from the last photo is new, not the same as the one already listed)
- N is X             (rename item N, e.g. 3 is a walnut dresser)
- price N X          (the family's own price for item N, e.g. price 2 $450)
- undo               (undo the last change)
- export             (send the spreadsheet and PDF)
- call me            (start a memory conversation about things no photo shows)
- pause              (stop prompts and reminders for now)
- delete my case     (the owner wants everything deleted; Ashline then asks them to confirm with yes)
- help               (what Ashline can do)
- address X          (the family gives the home's street address, e.g. address 12 Oak St, Pasadena, CA)
- date X             (when the fire, flood or storm happened, e.g. date July 2025)

Rules:
- Only choose a command the person clearly wants. Fix typos ("lsit" means list).
- If the message is small talk, thanks, a feeling, or a question you can answer in a sentence, set command to
  null and give a short reply (one or two sentences, no lists). Do not promise anything Ashline cannot do.
- If it is unclear, set command to null and reply with one short question that would make it clear.
- Your reply is sent as Ashline: write in the first person ("I"), never about "Ashline" in the third person.
  Never say you can't save or remember something the family tells you, and never send them to "the Ashline
  team" or an adjuster for something a command above does; if no command fits, ask what they would like
  you to do with it.
- Choose "delete my case" only when the person clearly asks to delete or erase their case, list or data.
  It does not delete anything by itself: Ashline asks them to confirm first."""

ALLOWED = ("new", "1", "2", "list", "check my bank", "map", "how much so far", "review", "ok", "undo", "export", "call me", "pause", "help",
           "delete my case")
PREFIXES = ("show ", "remove ", "split ", "price ", "address ", "date ")


def valid_command(command: str | None) -> str | None:
    """The command if it is one Ashline knows (exact, or one of its patterns), else None."""
    if not command:
        return None
    command = command.strip().lower()
    if command in ALLOWED or command.startswith(PREFIXES):
        return command
    head = command.split(" ", 2)
    if len(head) >= 3 and head[0].lstrip("#").isdigit() and head[1] in ("is", "was"):   # also "N is new"
        return command
    return None


class Interpreter:
    def __init__(self, model: str | None = None, client=None):
        config.load_env()
        self.model = model or config.env("ASHLINE_INTENT_MODEL", DEFAULT_MODEL)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def __call__(self, message: str, context: str = "") -> dict:
        """{'command': exact command or None, 'reply': text or None}. Never raises."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": PROMPT.format(message=message[:800], context=context or "none")}],
            )
            text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "{}")
            data = json.loads(text)
        except Exception as exc:                          # no network, bad key, odd reply: fall back quietly
            print(f"ashline: could not interpret message ({exc})")
            return {"command": None, "reply": None}
        reply = data.get("reply")
        return {"command": valid_command(data.get("command")),
                "reply": reply.strip()[:600] if isinstance(reply, str) and reply.strip() else None}


def get_interpreter():
    """An Interpreter when an Anthropic key is set, else None (exact commands only)."""
    config.load_env()
    if (config.env("ASHLINE_INTERPRETER") or "").lower() == "off" or not config.env("ANTHROPIC_API_KEY"):
        return None
    return Interpreter()
