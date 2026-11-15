"""Receipts and order pages: what the family paid, read from a picture or a PDF.

A photo of a paper receipt, a screenshot of an order page or an emailed receipt as a PDF, sent
to Ashline with the word "receipt", is read by Claude into its lines: the store, the date, and
each thing bought with what it cost. Each line is then matched to an inventory item exactly as
a card purchase is (`bank.match_purchases`), and after the owner says yes it becomes the item's
"paid" price, with the receipt kept as the proof. A line for something not on the list yet can
be added as a suggestion.

The reader only reads: it reports what the document says. It cannot add an item to the
inventory or change a price by itself; a line with no amount, or a document that is not a
receipt, yields nothing.
"""
from __future__ import annotations

import base64
import io
import json
import re

from . import config

CAPTION = re.compile(r"\b(receipts?|invoices?|order (confirmation|page|history|summary)|orders?|proof of purchase|bill of sale)\b", re.I)

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_receipt": {"type": "boolean",
                       "description": "True for a receipt, invoice, order confirmation or order history page"},
        "merchant": {"type": ["string", "null"], "description": "The store or seller, e.g. IKEA, Amazon"},
        "date": {"type": ["string", "null"], "description": "Purchase date as YYYY-MM-DD, or null if not shown"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "The thing bought, as written, tidied into plain words"},
                    "quantity": {"type": "integer"},
                    "amount": {"type": "number", "description": "What was paid for this line in total, in dollars"},
                },
                "required": ["description", "quantity", "amount"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["is_receipt", "merchant", "date", "lines"],
    "additionalProperties": False,
}

PROMPT = """This is a document a family sent while rebuilding the list of what was in their home for an
insurance claim. If it is a receipt, an invoice, an order confirmation or an order history page,
read it.

Give the store or seller, the purchase date (YYYY-MM-DD; null if it is not shown), and one line
for each thing bought: what it is in plain words (expand abbreviations you are sure of, keep brand
and model), how many, and the total paid for that line in dollars. On an order history page, each
order is its own line, and use the date of the most recent order as the date.

Leave out lines that are not things: tax, shipping, tips, discounts, fees, gift cards, subtotals
and totals. Leave out groceries, meals, fuel and other things that are used up; the list is for
belongings. Never guess an amount that is not printed: leave the line out instead.

If the document is not a receipt or order of any kind, set is_receipt to false and give no lines."""


class ReceiptError(RuntimeError):
    pass


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def wanted(text: str, attachments: list) -> bool:
    """True when a message's attachments should be read as receipts: it says so, or one is a PDF."""
    return bool(attachments) and (bool(CAPTION.search(text or "")) or any(is_pdf(data) for _, data in attachments))


class ClaudeReceiptReader:
    def __init__(self, model: str | None = None, client=None):
        from .detect_claude import DEFAULT_MODEL
        config.load_env()
        self.model = model or config.env("ASHLINE_CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = config.env("ASHLINE_CLAUDE_EFFORT", "medium")
        self._client = client

    def read(self, data: bytes) -> dict:
        """{is_receipt, merchant, date, lines: [{description, quantity, amount}]} for an image or a PDF."""
        import anthropic

        from .detect_claude import FALLBACK_BETA, FALLBACK_MODELS
        if is_pdf(data):
            document = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                       "data": base64.standard_b64encode(data).decode()}}
        else:
            from PIL import Image
            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                raise ReceiptError("That file could not be opened as a picture or a PDF.") from exc
            image.thumbnail((2000, 2000))                   # receipts are small print: keep them large
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=88)
            document = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(buf.getvalue()).decode()}}
        call = dict(model=self.model, max_tokens=16000,
                    output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": RECEIPT_SCHEMA}},
                    messages=[{"role": "user", "content": [document, {"type": "text", "text": PROMPT}]}])
        client = self._client or anthropic.Anthropic()
        try:
            if self.model.startswith(FALLBACK_MODELS):
                response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **call)
            else:
                response = client.messages.create(**call)
        except anthropic.APIError as exc:
            raise ReceiptError(f"The receipt could not be read just now: {exc}") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ReceiptError("The receipt could not be read.")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            return clean(json.loads(text))
        except ValueError as exc:
            raise ReceiptError("The receipt came back unreadable.") from exc


def clean(reading: dict) -> dict:
    """Only lines with a description and a positive amount; a date only in YYYY-MM-DD form."""
    if not isinstance(reading, dict) or not reading.get("is_receipt"):
        return {"is_receipt": False, "merchant": None, "date": None, "lines": []}
    lines = []
    for line in reading.get("lines") or []:
        if not isinstance(line, dict):
            continue
        description = str(line.get("description") or "").strip()
        try:
            amount, quantity = float(line.get("amount")), max(1, int(line.get("quantity") or 1))
        except (TypeError, ValueError):
            continue
        if description and amount > 0:
            lines.append({"description": description[:120], "quantity": quantity, "amount": round(amount, 2)})
    date = reading.get("date") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(reading.get("date") or "")) else None
    merchant = str(reading.get("merchant") or "").strip()[:60] or None
    return {"is_receipt": True, "merchant": merchant, "date": date, "lines": lines[:40]}


def as_purchases(reading: dict, receipt_id: str) -> list[dict]:
    """A receipt's lines in the shape card purchases have, so the same matching and applying work."""
    return [{"id": f"{receipt_id}:{n}", "date": reading.get("date"), "amount": line["amount"],
             "description": (f"{line['quantity']} x " if line["quantity"] > 1 else "") + line["description"],
             "merchant": reading.get("merchant") or "", "receipt": receipt_id, "quantity": line["quantity"]}
            for n, line in enumerate(reading.get("lines") or [], 1)]


def get_reader():
    """A reader when an Anthropic key is set, else None."""
    return ClaudeReceiptReader() if config.env("ANTHROPIC_API_KEY") else None
