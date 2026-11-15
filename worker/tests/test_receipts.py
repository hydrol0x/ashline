"""A receipt or order page sent in the chat: read, matched to items, kept as proof of what was paid."""
import io
import json
from types import SimpleNamespace

from PIL import Image

from ashline import receipts
from ashline.meter import meter
from test_chat import MAYA, MOM, Family, photo, texts

IKEA = {"is_receipt": True, "merchant": "IKEA", "date": "2024-03-12", "lines": [
    {"description": "Blue velvet 3-seat sofa", "quantity": 1, "amount": 1149.0},
    {"description": "Walnut bookcase", "quantity": 2, "amount": 398.0},
    {"description": "Paper napkins", "quantity": 1, "amount": 3.99}]}


class Reader:
    def __init__(self, reading=IKEA):
        self.reading, self.read_bytes = reading, []

    def read(self, data):
        self.read_bytes.append(len(data))
        return receipts.clean(self.reading)


def picture():
    buf = io.BytesIO()
    Image.new("RGB", (600, 900), (250, 250, 245)).save(buf, format="JPEG")
    return ("IMG_9001.jpg", buf.getvalue())


def family(reader=None):
    f = Family().setup()
    f.engine.receipt_reader = reader or Reader()
    f.say(MOM, photos=[photo(640)])                       # sofa, lamp, coffee table
    f.wait()
    return f


def test_a_message_says_when_its_attachment_is_a_receipt():
    files = [picture()]
    assert receipts.wanted("here's the receipt for the couch", files)
    assert receipts.wanted("Amazon order confirmation", files) and receipts.wanted("invoice", files)
    assert not receipts.wanted("thanksgiving 2019", files) and not receipts.wanted("receipt", [])
    assert receipts.wanted("", [("scan.pdf", b"%PDF-1.7 ...")])                 # a PDF is never a photo of the home


def test_only_things_with_a_price_are_kept_from_a_reading():
    reading = receipts.clean({"is_receipt": True, "merchant": " IKEA ", "date": "March 2024", "lines": [
        {"description": "Sofa", "quantity": 1, "amount": 1149}, {"description": "", "quantity": 1, "amount": 5},
        {"description": "Lamp", "quantity": 0, "amount": "abc"}, {"description": "Rug", "quantity": 2, "amount": -3}]})
    assert reading == {"is_receipt": True, "merchant": "IKEA", "date": None,
                       "lines": [{"description": "Sofa", "quantity": 1, "amount": 1149.0}]}
    assert receipts.clean({"is_receipt": False, "lines": [{"description": "Sofa", "quantity": 1, "amount": 9}]})["lines"] == []
    bought = receipts.as_purchases(receipts.clean(IKEA), "rcpt-1")
    assert bought[1] == {"id": "rcpt-1:2", "date": "2024-03-12", "amount": 398.0, "description": "2 x Walnut bookcase",
                         "merchant": "IKEA", "receipt": "rcpt-1", "quantity": 2}


def test_a_receipt_prices_the_item_it_names_and_offers_the_rest(tmp_path):
    f = family()
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    estimate = sofa["price_mid"]
    out = texts(f.say(MOM, "receipt from when we bought the couch", photos=[picture()]))
    assert len(out) == 1 and out[0].startswith("I read the receipt from IKEA: 3 things bought.")
    assert "This matches things on your list:\n1. Blue velvet sofa: $1,149 at IKEA on 2024-03-12" in out[0]
    assert "Also on it, not on your list yet: 2 x Walnut bookcase ($398). Reply add to list it." in out[0]
    assert "napkins" not in out[0]                                              # too small to be a belonging worth listing
    assert f.store.get_item(f.case_id, sofa["item_id"])["price_mid"] == estimate    # nothing changes before a yes
    assert not f.store.get_photos(f.case_id, source="photo")[1:]                # the receipt is not a photo of the home

    out = texts(f.say(MOM, "yes"))
    assert out[0].startswith("Done. 1 item now carries the price you paid, marked as a receipt.")
    assert out[0].endswith("Reply add to list the other things on the receipt.")
    after = f.store.get_item(f.case_id, sofa["item_id"])
    assert (after["price_mid"], after["price_source"]) == (1149.0, "receipt: IKEA")
    kept = json.loads(after["price_ref"])
    assert kept["paid"]["receipt"] == "rcpt-1" and kept["estimate"]["mid"] == estimate

    out = texts(f.say(MOM, "add"))
    assert out[0].startswith("Added 1 thing from the receipt as a suggestion, priced at what you paid.")
    bookcase = next(i for i in f.store.get_items(f.case_id) if "bookcase" in i["label"])
    assert (bookcase["source"], bookcase["status"], bookcase["qty"], bookcase["price_mid"]) == ("receipt", "suggested", 2, 199.0)
    assert bookcase["price_source"] == "receipt: IKEA" and bookcase["evidence_date"] == "2024-03-12"

    assert texts(f.say(MOM, "undo")) == ["Undone: I've taken those off the list."]
    assert f.store.get_item(f.case_id, bookcase["item_id"])["status"] == "rejected"
    assert texts(f.say(MOM, "undo")) == ["Undone: those prices are back to the estimates."]
    assert f.store.get_item(f.case_id, sofa["item_id"])["price_mid"] == estimate


def test_the_item_sheet_and_the_export_name_the_receipt():
    from fastapi.testclient import TestClient

    from ashline.export import export_rows
    from ashline.web import create_app
    f = family()
    f.say(MOM, "receipt", photos=[picture()])
    f.say(MOM, "yes")
    sofa = next(i for i in f.store.get_items(f.case_id) if "sofa" in i["label"])
    token = f.store.get_case(f.case_id)["owner_token"]
    detail = TestClient(create_app(lambda: f.store)).get(f"/p/{token}/item/{sofa['item_id']}.json").json()
    assert detail["price_basis"] == "What the family paid, from a receipt or order record they sent."
    assert (detail["price_sources"][0]["kind"], detail["price_sources"][0]["detail"]) == ("paid", "Receipt from IKEA, 2024-03-12")
    assert detail["badges"]["receipt"] == 1 and detail["price_from"] == "paid"
    assert [l["url"] for l in detail["shop_search"]] == [
        "https://www.amazon.com/s?k=blue+velvet+sofa", "https://www.google.com/search?tbm=shop&q=blue+velvet+sofa"]
    f.store.set_item_status(f.case_id, sofa["item_id"], "confirmed")
    row = next(r for r in export_rows(f.case_id, f.store) if "sofa" in r["item"])
    assert row["evidence"].endswith("receipt from IKEA dated 2024-03-12 ($1,149.00)") and row["basis"].startswith("Paid")
    assert meter(f.case_id, f.store)["backed"] == 1149


def test_what_is_not_a_receipt_and_who_may_send_one():
    f = family(Reader({"is_receipt": False, "merchant": None, "date": None, "lines": []}))
    out = texts(f.say(MOM, "receipt", photos=[picture()]))
    assert out == ["I couldn't read that as a receipt or an order. If it is a photo of the home, send it again without "
                   "the word receipt."]
    out = texts(f.say(MAYA, "receipt", photos=[picture()]))
    assert out[0].startswith("Thank you. Only Mom can add receipts")
    f.engine.receipt_reader = None                                  # no key in the test environment: no reader
    assert texts(f.say(MOM, "receipt", photos=[picture()]))[0].startswith("I can't read receipts yet on this setup.")


def test_the_reader_sends_a_picture_or_a_pdf_and_reads_the_reply():
    calls = []
    reply = json.dumps({**IKEA, "lines": IKEA["lines"] + [{"description": "Total", "quantity": 1, "amount": 0}]})

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=reply)])

    client = SimpleNamespace(messages=Messages(), beta=SimpleNamespace(messages=Messages()))
    reader = receipts.ClaudeReceiptReader(client=client)
    reading = reader.read(picture()[1])
    assert reading["merchant"] == "IKEA" and len(reading["lines"]) == 3            # the zero-amount line is dropped
    document, text = calls[0]["messages"][0]["content"]
    assert document["type"] == "image" and "Never guess an amount" in text["text"]
    assert calls[0]["output_config"]["format"]["schema"] == receipts.RECEIPT_SCHEMA
    reader.read(b"%PDF-1.7 fake")
    assert calls[1]["messages"][0]["content"][0] == {"type": "document", "source": {
        "type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjcgZmFrZQ=="}}
