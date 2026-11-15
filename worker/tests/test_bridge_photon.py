"""The Photon bridge against a stand-in client. This checks the bridge's own logic; it has not
been run against a live Photon project."""
import io
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("photon_api")

from PIL import Image

from ashline.bridge_photon import PhotonBridge, parts_of
from ashline.chat import Engine
from ashline.pricing import DefaultPricer
from ashline.store import Store


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def received(mid, text=None, attachment=None, sender="+15550001", space="space-mom", kind="direct", at="2026-11-14T15:00:00.000Z"):
    parts = ([NS(type="text", format="plain", text=text)] if text else []) + \
            ([NS(type="attachment", attachmentId=attachment)] if attachment else [])
    content = parts[0] if len(parts) == 1 else NS(type="group", parts=[NS(root=p) for p in parts])
    return NS(root=NS(root=NS(id=mid, status="received", createdAt=at, content=content,
                              sentBy=NS(root=NS(kind="human", handle=sender, id="user-" + sender)),
                              receivedBy=NS(root=NS(kind="agent", handle="+15559990000", id="agent-1")),
                              space=NS(root=NS(id=space, kind=kind)))))


class FakeProjects:
    def __init__(self):
        self.inbox, self.sent, self.uploads = [], [], []

    def list_messages(self, request):
        items, self.inbox = self.inbox, []
        return NS(items=items, nextPageToken=None)

    def send_message(self, request):
        self.sent.append(request.body)

    def download_attachment(self, request):
        return jpeg()

    def upload_attachment(self, request):
        self.uploads.append(request.headers)
        return NS(id=f"att-{len(self.uploads)}")


class Sees:
    def detect(self, path, room_hint=None, rooms=None):
        return [{"label": "blue velvet sofa", "category": "furniture", "confidence": 0.8, "bbox": [50, 200, 400, 420]}]


@pytest.fixture
def bridge():
    projects = FakeProjects()
    engine = Engine(Store(), detector=Sees(), pricer=DefaultPricer(), batch_seconds=0)
    return PhotonBridge(photon=NS(projects=projects), engine=engine, project="pho_prj_test"), projects


def sent_text(projects):
    return [unwrapped.text for unwrapped in (getattr(b.content, "root", b.content) for b in projects.sent)
            if getattr(unwrapped, "type", None) == "text"]


def test_text_round_trip_and_dedup(bridge):
    b, projects = bridge
    projects.inbox = [received("m1", "hi")]
    assert b.poll() == 1
    assert sent_text(projects)[0].startswith("I'm Ashline.")
    body = projects.sent[0]
    assert (body.spaceId, body.senderId) == ("space-mom", "agent-1")           # replies as the agent, in that chat
    projects.inbox = [received("m1", "hi")]                                     # the same message listed again
    assert b.poll() == 0 and len(projects.sent) == 1
    assert Store().get_member("+15550001")["dm_chat"] == "space-mom"


def test_photo_in_reply_with_annotated_image(bridge):
    b, projects = bridge
    for n, text in enumerate(["hi", "Example home", "skip", "living room"]):
        projects.inbox = [received(f"m{n}", text)]
        b.poll()
    projects.sent.clear()
    projects.inbox = [received("m9", "from thanksgiving", attachment="att-in-1")]
    assert b.poll() == 1 and sent_text(projects) == ["Got it, looking at it."]
    b.send(b.engine.tick())
    kinds = [getattr(getattr(body.content, "root", body.content), "type", None) for body in projects.sent]
    assert kinds == ["text", "attachment", "text"]                              # ack, annotated photo, legend
    assert projects.uploads[0].content_type.startswith("multipart/related; boundary=") and "1. Blue velvet sofa" in sent_text(projects)[-1]


def test_group_chat_agent_events_and_unknown_spaces(bridge):
    b, projects = bridge
    projects.inbox = [received("g1", "I'll be late", sender="+15550002", space="space-group", kind="group")]
    assert b.poll() == 1 and projects.sent == []                                # not Ashline's group: silence
    own = received("a1", "hello")
    own.root.root.sentBy.root.kind = "agent"
    typing = NS(root=NS(root=NS(id="t1", status="received", createdAt="2026-11-14T15:00:01.000Z",
                                content=NS(type="space_typing_update"), sentBy=NS(root=NS(kind="human", handle="+1", id="u")),
                                receivedBy=NS(root=NS(kind="agent", handle="+2", id="agent-1")),
                                space=NS(root=NS(id="space-x", kind="direct")))))
    projects.inbox = [own, typing]
    assert b.poll() == 0
    from ashline.chat import Outbound
    b.send([Outbound("space-never-seen", "hello")])                             # nothing to address it with
    assert projects.sent == []
    assert parts_of(NS(type="text", text="x"))[0].text == "x"
