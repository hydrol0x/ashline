"""The deepagents wiring, driven by a scripted model: no LLM call is made."""
import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ashline import agent as agent_mod
from ashline.store import Store


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_orchestrator_delegates_outside_work_to_a_subagent(monkeypatch):
    seen = {}

    def fake_outside(case_id, lat, lon, loss_date=None):
        seen.update(case_id=case_id, lat=lat, lon=lon, loss_date=loss_date)
        return {"covered": True, "capture_date": "2024-08-13", "outbuildings": 1, "yard_findings": ["trampoline"]}

    monkeypatch.setattr(agent_mod.seed, "outside", fake_outside)
    script = iter([
        call("task", {"subagent_type": "outside",
                      "description": "case_id fam1, lat 34.05, lon -118.4, loss date 2025-01-07"}, "t1"),
        call("seed_outside", {"case_id": "fam1", "lat": 34.05, "lon": -118.4, "loss_date": "2025-01-07"}, "t2"),
        AIMessage(content="Aerial from August 2024: one outbuilding, a trampoline."),       # subagent summary
        AIMessage(content="I can see a second building on the lot. Was that a garage?"),    # orchestrator
    ])
    graph = agent_mod.build_agent(model=Scripted(messages=script))
    out = graph.invoke({"messages": [HumanMessage("hi, our home was at 12 Example Ave")]})

    assert seen == {"case_id": "fam1", "lat": 34.05, "lon": -118.4, "loss_date": "2025-01-07"}
    tool_replies = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "one outbuilding" in tool_replies[0].content        # only the short summary comes back
    assert out["messages"][-1].content.startswith("I can see a second building")


def test_confirming_an_item_waits_for_the_family():
    store = Store()
    store.upsert_case("fam1")
    item_id = store.add_item("fam1", {"label": "floor lamp"}, source="scan")
    script = iter([call("confirm_item", {"case_id": "fam1", "item_id": item_id}, "c1"),
                   AIMessage(content="Added the floor lamp.")])
    graph = agent_mod.build_agent(model=Scripted(messages=script), checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "fam1"}}

    paused = graph.invoke({"messages": [HumanMessage("yes the lamp is ours")]}, thread)
    assert "__interrupt__" in paused                                           # stopped before the tool ran
    assert Store().get_items("fam1")[0]["status"] == "suggested"

    graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), thread)
    assert Store().get_items("fam1")[0]["status"] == "confirmed"


def test_subagents_and_gates_are_declared():
    assert [s["name"] for s in agent_mod.SUBAGENTS] == ["inside", "outside"]
    assert set(agent_mod.GATED_TOOLS) == {"confirm_item", "reject_item", "answer_finding"}
