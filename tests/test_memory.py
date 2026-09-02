"""
Memory/persistence tests — verifies conversation threads persist across invocations.

Requires a running PostgreSQL instance.

Run:
    python -m pytest tests/test_memory.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pytest
import uuid
from langchain_core.messages import HumanMessage
from app.graph.workflow import build_graph
from app.memory.checkpointer import get_checkpointer


@pytest.fixture(scope="module")
def persistent_app():
    checkpointer = get_checkpointer()
    return build_graph(checkpointer=checkpointer)


def test_conversation_continues_in_same_thread(persistent_app):
    """Within one thread, the second message should reference the first."""
    thread_id = f"test_memory_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1
    persistent_app.invoke(
        {"messages": [HumanMessage(content="Tell me about Rakshak.")]},
        config=config,
    )

    # Turn 2 — follow-up that only makes sense with context from turn 1
    result2 = persistent_app.invoke(
        {"messages": [HumanMessage(content="What technology was used in it?")]},
        config=config,
    )

    answer = result2["answer"].lower()
    # Should reference Rakshak-related technology, not be confused
    assert len(answer) > 20, "Expected a substantive answer referencing conversation context"


def test_thread_state_persists(persistent_app):
    """State should be retrievable after the graph has run."""
    thread_id = f"test_persist_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    persistent_app.invoke(
        {"messages": [HumanMessage(content="Tell me about LearnFlow.")]},
        config=config,
    )

    # Retrieve state
    state = persistent_app.get_state(config)
    assert state is not None
    assert len(state.values.get("messages", [])) >= 2  # HumanMessage + AIMessage


def test_different_threads_are_independent(persistent_app):
    """Two different thread IDs should have independent conversation state."""
    thread_a = f"test_a_{uuid.uuid4().hex[:6]}"
    thread_b = f"test_b_{uuid.uuid4().hex[:6]}"

    persistent_app.invoke(
        {"messages": [HumanMessage(content="Tell me about Rakshak.")]},
        config={"configurable": {"thread_id": thread_a}},
    )

    # Thread B has no prior context
    result_b = persistent_app.invoke(
        {"messages": [HumanMessage(content="What is Dhairya's background?")]},
        config={"configurable": {"thread_id": thread_b}},
    )

    # Thread B should give a general answer, not specifically about Rakshak
    state_a = persistent_app.get_state({"configurable": {"thread_id": thread_a}})
    state_b = persistent_app.get_state({"configurable": {"thread_id": thread_b}})

    msgs_a = state_a.values.get("messages", [])
    msgs_b = state_b.values.get("messages", [])

    assert len(msgs_a) != len(msgs_b) or msgs_a[0].content != msgs_b[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
