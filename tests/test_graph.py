"""
Graph tests — verifies LangGraph nodes and workflow execute correctly.

Run:
    python -m pytest tests/test_graph.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pytest
from langchain_core.messages import HumanMessage
from app.graph.workflow import build_graph


@pytest.fixture(scope="module")
def app():
    # Run without checkpointer for unit tests
    return build_graph(checkpointer=None)


def test_basic_question(app):
    """The graph should return an answer for a basic question."""
    result = app.invoke(
        {"messages": [HumanMessage(content="Tell me about Rakshak.")]},
        config={"configurable": {"thread_id": "test_001"}},
    )
    assert "answer" in result
    assert len(result["answer"]) > 10


def test_unknown_question_no_hallucination(app):
    """Unknown questions like salary should not return fabricated information."""
    result = app.invoke(
        {"messages": [HumanMessage(content="What is Dhairya's salary?")]},
        config={"configurable": {"thread_id": "test_002"}},
    )
    answer = result["answer"].lower()
    # Should say it doesn't know, not invent a number
    assert any(phrase in answer for phrase in [
        "don't have", "not available", "not provided", "no information",
        "cannot", "i don't", "unavailable", "not in"
    ]), f"Expected 'don't know' style response, got: {result['answer']}"


def test_sources_populated(app):
    """Retrieved docs should populate sources."""
    result = app.invoke(
        {"messages": [HumanMessage(content="What is Rakshak?")]},
        config={"configurable": {"thread_id": "test_003"}},
    )
    assert "sources" in result
    assert isinstance(result["sources"], list)


def test_tensorflow_lite_answer(app):
    """Technical question about TensorFlow Lite should give a grounded answer."""
    result = app.invoke(
        {"messages": [HumanMessage(content="Why was TensorFlow Lite used in Rakshak?")]},
        config={"configurable": {"thread_id": "test_004"}},
    )
    answer = result["answer"].lower()
    assert "tensorflow" in answer or "lite" in answer or "edge" in answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
