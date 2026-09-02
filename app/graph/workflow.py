"""
LangGraph workflow — wires together input_guardrail → retrieve_context → generate_answer.
Supports PostgreSQL checkpointing for persistent conversation threads and guardrail refusals.
"""

from langgraph.graph import StateGraph, START, END
from app.graph.state import InterviewState
from app.graph.nodes import (
    input_guardrail_node,
    safe_refusal_node,
    retrieve_context,
    generate_answer,
)


def route_after_guardrail(state: InterviewState) -> str:
    """Route to safe_refusal if blocked by guardrails, otherwise proceed to retrieval."""
    if state.get("is_blocked", False):
        return "safe_refusal"
    return "retrieve_context"


def build_graph(checkpointer=None):
    """
    Build and compile the interview agent graph with security guardrails.
    
    Flow:
        START -> input_guardrail -> (conditional)
                                  ├──> [safe] -> retrieve_context -> generate_answer -> END
                                  └──> [blocked] -> safe_refusal -> END
    """
    builder = StateGraph(InterviewState)

    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("safe_refusal", safe_refusal_node)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {
            "retrieve_context": "retrieve_context",
            "safe_refusal": "safe_refusal",
        },
    )
    builder.add_edge("retrieve_context", "generate_answer")
    builder.add_edge("generate_answer", END)
    builder.add_edge("safe_refusal", END)

    return builder.compile(checkpointer=checkpointer)

