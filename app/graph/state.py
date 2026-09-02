"""
LangGraph state definition for the interview agent.
Extensible for future security, evaluation, and caching nodes.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InterviewState(TypedDict):
    # Conversation history — managed by LangGraph's add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Latest user query (extracted for retrieval)
    query: str

    # Guardrail evaluation
    is_blocked: bool
    guardrail_reason: str | None

    # Retrieved documents from Qdrant
    retrieved_docs: list[dict]

    # Formatted context string passed to the LLM
    context: str

    # Final generated answer
    answer: str

    # Source filenames used in this turn
    sources: list[str]

