"""
LangGraph nodes — input_guardrail, retrieve_context, generate_answer, and safe_refusal.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.graph.state import InterviewState
from app.rag.retriever import retrieve, format_context, get_sources
from app.llm.groq_model import get_llm
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.guardrails.input_guardrail import validate_input_guardrail
from app.guardrails.output_guardrail import validate_output_guardrail
from app.cache.redis_cache import (
    get_cached_retrieval, set_cached_retrieval,
    get_cached_answer, set_cached_answer,
)


def input_guardrail_node(state: InterviewState) -> dict:
    """
    Node 0: Check the incoming query against safety and prompt injection guardrails.
    """
    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {
            "query": "",
            "is_blocked": False,
            "guardrail_reason": None,
        }

    is_safe, reason = validate_input_guardrail(query)
    return {
        "query": query,
        "is_blocked": not is_safe,
        "guardrail_reason": reason,
    }


def safe_refusal_node(state: InterviewState) -> dict:
    """
    Refusal Node: Returns a safe refusal message when a query is flagged by guardrails.
    """
    reason = state.get(
        "guardrail_reason",
        "Security Notice: This request could not be processed due to safety guidelines.",
    )
    return {
        "messages": [AIMessage(content=reason)],
        "answer": reason,
        "sources": [],
    }


def retrieve_context(state: InterviewState) -> dict:
    """
    Node 1: Query Qdrant with the user query, build grounded context string.
    Checks Redis cache first — skips Qdrant on cache hit.
    """
    query = state.get("query", "")
    if not query:
        return {"retrieved_docs": [], "context": "", "sources": []}

    # Check cache
    cached = get_cached_retrieval(query)
    if cached:
        return {
            "retrieved_docs": cached["docs"],
            "context": cached["context"],
            "sources": cached["sources"],
        }

    # Cache miss — hit Qdrant
    docs = retrieve(query)
    context = format_context(docs)
    sources = get_sources(docs)
    serialized_docs = [
        {"content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]

    # Store in cache
    set_cached_retrieval(query, context, sources, serialized_docs)

    return {
        "retrieved_docs": serialized_docs,
        "context": context,
        "sources": sources,
    }


def generate_answer(state: InterviewState) -> dict:
    """
    Node 2: Build the prompt with history + context, call Groq, validate with output guardrail.
    Checks Redis cache for identical (query, context) pairs before calling the LLM.
    """
    llm = get_llm()
    context = state.get("context", "")
    query = state.get("query", "")
    messages = state["messages"]

    # Check LLM cache (only for single-turn queries, not multi-turn conversations)
    if query and len(messages) <= 2:
        cached_answer = get_cached_answer(query, context)
        if cached_answer:
            return {
                "messages": [AIMessage(content=cached_answer)],
                "answer": cached_answer,
            }

    # Build the system message with retrieved context injected
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n## Retrieved Knowledge Base Context\n\n{context}"
    else:
        system_content += "\n\n## Retrieved Knowledge Base Context\n\nNo relevant context was retrieved for this query."

    prompt_messages = [SystemMessage(content=system_content)] + list(messages)
    response = llm.invoke(prompt_messages)
    raw_answer = response.content

    # Apply output guardrail sanitization
    safe_answer = validate_output_guardrail(raw_answer)

    # Store in LLM cache
    if query:
        set_cached_answer(query, context, safe_answer)

    return {
        "messages": [AIMessage(content=safe_answer)],
        "answer": safe_answer,
    }

