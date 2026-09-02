"""
Groq LLM — initializes ChatGroq once and reuses it.
Model is fully configurable through environment variables.
"""

from functools import lru_cache
from langchain_groq import ChatGroq
from app.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Returns a cached ChatGroq instance.
    Model name and API key are loaded from environment variables.
    """
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.2,
    )
