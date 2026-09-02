"""
Retrieval tests — verifies that Qdrant returns semantically relevant documents.

Run after ingestion:
    python -m pytest tests/test_retrieval.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pytest
from app.rag.retriever import retrieve, get_sources


def sources_contain(sources: list[str], keyword: str) -> bool:
    return any(keyword.lower() in s.lower() for s in sources)


def test_rakshak_retrieval():
    """Query about Rakshak contribution should return rakshak.md."""
    docs = retrieve("What was Dhairya's contribution to Rakshak?")
    sources = get_sources(docs)
    assert sources_contain(sources, "rakshak"), (
        f"Expected rakshak.md in sources, got: {sources}"
    )


def test_internship_retrieval():
    """Query about Honda internship should return internships.md."""
    docs = retrieve("What did Dhairya do at Honda?")
    sources = get_sources(docs)
    assert sources_contain(sources, "internship"), (
        f"Expected internships.md in sources, got: {sources}"
    )


def test_education_retrieval():
    """Query about education should return education.md or resume.md."""
    docs = retrieve("What is Dhairya studying?")
    sources = get_sources(docs)
    assert sources_contain(sources, "education") or sources_contain(sources, "resume"), (
        f"Expected education.md or resume.md in sources, got: {sources}"
    )


def test_tensorflow_lite_retrieval():
    """Query about TensorFlow Lite should return rakshak.md."""
    docs = retrieve("Why was TensorFlow Lite used?")
    sources = get_sources(docs)
    assert sources_contain(sources, "rakshak"), (
        f"Expected rakshak.md in sources, got: {sources}"
    )


def test_skills_retrieval():
    """Query about programming languages should return skills.md."""
    docs = retrieve("What programming languages does Dhairya know?")
    sources = get_sources(docs)
    assert sources_contain(sources, "skill"), (
        f"Expected skills.md in sources, got: {sources}"
    )


def test_returns_k_docs():
    """Should return at most TOP_K documents."""
    from app.config import get_settings
    docs = retrieve("Tell me about Dhairya")
    assert len(docs) <= get_settings().top_k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
