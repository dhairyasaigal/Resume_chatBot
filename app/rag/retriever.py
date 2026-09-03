"""
Retriever — queries Qdrant and returns relevant documents with source info.
Uses hybrid approach: keyword-based source filtering + semantic similarity.
"""

from langchain_core.documents import Document
from app.config import get_settings
from app.rag.vectorstore import get_vector_store

# Maps query keywords to source files for direct retrieval
KEYWORD_SOURCE_MAP = {
    "intern": "internships.md",
    "internship": "internships.md",
    "hmsi": "internships.md",
    "honda": "internships.md",
    "ss medi": "internships.md",
    "project": "projects.md",
    "projects": "projects.md",
    "built": "projects.md",
    "learnflow": "projects.md",
    "rag system": "projects.md",
    "onboarding": "projects.md",
    "capex": "projects.md",
    "mail router": "projects.md",
    "skill": "skills.md",
    "education": "education.md",
    "degree": "education.md",
    "college": "education.md",
    "university": "education.md",
    "research": "research.md",
    "publication": "research.md",
    "achievement": "achievements.md",
    "award": "achievements.md",
    "hackathon": "achievements.md",
    "certification": "achievements.md",
}


def _detect_target_source(query: str) -> str | None:
    """Return a source filename if the query strongly targets a specific file."""
    q = query.lower()
    for keyword, source in KEYWORD_SOURCE_MAP.items():
        if keyword in q:
            return source
    return None


def retrieve(query: str) -> list[Document]:
    """
    Hybrid retrieval: if query targets a specific source, fetch a larger candidate pool,
    bubble matching chunks to the top, then fill remaining slots with other results.
    """
    settings = get_settings()
    vector_store = get_vector_store()
    target_source = _detect_target_source(query)

    # Fetch a larger pool so targeted chunks have a chance to appear
    fetch_k = max(settings.top_k * 3, 30)
    all_docs = vector_store.similarity_search(query, k=fetch_k)

    if target_source:
        targeted = [d for d in all_docs if d.metadata.get("source") == target_source]
        others = [d for d in all_docs if d.metadata.get("source") != target_source]
        return (targeted + others)[: settings.top_k]

    return all_docs[: settings.top_k]


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_sources(docs: list[Document]) -> list[str]:
    """Return deduplicated list of source filenames."""
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources
