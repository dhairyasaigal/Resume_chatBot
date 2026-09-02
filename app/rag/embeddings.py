"""
Embeddings — initializes the local sentence-transformers model once and reuses it.
"""

from functools import lru_cache
from app.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns a cached HuggingFaceEmbeddings instance.
    The model is downloaded on first call and cached for subsequent calls.
    """
    settings = get_settings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
