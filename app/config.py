import os
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Groq
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-8b-instant", validation_alias="GROQ_MODEL")

    # Qdrant
    qdrant_url: str = Field("", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field("", validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("resume_knowledge", validation_alias="QDRANT_COLLECTION")
    qdrant_local_path: str = Field("./qdrant_storage", validation_alias="QDRANT_LOCAL_PATH")

    # PostgreSQL
    postgres_url: str = Field(..., validation_alias="POSTGRES_URL")

    # Embeddings
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL"
    )

    # RAG
    top_k: int = Field(12, validation_alias="TOP_K")
    chunk_size: int = Field(1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(150, validation_alias="CHUNK_OVERLAP")

    # Redis cache (optional)
    redis_url: str = Field("", validation_alias="REDIS_URL")

    # LangSmith (optional)
    langsmith_tracing: bool = Field(False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field("", validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("resume-interview-agent", validation_alias="LANGSMITH_PROJECT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache()
def get_settings() -> Settings:
    # On Streamlit Cloud, inject st.secrets into os.environ first
    try:
        import streamlit as st
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass

    settings = Settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return settings
