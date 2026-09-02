import os
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Groq
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-8b-instant", env="GROQ_MODEL")

    # Qdrant
    qdrant_url: str = Field("", env="QDRANT_URL")
    qdrant_api_key: str = Field("", env="QDRANT_API_KEY")
    qdrant_collection: str = Field("resume_knowledge", env="QDRANT_COLLECTION")
    qdrant_local_path: str = Field("./qdrant_storage", env="QDRANT_LOCAL_PATH")

    # PostgreSQL
    postgres_url: str = Field(..., env="POSTGRES_URL")

    # Embeddings
    embedding_model: str = Field(
        "sentence-transformers/all-mpnet-base-v2", env="EMBEDDING_MODEL"
    )

    # RAG
    top_k: int = Field(12, env="TOP_K")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(150, env="CHUNK_OVERLAP")

    # Redis cache (optional — leave empty to disable)
    redis_url: str = Field("", env="REDIS_URL")

    # Observability & Traceability (LangSmith)
    langsmith_tracing: bool = Field(False, env="LANGSMITH_TRACING")
    langsmith_api_key: str = Field("", env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("resume-interview-agent", env="LANGSMITH_PROJECT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # If LangSmith is configured, inject standard LangChain tracing environment variables
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return settings
