"""
Qdrant vector store — handles collection creation and document upload.
Supports both local Qdrant and Qdrant Cloud via environment variables.
"""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings
from app.rag.embeddings import get_embeddings


def get_qdrant_client() -> QdrantClient:
    """
    Returns a QdrantClient configured for local or cloud based on settings.
    """
    settings = get_settings()

    if settings.qdrant_url:
        # Qdrant Cloud
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    else:
        # Local Qdrant (file-based persistence)
        return QdrantClient(path=settings.qdrant_local_path)


def get_vector_store() -> QdrantVectorStore:
    """
    Returns a LangChain QdrantVectorStore for similarity search.
    """
    settings = get_settings()
    client = get_qdrant_client()
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int):
    """Create the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")


def ingest_documents(documents: list[Document]) -> QdrantVectorStore:
    """
    Embed and upload documents to Qdrant.
    Creates the collection if it doesn't exist.
    Returns the vector store.
    """
    settings = get_settings()
    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Determine vector size from a test embedding
    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)

    create_collection_if_not_exists(client, settings.qdrant_collection, vector_size)

    # Delete existing vectors to avoid duplicates on re-ingestion
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        count = client.count(settings.qdrant_collection).count
        if count > 0:
            print(f"Clearing {count} existing vectors before re-ingestion...")
            client.delete_collection(settings.qdrant_collection)
            create_collection_if_not_exists(client, settings.qdrant_collection, vector_size)

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=settings.qdrant_url or None,
        api_key=settings.qdrant_api_key or None,
        path=settings.qdrant_local_path if not settings.qdrant_url else None,
        collection_name=settings.qdrant_collection,
    )

    return vector_store
