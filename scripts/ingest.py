"""
Ingestion pipeline — loads, chunks, embeds, and uploads knowledge base to Qdrant.

Usage:
    python scripts/ingest.py
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.vectorstore import ingest_documents
from app.config import get_settings
from app.rag.chunker import chunk_documents


def main():
    settings = get_settings()

    print("\n=== Resume Knowledge Base Ingestion ===\n")

    # Step 1: Load documents
    print("Loading documents...")
    try:
        documents = load_documents("data")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure you are running this script from the project root directory.")
        sys.exit(1)

    if not documents:
        print("ERROR: No documents found in data/. Add .md files and retry.")
        sys.exit(1)

    print(f"Documents loaded: {len(documents)}")
    for doc in documents:
        print(f"  - {doc.metadata['source']} [{doc.metadata['category']}]")

    # Step 2: Chunk documents
    print("\nChunking documents...")
    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    # Step 3: Embed and upload
    print("\nEmbedding documents (this may take a moment on first run)...")
    print("Creating Qdrant collection if needed...")
    print("Uploading vectors...")

    try:
        vector_store = ingest_documents(chunks)
    except Exception as e:
        print(f"ERROR during ingestion: {e}")
        sys.exit(1)

    print(f"\n=== Ingestion Complete ===")
    print(f"Collection : {settings.qdrant_collection}")
    print(f"Vectors    : {len(chunks)}")
    print(f"Model      : {settings.embedding_model}")
    print(f"Chunk size : {settings.chunk_size} / overlap: {settings.chunk_overlap}")
    print()


if __name__ == "__main__":
    main()
