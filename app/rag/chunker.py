"""
Chunking pipeline — splits documents into meaningful chunks while preserving metadata.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter.
    Metadata is preserved on every chunk.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Filter out chunks that are too short to be meaningful
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    return chunks
