"""
Document loader — recursively discovers and loads all .md files from data/.
Automatically assigns category metadata based on file path.
"""

from pathlib import Path
from langchain_core.documents import Document


CATEGORY_MAP = {
    "resume": "resume",
    "education": "education",
    "skills": "skills",
    "internships": "internship",
    "achievements": "achievement",
    "research": "research",
    "projects": "project",
}


def _infer_metadata(file_path: Path, data_root: Path) -> dict:
    """Infer category and other metadata from the file path."""
    relative = file_path.relative_to(data_root)
    stem = file_path.stem.lower()
    metadata = {"source": str(relative).replace("\\", "/")}
    metadata["category"] = CATEGORY_MAP.get(stem, "general")
    return metadata


def load_documents(data_dir: str = "data") -> list[Document]:
    """
    Recursively scan data_dir for .md files and return LangChain Documents.
    """
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root.resolve()}")

    documents = []
    for md_file in sorted(data_root.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        if not text.strip():
            continue
        metadata = _infer_metadata(md_file, data_root)
        documents.append(Document(page_content=text, metadata=metadata))

    return documents
