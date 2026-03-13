from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from pypdf import PdfReader


def load_documents(dir_path: Path) -> List:
    docs = []
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return docs

    for p in dir_path.rglob("*"):
        if p.is_dir():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix == ".txt":
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif suffix == ".md":
                # Treat Markdown as plain text to avoid extra dependencies
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif suffix == ".pdf":
                reader = PdfReader(str(p))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(p),
                                "page": i + 1,
                            },
                        )
                    )
        except Exception:
            # Skip files that fail to load
            continue
    return docs
