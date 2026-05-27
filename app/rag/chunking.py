import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[TextChunk]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    documents: List[Document] = splitter.create_documents([cleaned])
    return [
        TextChunk(index=i, content=document.page_content.strip())
        for i, document in enumerate(documents)
        if document.page_content.strip()
    ]
