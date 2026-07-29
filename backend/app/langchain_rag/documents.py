"""Converts the app's internal `Chunk` (from `app/chunking.py`) into LangChain
`Document` objects. This is the only place that translation happens — the
retriever, chains, and evaluation code all work in terms of `Document` from
here on, so LangChain-based tooling (retrievers, output parsers, evaluators)
can be used without every module needing to know about `Chunk`.
"""
from typing import List

from langchain_core.documents import Document

from ..chunking import Chunk


def chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.content,
        metadata={
            "repo": chunk.repo,
            "file": chunk.file,
            "language": chunk.language,
            "symbol": chunk.symbol,
            "chunk_type": chunk.kind,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "imports": chunk.imports,
            "commit_sha": chunk.commit_sha,
            "author": chunk.author,
            "branch": chunk.branch,
        },
    )


def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    return [chunk_to_document(c) for c in chunks]


def document_citation(doc: Document) -> str:
    """The `file:start-end` citation format used throughout the app,
    e.g. 'backend/app/main.py:40-82'."""
    m = doc.metadata
    return f"{m.get('file')}:{m.get('start_line')}-{m.get('end_line')}"
