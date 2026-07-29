"""A LangChain-compatible retriever backed by the app's existing hybrid search
engine (`app/search.py`'s `HybridSearchIndex`: BM25 + dense retrieval via
Qdrant, weighted Reciprocal Rank Fusion, one-chunk-per-file dedupe, and a
reranking stage).

This is a deliberate choice, not laziness: `HybridSearchIndex` is already a
production BM25+dense+RRF+rerank pipeline, tuned and tested against real
repositories. Reimplementing it with LangChain's own `BM25Retriever` +
`EnsembleRetriever` would mean throwing that away to rebuild the same thing
with a different API surface — the master prompt itself says not to replace
working retrieval just to use a different framework. Instead, this class is
the seam: it satisfies `langchain_core.retrievers.BaseRetriever`, so anything
built for LangChain (chains, `MultiQueryRetriever`, evaluation tooling) can use
repository search without knowing it isn't a native LangChain retriever
underneath.
"""
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..search import HybridSearchIndex
from .documents import chunk_to_document


class HybridRepoRetriever(BaseRetriever):
    """Wraps a `HybridSearchIndex` for one indexed repository/session."""

    search_index: HybridSearchIndex
    top_k: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        chunks = self.search_index.search(query, top_k=self.top_k)
        return [chunk_to_document(c) for c in chunks]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # HybridSearchIndex.search is sync (BM25/TF-IDF/Qdrant local calls are
        # all fast, in-process); no real async work to do, so just call it.
        return self._get_relevant_documents(query, run_manager=run_manager)
