"""Reranking stage.

lexical -- default, no API key needed. Re-scores the fused candidate list by
           query/chunk token overlap (a real signal, distinct from BM25/TF-IDF
           ranking) so it's not just a passthrough of the fusion order.
cohere  -- true cross-encoder reranking via Cohere's Rerank API, used when
           RERANK_PROVIDER=cohere and COHERE_API_KEY is set.
"""
from typing import List

from . import config
from .chunking import Chunk
from .search_tokenize import tokenize  # shared tokenizer


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:
        q_tokens = set(tokenize(query))
        if not q_tokens or not candidates:
            return candidates[:top_k]

        def overlap_score(chunk: Chunk) -> float:
            c_tokens = set(tokenize(chunk.content))
            symbol_tokens = set(tokenize(chunk.symbol or ""))
            overlap = len(q_tokens & c_tokens)
            symbol_bonus = 2 * len(q_tokens & symbol_tokens)  # matching the function/class name matters more
            return overlap + symbol_bonus

        scored = sorted(candidates, key=overlap_score, reverse=True)
        return scored[:top_k]


class CohereReranker:
    name = "cohere"

    def __init__(self):
        if not config.COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY not set")
        import cohere
        self.client = cohere.Client(config.COHERE_API_KEY)

    def rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:
        if not candidates:
            return []
        docs = [c.content[:2000] for c in candidates]
        try:
            results = self.client.rerank(
                model=config.COHERE_RERANK_MODEL, query=query, documents=docs, top_n=top_k,
            )
            return [candidates[r.index] for r in results.results]
        except Exception:
            # fail open to lexical rather than breaking the whole request
            return LexicalReranker().rerank(query, candidates, top_k)


def get_reranker():
    if config.RERANK_PROVIDER == "cohere":
        try:
            return CohereReranker()
        except Exception:
            return LexicalReranker()
    if config.RERANK_PROVIDER == "none":
        class _Passthrough:
            name = "none"
            def rerank(self, query, candidates, top_k):
                return candidates[:top_k]
        return _Passthrough()
    return LexicalReranker()
