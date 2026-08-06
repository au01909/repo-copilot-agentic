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
from .source_boost import source_boost_score


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, candidates: List[Chunk], top_k: int, plan=None) -> List[Chunk]:
        q_tokens = set(tokenize(query))
        if not candidates:
            return []

        def overlap_score(chunk: Chunk) -> float:
            c_tokens = set(tokenize(chunk.content))
            symbol_tokens = set(tokenize(chunk.symbol or ""))
            overlap = len(q_tokens & c_tokens)
            symbol_bonus = 2 * len(q_tokens & symbol_tokens)  # matching the function/class name matters more
            boost = source_boost_score(chunk.file, plan) if plan is not None else 0.0
            return overlap + symbol_bonus + boost

        if not q_tokens and plan is None:
            return candidates[:top_k]

        scored = sorted(candidates, key=overlap_score, reverse=True)
        return scored[:top_k]


class CohereReranker:
    name = "cohere"

    def __init__(self):
        if not config.COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY not set")
        import cohere
        self.client = cohere.Client(config.COHERE_API_KEY)

    def rerank(self, query: str, candidates: List[Chunk], top_k: int, plan=None) -> List[Chunk]:
        if not candidates:
            return []
        docs = [c.content[:2000] for c in candidates]
        try:
            results = self.client.rerank(
                model=config.COHERE_RERANK_MODEL, query=query, documents=docs, top_n=top_k,
            )
            reranked = [candidates[r.index] for r in results.results]
            if plan is not None:
                # cross-encoder score doesn't know about the retrieval plan; re-sort by
                # (cross-encoder rank position, source boost) to fold in plan preference.
                boosted = sorted(
                    enumerate(reranked),
                    key=lambda pair: (source_boost_score(pair[1].file, plan), -pair[0]),
                    reverse=True,
                )
                reranked = [c for _, c in boosted]
            return reranked
        except Exception:
            # fail open to lexical rather than breaking the whole request
            return LexicalReranker().rerank(query, candidates, top_k, plan)


def get_reranker():
    if config.RERANK_PROVIDER == "cohere":
        try:
            return CohereReranker()
        except Exception:
            return LexicalReranker()
    if config.RERANK_PROVIDER == "none":
        class _Passthrough:
            name = "none"
            def rerank(self, query, candidates, top_k, plan=None):
                return candidates[:top_k]
        return _Passthrough()
    return LexicalReranker()
