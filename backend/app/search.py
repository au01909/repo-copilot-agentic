"""Hybrid search pipeline.

Query
  -> query expansion (thesaurus) + multi-query fan-out (LLM, if configured)
  -> BM25 search (each query variant)
  -> Dense search via vector store (each query variant)
  -> score filtering (drop near-zero BM25 / weak dense matches)
  -> weighted Reciprocal Rank Fusion across all variant result lists
  -> dedupe (one chunk per file, keep the highest-scoring one)
  -> fetch top-K candidates
  -> rerank (lexical by default, Cohere cross-encoder if configured)
  -> return top-K chunks
"""
import uuid
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from . import config
from .chunking import Chunk
from .embeddings import get_embedder
from .query_expansion import expand_query, generate_multi_queries
from .reranker import get_reranker
from .search_tokenize import tokenize
from .source_boost import source_boost_score
from .vectorstore import VectorStore


class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:top_k] if s > config.MIN_BM25_SCORE]


class DenseIndex:
    """Real dense retrieval: an embedder (TF-IDF by default, OpenAI if configured)
    plus a Qdrant vector store (embedded, no server required by default)."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.embedder = get_embedder()
        self.store = None
        if not chunks:
            return
        texts = [c.content for c in chunks]
        vectors = self.embedder.fit_transform(texts)
        dim = vectors.shape[1]
        collection_name = f"repo_{uuid.uuid4().hex[:12]}"
        self.store = VectorStore(collection_name, dim)
        ids = [c.id for c in chunks]
        payloads = [{"idx": i} for i in range(len(chunks))]
        self.store.upsert(ids, vectors, payloads)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.store is None:
            return []
        q_vec = self.embedder.transform([query])[0]
        hits = self.store.search(q_vec, top_k)
        results = []
        for _ext_id, score, payload in hits:
            if score >= config.MIN_DENSE_SCORE:
                results.append((payload["idx"], score))
        return results

    def close(self):
        if self.store:
            self.store.close()


def weighted_rrf(
    ranked_lists: List[Tuple[List[Tuple[int, float]], float]], k: int = 60
) -> List[Tuple[int, float]]:
    """RRF where each contributing ranked list has a configurable weight.
    ranked_lists: list of (ranked_results, weight) pairs."""
    fused: Dict[int, float] = {}
    for ranked, weight in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + weight * (1.0 / (k + rank + 1))
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridSearchIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex(chunks)
        self.reranker = get_reranker()
        # telemetry from the most recent search() call (see main.py /api/query)
        self.last_retrieval_strategy = "hybrid_only"
        self.last_boosted_files: List[str] = []

    def _query_variants(self, query: str, plan=None) -> List[str]:
        variants = set(expand_query(query, plan))
        if config.ENABLE_MULTI_QUERY:
            variants.update(generate_multi_queries(query))
        else:
            variants.add(query)
        return list(variants)[:6]  # cap fan-out so latency stays reasonable

    def _dedupe_one_per_file(self, ranked: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not config.DEDUPE_ONE_CHUNK_PER_FILE:
            return ranked
        seen_files = set()
        deduped = []
        for idx, score in ranked:
            f = self.chunks[idx].file
            if f in seen_files:
                continue
            seen_files.add(f)
            deduped.append((idx, score))
        return deduped

    def _apply_source_boost(self, fused: List[Tuple[int, float]], plan) -> List[Tuple[int, float]]:
        """Boost (not filter) fused scores toward files the plan prefers, then
        re-sort. A plan with no source that has file patterns (e.g. only
        "code"/"git_history") is a no-op, so hybrid search behaves exactly as
        before when the planner has nothing source-specific to say.

        Also injects preferred-source files that BM25/dense missed entirely
        (e.g. a Dockerfile has little lexical/semantic overlap with "how do I
        deploy?") — otherwise plan preference would be capped by hybrid
        recall, defeating the point of source-aware retrieval."""
        if plan is None:
            return fused
        boosted = [
            (idx, score + source_boost_score(self.chunks[idx].file, plan))
            for idx, score in fused
        ]
        present = {idx for idx, _ in fused}
        injected = [
            (idx, source_boost_score(chunk.file, plan))
            for idx, chunk in enumerate(self.chunks)
            if idx not in present and source_boost_score(chunk.file, plan) > 0
        ]
        injected = self._dedupe_one_per_file(injected)
        return sorted(boosted + injected, key=lambda x: x[1], reverse=True)

    def search(self, query: str, plan=None, top_k: int = None, fetch_k: int = None) -> List[Chunk]:
        if not self.chunks:
            return []
        top_k = top_k or config.TOP_K
        fetch_k = fetch_k or config.FETCH_K

        variants = self._query_variants(query, plan) if (config.ENABLE_QUERY_EXPANSION or config.ENABLE_MULTI_QUERY) else [query]

        weighted_lists: List[Tuple[List[Tuple[int, float]], float]] = []
        for v in variants:
            weighted_lists.append((self.bm25.search(v, fetch_k), config.BM25_WEIGHT))
            weighted_lists.append((self.dense.search(v, fetch_k), config.DENSE_WEIGHT))

        fused = weighted_rrf(weighted_lists)
        fused = self._dedupe_one_per_file(fused)

        # Fallback: if nothing in the fused list matches the plan's preferred
        # sources, boosting is a no-op and we've already fallen back to plain
        # hybrid search — we never filter candidates out, so results can't go
        # empty just because a preferred source is missing from the repo.
        fused = self._apply_source_boost(fused, plan)

        candidate_idxs = [idx for idx, _ in fused[:fetch_k]]
        candidates = [self.chunks[i] for i in candidate_idxs]

        boosted_files = [c.file for c in candidates if source_boost_score(c.file, plan) > 0.0] if plan else []
        self.last_boosted_files = boosted_files
        self.last_retrieval_strategy = "source_boosting" if boosted_files else "hybrid_only"

        return self.reranker.rerank(query, candidates, top_k, plan)

    def close(self):
        self.dense.close()
