"""Retrieval evaluation you can run today, without RAGAS/DeepEval/an LLM judge.

Those frameworks need either an LLM-as-judge call or extra heavy dependencies.
This module implements the retrieval-metric math directly (Recall@K, MRR, nDCG)
against a hand-labeled query set, which is exactly the kind of offline benchmark
the PRD describes — just without the auto-generation-from-docs step, which
would itself need an LLM to write the 500 questions.

Usage:
    labeled = [
        {"query": "How does auth work?", "relevant_files": ["backend/auth.py"]},
        ...
    ]
    report = evaluate(index, labeled)
"""
import math
from typing import Dict, List

from .search import HybridSearchIndex


def _dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate(index: HybridSearchIndex, labeled_queries: List[Dict], k: int = 10) -> Dict:
    recalls, mrrs, ndcgs = [], [], []

    for item in labeled_queries:
        query = item["query"]
        relevant_files = set(item["relevant_files"])
        results = index.search(query, top_k=k)
        retrieved_files = [c.file for c in results]

        hits = [1 if f in relevant_files else 0 for f in retrieved_files]
        recalls.append(1.0 if any(hits) else 0.0)

        rr = 0.0
        for i, h in enumerate(hits):
            if h:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)

        ideal = sorted(hits, reverse=True)
        dcg, idcg = _dcg(hits), _dcg(ideal)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    n = max(len(labeled_queries), 1)
    return {
        "n_queries": len(labeled_queries),
        "recall@k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "ndcg@k": sum(ndcgs) / n,
        "k": k,
    }
