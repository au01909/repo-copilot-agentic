"""RAGAS integration (faithfulness, context precision/recall, answer relevancy).

Known issue: at the time of writing, `pip install ragas` pulls in a langchain-
community version chain that can fail to import cleanly in some environments
(`ModuleNotFoundError: langchain_community.chat_models.vertexai`) — this is a
real upstream dependency-pinning problem with the ragas/langchain ecosystem,
not something specific to this codebase. If you hit it, try:

    pip install "langchain-community>=0.3" --upgrade

or pin ragas to a specific compatible release. This module fails soft (raises
RagasUnavailable rather than crashing the app) so the rest of the copilot works
regardless of whether ragas is importable in your environment.
"""
from typing import Dict, List

from . import config
from .chunking import Chunk


class RagasUnavailable(Exception):
    pass


def evaluate_answer(question: str, answer: str, context_chunks: List[Chunk],
                     ground_truth: str = "") -> Dict:
    if not config.OPENAI_API_KEY:
        raise RagasUnavailable("RAGAS's default metrics need OPENAI_API_KEY as the judge model.")
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                    context_recall, faithfulness)
    except ImportError as e:
        raise RagasUnavailable(f"ragas is not importable in this environment: {e}")

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[c.content[:1000] for c in context_chunks]],
        "ground_truth": [ground_truth or answer],
    }
    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    try:
        result = ragas_evaluate(dataset, metrics=metrics)
        return dict(result)
    except Exception as e:
        raise RagasUnavailable(f"ragas evaluation failed: {e}")
