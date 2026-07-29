"""DeepEval integration (hallucination, answer relevancy, correctness).

`pip install deepeval` pulls in real metric implementations, but every metric
that isn't purely statistical needs an LLM to act as the judge model — DeepEval
defaults to OpenAI, so this requires OPENAI_API_KEY regardless of which
provider is used for the copilot's own answers. Import is lazy so the rest of
the app works fine if deepeval isn't installed or configured.
"""
from typing import Dict, List

from . import config
from .chunking import Chunk


class DeepEvalUnavailable(Exception):
    pass


def evaluate_answer(question: str, answer: str, context_chunks: List[Chunk]) -> Dict:
    if not config.OPENAI_API_KEY:
        raise DeepEvalUnavailable(
            "DeepEval's default metrics need OPENAI_API_KEY as the judge model, "
            "even if you're using a different provider for answer generation."
        )
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import (AnswerRelevancyMetric,
                                       FaithfulnessMetric,
                                       HallucinationMetric)
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        raise DeepEvalUnavailable(f"deepeval is not installed: {e}")

    context_texts = [c.content[:1000] for c in context_chunks]
    test_case = LLMTestCase(
        input=question, actual_output=answer,
        retrieval_context=context_texts, context=context_texts,
    )
    metrics = [
        AnswerRelevancyMetric(), FaithfulnessMetric(), HallucinationMetric(threshold=0.5),
    ]
    results = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            results[metric.__class__.__name__] = {
                "score": metric.score, "reason": getattr(metric, "reason", None),
            }
        except Exception as e:
            results[metric.__class__.__name__] = {"error": str(e)}
    return results
