"""LLM-as-a-judge: scores a generated answer against the retrieved context on
correctness, usefulness, completeness, and groundedness. Requires an LLM
provider to be configured — there's no heuristic fallback for this one because
judging answer quality genuinely needs a language model, not just keyword math.
"""
import json
from dataclasses import dataclass
from typing import List, Optional

from . import providers
from .chunking import Chunk
from .providers import ProviderUnavailable

JUDGE_PROMPT = """You are grading an AI repository copilot's answer. Given the \
question, the retrieved context it was allowed to use, and its answer, score it \
from 1-5 on each of: correct, useful, complete, grounded (grounded = doesn't \
claim things the context doesn't support). Respond ONLY with JSON in the form: \
{"correct": int, "useful": int, "complete": int, "grounded": int, "notes": "short reason"}"""


@dataclass
class JudgeResult:
    correct: int
    useful: int
    complete: int
    grounded: int
    notes: str


def judge_answer(question: str, context_chunks: List[Chunk], answer: str) -> Optional[JudgeResult]:
    if not providers.is_configured():
        return None
    context = "\n\n".join(f"[{c.file}] {c.content[:500]}" for c in context_chunks)
    user_msg = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
    try:
        text, _debug = providers.complete(JUDGE_PROMPT, user_msg)
        cleaned = text.strip().strip("`").removeprefix("json").strip()
        data = json.loads(cleaned)
        return JudgeResult(
            correct=int(data.get("correct", 0)), useful=int(data.get("useful", 0)),
            complete=int(data.get("complete", 0)), grounded=int(data.get("grounded", 0)),
            notes=str(data.get("notes", "")),
        )
    except (ProviderUnavailable, json.JSONDecodeError, ValueError):
        return None
