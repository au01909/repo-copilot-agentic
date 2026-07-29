"""Automatic evaluation-dataset generation: turns README/docstring content into
candidate (question, relevant_file) pairs for the offline retrieval benchmark
in eval_stub.py. Requires an LLM provider — generating plausible questions from
docs is a genuine language task, not something a heuristic can fake well.

The PRD's "500 repository questions" benchmark is intentionally not attempted
in one call; this generates a batch per source file so it stays within a
reasonable prompt size, and the caller decides how many files to sample.
"""
import json
from typing import Dict, List

from . import providers
from .chunking import Chunk
from .providers import ProviderUnavailable

GEN_PROMPT = """Given this content from a repository file, write 2 realistic questions \
a developer new to the codebase might ask that this specific file would help answer. \
Respond ONLY with a JSON list of strings, e.g. ["question 1", "question 2"]."""


def generate_questions_for_chunk(chunk: Chunk) -> List[str]:
    if not providers.is_configured():
        return []
    try:
        text, _debug = providers.complete(GEN_PROMPT, f"File: {chunk.file}\n\n{chunk.content[:2000]}")
        cleaned = text.strip().strip("`").removeprefix("json").strip()
        questions = json.loads(cleaned)
        return [q for q in questions if isinstance(q, str)][:2]
    except (ProviderUnavailable, json.JSONDecodeError):
        return []


def generate_benchmark(chunks: List[Chunk], max_files: int = 25) -> List[Dict]:
    """Samples up to `max_files` distinct files (prioritizing README and
    top-level modules) and generates labeled queries for each."""
    seen_files = set()
    sampled: List[Chunk] = []
    # prioritize README / module-level chunks, which tend to produce clearer questions
    prioritized = sorted(chunks, key=lambda c: (c.kind not in ("heading", "module", "class"),))
    for c in prioritized:
        if c.file in seen_files:
            continue
        seen_files.add(c.file)
        sampled.append(c)
        if len(sampled) >= max_files:
            break

    benchmark = []
    for c in sampled:
        for q in generate_questions_for_chunk(c):
            benchmark.append({"query": q, "relevant_files": [c.file]})
    return benchmark
