"""Agentic planning: instead of always retrieving the same way, decide what
kinds of sources the question actually needs (code, README, issues, PRs, git
history) before retrieving.

Uses an LLM call when a provider is configured (real planning); otherwise
falls back to keyword heuristics (deterministic, zero-cost, always available).
"""
import re
from dataclasses import dataclass
from typing import List

from . import providers
from .providers import ProviderUnavailable

PLANNER_PROMPT = """Given a question about a code repository, decide which of these \
sources are needed to answer it well: code, readme, issues, pull_requests, git_history. \
Return a comma-separated list of only the needed sources, nothing else."""

_HEURISTICS = [
    (re.compile(r"\bissue\s*#?\d+\b", re.I), ["issues"]),
    (re.compile(r"\bpr\s*#?\d+\b|\bpull request\b", re.I), ["pull_requests"]),
    (re.compile(r"\bcommit\b|\bchangelog\b|\bhistory\b|\bwho (wrote|changed)\b", re.I), ["git_history"]),
    (re.compile(r"\breadme\b|\bhow do i (install|set ?up|run)\b|\bgetting started\b", re.I), ["readme"]),
]


@dataclass
class Plan:
    sources: List[str]
    reasoning: str


def _heuristic_plan(question: str) -> Plan:
    sources = set()
    for pattern, tags in _HEURISTICS:
        if pattern.search(question):
            sources.update(tags)
    sources.add("code")  # code retrieval is always relevant as a baseline
    return Plan(sources=sorted(sources), reasoning="keyword heuristics (no LLM provider configured)")


def plan(question: str) -> Plan:
    if not providers.is_configured():
        return _heuristic_plan(question)
    try:
        text, _debug = providers.complete(PLANNER_PROMPT, question)
        sources = [s.strip().lower() for s in text.split(",") if s.strip()]
        valid = {"code", "readme", "issues", "pull_requests", "git_history"}
        sources = [s for s in sources if s in valid] or ["code"]
        return Plan(sources=sources, reasoning="LLM planner")
    except ProviderUnavailable:
        return _heuristic_plan(question)
