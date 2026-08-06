"""Agentic planning: instead of always retrieving the same way, decide what
kinds of sources the question actually needs (code, README, issues, PRs, git
history) before retrieving.

Uses an LLM call when a provider is configured (real planning); otherwise
falls back to keyword heuristics (deterministic, zero-cost, always available).
"""
import re
from dataclasses import dataclass, field
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
    (re.compile(r"\breadme\b|\bhow do i (install|set ?up|run)\b|\bgetting started\b|\bwhat does this repo(sitory)? do\b|\bwhat is this (repo|project)\b|\boverview\b", re.I), ["readme"]),
    (re.compile(r"\barchitecture\b|\bdesign\b|\bhow (is|does) .* (structured|organized|work)\b|\bentry ?point\b", re.I), ["architecture"]),
    (re.compile(r"\btest(s|ing)?\b|\bpytest\b|\bunit tests?\b", re.I), ["testing"]),
    (re.compile(r"\bdeploy(ment|ing)?\b|\bdocker\b|\bkubernetes\b|\bk8s\b|\bhelm\b|\bci/?cd\b", re.I), ["deployment"]),
    (re.compile(r"\bdepend(enc(y|ies))?\b|\brequirements\b|\bpackages?\b", re.I), ["dependencies"]),
    (re.compile(r"\bapi\b|\bendpoint\b|\broute\b|\bcontroller\b", re.I), ["api"]),
]

# File-glob preferences per retrieval source, used to boost matching results
# rather than hard-filter them (see search.py: source_boost_score).
SOURCE_FILE_PATTERNS = {
    "readme": ["readme.md", "readme.rst", "readme.txt", "docs/*", "overview/*", "introduction/*"],
    "architecture": ["architecture/*", "design/*", "docs/*", "main.py", "app.py", "server.py"],
    "api": ["routes/*", "controllers/*", "api/*", "main.py"],
    "testing": ["tests/*", "test_*", "*_test.py"],
    "deployment": ["dockerfile", "docker-compose.yml", "deploy*", "docs/deployment*", "cloud*", "k8s*", "helm*"],
    "dependencies": ["requirements.txt", "pyproject.toml", "package.json", "cargo.toml", "pom.xml", "go.mod"],
    # git_history is served from git metadata directly, not file-glob boosting.
}


@dataclass
class Plan:
    sources: List[str]
    reasoning: str
    # per-source boost weight, e.g. {"readme": 3.0}; defaults applied in search.py
    # when a source has no explicit weight.
    source_weights: dict = field(default_factory=dict)


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
        valid = {"code", "readme", "issues", "pull_requests", "git_history", "architecture", "testing", "deployment", "dependencies", "api"}
        sources = [s for s in sources if s in valid] or ["code"]
        # LLM planning skips file-glob heuristics for sources it didn't already infer;
        # layer the keyword heuristics on top so boosting still has patterns to match.
        heuristic = _heuristic_plan(question)
        sources = sorted(set(sources) | (set(heuristic.sources) & set(SOURCE_FILE_PATTERNS)))
        return Plan(sources=sources, reasoning="LLM planner")
    except ProviderUnavailable:
        return _heuristic_plan(question)
