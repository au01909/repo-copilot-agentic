"""Query expansion and multi-query retrieval.

expand_query()    -- rule-based synonym expansion using a small built-in
                      software/infra thesaurus. Needs no external calls, so it
                      always works, but only covers common terms.
generate_multi_queries() -- LLM-based reformulation into 3-5 query variants
                      (architecture / keyword / synonym / API-focused), used
                      when an LLM provider is configured. Falls back to just
                      the original query otherwise.
"""
from typing import List

from . import providers
from .providers import ProviderUnavailable

# Small domain thesaurus covering common software/infra terms. Not exhaustive —
# this is the "works with zero setup" tier; swap in WordNet or an LLM call for
# broader coverage.
THESAURUS = {
    "auth": ["authentication", "login", "jwt", "oauth", "token validation"],
    "authentication": ["auth", "login", "jwt", "oauth"],
    "jwt": ["json web token", "bearer token", "authentication"],
    "db": ["database", "storage", "persistence"],
    "database": ["db", "storage", "persistence", "sql"],
    "api": ["endpoint", "route", "controller", "handler"],
    "endpoint": ["api", "route", "handler"],
    "queue": ["kafka", "rabbitmq", "message broker", "pubsub"],
    "cache": ["redis", "memoization", "caching layer"],
    "test": ["unit test", "pytest", "spec", "assertion"],
    "config": ["configuration", "settings", "env", "environment variable"],
    "log": ["logging", "logger", "observability"],
    "error": ["exception", "failure", "bug", "traceback"],
    "deploy": ["deployment", "ci/cd", "release", "pipeline"],
}


# Extra search terms to fan out to when the planner identifies a source intent,
# so retrieval isn't limited to generic thesaurus synonyms of the raw question.
_INTENT_TERMS = {
    "readme": ["repository purpose", "repository overview", "readme", "introduction"],
    "architecture": ["architecture", "system design", "entry point"],
    "api": ["api endpoint", "route handler"],
    "testing": ["unit tests", "test suite"],
    "deployment": ["deployment", "docker", "how to deploy"],
    "dependencies": ["dependencies", "requirements"],
}


def expand_query(query: str, plan=None) -> List[str]:
    """Return the original query plus synonym-expanded and, if a plan is given,
    intent-expanded variants (e.g. a README-intent question also searches for
    "repository overview" rather than only generic synonyms)."""
    lowered = query.lower()
    expansions = set()
    for term, synonyms in THESAURUS.items():
        if term in lowered:
            for syn in synonyms:
                expansions.add(lowered.replace(term, syn))

    intent_expansions = []
    if plan is not None:
        for source in plan.sources:
            intent_expansions.extend(_INTENT_TERMS.get(source, []))

    return [query] + intent_expansions[:4] + sorted(expansions)[:4]


MULTI_QUERY_PROMPT = """Given the user's question about a code repository, generate \
4 alternative search queries that would help retrieve relevant code and docs. \
Cover different angles: one architecture-level query, one keyword/identifier-focused \
query, one synonym-based rephrasing, and one API/interface-focused query. \
Return ONLY the 4 queries, one per line, no numbering, no extra text."""


def generate_multi_queries(question: str) -> List[str]:
    """LLM-based query fan-out. Returns [question] alone if no provider is configured
    or the call fails — callers should treat this as best-effort, not required."""
    if not providers.is_configured():
        return [question]
    try:
        text, _debug = providers.complete(MULTI_QUERY_PROMPT, question)
        variants = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
        variants = [v for v in variants if v][:4]
        return [question] + variants
    except ProviderUnavailable:
        return [question]
