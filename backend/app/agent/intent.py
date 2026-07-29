"""Intent classification for the LangGraph workflow's first node.

Deterministic keyword rules run first and are free/instant — most of the
acceptance-test questions in the spec ("What does this repository do?", "How
do I run this?", "Draw the architecture diagram") match a rule directly. LLM
classification is only used as a fallback for genuinely ambiguous questions,
and only if a provider is configured — this matches "use deterministic
routing where obvious and LLM classification only when necessary."
"""
import re
from typing import Optional

from .state import INTENT_LABELS

_RULES = [
    ("REPOSITORY_SUMMARY", re.compile(
        r"\bwhat does (this|the) repo(sitory)?\b|\bsummar(y|ize|ise)\b|\bmain functionality\b|\boverview\b", re.I)),
    ("ARCHITECTURE", re.compile(
        r"\barchitecture\b|\bhow (is|does) (this|the) (app|system|repo) (structured|organi[sz]ed)\b|\bdraw\b.*\bdiagram\b", re.I)),
    ("SETUP", re.compile(
        r"\bhow do i (run|install|set ?up|deploy)\b|\bsetup instructions\b|\bgetting started\b|\bstep.by.step\b", re.I)),
    ("TECH_STACK", re.compile(
        r"\btechnolog(y|ies)\b|\btech stack\b|\bwhat.*(framework|library|libraries) (is|are) used\b", re.I)),
    ("ENTRYPOINT", re.compile(
        r"\bentry ?point\b|\bwhere does (the|this) (app|application|program) start\b|\bmain\(\)", re.I)),
    ("DATA_FLOW", re.compile(
        r"\bdata flow\b|\bhow does data flow\b|\brequest lifecycle\b", re.I)),
    ("DEPENDENCY_ANALYSIS", re.compile(
        r"\bdependenc(y|ies)\b|\bwhy (are we|is) using\b|\bpackage(s)? (does|do) (it|this) (use|depend on)\b", re.I)),
    ("CODE_QUESTION", re.compile(
        r"\bexplain (this|the) function\b|\bwhere is .* implemented\b|\bwhich file\b|\bapi routes?\b|\bauthentication\b|\bdatabase\b", re.I)),
]


def classify_deterministic(question: str) -> Optional[str]:
    for label, pattern in _RULES:
        if pattern.search(question):
            return label
    return None


def classify(question: str) -> str:
    """Deterministic first; falls back to GENERAL (not an LLM call) when no
    rule matches and no provider is configured, or as a last resort otherwise.
    Kept simple per 'avoid unnecessary graph complexity' — an LLM classifier
    is easy to add here later if GENERAL turns out to catch too much."""
    label = classify_deterministic(question)
    if label:
        return label
    return "GENERAL"
