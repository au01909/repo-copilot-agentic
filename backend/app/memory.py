"""Conversation memory beyond a raw Q&A transcript.

Keeps a running summary (LLM-generated if a provider is configured, otherwise a
simple truncating concatenation) plus which files were surfaced in past turns —
useful for "what does refresh token handling look like" as a follow-up to
"explain auth" without repeating the earlier retrieval.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from . import providers
from .providers import ProviderUnavailable

SUMMARY_PROMPT = """Summarize this conversation about a code repository in 2-3 \
sentences, focused on what topics/files have already been discussed so future \
questions can build on that context."""


@dataclass
class ConversationMemory:
    history: List[Tuple[str, str]] = field(default_factory=list)
    files_discussed: List[str] = field(default_factory=list)
    running_summary: str = ""

    def add_turn(self, question: str, answer: str, files: List[str]):
        self.history.append((question, answer))
        for f in files:
            if f not in self.files_discussed:
                self.files_discussed.append(f)
        self._update_summary()

    def _update_summary(self):
        if len(self.history) < 2:
            return
        if providers.is_configured():
            try:
                transcript = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in self.history[-6:])
                text, _debug = providers.complete(SUMMARY_PROMPT, transcript)
                self.running_summary = text.strip()
                return
            except ProviderUnavailable:
                pass
        # naive fallback: just note the topics (question texts) covered so far
        topics = "; ".join(q for q, _ in self.history[-6:])
        self.running_summary = f"Topics discussed so far: {topics}"

    def context_block(self) -> str:
        parts = []
        if self.running_summary:
            parts.append(f"Conversation summary: {self.running_summary}")
        if self.files_discussed:
            parts.append(f"Files already discussed: {', '.join(self.files_discussed[-10:])}")
        return "\n".join(parts)
