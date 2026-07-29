"""Answer generation with citations.

Calls whichever LLM_PROVIDER is configured (anthropic/openai/nvidia/deepseek).
On any failure — missing key, network error, bad response — falls back to an
extractive answer instead of crashing, and reports why via LLMDebugInfo.
"""
from dataclasses import asdict
from typing import List, Tuple

from . import providers
from .chunking import Chunk
from .models import Citation
from .providers import LLMDebugInfo, ProviderUnavailable

SYSTEM_PROMPT = """You are a repository copilot. Answer the user's question about a \
codebase using ONLY the provided context chunks. Every factual claim must be \
grounded in the given chunks. Cite sources inline like [1], [2] matching the \
numbered context blocks. If the context doesn't contain the answer, say so \
explicitly instead of guessing."""


def _format_context(chunks: List[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.file} (lines {c.start_line}-{c.end_line})"
        if c.symbol:
            header += f" — {c.kind}: {c.symbol}"
        blocks.append(f"{header}\n```{c.language}\n{c.content}\n```")
    return "\n\n".join(blocks)


def _to_citations(chunks: List[Chunk]) -> List[Citation]:
    citations = []
    for c in chunks:
        snippet = c.content.strip().splitlines()
        preview = "\n".join(snippet[:3])
        citations.append(Citation(
            file=c.file, start_line=c.start_line, end_line=c.end_line,
            symbol=c.symbol, snippet=preview,
        ))
    return citations


def _extractive_answer(chunks: List[Chunk], reason: str = "") -> str:
    if not chunks:
        return "No relevant code or docs were found for this question in the indexed repository."
    prefix = f"(Extractive mode{': ' + reason if reason else ''} — showing the most " \
             "relevant retrieved context rather than a generated explanation.)\n"
    lines = [prefix]
    for i, c in enumerate(chunks, start=1):
        loc = f"{c.file}:{c.start_line}-{c.end_line}"
        symbol = f" ({c.kind}: {c.symbol})" if c.symbol else ""
        preview = "\n".join(c.content.strip().splitlines()[:8])
        lines.append(f"[{i}] {loc}{symbol}\n{preview}\n")
    return "\n".join(lines)


def _build_user_message(question: str, chunks: List[Chunk],
                         history: List[Tuple[str, str]],
                         repo_context: str = "") -> str:
    """Prompt includes repository context, conversation history, question,
    instructions, and numbered sources — not just the raw chunks."""
    parts = []
    if repo_context:
        parts.append(f"Repository context:\n{repo_context}\n")
    if history:
        convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-3:])
        parts.append(f"Conversation so far:\n{convo}\n")
    parts.append(f"Sources:\n{_format_context(chunks)}\n")
    parts.append(f"Question: {question}")
    parts.append("Answer using only the sources above, with inline [n] citations.")
    return "\n\n".join(parts)


def generate_answer(
    question: str, chunks: List[Chunk], history: List[Tuple[str, str]],
    repo_context: str = "",
) -> Tuple[str, List[Citation], str, dict]:
    """Returns (answer, citations, mode, debug_info_dict)."""
    citations = _to_citations(chunks)
    user_msg = _build_user_message(question, chunks, history, repo_context)

    if providers.is_configured():
        try:
            answer, debug = providers.complete(SYSTEM_PROMPT, user_msg)
            return answer, citations, "generative", asdict(debug)
        except ProviderUnavailable as e:
            debug = providers.get_active_provider_debug()
            debug.status, debug.error = "error", str(e)
            answer = _extractive_answer(chunks, reason=f"LLM call failed ({e})")
            return answer, citations, "extractive", asdict(debug)

    debug = providers.get_active_provider_debug()
    answer = _extractive_answer(chunks, reason="no LLM provider configured")
    return answer, citations, "extractive", asdict(debug)
