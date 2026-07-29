"""Typed state for the LangGraph repository workflow.

Kept deliberately small — the master prompt explicitly asks to avoid
unnecessary graph complexity, so this state only has the fields the five
nodes (intent_detection, retrieve, validate_context, generate, citations)
actually read or write.
"""
from typing import Any, Dict, List, Optional, TypedDict


class RepoWorkflowState(TypedDict, total=False):
    session_id: str
    question: str
    intent: str                    # one of INTENT_LABELS below
    retrieved_documents: List[Any]  # List[langchain_core.documents.Document] — the
                                     # LangChain-layer view of retrieval results
    retrieved_chunks: List[Any]     # List[app.chunking.Chunk] — same results, native
                                     # type, used internally by generate/citations so
                                     # the existing (tested) llm.generate_answer and
                                     # citation formatting don't need to change
    context_sufficient: bool
    answer: str
    citations: List[Dict]
    mode: str                      # "generative" | "extractive"
    retry_count: int
    errors: List[str]


INTENT_LABELS = [
    "REPOSITORY_SUMMARY",
    "ARCHITECTURE",
    "SETUP",
    "TECH_STACK",
    "CODE_QUESTION",
    "DATA_FLOW",
    "ENTRYPOINT",
    "DEPENDENCY_ANALYSIS",
    "GENERAL",
]
