"""LangSmith tracing.

There's deliberately almost no code here: LangSmith tracing in LangChain/
LangGraph is activated purely by environment variables —

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=repo-copilot   # optional, defaults to "default"

— which LangChain reads automatically at import time. Every `HybridRepoRetriever`
call and every LangGraph node in `agent/workflow.py` gets traced for free once
those are set, with zero code changes. This module exists only so `/api/health`
can report whether tracing is actually active, without the rest of the app
needing to know how LangSmith's env-var contract works.
"""
import os


def is_enabled() -> bool:
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")


def status() -> dict:
    return {
        "enabled": is_enabled(),
        "project": os.environ.get("LANGCHAIN_PROJECT", "default"),
        "api_key_present": bool(os.environ.get("LANGCHAIN_API_KEY")),
    }
