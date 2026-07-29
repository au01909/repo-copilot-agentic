"""LangGraph repository workflow.

    START -> intent_detection -> retrieve -> validate_context
                                        |-- insufficient (retry_count < max) --> retrieve (widen fetch_k)
                                        `-- sufficient/out of retries --> generate -> citations -> END

Deliberately flat — five real nodes, one conditional retry loop with a strict
bound (`config.AGENT_MAX_RETRIES`), no nested subgraphs. This is the
orchestration layer the ADK Repository Agent calls into for its single
"answer_repository_question" tool (see `repository_agent.py`), and it's also
callable directly (`AGENT_FRAMEWORK=direct`) if ADK/its model backend isn't
configured.
"""
from typing import List

from langgraph.graph import END, StateGraph

from .. import config, llm as llm_module
from ..chunking import Chunk
from ..langchain_rag.documents import chunks_to_documents, document_citation
from .intent import classify
from .state import RepoWorkflowState


def node_intent_detection(state: RepoWorkflowState) -> dict:
    return {"intent": classify(state["question"]), "retry_count": 0, "errors": []}


def _make_retrieve_node(search_index):
    # Calls search_index + chunks_to_documents directly rather than through
    # HybridRepoRetriever.invoke(): fetch_k needs to widen per retry, which
    # BaseRetriever's plain `invoke(query)` signature doesn't expose. The
    # conversion step is identical either way — chunks_to_documents is exactly
    # what HybridRepoRetriever calls internally — so this still produces the
    # same LangChain Document contract; HybridRepoRetriever itself is the
    # right entry point for anything that doesn't need per-call fetch_k
    # control (e.g. LangChain's MultiQueryRetriever/EnsembleRetriever, or a
    # notebook/script using the retriever standalone).
    def node_retrieve(state: RepoWorkflowState) -> dict:
        retry = state.get("retry_count", 0)
        fetch_k = config.FETCH_K * (1 + retry)  # widen on retry, same top_k returned
        chunks: List[Chunk] = search_index.search(state["question"], fetch_k=fetch_k)
        documents = chunks_to_documents(chunks)
        return {"retrieved_documents": documents, "retrieved_chunks": chunks}
    return node_retrieve


def node_validate_context(state: RepoWorkflowState) -> dict:
    docs = state.get("retrieved_chunks", [])
    # "sufficient" here is a cheap, real heuristic — at least one chunk was
    # retrieved. This isn't a semantic sufficiency check (that would need
    # another LLM call, which the spec says to avoid running per-query
    # more than necessary); it catches the true-empty case reliably.
    sufficient = len(docs) > 0
    return {"context_sufficient": sufficient}


def _should_retry(state: RepoWorkflowState) -> str:
    if state.get("context_sufficient"):
        return "generate"
    if state.get("retry_count", 0) >= config.AGENT_MAX_RETRIES:
        return "generate"  # give up widening, let `generate` produce the
        # "couldn't find that information" response via the empty-context path
    return "retry"


def node_bump_retry(state: RepoWorkflowState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def _make_generate_node(get_history):
    def node_generate(state: RepoWorkflowState) -> dict:
        chunks = state.get("retrieved_chunks", [])
        history = get_history(state["session_id"]) if get_history else []
        answer, citations, mode, _debug = llm_module.generate_answer(
            state["question"], chunks, history,
        )
        return {"answer": answer, "mode": mode}
    return node_generate


def node_citations(state: RepoWorkflowState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    citations = []
    for c in chunks:
        citations.append({
            "file": c.file, "start_line": c.start_line, "end_line": c.end_line,
            "symbol": c.symbol, "citation": f"{c.file}:{c.start_line}-{c.end_line}",
        })
    return {"citations": citations}


def build_workflow(search_index, get_history=None):
    """Compiles a fresh graph bound to one repository's search index (and
    optionally a conversation-history lookup for multi-turn context)."""
    graph = StateGraph(RepoWorkflowState)

    graph.add_node("intent_detection", node_intent_detection)
    graph.add_node("retrieve", _make_retrieve_node(search_index))
    graph.add_node("validate_context", node_validate_context)
    graph.add_node("bump_retry", node_bump_retry)
    graph.add_node("generate", _make_generate_node(get_history))
    graph.add_node("build_citations", node_citations)

    graph.set_entry_point("intent_detection")
    graph.add_edge("intent_detection", "retrieve")
    graph.add_edge("retrieve", "validate_context")
    graph.add_conditional_edges(
        "validate_context", _should_retry, {"generate": "generate", "retry": "bump_retry"},
    )
    graph.add_edge("bump_retry", "retrieve")
    graph.add_edge("generate", "build_citations")
    graph.add_edge("build_citations", END)

    return graph.compile()


def run_workflow(search_index, session_id: str, question: str, get_history=None) -> RepoWorkflowState:
    app = build_workflow(search_index, get_history)
    initial: RepoWorkflowState = {"session_id": session_id, "question": question}
    return app.invoke(initial)
