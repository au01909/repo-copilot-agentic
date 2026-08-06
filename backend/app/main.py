import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import (architecture_detect, chunking, code_exec, config, eval_dataset_gen,
               eval_deepeval, eval_judge, eval_ragas, eval_stub, github_integration,
               incremental, ingest, llm, memory, persistence, planner, providers)
from .agent.repository_agent import ADKAgentUnavailable, run_adk_agent
from .agent.tools import RepoToolContext
from .agent.workflow import run_workflow
from .chunking import Chunk
from .github_integration import GitHubAPIError
from .graph import build_import_graph, graph_stats, to_mermaid
from .observability import langsmith_setup
from .observability.mlflow_tracking import track_query
from .models import (AddRepoRequest, ArchitectureResponse, ChatRequest,
                      ChatResponse, Citation, CommitSummary, EvalRunRequest,
                      EvalRunResponse, ExecRequest, ExecResponse,
                      GenerateBenchmarkRequest, GraphResponse, HistoryResponse,
                      HistoryTurn, IndexRequest, IndexResponse, IssueDetail,
                      IssueSummary, JudgeRequest, JudgeResponse,
                      LLMDebugResponse, PullRequestDetail, QueryRequest,
                      QueryResponse, ReindexRequest, ReindexResponse,
                      RepositorySummaryResponse, SessionInfo)
from .search import HybridSearchIndex

app = FastAPI(title="GitHub Repository Copilot — Agentic Repository Intelligence Platform")

app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)

session_store = persistence.get_session_store()


@dataclass
class RepoState:
    repo_url: str
    local_dir: str
    head_sha: str


@dataclass
class Session:
    repos: Dict[str, RepoState] = field(default_factory=dict)   # repo_url -> RepoState
    chunks: List[Chunk] = field(default_factory=list)
    index: Optional[HybridSearchIndex] = None
    conversation: "memory.ConversationMemory" = field(default_factory=lambda: memory.ConversationMemory())


SESSIONS: Dict[str, Session] = {}


def _ingest_repo(repo_url: str, branch: Optional[str] = None) -> Tuple[RepoState, List[Chunk]]:
    local_dir = ingest.clone_repository(repo_url, branch)
    source_files = ingest.walk_repository(local_dir)
    if not source_files:
        shutil.rmtree(local_dir, ignore_errors=True)
        raise HTTPException(400, f"No supported files found in {repo_url}")

    chunks: List[Chunk] = []
    for sf in source_files:
        chunks += chunking.chunk_file(repo_url, sf.path, sf.language, sf.content)
    ingest.enrich_chunks_with_git_metadata(local_dir, chunks)

    head_sha = ingest.get_repo_head_sha(local_dir)
    return RepoState(repo_url=repo_url, local_dir=local_dir, head_sha=head_sha), chunks


def _rebuild_index(session: Session):
    if session.index:
        session.index.close()
    session.index = HybridSearchIndex(session.chunks)


def _get_session(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Unknown session_id. Call /api/index first.")
    return session


def _citations_from_chunks(chunks: List[Chunk]) -> List[Citation]:
    out = []
    for c in chunks:
        preview = "\n".join(c.content.strip().splitlines()[:3])
        out.append(Citation(
            file=c.file, start_line=c.start_line, end_line=c.end_line,
            symbol=c.symbol, snippet=preview, commit_sha=c.commit_sha, author=c.author,
        ))
    return out


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

@app.post("/api/index", response_model=IndexResponse)
def index_repository(req: IndexRequest):
    start = time.time()
    repo_state, chunks = _ingest_repo(req.repo_url, req.branch)

    session_id = str(uuid.uuid4())
    session = Session(repos={req.repo_url: repo_state}, chunks=chunks)
    _rebuild_index(session)
    SESSIONS[session_id] = session
    session_store.create_session(session_id, [req.repo_url])

    return IndexResponse(
        session_id=session_id, repo_urls=[req.repo_url],
        files_indexed=len({c.file for c in chunks}), chunks_indexed=len(chunks),
        elapsed_seconds=round(time.time() - start, 2), head_sha=repo_state.head_sha,
    )


@app.post("/api/sessions/add_repo", response_model=IndexResponse)
def add_repo_to_session(req: AddRepoRequest):
    """Multi-repository search: add another repo into an existing session so
    queries retrieve across all of them at once."""
    start = time.time()
    session = _get_session(req.session_id)
    if req.repo_url in session.repos:
        raise HTTPException(400, f"{req.repo_url} is already in this session.")

    repo_state, new_chunks = _ingest_repo(req.repo_url, req.branch)
    session.repos[req.repo_url] = repo_state
    session.chunks += new_chunks
    _rebuild_index(session)
    session_store.create_session(req.session_id, list(session.repos.keys()))

    return IndexResponse(
        session_id=req.session_id, repo_urls=list(session.repos.keys()),
        files_indexed=len({c.file for c in session.chunks}), chunks_indexed=len(session.chunks),
        elapsed_seconds=round(time.time() - start, 2), head_sha=repo_state.head_sha,
    )


@app.post("/api/reindex", response_model=ReindexResponse)
def reindex_repository(req: ReindexRequest):
    """Incremental reindexing: only re-chunks files that changed since the last
    indexed commit (via git diff), instead of re-walking the whole repo."""
    session = _get_session(req.session_id)
    repo_state = session.repos.get(req.repo_url)
    if not repo_state:
        raise HTTPException(404, f"{req.repo_url} is not part of this session.")

    incremental.unshallow(repo_state.local_dir)
    diff = incremental.diff_since(repo_state.local_dir, repo_state.head_sha)

    changed_paths = set(diff.added) | set(diff.modified) | {new for _old, new in diff.renamed}
    removed_paths = set(diff.deleted) | {old for old, _new in diff.renamed}

    # drop chunks for removed/changed files belonging to this repo
    session.chunks = [
        c for c in session.chunks
        if not (c.repo == req.repo_url and (c.file in removed_paths or c.file in changed_paths))
    ]

    new_chunks: List[Chunk] = []
    for rel_path in changed_paths:
        import os
        abs_path = os.path.join(repo_state.local_dir, rel_path)
        if not os.path.exists(abs_path) or not ingest.is_supported_file(rel_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError:
            continue
        language = ingest._detect_language(rel_path)
        new_chunks += chunking.chunk_file(req.repo_url, rel_path, language, content)

    ingest.enrich_chunks_with_git_metadata(repo_state.local_dir, new_chunks)
    session.chunks += new_chunks
    repo_state.head_sha = diff.new_head_sha
    _rebuild_index(session)

    return ReindexResponse(
        session_id=req.session_id, added=diff.added, modified=diff.modified,
        deleted=diff.deleted, chunks_reindexed=len(new_chunks), new_head_sha=diff.new_head_sha,
    )


@app.get("/api/sessions", response_model=List[SessionInfo])
def list_sessions():
    return [SessionInfo(**s) for s in session_store.list_sessions()]


# ---------------------------------------------------------------------------
# Query (retrieval + planning + generation)
# ---------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
def query_repository(req: QueryRequest):
    session = _get_session(req.session_id)
    if not session.index:
        raise HTTPException(400, "Session has no indexed repositories.")

    plan = planner.plan(req.question)
    retrieved = session.index.search(req.question, plan, top_k=req.top_k)

    repo_context = session.conversation.context_block()
    answer, citations, mode, debug = llm.generate_answer(
        req.question, retrieved, session.conversation.history, repo_context=repo_context,
    )

    session.conversation.add_turn(req.question, answer, [c.file for c in retrieved])
    session_store.append_turn(req.session_id, req.question, answer)

    return QueryResponse(
        answer=answer, citations=_citations_from_chunks(retrieved), mode=mode,
        retrieved_chunks=len(retrieved), llm_debug=LLMDebugResponse(**debug),
        plan_sources=plan.sources,
        retrieval_strategy=session.index.last_retrieval_strategy,
        boosted_files=session.index.last_boosted_files,
    )


def _build_tool_context(session: Session) -> RepoToolContext:
    local_dirs = {url: state.local_dir for url, state in session.repos.items()}
    return RepoToolContext(session.index, session.chunks, local_dirs)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """The agentic entry point: `AGENT_FRAMEWORK=adk` (default) tries the
    Google ADK Repository Agent first — real tool-calling, the model decides
    which of search_repository/read_repository_file/get_repository_tree/
    find_symbol/get_dependencies to call and in what order. If ADK isn't
    configured (no NVIDIA_NIM_API_KEY) or construction fails for any reason, this
    falls back to `AGENT_FRAMEWORK=direct`: the LangGraph workflow
    (intent -> retrieve -> validate -> generate -> citations) called directly,
    which is what /api/query already uses under the hood — so /api/chat always
    answers, with or without ADK's model configured.
    """
    session = _get_session(req.session_id)
    if not session.index:
        raise HTTPException(400, "Session has no indexed repositories.")

    provider_debug = providers.get_active_provider_debug()

    with track_query(req.session_id, req.message, provider_debug.provider, provider_debug.model) as run:
        agent_framework_used = "direct"
        tool_calls: List[str] = []

        if config.AGENT_FRAMEWORK == "adk":
            try:
                ctx = _build_tool_context(session)
                adk_result = run_adk_agent(ctx, req.message)
                # ADK's own answer text, but still run our own retrieval pass so
                # the citations returned to the client are grounded in the same
                # chunk metadata (file/lines/commit) as the rest of the app,
                # rather than re-parsing them out of the model's free text.
                retrieved = session.index.search(req.message, top_k=req.top_k)
                result = {
                    "answer": adk_result["answer"] or llm._extractive_answer(retrieved),
                    "citations": _citations_from_chunks(retrieved),
                    "mode": "generative" if adk_result["answer"] else "extractive",
                    "intent": "GENERAL",
                    "retry_count": 0,
                }
                tool_calls = adk_result["tool_calls"]
                agent_framework_used = "adk"
            except ADKAgentUnavailable:
                result = None  # fall through to direct workflow below
        else:
            result = None

        if result is None:
            wf_result = run_workflow(
                session.index, req.session_id, req.message,
                get_history=lambda sid: session.conversation.history,
            )
            result = {
                "answer": wf_result["answer"],
                "citations": [
                    Citation(file=c["file"], start_line=c["start_line"], end_line=c["end_line"],
                             symbol=c["symbol"], snippet="")
                    for c in wf_result["citations"]
                ],
                "mode": wf_result["mode"],
                "intent": wf_result["intent"],
                "retry_count": wf_result.get("retry_count", 0),
            }
            agent_framework_used = "direct"

        session.conversation.add_turn(req.message, result["answer"],
                                       [c.file for c in result["citations"]])
        session_store.append_turn(req.session_id, req.message, result["answer"])

        run.log(
            mode=result["mode"], intent=result["intent"],
            agent_framework=agent_framework_used, retrieved_chunks=len(result["citations"]),
            retry_count=result["retry_count"],
        )

    return ChatResponse(
        answer=result["answer"], citations=result["citations"], mode=result["mode"],
        intent=result["intent"], agent_framework=agent_framework_used,
        tool_calls=tool_calls, retry_count=result["retry_count"],
    )


@app.get("/api/sessions/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str):
    _get_session(session_id)
    turns = [HistoryTurn(question=q, answer=a) for q, a in session_store.get_history(session_id)]
    return HistoryResponse(session_id=session_id, turns=turns)


@app.get("/api/debug/llm", response_model=LLMDebugResponse)
def debug_llm():
    return LLMDebugResponse(**providers.get_active_provider_debug().__dict__)


# ---------------------------------------------------------------------------
# Repository graph & architecture
# ---------------------------------------------------------------------------

@app.get("/api/sessions/{session_id}/graph", response_model=GraphResponse)
def get_graph(session_id: str):
    session = _get_session(session_id)
    graph = build_import_graph(session.chunks)
    return GraphResponse(mermaid=to_mermaid(graph), stats=graph_stats(graph))


@app.get("/api/sessions/{session_id}/architecture", response_model=ArchitectureResponse)
def get_architecture(session_id: str):
    session = _get_session(session_id)
    return ArchitectureResponse(**architecture_detect.detect(session.chunks))


_SUMMARY_EVIDENCE_FILENAMES = (
    "readme", "requirements.txt", "pyproject.toml", "package.json", "setup.py",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml", "makefile",
    "main.py", "app.py", "index.js", "index.ts", "server.py",
)


def _gather_summary_evidence(session: Session) -> List[Chunk]:
    """Repository Summary Agent's retrieval step: pulls README, dependency
    manifests, Docker/deployment files, and likely entrypoints directly by
    filename (cheap, deterministic, no LLM call needed to find them), plus a
    hybrid-search pass for 'entry point', 'API routes', 'configuration' to
    catch anything the filename heuristic misses."""
    evidence: List[Chunk] = []
    seen_files = set()
    for c in session.chunks:
        fname = c.file.lower().split("/")[-1]
        if any(fname.startswith(marker) or marker in fname for marker in _SUMMARY_EVIDENCE_FILENAMES):
            if c.file not in seen_files:
                evidence.append(c)
                seen_files.add(c.file)

    for query in ("application entry point", "API routes and endpoints", "configuration and environment variables"):
        for c in session.index.search(query, top_k=2):
            if c.file not in seen_files:
                evidence.append(c)
                seen_files.add(c.file)

    return evidence[:25]  # keep the context small — per-file dedupe already applied


SUMMARY_INSTRUCTIONS = (
    "Produce a structured repository summary with these exact sections: "
    "Repository Purpose, Main Functionality, Architecture, Important Components, "
    "Technology Stack, Entry Point, APIs, Data Stores, External Services, "
    "How the Components Interact. Every factual claim must be grounded in the "
    "provided sources — if a section can't be supported by the sources, say "
    "'Not evident from the indexed repository content' for that section rather "
    "than inventing functionality."
)


@app.get("/api/sessions/{session_id}/summary", response_model=RepositorySummaryResponse)
def get_repository_summary(session_id: str):
    session = _get_session(session_id)
    if not session.index:
        raise HTTPException(400, "Session has no indexed repositories.")

    evidence = _gather_summary_evidence(session)
    arch = architecture_detect.detect(session.chunks)
    repo_context = (
        f"Detected architecture patterns: {', '.join(arch['likely_architecture_patterns'])}\n"
        f"Detected frameworks/infra: {', '.join(arch['detected_frameworks_and_infra'])}"
    )

    answer, citations, mode, _debug = llm.generate_answer(
        SUMMARY_INSTRUCTIONS, evidence, [], repo_context=repo_context,
    )

    return RepositorySummaryResponse(
        repo_urls=list(session.repos.keys()), summary=answer,
        citations=_citations_from_chunks(evidence),
        architecture_patterns=arch["likely_architecture_patterns"],
        detected_frameworks=arch["detected_frameworks_and_infra"], mode=mode,
    )


@app.get("/api/repository/summary", response_model=RepositorySummaryResponse)
def get_repository_summary_alias(session_id: str):
    """Alias matching the spec's literal `GET /api/repository/summary` path.
    This codebase's existing convention keys everything off `session_id`
    (a session already represents one indexed-repository scope, potentially
    spanning multiple repos via /api/sessions/add_repo), so `repo_id` and
    `session_id` are treated as the same identifier here rather than adding a
    second, parallel ID space."""
    return get_repository_summary(session_id)


# ---------------------------------------------------------------------------
# GitHub integration (issues / PRs / commits)
# ---------------------------------------------------------------------------

@app.get("/api/github/issues", response_model=List[IssueSummary])
def github_list_issues(repo_url: str, state: str = "all", limit: int = 20):
    try:
        return [IssueSummary(**i) for i in github_integration.list_issues(repo_url, state, limit)]
    except GitHubAPIError as e:
        raise HTTPException(502, str(e))


@app.get("/api/github/issues/{issue_number}", response_model=IssueDetail)
def github_get_issue(repo_url: str, issue_number: int):
    try:
        return IssueDetail(**github_integration.get_issue(repo_url, issue_number))
    except GitHubAPIError as e:
        raise HTTPException(502, str(e))


@app.get("/api/github/pulls/{pr_number}", response_model=PullRequestDetail)
def github_get_pull_request(repo_url: str, pr_number: int):
    try:
        return PullRequestDetail(**github_integration.get_pull_request(repo_url, pr_number))
    except GitHubAPIError as e:
        raise HTTPException(502, str(e))


@app.get("/api/github/commits", response_model=List[CommitSummary])
def github_list_commits(repo_url: str, path: Optional[str] = None, limit: int = 20):
    try:
        return [CommitSummary(**c) for c in github_integration.list_commits(repo_url, path, limit)]
    except GitHubAPIError as e:
        raise HTTPException(502, str(e))


# ---------------------------------------------------------------------------
# Code execution (off by default)
# ---------------------------------------------------------------------------

@app.post("/api/exec", response_model=ExecResponse)
def exec_command(req: ExecRequest):
    session = _get_session(req.session_id)
    repo_state = session.repos.get(req.repo_url)
    if not repo_state:
        raise HTTPException(404, f"{req.repo_url} is not part of this session.")
    try:
        result = code_exec.run(repo_state.local_dir, req.command_key, req.extra_args)
        return ExecResponse(**result.__dict__)
    except code_exec.CodeExecutionDisabled as e:
        raise HTTPException(403, str(e))
    except code_exec.UnknownCommand as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.post("/api/eval/run", response_model=EvalRunResponse)
def eval_run(req: EvalRunRequest):
    session = _get_session(req.session_id)
    report = eval_stub.evaluate(session.index, req.labeled_queries, k=req.k)
    return EvalRunResponse(
        n_queries=report["n_queries"], recall_at_k=report["recall@k"],
        mrr=report["mrr"], ndcg_at_k=report["ndcg@k"], k=report["k"],
    )


@app.post("/api/eval/generate_benchmark")
def eval_generate_benchmark(req: GenerateBenchmarkRequest):
    session = _get_session(req.session_id)
    if not providers.is_configured():
        raise HTTPException(400, "Benchmark generation needs a configured LLM provider.")
    benchmark = eval_dataset_gen.generate_benchmark(session.chunks, max_files=req.max_files)
    return {"benchmark": benchmark, "n_generated": len(benchmark)}


@app.post("/api/eval/judge", response_model=JudgeResponse)
def eval_judge_answer(req: JudgeRequest):
    session = _get_session(req.session_id)
    retrieved = session.index.search(req.question, top_k=config.TOP_K)
    result = eval_judge.judge_answer(req.question, retrieved, req.answer)
    if result is None:
        return JudgeResponse(available=False)
    return JudgeResponse(**result.__dict__, available=True)


@app.post("/api/eval/deepeval")
def eval_deepeval_run(req: JudgeRequest):
    session = _get_session(req.session_id)
    retrieved = session.index.search(req.question, top_k=config.TOP_K)
    try:
        return eval_deepeval.evaluate_answer(req.question, req.answer, retrieved)
    except eval_deepeval.DeepEvalUnavailable as e:
        raise HTTPException(400, str(e))


@app.post("/api/eval/ragas")
def eval_ragas_run(req: JudgeRequest):
    session = _get_session(req.session_id)
    retrieved = session.index.search(req.question, top_k=config.TOP_K)
    try:
        return eval_ragas.evaluate_answer(req.question, req.answer, retrieved)
    except eval_ragas.RagasUnavailable as e:
        raise HTTPException(400, str(e))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "active_sessions": len(SESSIONS),
        "llm_provider": config.LLM_PROVIDER,
        "agent_framework": config.AGENT_FRAMEWORK,
        "adk_configured": bool(config.NVIDIA_NIM_API_KEY),
        "mlflow_enabled": config.ENABLE_MLFLOW,
        "langsmith": langsmith_setup.status(),
    }


@app.get("/api/ready")
def ready():
    """Readiness probe for container orchestration: distinct from /api/health
    (which reports 'ok' as long as the process is up) — this checks that the
    process can actually do its job, i.e. import and construct the search
    engine's dependencies. No repository needs to be indexed for this to pass."""
    try:
        from .embeddings import get_embedder
        get_embedder()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"Not ready: {e}")
