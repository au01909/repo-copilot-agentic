from typing import Dict, List, Optional
from pydantic import BaseModel


class IndexRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = None


class IndexResponse(BaseModel):
    session_id: str
    repo_urls: List[str]
    files_indexed: int
    chunks_indexed: int
    elapsed_seconds: float
    head_sha: Optional[str] = None


class AddRepoRequest(BaseModel):
    session_id: str
    repo_url: str
    branch: Optional[str] = None


class ReindexRequest(BaseModel):
    session_id: str
    repo_url: str


class ReindexResponse(BaseModel):
    session_id: str
    added: List[str]
    modified: List[str]
    deleted: List[str]
    chunks_reindexed: int
    new_head_sha: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = None


class Citation(BaseModel):
    file: str
    start_line: int
    end_line: int
    symbol: Optional[str] = None
    snippet: str
    commit_sha: Optional[str] = None
    author: Optional[str] = None


class LLMDebugResponse(BaseModel):
    provider: str
    model: str
    api_key_present: bool
    api_key_prefix: Optional[str] = None
    latency_seconds: Optional[float] = None
    status: str
    error: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    mode: str  # "generative" or "extractive"
    retrieved_chunks: int
    llm_debug: LLMDebugResponse
    plan_sources: List[str] = []


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    mode: str              # "generative" | "extractive"
    intent: str
    agent_framework: str   # "adk" | "direct"
    tool_calls: List[str] = []
    retry_count: int = 0


class RepositorySummaryResponse(BaseModel):
    repo_urls: List[str]
    summary: str
    citations: List[Citation]
    architecture_patterns: List[str]
    detected_frameworks: List[str]
    mode: str


class HistoryTurn(BaseModel):
    question: str
    answer: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: List[HistoryTurn]


class SessionInfo(BaseModel):
    session_id: str
    repo_urls: List[str]
    created_at: float


class GraphResponse(BaseModel):
    mermaid: str
    stats: Dict


class ArchitectureResponse(BaseModel):
    likely_architecture_patterns: List[str]
    detected_frameworks_and_infra: List[str]
    signal_strength: Dict


class IssueSummary(BaseModel):
    number: int
    title: str
    state: str
    is_pull_request: bool
    created_at: str


class IssueDetail(BaseModel):
    number: int
    title: str
    state: str
    body: str
    author: str
    created_at: str
    closed_at: Optional[str] = None
    labels: List[str]
    is_pull_request: bool
    comments: List[Dict]


class PullRequestDetail(BaseModel):
    number: int
    title: str
    state: str
    body: str
    author: str
    merged: bool
    merged_at: Optional[str] = None
    base: str
    head: str
    files_changed: List[Dict]


class CommitSummary(BaseModel):
    sha: str
    message: str
    author: str
    date: Optional[str] = None


class ExecRequest(BaseModel):
    session_id: str
    repo_url: str
    command_key: str
    extra_args: Optional[List[str]] = None


class ExecResponse(BaseModel):
    command: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class EvalRunRequest(BaseModel):
    session_id: str
    labeled_queries: List[Dict]
    k: int = 10


class EvalRunResponse(BaseModel):
    n_queries: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


class GenerateBenchmarkRequest(BaseModel):
    session_id: str
    max_files: int = 25


class JudgeRequest(BaseModel):
    session_id: str
    question: str
    answer: str


class JudgeResponse(BaseModel):
    correct: Optional[int] = None
    useful: Optional[int] = None
    complete: Optional[int] = None
    grounded: Optional[int] = None
    notes: Optional[str] = None
    available: bool = True
