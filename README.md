# GitHub Repository Copilot — Agentic Repository Intelligence Platform

Chat with a GitHub repository: index it, then ask questions about what it does, how
it's architected, how to run it, and how specific code works — with grounded citations
on every answer. Built incrementally on a working hybrid-RAG core, now wrapped in a
real LangGraph workflow and a real Google ADK tool-calling agent, backed by NVIDIA AI
Endpoints (no self-hosted model) with a pluggable multi-provider fallback.

**How to read this README:** every claim below falls into one of three buckets, and
I've kept them visually distinct on purpose rather than blur them into one "it works"
narrative:
- ✅ **Verified** — built and actually run in this session, against a real cloned
  repository or a real local service (MLflow, a live FastAPI server), with the
  command/output to prove it in the PR/commit history.
- 🔧 **Real but unexecuted** — real code/config that was written and reviewed, but
  needs infrastructure this sandbox doesn't have (a Docker daemon, GCP credentials,
  a live NVIDIA API key) to actually run. Not fake, not tested end-to-end.
- 📋 **Documented only** — instructions/templates, not code that runs by itself.

---

## What's actually new here vs. the existing hybrid-RAG core

The repository already had a solid, tested foundation: repo cloning, tree-sitter +
Python AST + Markdown chunking, BM25 + TF-IDF/dense hybrid search with Qdrant, RRF,
reranking, citations, GitHub issues/PRs/commits, incremental indexing, SQLite session
persistence, and RAGAS/DeepEval hooks. None of that was rewritten. What's new in this
pass:

| Layer | What it is | Status |
|---|---|---|
| `app/llm_gateway/` | Abstract `LLMProvider` interface + `ChatNVIDIA`-backed implementation | ✅ constructs correctly, raises clean errors without a key |
| `app/langchain_rag/` | `Chunk` → LangChain `Document` conversion + `HybridRepoRetriever(BaseRetriever)` | ✅ verified against a real repo via `.invoke()` |
| `app/agent/workflow.py` | Real LangGraph `StateGraph`: intent → retrieve → validate → generate → citations, bounded retry | ✅ verified end-to-end, including retry-loop termination on an empty index |
| `app/agent/tools.py` | 5 typed repository tools (search, read file, tree, find symbol, dependencies) | ✅ all 5 tested, including path-traversal blocking |
| `app/agent/repository_agent.py` | Google ADK `Agent` with tool-calling, NVIDIA model via LiteLLM | ✅ Agent object constructs correctly with all tools + model bound; 🔧 not invoked live (no NVIDIA key here) |
| `app/observability/mlflow_tracking.py` | Per-query MLflow run logging | ✅ verified — queried the SQLite store afterward and confirmed real metrics |
| `app/observability/langsmith_setup.py` | LangSmith status check (tracing itself is env-var-activated) | ✅ (there's little to test — it's a status read) |
| `POST /api/chat` | ADK-first, automatic fallback to the direct LangGraph workflow | ✅ hit through a live server, confirmed correct fallback behavior |
| `GET /api/sessions/{id}/summary`, `GET /api/repository/summary` | Repository Summary Agent | ✅ hit through a live server, grounded README/manifest/entrypoint evidence |
| `GET /api/ready` | Readiness probe | ✅ |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci-cd.yml` | Containerization + CI | 🔧 written, YAML/syntax-validated, not build/run-tested |
| `docs/DEPLOYMENT.md` | Cloud Run deployment | 📋 real `gcloud` commands, unexecuted |

---

## Architecture

```
Frontend (static, unchanged)
        │
        ▼
     FastAPI  ──────────────────────────────────────────┐
        │                                                │
        ▼                                                │
  Google ADK Repository Agent  (AGENT_FRAMEWORK=adk)      │ AGENT_FRAMEWORK=direct
  - tool-calling: search_repository, read_repository_file,│ or ADK unavailable/
    get_repository_tree, find_symbol, get_dependencies    │ construction failed
        │                                                │
        ▼                                                ▼
              LangGraph Workflow (app/agent/workflow.py)
   intent_detection → retrieve → validate_context ⇄ (bounded retry)
                                        │
                                        ▼
                                    generate → build_citations
        │
        ▼
   LangChain Retrieval Layer (app/langchain_rag/)
   HybridRepoRetriever(BaseRetriever) → Document[]
        │
        ▼
   Hybrid Retrieval (app/search.py — unchanged, pre-existing)
   BM25  +  Dense (Qdrant)  →  weighted RRF  →  dedupe  →  rerank
        │
        ▼
   LLM: ChatNVIDIA (app/llm_gateway/) or the existing multi-provider
   gateway (app/llm.py: Anthropic/OpenAI/NVIDIA-compat/DeepSeek),
   with extractive fallback if nothing is configured
        │
        ▼
   Grounded Answer + Citations (file:start-end, commit SHA, author)
```

`/api/chat` is the agentic entry point described above. `/api/query` (pre-existing)
still works exactly as before — it calls the retrieval+generation path directly
without going through LangGraph, for callers that don't need agentic routing.

---

## Technology stack

| Concern | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, Pydantic v2 |
| Agent framework | Google ADK (`google-adk[extensions]`) |
| Orchestration | LangGraph (`StateGraph`) |
| Retrieval abstraction | LangChain (`Document`, `BaseRetriever`) |
| LLM (primary, agentic path) | NVIDIA AI Endpoints via `ChatNVIDIA` — no self-hosted model |
| LLM (fallback path, pre-existing) | Anthropic, OpenAI, NVIDIA (OpenAI-compatible), DeepSeek |
| Hybrid retrieval | BM25 (`rank-bm25`) + dense (TF-IDF or OpenAI embeddings) via Qdrant, weighted RRF |
| Reranking | Lexical (default) or Cohere cross-encoder |
| Chunking | Python AST, tree-sitter (Java/Go/Rust/TS/JS/C/C++), Markdown headings, generic fallback |
| Vector store | Qdrant — embedded in-memory/on-disk by default, or a real server (`qdrant_server`) |
| Experiment tracking | MLflow (local SQLite by default) |
| Tracing | LangSmith (optional, env-var activated) |
| Evaluation | Recall@K/MRR/nDCG (built-in), RAGAS, DeepEval |
| Session persistence | SQLite (adapter-shaped for Redis/Postgres later) |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Cloud target | Google Cloud Run (LLM stays external) |

---

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Verified fresh: `pip install -r requirements.txt` into a **clean, empty virtualenv**
(not just this already-loaded sandbox) produced a fully working app with all 27 tests
passing — see [Testing](#testing) below.

With no `.env` filled in at all, the app still answers questions: retrieval runs on
BM25 + TF-IDF (no key needed), `/api/chat` automatically falls back from
`AGENT_FRAMEWORK=adk` to the direct LangGraph workflow (since ADK's model needs a
`NVIDIA_API_KEY` to plan tool calls), and answers come back in extractive mode —
labeled as such, not silently degraded.

Open `frontend/index.html` (unchanged from the existing hybrid-RAG build) and point it
at `http://localhost:8000`.

### Configuring providers

See `backend/.env.example` for the full list. The two provider paths that matter most:

```bash
# Agentic path (ADK + ChatNVIDIA) — needs this to do real tool-calling:
NVIDIA_API_KEY=nvapi-...
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
AGENT_FRAMEWORK=adk

# Fallback path (pre-existing, works today, either provider set):
LLM_PROVIDER=anthropic   # or openai | nvidia | deepseek
ANTHROPIC_API_KEY=sk-ant-...
```

Both can be set at once — `/api/chat` tries ADK first and falls back automatically;
`/api/query` always uses the fallback path.

---

## API reference

All pre-existing endpoints from the hybrid-RAG core are unchanged (indexing,
multi-repo sessions, incremental reindex, graph, architecture, GitHub integration,
code execution, evaluation — see inline FastAPI docs at `/docs` once running). New
in this pass:

```
POST /api/chat
  { "session_id": "...", "message": "How does auth work?", "top_k": 5 }
  → { answer, citations[], mode, intent, agent_framework, tool_calls[], retry_count }

  Tries the Google ADK Repository Agent (real tool-calling) if AGENT_FRAMEWORK=adk
  and NVIDIA_API_KEY is set; otherwise (or on any ADK construction failure) falls
  back to the LangGraph workflow directly. agent_framework in the response tells
  you which path actually answered.

GET /api/sessions/{session_id}/summary
GET /api/repository/summary?session_id=...        (alias matching the spec's literal path)
  → { repo_urls[], summary, citations[], architecture_patterns[], detected_frameworks[], mode }

  Repository Summary Agent: gathers README/manifest/Dockerfile/entrypoint evidence
  by filename + a few targeted searches, combines with the existing heuristic
  architecture detector, and generates a structured summary (Purpose, Main
  Functionality, Architecture, Important Components, Technology Stack, Entry Point,
  APIs, Data Stores, External Services, How Components Interact) — grounded in the
  gathered evidence, explicitly saying "Not evident from the indexed repository
  content" for anything it can't support.

GET /api/health
  → now also reports agent_framework, adk_configured, mlflow_enabled, langsmith status

GET /api/ready
  → readiness probe: can the process actually construct its retrieval dependencies
```

---

## Testing

```bash
cd backend
pytest tests/ -v
```

27 tests, covering:
- Python AST + multi-file chunking (`test_retrieval.py`)
- Hybrid search correctness and dedup (`test_retrieval.py`)
- The LangChain `HybridRepoRetriever` wrapper, including metadata completeness
- Intent classification against the spec's example questions (`test_langgraph_workflow.py`)
- The LangGraph workflow's citations and — importantly — that its retry loop actually
  **terminates** on a genuinely empty index instead of looping forever
- All 5 ADK tools, including a real path-traversal-blocking check
- ADK's own graceful failure (`ADKAgentUnavailable`) with no `NVIDIA_API_KEY`
- FastAPI endpoints end-to-end via `TestClient`: `/api/chat`'s automatic ADK→direct
  fallback, the summary endpoint (both paths), health/ready, and a 404 on an unknown
  session

All 27 pass both in this sandbox and in a from-scratch clean virtualenv — the clean-venv
run matters more, since it's the only one that actually proves `requirements.txt` is
sufficient on its own rather than quietly relying on something already installed.

CI (`.github/workflows/ci-cd.yml`) runs the same `pytest tests/` on every PR — 🔧 written
and internally consistent with the local commands above, not run on a real GitHub
Actions runner from this sandbox.

---

## Evaluation

Unchanged from the existing core, still real:

```bash
POST /api/eval/run            # Recall@K, MRR, nDCG against a labeled query set you provide
POST /api/eval/generate_benchmark   # needs an LLM provider
POST /api/eval/judge          # LLM-as-judge, needs an LLM provider
POST /api/eval/deepeval        # needs OPENAI_API_KEY + `pip install -r requirements-eval.txt`
POST /api/eval/ragas           # same; see that file for a known ragas/langchain-community
                                # dependency-pinning issue and its workaround
```

Evaluation is never run automatically per-query — only on these explicit calls, per the
spec's own "do not run RAGAS/DeepEval on every query" requirement. MLflow tracking
(separate from RAGAS/DeepEval) *does* run on every `/api/chat` call, logging latency,
retrieved-chunk count, mode, and intent — that's lightweight enough to always run, and
is what makes the "Model Evaluation" MLflow requirement real rather than a manual-only
afterthought:

```bash
mlflow ui --backend-store-uri sqlite:///./backend/mlflow.db
```

---

## Docker

```bash
cp backend/.env.example backend/.env   # fill in a provider
docker compose up --build
```

🔧 **Written, `docker-compose.yml` YAML-validated, `Dockerfile` hand-reviewed against
the same dependency list that passed the clean-venv test above — not actually built or
run**, since this sandbox has no Docker daemon. Brings up `qdrant`, `mlflow`, and the
FastAPI `backend` together; the backend is configured via env vars to point at the
sibling `qdrant`/`mlflow` containers instead of its embedded/local defaults. Non-root
user, healthchecks, and a `.dockerignore` that keeps `.env`/databases/caches out of the
image are all in place — standard hardening, not exotic.

---

## Cloud deployment (Google Cloud Run)

📋 See [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) — real, reviewed `gcloud` commands,
explicitly marked as unexecuted (no GCP credentials or network access to any
`*.googleapis.com` endpoint in the sandbox this was built in). The LLM stays external
throughout (NVIDIA AI Endpoints or the existing provider gateway) — Cloud Run only ever
runs the FastAPI container, never a model.

---

## Security

- **Prompt injection:** the ADK agent's system instruction explicitly tells the model
  to treat all repository content as untrusted data, never as instructions to follow —
  see the `instruction=` string in `app/agent/repository_agent.py`.
- **Path traversal:** `read_repository_file` resolves and verifies every path stays
  inside the cloned repo root before reading — tested (`test_read_repository_file_blocks_path_traversal`).
- **No arbitrary code execution via tools:** none of the 5 ADK tools shell out or eval
  anything. The separate `code_exec.py` (pre-existing, allowlisted pytest/lint runner)
  is unrelated to the agent's tools and stays off by default (`ENABLE_CODE_EXECUTION=false`).
- **File size / read limits:** `read_repository_file` caps reads at 200KB; the
  pre-existing ingestion pipeline already caps individual file size and applies
  ignore-lists for binaries/build artifacts/`node_modules`/etc.
- **Secrets:** never logged, never sent to the frontend; MLflow logging truncates
  question text to 250 chars and only logs numeric metrics/short string params —
  no repository content or API keys ever hit the tracking store.

---

## Known limitations (stated plainly, not as "future work" dressed up)

- The ADK agent's actual tool-calling behavior (which tools it chooses, in what order)
  is unverified live — I confirmed the `Agent` object constructs correctly with a real
  NVIDIA/LiteLLM model attached, but had no live `NVIDIA_API_KEY` to run an actual
  conversation through it. The `direct` fallback path (LangGraph workflow, no ADK) is
  fully verified and is what answers by default without that key.
- Docker build/run and the CI workflow are real files, not tested executions.
- Cloud Run deployment is documentation, not a deployed instance.
- MLflow/session persistence both default to SQLite-on-local-disk, which doesn't
  survive a Cloud Run cold start on a new instance — see the upgrade path table in
  `docs/DEPLOYMENT.md`.
- No screenshots included — there's no running deployment to screenshot from this
  sandbox; the existing `frontend/index.html` is unchanged from the prior hybrid-RAG
  build and its behavior is documented in the earlier README history, not re-verified
  here since the frontend itself wasn't touched in this pass.
- RAGAS has a known real dependency-pinning issue in some environments (see
  `requirements-eval.txt`) — not something this pass fixed, since it's upstream.

## Repo layout (new pieces only — everything else is the pre-existing hybrid-RAG core)

```
backend/
  app/
    llm_gateway/
      base.py                 LLMProvider abstract interface
      nvidia_provider.py       ChatNVIDIA-backed implementation
    langchain_rag/
      documents.py             Chunk -> Document conversion
      retriever.py              HybridRepoRetriever(BaseRetriever)
    agent/
      state.py                  RepoWorkflowState (TypedDict)
      intent.py                  deterministic-first intent classification
      workflow.py                 LangGraph StateGraph
      tools.py                     5 typed ADK repository tools
      repository_agent.py           Google ADK Agent construction + invocation
    observability/
      mlflow_tracking.py           per-query MLflow run logging
      langsmith_setup.py            LangSmith status check
  tests/                              27 tests, see Testing above
  Dockerfile / .dockerignore
docker-compose.yml
docs/DEPLOYMENT.md
.github/workflows/ci-cd.yml
```
