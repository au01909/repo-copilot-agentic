"""Proves the planner is an active participant in retrieval (search.py boosts
plan-preferred sources) rather than a telemetry-only side channel."""
import pytest

from app import planner
from app.chunking import Chunk
from app.search import HybridSearchIndex

REPO = "test-repo"


def _chunk(file: str, content: str, symbol: str = None) -> Chunk:
    return Chunk(repo=REPO, file=file, language="python", content=content,
                 start_line=1, end_line=len(content.splitlines()) or 1, symbol=symbol)


@pytest.fixture(scope="module")
def index():
    chunks = [
        _chunk("README.md", "This repository implements a small web service for managing tasks."),
        _chunk("app/intent.py", "def classify_intent(question): return detect_source(question)"),
        _chunk("app/tools.py", "def search_repository(query, top_k): return index.search(query)"),
        _chunk("tests/test_intent.py", "def test_classify_intent(): assert classify_intent('x')"),
        _chunk("app/repository_agent.py", "class RepositoryAgent: def run(self, question): pass"),
        _chunk("docs/architecture.md", "System architecture: request flows through main.py entrypoint into the agent workflow."),
        _chunk("main.py", "app = FastAPI()  # entrypoint wiring routes and the agent workflow"),
        _chunk("Dockerfile", "FROM python:3.10\nCOPY . /app\nCMD [\"uvicorn\", \"app.main:app\"]"),
        _chunk("docs/deployment.md", "Deployment guide: build the Docker image and run docker-compose up."),
        _chunk("requirements.txt", "fastapi\nuvicorn\nrank-bm25"),
        _chunk("backend/app/search.py", "class HybridSearchIndex: def search(self, query, plan=None, top_k=None): pass"),
    ]
    idx = HybridSearchIndex(chunks)
    yield idx
    idx.close()


def _rank_of(files, target):
    return next(i for i, f in enumerate(files) if f == target)


def test_repository_summary_question_ranks_readme_first(index):
    question = "What does this repository do?"
    plan = planner.plan(question)
    assert "readme" in plan.sources

    results = index.search(question, plan, top_k=11)  # full corpus, so rank order is meaningful
    files = [c.file for c in results]
    assert files[0] == "README.md"
    if "app/repository_agent.py" in files:
        assert _rank_of(files, "README.md") < _rank_of(files, "app/repository_agent.py")


def test_architecture_question_ranks_architecture_docs_before_tests(index):
    question = "Explain the architecture of this system"
    plan = planner.plan(question)
    assert "architecture" in plan.sources

    results = index.search(question, plan, top_k=11)  # full corpus, so rank order is meaningful
    files = [c.file for c in results]
    assert files[0] == "docs/architecture.md"
    if "tests/test_intent.py" in files:
        assert _rank_of(files, "docs/architecture.md") < _rank_of(files, "tests/test_intent.py")


def test_deployment_question_surfaces_dockerfile_and_docs(index):
    question = "How do I deploy this application?"
    plan = planner.plan(question)
    assert "deployment" in plan.sources

    results = index.search(question, plan, top_k=6)
    files = {c.file for c in results}
    assert "Dockerfile" in files or "docs/deployment.md" in files


def test_code_symbol_question_still_finds_the_right_file(index):
    question = "Explain HybridSearchIndex"
    plan = planner.plan(question)

    results = index.search(question, plan, top_k=5)
    files = [c.file for c in results]
    assert "backend/app/search.py" in files


def test_search_without_plan_is_backward_compatible(index):
    # old call signature: search(query, top_k=...) with no plan
    results = index.search("HybridSearchIndex", top_k=3)
    assert results
    assert index.last_retrieval_strategy == "hybrid_only"


def test_fallback_never_empties_results_when_preferred_source_missing(index):
    # "issues"/"pull_requests" have no file-glob patterns in this corpus at all
    plan = planner.Plan(sources=["issues"], reasoning="test")
    results = index.search("What does this repository do?", plan, top_k=3)
    assert results
