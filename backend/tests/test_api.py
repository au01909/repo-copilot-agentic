import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TEST_REPO_URL = "https://github.com/karpathy/micrograd.git"


@pytest.fixture(scope="module")
def indexed_session():
    resp = client.post("/api/index", json={"repo_url": TEST_REPO_URL})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_health_endpoint_reports_agent_and_observability_status():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "agent_framework" in body
    assert "mlflow_enabled" in body
    assert "langsmith" in body


def test_ready_endpoint():
    resp = client.get("/api/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_index_endpoint_returns_real_session(indexed_session):
    assert indexed_session  # non-empty session id


def test_chat_endpoint_falls_back_to_direct_workflow_without_nvidia_key(indexed_session):
    resp = client.post("/api/chat", json={
        "session_id": indexed_session, "message": "How does the Value class work?",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_framework"] == "direct"  # no NVIDIA_API_KEY in test env
    assert len(body["citations"]) > 0
    assert body["mode"] in ("generative", "extractive")


def test_summary_endpoint_returns_grounded_evidence(indexed_session):
    resp = client.get(f"/api/sessions/{indexed_session}/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["repo_urls"] == [TEST_REPO_URL]
    assert len(body["citations"]) > 0


def test_summary_alias_matches_spec_path(indexed_session):
    resp = client.get(f"/api/repository/summary?session_id={indexed_session}")
    assert resp.status_code == 200
    assert resp.json()["repo_urls"] == [TEST_REPO_URL]


def test_query_endpoint_still_works(indexed_session):
    resp = client.post("/api/query", json={
        "session_id": indexed_session, "question": "What license is this under?",
    })
    assert resp.status_code == 200
    assert len(resp.json()["citations"]) > 0


def test_unknown_session_returns_404():
    resp = client.post("/api/chat", json={"session_id": "does-not-exist", "message": "hi"})
    assert resp.status_code == 404
