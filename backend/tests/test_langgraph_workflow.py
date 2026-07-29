from app.agent.intent import classify
from app.agent.workflow import run_workflow
from app.search import HybridSearchIndex


def test_intent_classification_deterministic_rules():
    assert classify("What does this repository do?") == "REPOSITORY_SUMMARY"
    assert classify("Explain the architecture.") == "ARCHITECTURE"
    assert classify("How do I run this repository?") == "SETUP"
    assert classify("What technologies are used?") == "TECH_STACK"
    assert classify("Where does the application start?") == "ENTRYPOINT"
    assert classify("asdkfjhaslkdfj random text") == "GENERAL"


def test_workflow_returns_grounded_citations(search_index):
    result = run_workflow(search_index, "test-session", "How does the Value class work?")
    assert result["mode"] in ("generative", "extractive")
    assert len(result["citations"]) > 0
    for c in result["citations"]:
        assert c["file"]
        assert c["citation"] == f"{c['file']}:{c['start_line']}-{c['end_line']}"


def test_workflow_intent_is_set(search_index):
    result = run_workflow(search_index, "test-session", "Explain the architecture of this repo.")
    assert result["intent"] == "ARCHITECTURE"


def test_workflow_retry_terminates_on_empty_index():
    empty_index = HybridSearchIndex([])
    result = run_workflow(empty_index, "test-session", "anything at all")
    empty_index.close()
    assert result["retry_count"] <= 2  # AGENT_MAX_RETRIES default; must not loop forever
    assert "couldn't find" in result["answer"].lower() or "No relevant" in result["answer"]


def test_workflow_langchain_documents_have_metadata(search_index):
    result = run_workflow(search_index, "test-session", "how does backpropagation work")
    for doc in result["retrieved_documents"]:
        assert doc.metadata.get("file")
