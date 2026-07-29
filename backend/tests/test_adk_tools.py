import pytest

from app.agent.repository_agent import ADKAgentUnavailable, build_repository_agent
from app.agent.tools import RepoToolContext, bind_repository_tools

TEST_REPO_URL = "https://github.com/karpathy/micrograd.git"


@pytest.fixture(scope="module")
def tool_context(cloned_repo, repo_chunks, search_index):
    return RepoToolContext(search_index, repo_chunks, {TEST_REPO_URL: cloned_repo})


def test_search_repository_tool_returns_grounded_results(tool_context):
    search_repository, *_ = bind_repository_tools(tool_context)
    result = search_repository("neural network layer", top_k=3)
    assert "results" in result
    assert len(result["results"]) > 0
    for r in result["results"]:
        assert r["citation"] == f"{r['file']}:{r['start_line']}-{r['end_line']}"


def test_read_repository_file_tool_reads_real_content(tool_context):
    _, read_repository_file, *_ = bind_repository_tools(tool_context)
    result = read_repository_file(TEST_REPO_URL, "micrograd/engine.py", 1, 5)
    assert "content" in result
    assert "class Value" in result["content"] or "Value" in result["content"]


def test_read_repository_file_blocks_path_traversal(tool_context):
    _, read_repository_file, *_ = bind_repository_tools(tool_context)
    result = read_repository_file(TEST_REPO_URL, "../../../etc/passwd")
    assert "error" in result
    assert "escapes" in result["error"]


def test_read_repository_file_rejects_unknown_repo(tool_context):
    _, read_repository_file, *_ = bind_repository_tools(tool_context)
    result = read_repository_file("https://github.com/not/indexed.git", "README.md")
    assert "error" in result


def test_get_repository_tree_tool(tool_context):
    _, _, get_repository_tree, _, _ = bind_repository_tools(tool_context)
    result = get_repository_tree("micrograd")
    assert "micrograd/engine.py" in result["files"]
    assert "micrograd/nn.py" in result["files"]


def test_find_symbol_tool(tool_context):
    *_, find_symbol, _ = bind_repository_tools(tool_context)
    result = find_symbol("MLP")
    assert result["count"] >= 1
    assert any(m["symbol"] == "MLP" for m in result["matches"])


def test_get_dependencies_tool(tool_context):
    *_, get_dependencies = bind_repository_tools(tool_context)
    result = get_dependencies("micrograd/nn.py")
    assert "imports" in result


def test_adk_agent_raises_clean_error_without_nvidia_key(tool_context, monkeypatch):
    monkeypatch.setattr("app.config.NVIDIA_API_KEY", None)
    with pytest.raises(ADKAgentUnavailable):
        build_repository_agent(tool_context)
