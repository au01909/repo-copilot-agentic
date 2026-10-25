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
    result = read_repository_file(file_path="micrograd/engine.py", repo_url=TEST_REPO_URL, start_line=1, end_line=5)
    assert "content" in result
    assert "class Value" in result["content"] or "Value" in result["content"]


def test_read_repository_file_blocks_path_traversal(tool_context):
    _, read_repository_file, *_ = bind_repository_tools(tool_context)
    result = read_repository_file(file_path="../../../etc/passwd", repo_url=TEST_REPO_URL)
    assert "error" in result
    assert "escapes" in result["error"]


def test_read_repository_file_rejects_unknown_repo(tool_context):
    _, read_repository_file, *_ = bind_repository_tools(tool_context)
    result = read_repository_file(file_path="README.md", repo_url="https://github.com/not/indexed.git")
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
    monkeypatch.setattr("app.config.NVIDIA_NIM_API_KEY", None)
    with pytest.raises(ADKAgentUnavailable):
        build_repository_agent(tool_context)


def test_agent_caps_tool_calls_per_answer(monkeypatch):
    from app import config
    from app.agent.repository_agent import build_repository_agent
    from app.agent.tools import RepoToolContext

    monkeypatch.setattr(config, "NVIDIA_NIM_API_KEY", "test-key")  # builds the agent, no network call
    monkeypatch.setattr(config, "AGENT_MAX_TOOL_CALLS", 2)
    agent = build_repository_agent(RepoToolContext(None, [], {}))

    results = [agent.before_tool_callback(None, {}, None) for _ in range(3)]
    assert results[0] is None and results[1] is None  # first two tool calls run
    assert "budget" in results[2]["error"]            # third is refused, model must answer


def test_read_repository_file_tolerates_url_variants_and_missing_repo_url(tool_context):
    _, read_repository_file, *_ = bind_repository_tools(tool_context)
    bare = TEST_REPO_URL.removesuffix(".git")  # what a model writes: no ".git"
    for kwargs in ({"repo_url": bare}, {"repo_url": bare.upper() + "/"}, {}):
        result = read_repository_file(file_path="micrograd/engine.py", start_line=1, end_line=5, **kwargs)
        assert "content" in result, (kwargs, result)
