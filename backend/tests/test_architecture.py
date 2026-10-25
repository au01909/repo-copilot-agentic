"""Architecture analysis on small synthetic repos in different languages."""
from app import architecture_detect
from app.chunking import Chunk


def _file(path, language, content, imports=None, symbol=None, kind="generic"):
    return Chunk(repo="r", file=path, language=language, content=content, start_line=1,
                 end_line=content.count("\n") + 1, symbol=symbol, kind=kind, imports=imports or [])


def _edges(result):
    return {(c["name"], d) for c in result["components"] for d in c["depends_on"]}


def test_javascript_repo_components_edges_stack_and_entry_point():
    chunks = [
        _file("package.json", "json", '{"dependencies": {"express": "^4", "pg": "^8"}}'),
        _file("src/index.js", "javascript", 'const app = require("express")();\nrequire("./api/routes");\napp.listen(3000);'),
        _file("src/api/routes.js", "javascript", 'const users = require("../services/users");'),
        _file("src/services/users.js", "javascript", 'const db = require("pg");'),
    ] + [_file(f"src/util/u{i}.js", "javascript", "module.exports = {};") for i in range(14)]
    result = architecture_detect.detect(chunks)

    assert ("src", "src/api") in _edges(result) and ("src/api", "src/services") in _edges(result)
    assert "Express" in result["detected_frameworks_and_infra"]
    assert result["entry_points"][0]["file"] == "src/index.js"
    assert "API / web service" in result["likely_architecture_patterns"]
    assert result["diagram"].startswith("flowchart TD") and "-->" in result["diagram"]


def test_python_relative_import_resolves_and_library_is_detected():
    chunks = [
        _file("pkg/engine.py", "python", "class Value: ...", symbol="Value", kind="class"),
        _file("pkg/nn.py", "python", "from .engine import Value", imports=["engine"]),
        _file("setup.py", "python", "setup(name='pkg')"),
    ]
    result = architecture_detect.detect(chunks)

    assert ("pkg/nn.py", "pkg/engine.py") in _edges(result)
    assert result["likely_architecture_patterns"] == ["Library / package"]
    assert result["entry_points"] == []
    assert result["detected_frameworks_and_infra"] == ["none detected"]


def test_go_repo_manifest_import_and_main():
    chunks = [
        _file("go.mod", "text", "module x\n\nrequire github.com/gin-gonic/gin v1.9.0\n"),
        _file("cmd/server/main.go", "go",
              'package main\nimport (\n "x/internal/store"\n "github.com/gin-gonic/gin"\n)\nfunc main() {}'),
        _file("internal/store/store.go", "go", "package store"),
    ]
    result = architecture_detect.detect(chunks)

    assert ("cmd/server/main.go", "internal/store/store.go") in _edges(result)
    assert "Gin" in result["detected_frameworks_and_infra"]
    assert result["entry_points"][0]["file"] == "cmd/server/main.go"


def test_mentions_in_docs_and_tests_do_not_count_as_stack():
    chunks = [
        _file("README.md", "markdown", "We could use kafka, redis and graphql someday."),
        _file("tests/test_x.py", "python", "import flask", imports=["flask"]),
        _file("app.py", "python", "print('hi')"),
    ]
    assert architecture_detect.detect(chunks)["detected_frameworks_and_infra"] == ["none detected"]
