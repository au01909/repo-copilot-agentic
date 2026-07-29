from app.chunking import chunk_file
from app.langchain_rag.documents import chunks_to_documents, document_citation
from app.langchain_rag.retriever import HybridRepoRetriever


def test_python_ast_chunking_finds_class_and_function():
    src = (
        "def hello(name):\n    return f'hi {name}'\n\n"
        "class Greeter:\n    def greet(self, name):\n        return hello(name)\n"
    )
    chunks = chunk_file("repo", "app.py", "python", src)
    symbols = {c.symbol for c in chunks}
    assert "hello" in symbols
    assert "Greeter" in symbols


def test_repo_chunks_cover_multiple_files(repo_chunks):
    files = {c.file for c in repo_chunks}
    assert "micrograd/engine.py" in files
    assert "micrograd/nn.py" in files
    assert "README.md" in files


def test_hybrid_search_finds_value_class(search_index):
    results = search_index.search("Value class autograd gradient", top_k=3)
    assert any(c.file == "micrograd/engine.py" for c in results)


def test_hybrid_search_dedupes_one_chunk_per_file(search_index):
    results = search_index.search("class", top_k=10)
    files = [c.file for c in results]
    assert len(files) == len(set(files))


def test_langchain_retriever_returns_documents_with_full_metadata(search_index):
    retriever = HybridRepoRetriever(search_index=search_index, top_k=2)
    docs = retriever.invoke("neural network MLP")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["file"]
        assert doc.metadata["start_line"] > 0
        assert doc.metadata["end_line"] >= doc.metadata["start_line"]
        assert document_citation(doc)  # non-empty "file:start-end" string


def test_chunks_to_documents_preserves_git_metadata(repo_chunks):
    docs = chunks_to_documents(repo_chunks[:3])
    for doc in docs:
        assert "commit_sha" in doc.metadata
        assert "author" in doc.metadata
