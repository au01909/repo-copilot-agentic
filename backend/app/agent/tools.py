"""Repository tools for the Google ADK agent.

Every tool here is a plain typed Python function — ADK's `FunctionTool`
wrapping introspects the signature and docstring to build the tool schema, so
these double as the tool definitions themselves; no separate JSON schema to
keep in sync. Each one:
  - takes/returns plain JSON-serializable types (typed via annotations)
  - validates its own inputs and returns a structured error dict on failure
    instead of raising, since ADK tool-calling expects a result to hand back
    to the model, not an exception
  - never executes repository content — `read_file` reads bytes, `search_repository`
    queries the existing index, nothing here shells out

These are bound to one repository's search index / chunk list per call via a
small context object, since ADK tools are plain functions (no `self`) — see
`bind_repository_tools` below.
"""
import os
from typing import Dict, List, Optional

from ..chunking import Chunk
from ..search import HybridSearchIndex


class RepoToolContext:
    """Holds what the tools need for one indexed repository: the search
    index, the chunk list (for tree/symbol lookups), and the local clone
    path (for safe, size-bounded file reads)."""

    def __init__(self, search_index: HybridSearchIndex, chunks: List[Chunk],
                 local_dirs: Dict[str, str]):
        self.search_index = search_index
        self.chunks = chunks
        self.local_dirs = local_dirs  # repo_url -> local clone path


MAX_READ_BYTES = 200_000


def bind_repository_tools(ctx: RepoToolContext):
    """Returns the list of tool functions with `ctx` closed over, ready to
    pass to `google.adk.agents.Agent(tools=[...])`."""

    def search_repository(query: str, top_k: int = 5) -> Dict:
        """Search the indexed repository for code/docs relevant to a query.

        Args:
            query: natural-language or keyword search query.
            top_k: maximum number of results to return (default 5).

        Returns:
            A dict with a "results" list of {file, start_line, end_line,
            symbol, chunk_type, snippet, citation}.
        """
        try:
            chunks = ctx.search_index.search(query, top_k=top_k)
        except Exception as e:
            return {"error": str(e), "results": []}
        return {"results": [
            {
                "file": c.file, "start_line": c.start_line, "end_line": c.end_line,
                "symbol": c.symbol, "chunk_type": c.kind,
                "snippet": "\n".join(c.content.splitlines()[:8]),
                "citation": f"{c.file}:{c.start_line}-{c.end_line}",
            }
            for c in chunks
        ]}

    def read_repository_file(repo_url: str, file_path: str, start_line: Optional[int] = None,
                              end_line: Optional[int] = None) -> Dict:
        """Read a specific file (or line range) from an indexed repository.

        Args:
            repo_url: the repository this session indexed (must match a repo
                already added to this session — arbitrary paths are rejected).
            file_path: path relative to the repository root.
            start_line: optional 1-indexed start line.
            end_line: optional 1-indexed end line (inclusive).

        Returns:
            A dict with "content" (str) or "error" (str).
        """
        local_dir = ctx.local_dirs.get(repo_url)
        if not local_dir:
            return {"error": f"{repo_url} is not part of this session"}

        # resolve and verify the path stays inside the clone (no path traversal)
        abs_path = os.path.realpath(os.path.join(local_dir, file_path))
        if not abs_path.startswith(os.path.realpath(local_dir) + os.sep):
            return {"error": "file_path escapes the repository root"}
        if not os.path.isfile(abs_path):
            return {"error": f"{file_path} does not exist in this repository"}
        if os.path.getsize(abs_path) > MAX_READ_BYTES:
            return {"error": f"{file_path} exceeds the {MAX_READ_BYTES}-byte read limit"}

        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError as e:
            return {"error": str(e)}

        if start_line or end_line:
            s = max((start_line or 1) - 1, 0)
            e = end_line or len(lines)
            lines = lines[s:e]
        return {"content": "".join(lines)}

    def get_repository_tree(directory_prefix: str = "") -> Dict:
        """List indexed files, optionally filtered to a directory prefix.

        Args:
            directory_prefix: only return files under this path prefix (empty = all).

        Returns:
            A dict with a "files" list of relative paths.
        """
        files = sorted({c.file for c in ctx.chunks if c.file.startswith(directory_prefix)})
        return {"files": files, "count": len(files)}

    def find_symbol(symbol_name: str) -> Dict:
        """Find where a function/class/symbol is defined in the indexed repository.

        Args:
            symbol_name: exact or partial symbol name to search for.

        Returns:
            A dict with a "matches" list of {file, start_line, end_line, kind, symbol}.
        """
        needle = symbol_name.lower()
        matches = [
            {"file": c.file, "start_line": c.start_line, "end_line": c.end_line,
             "kind": c.kind, "symbol": c.symbol}
            for c in ctx.chunks
            if c.symbol and needle in c.symbol.lower()
        ]
        return {"matches": matches[:20], "count": len(matches)}

    def get_dependencies(file_path: str) -> Dict:
        """List the imports declared in a specific indexed file.

        Args:
            file_path: path relative to the repository root.

        Returns:
            A dict with an "imports" list.
        """
        imports = set()
        for c in ctx.chunks:
            if c.file == file_path:
                imports.update(c.imports)
        return {"file": file_path, "imports": sorted(imports)}

    return [search_repository, read_repository_file, get_repository_tree,
            find_symbol, get_dependencies]
