"""Repository ingestion: clone (or use local path), walk files, filter noise."""
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

SUPPORTED_EXTENSIONS = {
    ".py", ".java", ".go", ".rs", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".ts", ".tsx", ".js", ".jsx",
    ".md", ".markdown", ".yaml", ".yml", ".json",
    ".dockerfile", ".tf", ".sql",
}

SUPPORTED_FILENAMES = {
    "Dockerfile", "README", "README.md", "LICENSE", "Makefile",
}

IGNORE_DIRS = {
    "node_modules", "dist", "build", "venv", ".venv", "env",
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "target", "vendor", ".idea", ".vscode", "coverage",
}

IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".whl", ".pyc", ".so", ".dylib", ".dll",
    ".pdf", ".lock",
}

MAX_FILE_BYTES = 500_000  # skip pathologically large files


@dataclass
class SourceFile:
    path: str          # path relative to repo root
    abs_path: str
    language: str
    content: str


def _detect_language(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    mapping = {
        ".py": "python", ".java": "java", ".go": "go", ".rs": "rust",
        ".cpp": "cpp", ".cc": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
        ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
        ".md": "markdown", ".markdown": "markdown",
        ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".tf": "terraform", ".sql": "sql",
    }
    if name == "Dockerfile":
        return "dockerfile"
    return mapping.get(ext, "text")


def clone_repository(repo_url: str, branch: Optional[str] = None) -> str:
    """Clone a repo to a temp dir and return the local path. Raises on failure."""
    tmp_dir = tempfile.mkdtemp(prefix="repo_copilot_")
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, tmp_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return tmp_dir


def is_supported_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext in IGNORE_EXTENSIONS:
        return False
    if name in SUPPORTED_FILENAMES:
        return True
    return ext in SUPPORTED_EXTENSIONS


def get_file_git_metadata(repo_dir: str, rel_path: str) -> dict:
    """Last commit that touched this file: sha, author, date. Real git data,
    not inferred — used to enrich chunk metadata per the PRD's 'commit SHA' /
    'git author' fields. Silently returns empty fields on any git error (e.g.
    a --depth 1 clone with no history for this file) rather than failing indexing."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "log", "-1", "--format=%H|%an|%ad", "--date=short", "--", rel_path],
            capture_output=True, text=True, timeout=10,
        )
        line = result.stdout.strip()
        if not line or "|" not in line:
            return {"commit_sha": None, "author": None, "date": None}
        sha, author, date = line.split("|", 2)
        return {"commit_sha": sha[:10], "author": author, "date": date}
    except Exception:
        return {"commit_sha": None, "author": None, "date": None}


def get_repo_head_sha(repo_dir: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip()


def get_current_branch(repo_dir: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip()


def enrich_chunks_with_git_metadata(repo_dir: str, chunks: list, branch: str = None) -> None:
    """Mutates chunks in place, attaching commit_sha/author/branch. One git call
    per unique file (not per chunk) to keep this fast on large repos."""
    branch = branch or get_current_branch(repo_dir)
    cache: dict = {}
    for c in chunks:
        if c.file not in cache:
            cache[c.file] = get_file_git_metadata(repo_dir, c.file)
        meta = cache[c.file]
        c.commit_sha = meta["commit_sha"]
        c.author = meta["author"]
        c.branch = branch


def walk_repository(root_dir: str) -> List[SourceFile]:
    """Walk the repo, apply ignore rules, return readable source files."""
    files: List[SourceFile] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if not is_supported_file(fname):
                continue
            abs_path = os.path.join(dirpath, fname)
            try:
                if os.path.getsize(abs_path) > MAX_FILE_BYTES:
                    continue
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue
            if not content.strip():
                continue
            rel_path = os.path.relpath(abs_path, root_dir)
            files.append(SourceFile(
                path=rel_path,
                abs_path=abs_path,
                language=_detect_language(fname),
                content=content,
            ))
    return files
