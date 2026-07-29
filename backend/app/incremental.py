"""Incremental indexing: after the first full clone+index, re-indexing on a
`git pull` only needs to touch files that actually changed, found via `git diff
--name-status`, instead of re-walking and re-chunking the whole repository.
"""
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DiffResult:
    added: List[str]
    modified: List[str]
    deleted: List[str]
    renamed: List[Tuple[str, str]]  # (old_path, new_path)
    new_head_sha: str


def get_current_sha(repo_dir: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=15,
    )
    return result.stdout.strip()


def pull_latest(repo_dir: str) -> None:
    subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], capture_output=True, text=True, timeout=60)


def unshallow(repo_dir: str) -> None:
    """Repos are cloned with --depth 1 for fast initial indexing, which means
    they start with no history to diff against. Call this once before the
    first incremental diff to fetch full history."""
    subprocess.run(["git", "-C", repo_dir, "fetch", "--unshallow"], capture_output=True, text=True, timeout=120)


def diff_since(repo_dir: str, old_sha: str) -> DiffResult:
    """Compare old_sha to the current HEAD after a pull. Requires the clone to
    have enough history — a `--depth 1` clone only has the latest commit, so
    incremental diffing needs `git fetch --unshallow` first for repos cloned
    via `ingest.clone_repository` (which uses --depth 1 for speed)."""
    pull_latest(repo_dir)
    new_sha = get_current_sha(repo_dir)
    if new_sha == old_sha:
        return DiffResult([], [], [], [], new_sha)

    result = subprocess.run(
        ["git", "-C", repo_dir, "diff", "--name-status", old_sha, new_sha],
        capture_output=True, text=True, timeout=30,
    )
    added, modified, deleted, renamed = [], [], [], []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        status = parts[0]
        if status == "A":
            added.append(parts[1])
        elif status == "M":
            modified.append(parts[1])
        elif status == "D":
            deleted.append(parts[1])
        elif status.startswith("R"):
            renamed.append((parts[1], parts[2]))
    return DiffResult(added, modified, deleted, renamed, new_sha)
