"""GitHub integration: issues, pull requests, and commit history via the real
GitHub REST API (api.github.com). Works unauthenticated for public repos at
GitHub's default rate limit (60 req/hour); set GITHUB_TOKEN for 5000 req/hour.
"""
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from . import config

API_ROOT = "https://api.github.com"


class GitHubAPIError(Exception):
    pass


def parse_owner_repo(repo_url: str) -> Tuple[str, str]:
    """https://github.com/owner/repo(.git) -> (owner, repo)"""
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise GitHubAPIError(f"Could not parse owner/repo from {repo_url}")
    owner, repo = parts[0], parts[1]
    repo = re.sub(r"\.git$", "", repo)
    return owner, repo


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _get(path: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(f"{API_ROOT}{path}", headers=_headers(), params=params or {}, timeout=15)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise GitHubAPIError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN in your .env for a much "
            "higher limit (5000/hr vs 60/hr unauthenticated)."
        )
    if resp.status_code >= 400:
        raise GitHubAPIError(f"GitHub API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def get_issue(repo_url: str, issue_number: int) -> dict:
    owner, repo = parse_owner_repo(repo_url)
    issue = _get(f"/repos/{owner}/{repo}/issues/{issue_number}")
    comments = _get(f"/repos/{owner}/{repo}/issues/{issue_number}/comments")
    return {
        "number": issue["number"], "title": issue["title"], "state": issue["state"],
        "body": issue.get("body") or "", "author": issue["user"]["login"],
        "created_at": issue["created_at"], "closed_at": issue.get("closed_at"),
        "labels": [l["name"] for l in issue.get("labels", [])],
        "is_pull_request": "pull_request" in issue,
        "comments": [{"author": c["user"]["login"], "body": c["body"]} for c in comments],
    }


def list_issues(repo_url: str, state: str = "all", limit: int = 20) -> List[dict]:
    owner, repo = parse_owner_repo(repo_url)
    issues = _get(f"/repos/{owner}/{repo}/issues", {"state": state, "per_page": limit})
    return [
        {"number": i["number"], "title": i["title"], "state": i["state"],
         "is_pull_request": "pull_request" in i, "created_at": i["created_at"]}
        for i in issues
    ]


def get_pull_request(repo_url: str, pr_number: int) -> dict:
    owner, repo = parse_owner_repo(repo_url)
    pr = _get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    files = _get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
    return {
        "number": pr["number"], "title": pr["title"], "state": pr["state"],
        "body": pr.get("body") or "", "author": pr["user"]["login"],
        "merged": pr.get("merged", False), "merged_at": pr.get("merged_at"),
        "base": pr["base"]["ref"], "head": pr["head"]["ref"],
        "files_changed": [
            {"filename": f["filename"], "status": f["status"],
             "additions": f["additions"], "deletions": f["deletions"]}
            for f in files
        ],
    }


def list_commits(repo_url: str, path: Optional[str] = None, limit: int = 20) -> List[dict]:
    owner, repo = parse_owner_repo(repo_url)
    params = {"per_page": limit}
    if path:
        params["path"] = path
    commits = _get(f"/repos/{owner}/{repo}/commits", params)
    return [
        {"sha": c["sha"][:10], "message": c["commit"]["message"].splitlines()[0],
         "author": (c["commit"]["author"] or {}).get("name", "unknown"),
         "date": (c["commit"]["author"] or {}).get("date")}
        for c in commits
    ]
