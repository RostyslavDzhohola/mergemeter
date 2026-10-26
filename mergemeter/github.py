"""Fetch pull requests for a public GitHub repo and normalise them.

Two paths:
- GraphQL (needs GITHUB_TOKEN): sizes, file paths, comment and review
  timestamps for every PR in a handful of requests.
- REST without a token: the list endpoint only (no sizes or files), so the
  size features are left missing. TabPFN handles missing values natively.

Every fetch is cached as JSON on disk, so a repo costs its API calls once
per CACHE_TTL.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://api.github.com"
CACHE_DIR = Path(os.environ.get("MERGEMETER_CACHE", Path(__file__).resolve().parent.parent / "data" / "cache"))
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"
CACHE_TTL = 6 * 3600
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $states: [PullRequestState!], $first: Int!, $after: String) {
  rateLimit { remaining resetAt }
  repository(owner: $owner, name: $name) {
    pullRequests(states: $states, first: $first, after: $after,
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body url state merged createdAt closedAt mergedAt
        additions deletions changedFiles authorAssociation
        author { login __typename }
        labels(first: 20) { nodes { name } }
        files(first: 30) { nodes { path } }
        comments(first: 20) { nodes { createdAt author { login } } }
        reviews(first: 20) { nodes { createdAt author { login } } }
      }
    }
  }
}
"""


class GitHubError(RuntimeError):
    pass


def parse_repo(text: str) -> str:
    """Accept 'owner/repo' or a github.com URL and return 'owner/repo'."""
    text = (text or "").strip().rstrip("/")
    text = re.sub(r"^(https?://)?(www\.)?github\.com/", "", text)
    text = re.sub(r"\.git$", "", text)
    text = "/".join(text.split("/")[:2])
    if not REPO_RE.match(text) or any(part in {".", ".."} for part in text.split("/")):
        raise GitHubError(f"Expected owner/repo, got {text!r}")
    return text


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None


def _cache_path(repo: str, mode: str) -> Path:
    return CACHE_DIR / f"{repo.replace('/', '__')}.{mode}.json"


def _snapshot_path(repo: str) -> Path:
    return SNAPSHOT_DIR / f"{repo.replace('/', '__')}.json"


def _graphql_nodes(repo: str, states: list[str], limit: int, token: str) -> list[dict]:
    owner, name = repo.split("/")
    nodes: list[dict] = []
    after = None
    while len(nodes) < limit:
        page = min(50, limit - len(nodes))
        variables = {"owner": owner, "name": name, "states": states, "first": page, "after": after}
        for attempt in range(3):
            resp = requests.post(
                f"{API}/graphql",
                json={"query": GRAPHQL_QUERY, "variables": variables},
                headers={"Authorization": f"bearer {token}"},
                timeout=60,
            )
            if resp.status_code < 500:
                break
            time.sleep(2 * (attempt + 1))  # GitHub sometimes 502s on heavy pages
        if resp.status_code != 200:
            raise GitHubError(f"GitHub GraphQL HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        if payload.get("errors"):
            raise GitHubError(payload["errors"][0].get("message", "GraphQL error"))
        repo_data = payload["data"]["repository"]
        if repo_data is None:
            raise GitHubError(f"Repository {repo} not found")
        conn = repo_data["pullRequests"]
        nodes.extend(conn["nodes"])
        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]
    return nodes


def _from_graphql(node: dict) -> dict:
    author = node.get("author") or {}
    return {
        "number": node["number"],
        "title": node.get("title") or "",
        "body": node.get("body") or "",
        "url": node["url"],
        "state": "open" if node["state"] == "OPEN" else "closed",
        "merged": bool(node["merged"]),
        "created_at": node["createdAt"],
        "closed_at": node.get("closedAt"),
        "merged_at": node.get("mergedAt"),
        "author": author.get("login") or "ghost",
        "author_is_bot": author.get("__typename") == "Bot",
        "author_association": node.get("authorAssociation") or "NONE",
        "additions": node.get("additions"),
        "deletions": node.get("deletions"),
        "changed_files": node.get("changedFiles"),
        "files": [f["path"] for f in (node.get("files") or {}).get("nodes") or []],
        "labels": [lbl["name"] for lbl in node["labels"]["nodes"]],
        "comments": [
            {"at": c["createdAt"], "by": (c.get("author") or {}).get("login")}
            for c in node["comments"]["nodes"]
        ],
        "reviews": [
            {"at": r["createdAt"], "by": (r.get("author") or {}).get("login")}
            for r in node["reviews"]["nodes"]
        ],
    }


def _from_rest(item: dict) -> dict:
    user = item.get("user") or {}
    return {
        "number": item["number"],
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "url": item["html_url"],
        "state": item["state"],
        "merged": item.get("merged_at") is not None,
        "created_at": item["created_at"],
        "closed_at": item.get("closed_at"),
        "merged_at": item.get("merged_at"),
        "author": user.get("login") or "ghost",
        "author_is_bot": user.get("type") == "Bot",
        "author_association": item.get("author_association") or "NONE",
        # Not in the list endpoint; left missing on purpose.
        "additions": None,
        "deletions": None,
        "changed_files": None,
        "files": None,
        "labels": [lbl["name"] for lbl in item.get("labels") or []],
        "comments": None,
        "reviews": None,
    }


def _rest_list(repo: str, state: str, limit: int) -> list[dict]:
    items: list[dict] = []
    page = 1
    while len(items) < limit:
        resp = requests.get(
            f"{API}/repos/{repo}/pulls",
            params={"state": state, "per_page": 100, "page": page, "sort": "created", "direction": "desc"},
            headers={"Accept": "application/vnd.github+json"},
            timeout=30,
        )
        if resp.status_code == 404:
            raise GitHubError(f"Repository {repo} not found")
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            raise GitHubError("GitHub's unauthenticated rate limit is used up; set GITHUB_TOKEN or try a cached repo")
        if resp.status_code != 200:
            raise GitHubError(f"GitHub REST HTTP {resp.status_code}: {resp.text[:200]}")
        batch = resp.json()
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items[:limit]


def fetch_prs(repo: str, closed_limit: int = 300, open_limit: int = 100, use_cache: bool = True) -> dict:
    """Return {"repo", "mode", "fetched_at", "prs": [...]} for a public repo."""
    repo = parse_repo(repo)
    token = _token()
    mode = "graphql" if token else "rest"

    if use_cache:
        for path, ttl in ((_cache_path(repo, "graphql"), CACHE_TTL), (_cache_path(repo, mode), CACHE_TTL)):
            if path.exists() and time.time() - path.stat().st_mtime < ttl:
                return json.loads(path.read_text())
        # Bundled snapshots keep the demo and EVAL.md reproducible without a token.
        if not token and _snapshot_path(repo).exists():
            return json.loads(_snapshot_path(repo).read_text())

    if token:
        with ThreadPoolExecutor(2) as pool:
            closed_job = pool.submit(_graphql_nodes, repo, ["MERGED", "CLOSED"], closed_limit, token)
            opened_job = pool.submit(_graphql_nodes, repo, ["OPEN"], open_limit, token)
            closed, opened = closed_job.result(), opened_job.result()
        prs = [_from_graphql(n) for n in closed + opened]
    else:
        closed = _rest_list(repo, "closed", closed_limit)
        opened = _rest_list(repo, "open", open_limit)
        prs = [_from_rest(i) for i in closed + opened]

    data = {"repo": repo, "mode": mode, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prs": prs}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(repo, mode).write_text(json.dumps(data))
    return data
