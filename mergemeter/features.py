"""Turn normalised PR records into a feature table for an open-time model.

Rule: score a PR as it looks when it is opened. So:
- author history is counted only from PRs in `history` that closed before
  this PR was created, never from GitHub's author_association (which flips
  FIRST_TIME_CONTRIBUTOR to CONTRIBUTOR once a PR merges); callers pass the
  training PRs as `history` so held-out outcomes never leak in;
- comments, reviews and time-to-close are not features at all;
- labels that maintainers add as a verdict (stale, wontfix, accepted,
  invalid, ...) are ignored.
Known gap: GitHub only returns a PR's *current* size, files, title, body and
labels, so edits after opening (and late topic labels) can still seep in.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

FEATURES = [
    "additions",
    "deletions",
    "changed_files",
    "touches_tests",
    "docs_only",
    "is_maintainer",
    "is_bot",
    "author_prior_merged",
    "author_prior_unmerged",
    "has_linked_issue",
    "title_len",
    "body_len",
    "conventional_title",
    "label_bug",
    "label_docs",
    "label_feature",
    "label_hacktoberfest",
    "label_deps",
]

FEATURE_LABELS = {
    "additions": "Lines added",
    "deletions": "Lines deleted",
    "changed_files": "Files changed",
    "touches_tests": "Touches tests",
    "docs_only": "Docs-only change",
    "is_maintainer": "Author is maintainer",
    "is_bot": "Author is a bot",
    "author_prior_merged": "Author's earlier merged PRs",
    "author_prior_unmerged": "Author's earlier unmerged PRs",
    "has_linked_issue": "Links an issue",
    "title_len": "Title length",
    "body_len": "Description length",
    "conventional_title": "Conventional title (feat:/fix:)",
    "label_bug": "Label: bug",
    "label_docs": "Label: docs",
    "label_feature": "Label: feature",
    "label_hacktoberfest": "Label: hacktoberfest",
    "label_deps": "Label: dependencies",
}

MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
VERDICT_LABEL_RE = re.compile(r"accepted|stale|wontfix|won't fix|invalid|duplicate|spam|rejected|merged|approved|lgtm", re.I)
LINKED_ISSUE_RE = re.compile(
    r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?|refs?|related to|see)\b:?\s+([\w.-]+/[\w.-]+)?#\d+"
    r"|github\.com/[\w.-]+/[\w.-]+/issues/\d+",
    re.I,
)
CONVENTIONAL_RE = re.compile(r"^(feat|fix|docs|chore|refactor|test|perf|build|ci|style|revert)(\(.+?\))?!?:", re.I)
TEST_PATH_RE = re.compile(r"(^|/)(tests?|__tests__|spec|specs)(/|$)|(_test|\.test|\.spec|_spec)\.\w+$|(^|/)test_[^/]+$", re.I)
DOC_PATH_RE = re.compile(r"\.(md|mdx|rst|txt|adoc)$|(^|/)(docs?|documentation)/|(^|/)(README|CHANGELOG|LICENSE|CONTRIBUTING)", re.I)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def touches_tests(files: list[str] | None, changed_files: int | None = None) -> float:
    if files is None:
        return math.nan
    if any(TEST_PATH_RE.search(f) for f in files):
        return 1.0
    # GitHub returns only the first files; an unseen one could be a test.
    return math.nan if changed_files and changed_files > len(files) else 0.0


def docs_only(files: list[str] | None, changed_files: int | None = None) -> float:
    if not files:
        return math.nan if files is None else 0.0
    if not all(DOC_PATH_RE.search(f) for f in files):
        return 0.0
    return math.nan if changed_files and changed_files > len(files) else 1.0


def has_linked_issue(body: str) -> float:
    return float(bool(LINKED_ISSUE_RE.search(body or "")))


def label_flags(labels: list[str]) -> dict[str, float]:
    names = [lbl.lower() for lbl in labels if not VERDICT_LABEL_RE.search(lbl)]

    def any_of(*words: str) -> float:
        return float(any(w in name for name in names for w in words))

    return {
        "label_bug": any_of("bug", "fix", "regression"),
        "label_docs": any_of("doc"),
        "label_feature": any_of("feature", "enhancement"),
        "label_hacktoberfest": any_of("hacktoberfest"),
        "label_deps": any_of("dependencies", "deps", "dependabot", "renovate"),
    }


def _num(value) -> float:
    return math.nan if value is None else float(value)


def pr_features(pr: dict, history: list[dict]) -> dict[str, float]:
    """Features for one PR. `history` is the closed PRs whose outcomes may be used (any order)."""
    created = parse_time(pr["created_at"])
    author = pr.get("author")
    earlier = [] if author is None else [
        h for h in history
        if h.get("author") == author and h["number"] != pr["number"]
        and parse_time(h.get("closed_at")) is not None and parse_time(h["closed_at"]) < created
    ]
    title = pr.get("title") or ""
    body = pr.get("body") or ""
    feats = {
        "additions": _num(pr.get("additions")),
        "deletions": _num(pr.get("deletions")),
        "changed_files": _num(pr.get("changed_files")),
        "touches_tests": touches_tests(pr.get("files"), pr.get("changed_files")),
        "docs_only": docs_only(pr.get("files"), pr.get("changed_files")),
        "is_maintainer": float(pr.get("author_association") in MAINTAINER_ASSOCIATIONS),
        "is_bot": float(bool(pr.get("author_is_bot")) or (author or "").endswith("[bot]")),
        "author_prior_merged": float(sum(1 for h in earlier if h["merged"])),
        "author_prior_unmerged": float(sum(1 for h in earlier if not h["merged"])),
        "has_linked_issue": has_linked_issue(body),
        "title_len": float(len(title)),
        "body_len": float(len(body)),
        "conventional_title": float(bool(CONVENTIONAL_RE.match(title.strip()))),
    }
    feats.update(label_flags(pr.get("labels") or []))
    return feats


def build_dataset(prs: list[dict]):
    """Split PRs into (closed features, merged labels, closed PRs, open features, open PRs).

    Every closed outcome is usable here, which is right for scoring open PRs
    now. Evaluation must not use this: see model.split_features.
    """
    closed = [p for p in prs if p["state"] == "closed"]
    opened = [p for p in prs if p["state"] == "open"]
    X_closed = [pr_features(p, closed) for p in closed]
    y = [int(p["merged"]) for p in closed]
    X_open = [pr_features(p, closed) for p in opened]
    return X_closed, y, closed, X_open, opened


def draft_features(
    *,
    additions: float,
    deletions: float,
    changed_files: float,
    touches_tests: bool,
    docs_only: bool,
    is_maintainer: bool,
    prior_merged: float,
    prior_unmerged: float,
    title: str,
    body: str,
    labels: list[str],
) -> dict[str, float]:
    """Features for a PR that does not exist yet (the what-if form)."""
    feats = {
        "additions": float(additions),
        "deletions": float(deletions),
        "changed_files": float(changed_files),
        "touches_tests": float(touches_tests),
        "docs_only": float(docs_only),
        "is_maintainer": float(is_maintainer),
        "is_bot": 0.0,
        "author_prior_merged": float(prior_merged),
        "author_prior_unmerged": float(prior_unmerged),
        "has_linked_issue": has_linked_issue(body),
        "title_len": float(len(title)),
        "body_len": float(len(body)),
        "conventional_title": float(bool(CONVENTIONAL_RE.match(title.strip()))),
    }
    feats.update(label_flags(labels))
    return feats
