import math

from mergemeter.features import (
    FEATURES,
    build_dataset,
    docs_only,
    draft_features,
    early_activity,
    has_linked_issue,
    label_flags,
    parse_time,
    pr_features,
    touches_tests,
)
from mergemeter.github import GitHubError, parse_repo


def make_pr(number, *, author="alice", merged=False, state="closed", created="2026-10-01T00:00:00Z",
            closed="2026-10-02T00:00:00Z", **extra):
    pr = {
        "number": number, "title": "Fix typo", "body": "", "url": f"https://github.com/o/r/pull/{number}",
        "state": state, "merged": merged, "created_at": created, "closed_at": closed if state == "closed" else None,
        "merged_at": closed if merged else None, "author": author, "author_is_bot": False,
        "author_association": "NONE", "additions": 10, "deletions": 2, "changed_files": 1,
        "files": ["src/app.py"], "labels": [], "comments": [], "reviews": [],
    }
    pr.update(extra)
    return pr


def test_parse_repo_accepts_urls_and_slugs():
    assert parse_repo("encode/httpx") == "encode/httpx"
    assert parse_repo("https://github.com/encode/httpx/pulls") == "encode/httpx"
    assert parse_repo("github.com/encode/httpx.git") == "encode/httpx"


def test_parse_repo_rejects_garbage():
    for bad in ["", "httpx", "a b/c", "../etc/passwd"]:
        try:
            parse_repo(bad)
        except GitHubError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_touches_tests_paths():
    assert touches_tests(["tests/test_api.py"]) == 1.0
    assert touches_tests(["src/foo.test.ts"]) == 1.0
    assert touches_tests(["pkg/handler_test.go"]) == 1.0
    assert touches_tests(["src/contest.py", "README.md"]) == 0.0
    assert math.isnan(touches_tests(None))


def test_docs_only():
    assert docs_only(["README.md", "docs/guide/intro.rst"]) == 1.0
    assert docs_only(["README.md", "src/app.py"]) == 0.0
    assert docs_only([]) == 0.0
    assert math.isnan(docs_only(None))


def test_linked_issue_patterns():
    assert has_linked_issue("Fixes #12") == 1.0
    assert has_linked_issue("closes: encode/httpx#99") == 1.0
    assert has_linked_issue("See https://github.com/o/r/issues/3") == 1.0
    assert has_linked_issue("Adds a #hashtag and issue 12") == 0.0


def test_verdict_labels_are_ignored():
    flags = label_flags(["hacktoberfest-accepted", "stale", "wontfix"])
    assert flags["label_hacktoberfest"] == 0.0
    flags = label_flags(["hacktoberfest", "bug", "type: enhancement", "dependencies"])
    assert flags == {"label_bug": 1.0, "label_docs": 0.0, "label_feature": 1.0,
                     "label_hacktoberfest": 1.0, "label_deps": 1.0}


def test_early_activity_counts_others_in_first_day_before_close():
    created = parse_time("2026-10-01T00:00:00Z")
    closed = parse_time("2026-10-01T12:00:00Z")
    events = [
        {"at": "2026-10-01T01:00:00Z", "by": "maintainer"},  # counts
        {"at": "2026-10-01T02:00:00Z", "by": "alice"},       # the author: ignored
        {"at": "2026-10-01T11:55:00Z", "by": "maintainer"},  # the closing comment: ignored
        {"at": "2026-10-03T00:00:00Z", "by": "maintainer"},  # after 24 h: ignored
    ]
    assert early_activity(events, "alice", created, closed) == 1.0
    assert math.isnan(early_activity(None, "alice", created, closed))


def test_author_history_uses_only_earlier_prs():
    old_merged = make_pr(1, merged=True, created="2026-09-01T00:00:00Z", closed="2026-09-02T00:00:00Z")
    old_rejected = make_pr(2, created="2026-09-03T00:00:00Z", closed="2026-09-04T00:00:00Z")
    target = make_pr(3, created="2026-10-01T00:00:00Z", closed="2026-10-05T00:00:00Z")
    later = make_pr(4, merged=True, created="2026-10-10T00:00:00Z", closed="2026-10-11T00:00:00Z")
    other = make_pr(5, author="bob", merged=True, created="2026-09-01T00:00:00Z", closed="2026-09-02T00:00:00Z")
    feats = pr_features(target, [old_merged, old_rejected, target, later, other])
    assert feats["author_prior_merged"] == 1.0
    assert feats["author_prior_unmerged"] == 1.0


def test_maintainer_and_bot_flags():
    feats = pr_features(make_pr(1, author="dependabot[bot]", author_association="CONTRIBUTOR"), [])
    assert feats["is_bot"] == 1.0 and feats["is_maintainer"] == 0.0
    feats = pr_features(make_pr(2, author_association="MEMBER"), [])
    assert feats["is_maintainer"] == 1.0


def test_rest_records_leave_size_features_missing():
    pr = make_pr(1, additions=None, deletions=None, changed_files=None, files=None, comments=None, reviews=None)
    feats = pr_features(pr, [])
    for key in ["additions", "deletions", "changed_files", "touches_tests", "docs_only", "n_comments_24h"]:
        assert math.isnan(feats[key]), key


def test_build_dataset_splits_and_labels():
    prs = [make_pr(1, merged=True), make_pr(2), make_pr(3, state="open")]
    X, y, closed, X_open, opened = build_dataset(prs)
    assert y == [1, 0]
    assert [p["number"] for p in opened] == [3]
    assert all(set(row) == set(FEATURES) for row in X + X_open)


def test_draft_features_cover_every_feature():
    feats = draft_features(additions=40, deletions=5, changed_files=2, touches_tests=True, docs_only=False,
                           is_maintainer=False, prior_merged=0, prior_unmerged=0,
                           title="fix: handle empty body", body="Fixes #7", labels=["bug"])
    assert set(feats) == set(FEATURES)
    assert feats["conventional_title"] == 1.0 and feats["has_linked_issue"] == 1.0 and feats["label_bug"] == 1.0
