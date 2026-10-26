"""One call from 'owner/repo' to scored open PRs, factors and a what-if model."""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import github
from .features import FEATURE_LABELS, build_dataset
from .model import check_history, direction, make_tabpfn, permutation_importance, to_frame

_RESULTS: OrderedDict[str, dict] = OrderedDict()
_MAX_RESULTS = 8


def score_repo(repo: str, use_cache: bool = True) -> dict:
    repo = github.parse_repo(repo)
    if use_cache and repo in _RESULTS:
        _RESULTS.move_to_end(repo)
        return _RESULTS[repo]

    t0 = time.time()
    data = github.fetch_prs(repo, use_cache=use_cache)
    t_fetch = time.time() - t0

    X_rows, y, closed, X_open_rows, opened = build_dataset(data["prs"])
    y = np.asarray(y)
    check_history(y)
    X = to_frame(X_rows)
    created = pd.to_datetime(pd.Series([p["created_at"] for p in closed]))

    # Factors: fit on the oldest 80 %, shuffle features on the newest 20 %.
    order = np.argsort(created.to_numpy())
    cut = int(len(order) * 0.8)
    train, test = order[:cut], order[cut:]
    importance, holdout_auc = pd.Series(dtype=float), float("nan")
    if len(set(y[test])) == 2 and len(set(y[train])) == 2:
        model = make_tabpfn()
        model.fit(X.iloc[train], y[train])
        importance, holdout_auc = permutation_importance(model, X.iloc[test].reset_index(drop=True), y[test])

    # Scoring model: every closed PR is context.
    full = make_tabpfn()
    full.fit(X, y)
    open_table = pd.DataFrame()
    if X_open_rows:
        X_open = to_frame(X_open_rows)
        proba = full.predict_proba(X_open)[:, 1]
        open_table = pd.DataFrame({
            "PR": [f"#{p['number']}" for p in opened],
            "Title": [p["title"][:90] for p in opened],
            "Author": [p["author"] for p in opened],
            "Merge chance": np.round(proba * 100).astype(int),
            "Link": [p["url"] for p in opened],
        }).sort_values("Merge chance", ascending=False, ignore_index=True)

    factors = pd.DataFrame({
        "feature": importance.index,
        "factor": [FEATURE_LABELS[c] for c in importance.index],
        "auc_drop": importance.to_numpy(),
        "direction": [direction(X, y, c) for c in importance.index],
    }) if len(importance) else pd.DataFrame(columns=["feature", "factor", "auc_drop", "direction"])

    result = {
        "repo": repo,
        "mode": data["mode"],
        "fetched_at": data["fetched_at"],
        "n_closed": int(len(y)),
        "n_merged": int(y.sum()),
        "n_open": len(opened),
        "holdout_auc": holdout_auc,
        "factors": factors,
        "open_table": open_table,
        "model": full,
        "medians": X.median(numeric_only=True).to_dict(),
        "seconds": {"fetch": round(t_fetch, 1), "total": round(time.time() - t0, 1)},
    }
    _RESULTS[repo] = result
    while len(_RESULTS) > _MAX_RESULTS:
        _RESULTS.popitem(last=False)
    return result


def score_draft(result: dict, feats: dict) -> float:
    return float(result["model"].predict_proba(to_frame([feats]))[:, 1][0])
