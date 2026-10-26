"""TabPFN (local, open weights) vs a logistic-regression baseline."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURES, parse_time, pr_features

warnings.filterwarnings("ignore", message=".*Running on CPU with more than.*")

N_ESTIMATORS = int(os.environ.get("MERGEMETER_ESTIMATORS", "4"))
MIN_PER_CLASS = 8


class NotEnoughHistory(ValueError):
    pass


def make_tabpfn(seed: int = 0):
    """The open v2 classifier checkpoint: downloads once over plain HTTP, no token, runs on CPU.

    TabPFN 2.5+ checkpoints sit behind a browser licence click-through (and are
    non-commercial), so the demo pins v2.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cpu", n_estimators=N_ESTIMATORS, random_state=seed
    )


def make_baseline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log1p),  # every feature is a non-negative count or flag
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FEATURES).astype(float)


def check_history(y) -> None:
    y = np.asarray(y)
    merged, unmerged = int(y.sum()), int(len(y) - y.sum())
    if merged < MIN_PER_CLASS or unmerged < MIN_PER_CLASS:
        raise NotEnoughHistory(
            f"Need at least {MIN_PER_CLASS} merged and {MIN_PER_CLASS} closed-unmerged PRs; "
            f"found {merged} merged and {unmerged} unmerged."
        )


def split_features(train_prs: list[dict], test_prs: list[dict]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Features for a train/test split where only training outcomes count as author history.

    Building features once over every closed PR would let a training row's
    "author's earlier merged PRs" encode a held-out PR's label.
    """
    X_train = to_frame([pr_features(p, train_prs) for p in train_prs])
    X_test = to_frame([pr_features(p, train_prs) for p in test_prs])
    y_train = np.array([int(p["merged"]) for p in train_prs])
    y_test = np.array([int(p["merged"]) for p in test_prs])
    return X_train, y_train, X_test, y_test


def time_split(closed: list[dict], holdout: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Forecast split: test on the newest 20 % by creation time, train only on PRs closed before the cutoff.

    The cutoff is the creation time of the oldest test PR, so the model knows
    exactly what a maintainer could have known at that moment.
    """
    by_created = sorted(closed, key=lambda p: p["created_at"])
    cut = int(len(by_created) * (1 - holdout))
    cutoff = parse_time(by_created[cut]["created_at"])
    train = [p for p in by_created[:cut] if parse_time(p["closed_at"]) < cutoff]
    return train, by_created[cut:]


def _two_classes(*ys) -> bool:
    return all(len(set(np.asarray(y).tolist())) == 2 for y in ys)


def cross_validate(closed: list[dict], folds: int = 5, seed: int = 0) -> dict:
    """5-fold stratified CV ROC-AUC for TabPFN and the baseline on identical folds."""
    y = np.array([int(p["merged"]) for p in closed])
    check_history(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {"tabpfn": [], "logreg": []}
    for train, test in skf.split(np.zeros(len(y)), y):
        X_tr, y_tr, X_te, y_te = split_features([closed[i] for i in train], [closed[i] for i in test])
        for name, model in (("tabpfn", make_tabpfn(seed)), ("logreg", make_baseline())):
            model.fit(X_tr, y_tr)
            scores[name].append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    return {name: (float(np.mean(s)), float(np.std(s))) for name, s in scores.items()}


def time_split_auc(closed: list[dict], seed: int = 0) -> dict:
    """ROC-AUC of both models on the forecast split; {} when either side lacks a class."""
    train, test = time_split(closed)
    X_tr, y_tr, X_te, y_te = split_features(train, test)
    if not _two_classes(y_tr, y_te):
        return {}
    out = {"n_train": len(train), "n_test": len(test)}
    for name, model in (("tabpfn", make_tabpfn(seed)), ("logreg", make_baseline())):
        model.fit(X_tr, y_tr)
        out[name] = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    return out


def permutation_importance(model, X_test: pd.DataFrame, y_test, seed: int = 0) -> tuple[pd.Series, float]:
    """AUC drop when each feature is shuffled, plus the unshuffled AUC.

    All shuffled copies go through one predict_proba call: TabPFN's cost is per
    call (attention over the training set), so batching keeps this to seconds on CPU.
    """
    y_test = np.asarray(y_test)
    rng = np.random.default_rng(seed)
    blocks = [X_test]
    for col in FEATURES:
        shuffled = X_test.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        blocks.append(shuffled)
    proba = model.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1]
    n = len(X_test)
    base = roc_auc_score(y_test, proba[:n])
    drops = {
        col: base - roc_auc_score(y_test, proba[(i + 1) * n:(i + 2) * n])
        for i, col in enumerate(FEATURES)
    }
    return pd.Series(drops).sort_values(ascending=False), base


def direction(X: pd.DataFrame, y, col: str) -> str:
    """Plain-language sign for a factor: does more of it go with merging?"""
    x = X[col]
    y = pd.Series(np.asarray(y), index=X.index)
    mask = x.notna()
    if mask.sum() < 5 or x[mask].nunique() < 2:
        return ""
    corr = np.corrcoef(x[mask].rank(), y[mask])[0, 1]
    if np.isnan(corr) or abs(corr) < 0.05:
        return "mixed"
    return "helps" if corr > 0 else "hurts"
