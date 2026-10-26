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

from .features import FEATURES

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


def cross_validate(X: pd.DataFrame, y, folds: int = 5, seed: int = 0) -> dict:
    """5-fold stratified CV ROC-AUC for TabPFN and the baseline on identical folds."""
    y = np.asarray(y)
    check_history(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {"tabpfn": [], "logreg": []}
    for train, test in skf.split(X, y):
        for name, model in (("tabpfn", make_tabpfn(seed)), ("logreg", make_baseline())):
            model.fit(X.iloc[train], y[train])
            proba = model.predict_proba(X.iloc[test])[:, 1]
            scores[name].append(roc_auc_score(y[test], proba))
    return {name: (float(np.mean(s)), float(np.std(s))) for name, s in scores.items()}


def time_split_auc(X: pd.DataFrame, y, created: list[str], holdout: float = 0.2, seed: int = 0) -> dict:
    """Train on the oldest 80 %, score the newest 20 %: the honest 'predict the future' check."""
    y = np.asarray(y)
    order = np.argsort(pd.to_datetime(pd.Series(created)).to_numpy())
    cut = int(len(order) * (1 - holdout))
    train, test = order[:cut], order[cut:]
    if len(set(y[test])) < 2:
        return {}
    out = {}
    for name, model in (("tabpfn", make_tabpfn(seed)), ("logreg", make_baseline())):
        model.fit(X.iloc[train], y[train])
        out[name] = float(roc_auc_score(y[test], model.predict_proba(X.iloc[test])[:, 1]))
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
