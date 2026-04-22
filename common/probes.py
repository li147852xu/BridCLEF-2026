"""Per-class probes on Perch embeddings / temporal summary features.

Two probe flavors are supported and can be mixed:

- **Linear probe** (fast, robust baseline): ``sklearn.LogisticRegression`` on
  PCA-projected embeddings.
- **LightGBM probe** (stronger, slower): gradient boosting on raw or
  PCA-projected embeddings. Optional.

Both export a numeric ``(coef_mat, intercept_vec)`` payload for the linear
case, and a ``booster_dumps`` list for LightGBM. The online pipeline only
applies linear probes (tiny matmul); GBDTs are optional and, if used online,
need ``lightgbm`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm


@dataclass
class LinearProbe:
    """Per-class one-vs-rest logistic regression.

    Attributes
    ----------
    active_class_idx : (K,) int32 — indices of classes that had any positives.
    coef_mat : (K, D) float32 — weights per active class.
    intercept_vec : (K,) float32.
    """

    active_class_idx: np.ndarray
    coef_mat: np.ndarray
    intercept_vec: np.ndarray

    def predict_logits_active(self, X: np.ndarray) -> np.ndarray:
        """Return (N, K) logits for the *active* classes only.

        Caller is responsible for scattering into the full-class matrix.
        """
        return (X @ self.coef_mat.T + self.intercept_vec[None, :]).astype(np.float32)

    def to_artifact(self) -> dict:
        return {
            "active_class_idx": self.active_class_idx.astype(np.int32),
            "coef_mat": self.coef_mat.astype(np.float32),
            "intercept_vec": self.intercept_vec.astype(np.float32),
        }


def fit_linear_probe(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 300,
    solver: str = "liblinear",
    class_weight: str | None = "balanced",
    random_state: int = 42,
    tqdm_desc: str = "Training linear probes",
) -> LinearProbe:
    """Train one LogReg per class that has at least one positive label.

    ``X`` shape: (N, D). ``Y`` shape: (N, C) multi-label (0/1).
    """
    active = np.where(Y.sum(axis=0) > 0)[0].astype(np.int32)
    d = X.shape[1]
    coef = np.zeros((len(active), d), dtype=np.float32)
    intercept = np.zeros(len(active), dtype=np.float32)

    for k, j in enumerate(tqdm(active, desc=tqdm_desc)):
        yj = Y[:, j]
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state,
        )
        clf.fit(X, yj)
        coef[k] = clf.coef_[0].astype(np.float32)
        intercept[k] = np.float32(clf.intercept_[0])

    return LinearProbe(active, coef, intercept)


def apply_linear_probe_artifact(X: np.ndarray, artifact: dict) -> np.ndarray:
    """Return (N, K) logits for the K active classes using pure numpy."""
    return (X @ artifact["coef_mat"].T + artifact["intercept_vec"][None, :]).astype(np.float32)


def fit_linear_probe_masked(
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 300,
    solver: str = "liblinear",
    class_weight: str | None = "balanced",
    random_state: int = 42,
    min_rows: int = 20,
    min_pos: int = 2,
    min_neg: int = 2,
    neg_per_pos: int | None = 50,
    max_rows_per_class: int | None = 20000,
    tqdm_desc: str = "Training masked probes",
) -> LinearProbe:
    """Per-class LogReg with per-class row masks (for pseudo-labeled data).

    For each class ``j``:
      - Select rows where ``mask[:, j] == 1``.
      - Optional: subsample negatives to ``neg_per_pos * n_pos`` to avoid
        drowning positive signal under 127k pseudo-negatives.
      - Optional: cap total rows at ``max_rows_per_class`` for CPU budget.
      - Skip if fewer than ``min_rows`` total, or ``min_pos`` positives,
        or ``min_neg`` negatives.
      - Fit a :class:`LogisticRegression` and stash the coefficients.
    """
    rng = np.random.default_rng(random_state)
    n_classes = Y.shape[1]
    active: list[int] = []
    chosen_pos_idx: list[np.ndarray] = []
    chosen_neg_idx: list[np.ndarray] = []

    for j in range(n_classes):
        sel = mask[:, j] == 1
        if int(sel.sum()) < min_rows:
            continue
        sel_idx = np.where(sel)[0]
        y_sel = Y[sel_idx, j].astype(np.int32)
        pos_idx = sel_idx[y_sel == 1]
        neg_idx = sel_idx[y_sel == 0]
        if len(pos_idx) < min_pos or len(neg_idx) < min_neg:
            continue
        if neg_per_pos is not None and len(neg_idx) > neg_per_pos * max(len(pos_idx), 1):
            take = neg_per_pos * len(pos_idx)
            neg_idx = rng.choice(neg_idx, size=take, replace=False)
        if max_rows_per_class is not None and (len(pos_idx) + len(neg_idx)) > max_rows_per_class:
            keep_neg = max(max_rows_per_class - len(pos_idx), min_neg)
            neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False)
        active.append(j)
        chosen_pos_idx.append(pos_idx)
        chosen_neg_idx.append(neg_idx)

    active_arr = np.asarray(active, dtype=np.int32)
    coef = np.zeros((len(active_arr), X.shape[1]), dtype=np.float32)
    intercept = np.zeros(len(active_arr), dtype=np.float32)

    for k, (j, pidx, nidx) in enumerate(
        tqdm(list(zip(active_arr, chosen_pos_idx, chosen_neg_idx)), desc=tqdm_desc)
    ):
        idx = np.concatenate([pidx, nidx])
        Xj = X[idx]
        yj = np.concatenate([np.ones(len(pidx), dtype=np.int32), np.zeros(len(nidx), dtype=np.int32)])
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver,
            class_weight=class_weight, random_state=random_state,
        )
        clf.fit(Xj, yj)
        coef[k] = clf.coef_[0].astype(np.float32)
        intercept[k] = float(clf.intercept_[0])

    return LinearProbe(active_arr, coef, intercept)


# ---------------------------------------------------------------------- LightGBM


@dataclass
class LGBMProbe:
    active_class_idx: np.ndarray
    booster_dumps: list[dict]  # each element is booster.dump_model()

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        params: dict | None = None,
        num_boost_round: int = 300,
        tqdm_desc: str = "Training LightGBM probes",
    ) -> "LGBMProbe":
        import lightgbm as lgb

        params = params or {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbose": -1,
        }
        active = np.where(Y.sum(axis=0) > 0)[0].astype(np.int32)
        dumps: list[dict] = []
        for j in tqdm(active, desc=tqdm_desc):
            yj = Y[:, j]
            dtrain = lgb.Dataset(X, label=yj)
            booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
            dumps.append(booster.dump_model())

        return cls(active, dumps)
