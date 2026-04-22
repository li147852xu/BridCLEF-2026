"""Feature engineering on Perch outputs.

Two families:

1. **Embedding features** — StandardScaler + PCA projection from 1536-D to PCA_DIM.
   Used as input for per-class linear/GBDT probes.
2. **Temporal summary features** — per-class pooling statistics over the 12 windows
   of a single 60-second file (mean, max, std, last-first-delta, optional p90).
   Used by a tiny "temporal-lite" head that learns to boost file-level decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------- embedding pipeline


@dataclass
class EmbeddingPipeline:
    scaler: StandardScaler
    pca: PCA

    def transform(self, emb: np.ndarray) -> np.ndarray:
        scaled = (emb.astype(np.float32) - self.scaler.mean_[None, :]) / self.scaler.scale_[None, :]
        centered = scaled - self.pca.mean_[None, :]
        return (centered @ self.pca.components_.T).astype(np.float32)

    def to_artifact(self) -> dict:
        return {
            "scaler_mean": self.scaler.mean_.astype(np.float32),
            "scaler_scale": self.scaler.scale_.astype(np.float32),
            "pca_mean": self.pca.mean_.astype(np.float32),
            "pca_components": self.pca.components_.astype(np.float32),
        }


def fit_embedding_pipeline(emb: np.ndarray, pca_dim: int = 128, random_state: int = 42) -> EmbeddingPipeline:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(emb)
    pca = PCA(n_components=pca_dim, random_state=random_state)
    pca.fit(scaled)
    return EmbeddingPipeline(scaler=scaler, pca=pca)


def apply_embedding_artifact(emb: np.ndarray, artifact: dict) -> np.ndarray:
    """Runtime application without reloading scikit-learn objects."""
    scaled = (emb.astype(np.float32) - artifact["scaler_mean"][None, :]) / artifact["scaler_scale"][None, :]
    centered = scaled - artifact["pca_mean"][None, :]
    return (centered @ artifact["pca_components"].T).astype(np.float32)


# ----------------------------------------------------------- temporal features


def temporal_features_from_seq(
    seq_scores: np.ndarray,
    stats: tuple[str, ...] = ("mean", "max", "std", "delta"),
    n_classes: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute per-class temporal summary features from window-level logits.

    Parameters
    ----------
    seq_scores : ``(n_files, windows_per_file, n_classes)`` float32
    stats : tuple of stat names, subset of ``{"mean","max","std","delta","p90"}``
    n_classes : optional override; defaults to ``seq_scores.shape[-1]``

    Returns
    -------
    stacked : ``(n_files, len(stats), n_classes)`` float32 — order matches ``stats``
    lookup : dict name -> (n_files, n_classes)
    """
    n_classes = n_classes or seq_scores.shape[-1]
    lookup: dict[str, np.ndarray] = {}
    per_stat = []
    for s in stats:
        if s == "mean":
            v = seq_scores.mean(axis=1)
        elif s == "max":
            v = seq_scores.max(axis=1)
        elif s == "std":
            v = seq_scores.std(axis=1)
        elif s == "delta":
            v = seq_scores[:, -1, :] - seq_scores[:, 0, :]
        elif s == "p90":
            v = np.quantile(seq_scores, 0.9, axis=1)
        else:
            raise ValueError(f"unknown stat '{s}'")
        lookup[s] = v.astype(np.float32)
        per_stat.append(v.astype(np.float32))

    stacked = np.stack(per_stat, axis=1)  # (n_files, n_stats, n_classes)
    return stacked, lookup
