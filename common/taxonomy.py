"""Taxonomy + Perch label mapping.

The Kaggle ``sample_submission.csv`` defines the canonical ``primary_labels`` order.
Perch v2 emits logits over ~10,932 classes (see ``labels.csv`` inside the SavedModel);
most of them are not target species, and we need:

1. A **direct index mapping** for species that appear in both vocabularies.
2. A **genus-level proxy** for species present in the competition but absent from Perch
   (typical for Amphibia / Insecta — Perch is bird-centric).
3. An escape hatch for a few frog "sonotype" rows in the taxonomy that have no scientific
   name we could proxy (filtered out per legacy convention by the "son" substring rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from common.paths import comp_dir, perch_model_dir


@dataclass
class LabelMapping:
    """Everything needed to turn Perch logits into competition scores."""

    primary_labels: list[str]                # canonical order from sample_submission.csv
    label_to_idx: dict[str, int]
    bc_indices: np.ndarray                   # (C,) int32 — Perch index per class, or NO_LABEL
    mapped_pos: np.ndarray                   # (M,) int32 — positions in primary_labels that have a direct Perch index
    mapped_bc_indices: np.ndarray            # (M,) int32 — corresponding Perch indices
    proxy_pos_to_bc: dict[int, np.ndarray]   # position -> Perch indices for proxy reduction
    no_label_index: int                      # sentinel for "not mapped"
    class_name_map: dict[str, str]           # primary_label -> taxonomic class (Aves / Amphibia / …)


def load_primary_labels(comp: Path | None = None) -> list[str]:
    """Canonical ``primary_labels`` order from ``sample_submission.csv``."""
    comp = comp or comp_dir()
    sample_sub = pd.read_csv(comp / "sample_submission.csv")
    return sample_sub.columns[1:].tolist()


def load_taxonomy(comp: Path | None = None) -> pd.DataFrame:
    comp = comp or comp_dir()
    tax = pd.read_csv(comp / "taxonomy.csv")
    tax["primary_label"] = tax["primary_label"].astype(str)
    return tax


def load_perch_labels(perch_dir: Path | None = None, labels_name_col: str = "inat2024_fsd50k") -> pd.DataFrame:
    """Load the Perch label table with a stable ``bc_index`` column."""
    perch_dir = perch_dir or perch_model_dir()
    labels_path = Path(perch_dir) / "assets" / "labels.csv"
    bc_labels = (
        pd.read_csv(labels_path)
        .reset_index()
        .rename(columns={"index": "bc_index", labels_name_col: "scientific_name"})
    )
    bc_labels["bc_index"] = bc_labels["bc_index"].astype(int)
    bc_labels["scientific_name"] = bc_labels["scientific_name"].astype(str)
    return bc_labels


def build_label_mapping(
    *,
    primary_labels: Sequence[str] | None = None,
    taxonomy: pd.DataFrame | None = None,
    bc_labels: pd.DataFrame | None = None,
    proxy_taxa: Sequence[str] = ("Aves", "Amphibia", "Insecta"),
) -> LabelMapping:
    """Construct the :class:`LabelMapping` from the Kaggle and Perch asset tables."""
    primary_labels = list(primary_labels) if primary_labels is not None else load_primary_labels()
    taxonomy = taxonomy if taxonomy is not None else load_taxonomy()
    bc_labels = bc_labels if bc_labels is not None else load_perch_labels()

    no_label_index = int(len(bc_labels))

    taxonomy = taxonomy.copy()
    taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"]

    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
    mapping = taxonomy.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label_index).astype(int)

    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    bc_indices = np.array(
        [int(label_to_bc_index.loc[c]) for c in primary_labels], dtype=np.int32
    )

    mapped_mask = bc_indices != no_label_index
    mapped_pos = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc_indices = bc_indices[mapped_mask].astype(np.int32)

    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    proxy_pos_to_bc = _build_proxy(
        mapping=mapping,
        bc_labels=bc_labels,
        label_to_idx=label_to_idx,
        class_name_map=class_name_map,
        no_label_index=no_label_index,
        proxy_taxa=set(proxy_taxa),
    )

    return LabelMapping(
        primary_labels=primary_labels,
        label_to_idx=label_to_idx,
        bc_indices=bc_indices,
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        no_label_index=no_label_index,
        class_name_map=class_name_map,
    )


def _build_proxy(
    *,
    mapping: pd.DataFrame,
    bc_labels: pd.DataFrame,
    label_to_idx: dict[str, int],
    class_name_map: dict[str, str],
    no_label_index: int,
    proxy_taxa: set[str],
) -> dict[int, np.ndarray]:
    """Genus-level proxy for species Perch doesn't know.

    Logic (faithfully ported from the legacy ``A3`` cell):

    - Skip labels with ``"son"`` substring (frog sonotype rows — no scientific name).
    - For every other unmapped primary_label, find Perch labels whose scientific_name
      starts with the same genus (first whitespace-delimited token).
    - Keep only labels in ``proxy_taxa``.
    """
    proxy: dict[int, np.ndarray] = {}
    unmapped = mapping[mapping["bc_index"] == no_label_index].copy()
    unmapped = unmapped[~unmapped["primary_label"].astype(str).str.contains("son", na=False)]

    for _, row in unmapped.iterrows():
        target = str(row["primary_label"])
        sci = str(row["scientific_name"])
        cls = class_name_map.get(target)
        if cls not in proxy_taxa:
            continue

        genus = sci.split()[0] if sci else ""
        if not genus:
            continue

        hits = bc_labels[
            bc_labels["scientific_name"].str.match(rf"^{re.escape(genus)}\s", na=False)
        ]
        if len(hits) == 0:
            continue

        pos = label_to_idx[target]
        proxy[pos] = hits["bc_index"].astype(np.int32).to_numpy()

    return proxy


def describe_mapping(m: LabelMapping) -> dict:
    """Small dict summary for logs."""
    return {
        "n_classes": len(m.primary_labels),
        "n_mapped_direct": int(len(m.mapped_pos)),
        "n_proxy": int(len(m.proxy_pos_to_bc)),
        "n_unmapped_total": int(len(m.primary_labels) - len(m.mapped_pos) - len(m.proxy_pos_to_bc)),
    }
