# `10_baseline/` — historical / fallback submission notebooks

Kept around for rollback and comparison. **Don't edit; treat as immutable
reference points.**

## submit_v2_legacy.ipynb (PB 0.883)

Old Perch-distillation pipeline:

- Perch v2 zero-shot scores
- Linear probe on PCA(emb)
- MLP student trained on pseudo-labels from train_soundscapes
- Site/hour/month prior tables
- TopN=1 file-level smoothing

This was the best score before the hybrid path was wired. If anything
breaks our current submission flow and we need a "known good" file, this
is it. Requires the v2 `submission_bundle.pkl` Kaggle Dataset (legacy
slug — check the file's Cell 0).

## submit_v5_plan_y.ipynb (PB 0.866-0.868)

Plan Y standalone — the 5-fold `eca_nfnet_l0` OpenVINO ensemble we
trained on AutoDL (without the open-source 0.924 notebook on top):

- Adaptive ETA (probes first 8 files, picks 1/2/3 TTA shifts that fit
  in 86 min)
- Per-taxon temperature, neighbour smoothing, rank-aware amplification
- Placeholder zero CSV written first to avoid `Notebook Threw Exception`
- Auto fallback to dry-run on `train_soundscapes[:20]` when test isn't
  mounted (editor commit)

If the open-source notebook ever stops working, this is the strongest
**self-contained** submission we own. Mounts only the
`birdclef-2026-bundle-y1` dataset (no Perch / TF needed).
