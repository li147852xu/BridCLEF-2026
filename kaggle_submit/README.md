# `kaggle_submit/` — Kaggle submission notebooks

All notebooks here are designed to be **imported into Kaggle as-is** (File →
Import notebook → from GitHub URL). No code edits are needed before Save &
Run All.

## Folder map

```
kaggle_submit/
├── 00_setup/
│   └── hf_to_kaggle_dataset.ipynb       # one-time: HF bundle → Kaggle dataset
├── 10_baseline/
│   ├── submit_v2_legacy.ipynb           # PB 0.883 (legacy Perch-distill, archived)
│   └── submit_v5_plan_y.ipynb           # PB 0.866-0.868 (Plan Y CNN-only)
├── 20_hybrid_phase1/
│   ├── exp_A0_W020_rank_arith.ipynb     # W=0.20, conservative
│   ├── exp_A1_W030_rank_arith.ipynb     # ★ current best (PB 0.925)
│   ├── exp_A2_W040_rank_arith.ipynb
│   ├── exp_A3_W050_rank_arith.ipynb
│   ├── exp_B_W030_rank_geo.ipynb        # geometric mean blend
│   ├── exp_C_per_taxon_arith.ipynb      # per-taxon weights
│   └── exp_D_combo_geo_per_taxon.ipynb  # combo of best knobs
└── README.md (this file)
```

## What each folder is for

### `00_setup/` — one-time bundle bootstrap

`hf_to_kaggle_dataset.ipynb` mirrors our Hugging Face bundle
(`Tiantanghuaxiao/birdclef-2026-ckpts:bundle_y1/`) into a Kaggle Dataset
(`tiantanghuaxiao/birdclef-2026-bundle-y1`). Run it **once** when you've
generated a new bundle; afterwards every submission notebook in
`20_hybrid_phase1/` mounts the same Kaggle Dataset.

### `10_baseline/` — historical baselines

| File | What it does | PB |
|---|---|---|
| `submit_v2_legacy.ipynb` | Old Perch + linear probe + MLP student stack | 0.883 |
| `submit_v5_plan_y.ipynb` | 5-fold OpenVINO eca_nfnet_l0 alone (no fusion) | 0.866-0.868 |

Kept for reference and as a rollback path.

### `20_hybrid_phase1/` — current sweep

Each `exp_*.ipynb` is a clone of the open-source 0.924 ProtoSSM/MLP
notebook + our 5-fold OpenVINO ensemble fused via rank-blend. They differ
ONLY in the values baked into cell 52:

| File | HYBRID_W | PER_TAXON_W | BLEND_MODE | Notes |
|---|---|---|---|---|
| `exp_A0_W020_rank_arith` | 0.20 | – | rank_arith | conservative weight |
| `exp_A1_W030_rank_arith` | 0.30 | – | rank_arith | **current best, PB 0.925** |
| `exp_A2_W040_rank_arith` | 0.40 | – | rank_arith | bigger weight |
| `exp_A3_W050_rank_arith` | 0.50 | – | rank_arith | balanced 50/50 |
| `exp_B_W030_rank_geo` | 0.30 | – | rank_geo | geometric mean |
| `exp_C_per_taxon_arith` | 0.30 | Aves 0.20 / Insecta 0.45 / ... | rank_arith | per-taxon |
| `exp_D_combo_geo_per_taxon` | 0.40 | Aves 0.30 / Insecta 0.50 / ... | rank_geo | combo |

## Kaggle data sources to attach (same for all `20_hybrid_phase1/*` notebooks)

1. `birdclef-2026` — competition dataset
2. `tiantanghuaxiao/birdclef-2026-bundle-y1` — our 5-fold OV bundle
3. `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` — Perch v2
4. `ashok205/tf-wheels` — TF 2.20 wheels (offline install)
5. `<perch-meta>` — cached Perch outputs on train_soundscapes (the legacy
   notebook expects this; check the source for the exact slug)

## Recommended Phase-1 workflow

1. Pick an experiment that hasn't been submitted yet.
2. Open it on Kaggle (import from GitHub or click the file URL).
3. Make sure the 5 data sources above are attached.
4. **Save Version → Save & Run All** (~10 min editor commit).
5. Once "Succeeded": **Submit to Competition**.
6. Note the PB score next to the experiment name.
7. Repeat for the other experiments.

Each Save Version + Submit cycle is roughly 15–25 min wall-clock; you can
batch all 6 sweeps in an afternoon.

## Regenerating the experiments

If the canonical hybrid notebook (`exp_A1_W030_rank_arith.ipynb`) needs
to be rebuilt from the legacy 0.924 notebook + our injection cells,
run:

```bash
python scripts/build_submit_hybrid.py        # rebuilds exp_A1 (canonical)
python scripts/build_phase1_experiments.py   # propagates to A0/A2/A3/B/C/D
```

The second script clones the canonical notebook and replaces the three
literal lines for `HYBRID_W`, `PER_TAXON_W`, `BLEND_MODE` per the
`EXPERIMENTS` table inside it. Add or tweak experiments by editing that
table and re-running.

## Score history

| Stage | PB |
|---|---|
| v2 Perch-distill baseline | 0.883 |
| Plan Y standalone (CNN only) | 0.868 |
| Hybrid v2 (probability-space blend) | 0.920 |
| Hybrid v3 (rank-space blend, W=0.30, post-proc tail) | **0.925** |
| Phase-1 sweep target | ≥ 0.93 |
| Plan Z (2nd backbone, future) | ≥ 0.94 |
| Plan Phase-3 (XC-pretrain, future) | ≥ 0.95 |
