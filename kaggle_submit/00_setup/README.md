# `00_setup/` — one-time bundle bootstrap

## hf_to_kaggle_dataset.ipynb

Pulls our `bundle_y1/` directory from the private HF repo
`Tiantanghuaxiao/birdclef-2026-ckpts` and creates / updates the Kaggle
Dataset `tiantanghuaxiao/birdclef-2026-bundle-y1`.

### When to run

- **First time** setting up the Plan Y.1 bundle on Kaggle.
- **After a new bundle** is uploaded to HF (e.g. Plan Z adds a second
  backbone — see `bundle_y2/` etc.). In that case copy this notebook to
  a new file (e.g. `hf_to_kaggle_dataset_y2.ipynb`), edit `bundle_y1` →
  `bundle_y2` and the dataset slug, then run.

### Why we go through HF

AutoDL CN regions can't reach Kaggle's GCS resumable-upload endpoint
directly (specific IP range is blocked). HF Hub upload works fine from
AutoDL via `network_turbo`, and the Kaggle CLI inside a Kaggle Notebook
can talk to itself. So:

```
AutoDL  ──(hf upload-folder)──►  HF private repo
                                       │
                                       ▼
                           Kaggle Notebook (this file)
                          snapshot_download + kaggle datasets create
                                       │
                                       ▼
                       Kaggle Dataset (mounted by submit notebooks)
```

### Steps inside Kaggle

1. Add HF token as a Kaggle Notebook **Secret** named `HF_TOKEN`.
2. Settings → **Internet: On**.
3. Save & Run All.
4. Verify the dataset appears under Datasets → My datasets.
