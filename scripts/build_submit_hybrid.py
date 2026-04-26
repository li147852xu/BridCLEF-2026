"""Build kaggle_submit/submit_v6_hybrid.ipynb.

Takes the open-source 0.924 notebook
(``legacy/notebooks/pantanal-distill-birdclef2026-improvement-a4dc68.ipynb``)
and injects two cells so it ALSO runs our Plan Y.1 5-fold OpenVINO ensemble,
then late-fuses the per-window probability matrix with theirs in
**probability space** before their post-processing chain (rank-aware, delta
smooth, per-class thresholds) runs.

Why probability-space fusion (not logit-space): their final_test_scores is
the output of ProtoSSM + MLP probe ensemble in some unspecified scale; our
OV outputs are calibrated logits from BCE-style training. Adding logits at
different scales is fragile. Blending after their sigmoid keeps both models
on a common [0, 1] scale, and the rest of their post-proc chain (which is
the bulk of why they hit 0.924) still applies on top of the blended probs.

Two insertions, one edit:
    1. After original cell 1 (their TF 2.20 install), insert "Cell 1b": offline
       install OpenVINO from our Y.1 bundle's wheels/ dir. Never raises.
    2. After original cell 46 (their per-test-file Perch loop), insert
       "Cell 46b": load Y.1 manifest, compile 5 OV folds, iterate the same
       ``test_paths`` (so order matches meta_test row_ids), produce
       ``our_probs_test`` of shape (n_windows, N_CLASSES) in PRIMARY_LABELS order.
    3. Inside cell 50, immediately after ``probs = sigmoid(scaled_scores)``,
       inject the blend with our_probs_test. Their entire post-proc chain
       below (file-level scaling, rank-aware, delta smooth, per-class
       thresholds) runs on the blended probs.

Run: ``python scripts/build_submit_hybrid.py``
Output: ``kaggle_submit/submit_v6_hybrid.ipynb`` (overwritten)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_NB = REPO / "legacy" / "notebooks" / "pantanal-distill-birdclef2026-improvement-a4dc68.ipynb"
# Canonical hybrid output = the W=0.30 / rank_arith Phase-1 experiment A1.
# Variants A0, A2, A3, B, C, D are derived from this file by
# scripts/build_phase1_experiments.py.
OUT_NB = REPO / "kaggle_submit" / "20_hybrid_phase1" / "exp_A1_W030_rank_arith.ipynb"


# Cell inserted right after their TF 2.20 install (becomes cell index 2 in the new nb)
CELL_OV_INSTALL = r'''# Cell 1b — Install OpenVINO from our Plan Y.1 bundle (offline, no Internet on submit)
#
# We need OpenVINO at runtime to load our 5 fold IR files. The bundle's
# wheels/ dir contains numpy + openvino_telemetry + openvino, but we
# DELIBERATELY skip the numpy wheel: Kaggle's preinstalled numpy is the
# one scipy / sklearn / numpy-itself's C extensions were compiled against,
# and replacing it via --no-deps breaks them with errors like
# "cannot import name '_center' from 'numpy._core.umath'". The 0.924
# notebook needs sklearn (PCA, MLPClassifier) and scipy under the hood,
# so we install only telemetry + openvino and leave numpy alone.
import glob, subprocess, sys, os, time

# Wall-clock anchor for ETA budgeting in the hybrid OV cell later.
# Submitting Cells 0-1 (TF wheels) already burns ~80s; we measure from here so
# the OV ETA reflects "time the user has burned since the notebook started".
_WALL_START = time.time()

_OV_WHEEL_DIRS = [
    "/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y1/wheels",
    "/kaggle/input/birdclef-2026-bundle-y1/wheels",
    "/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y/wheels",
    "/kaggle/input/birdclef-2026-bundle-y/wheels",
]
_wheel_dir = next((d for d in _OV_WHEEL_DIRS if os.path.isdir(d)), None)
HYBRID_OV_OK = False
if _wheel_dir is None:
    print("HYBRID: no OV wheels found; hybrid blend will be skipped (still get 0.924-baseline output).")
else:
    # Skip numpy* wheels — Kaggle's preinstalled numpy must NOT be replaced.
    _whls = [w for w in sorted(glob.glob(_wheel_dir + "/*.whl"))
             if not os.path.basename(w).lower().startswith("numpy")]
    # Install order: telemetry first (openvino imports it at top level), then openvino.
    def _ord(p):
        n = os.path.basename(p).lower()
        return 0 if "telemetry" in n else 1 if n.startswith("openvino") else 2
    _whls.sort(key=_ord)
    print("HYBRID wheels to install:", [os.path.basename(w) for w in _whls])
    for _w in _whls:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _w], check=True)
        except subprocess.CalledProcessError as _e:
            print("HYBRID wheel install failed for", os.path.basename(_w), ":", _e)
    try:
        import openvino as _ov  # noqa: F401
        print("HYBRID: openvino", _ov.__version__, "loaded.")
        HYBRID_OV_OK = True
    except Exception as _e:
        print("HYBRID: openvino import failed (", type(_e).__name__, _e, "); skipping hybrid.")
'''


# Cell inserted right after their per-file Perch inference (their original cell 46)
CELL_OUR_INFER = r'''# Cell 46b — Plan Y.1 OpenVINO ensemble, file-batched + adaptively-degraded.
#
# Goal: extract maximum signal from our 5 folds within the 90-min Kaggle cap.
# Two performance levers and one safety net.
#
# Lever 1 — file-batching (BATCH_FILES = 4):
#   Decode + mel-spec ``BATCH_FILES`` files at a time, then run ONE forward
#   per fold on (BATCH_FILES * 12, 1, 128, T). OpenVINO amortises Python
#   overhead and parallelises across the bigger batch -> ~1.7x speedup vs
#   per-file forwards.
#
# Lever 2 — adaptive fold count (5 → 3 → 1):
#   Compile all 5 folds upfront. Probe with the chosen fold count on the
#   first PROBE_BATCHES batches; project remaining wall. If projection >
#   budget, drop fold count down the FOLD_LADDER and re-probe. Only bail
#   to baseline when even a single fold cannot fit. We keep MAXIMUM
#   diversity at every level rather than pre-cutting.
#
# Safety net — mid-loop wall guard (HARD_ABORT_WALL_S):
#   Inside the file loop, if wall clock crosses the abort threshold, stop
#   producing OV. If we have full coverage, blend; otherwise bail to
#   baseline (no partial blend — would crash the shape assert downstream).
#
# Output: ``our_probs_test`` (n_windows, N_CLASSES) PRIMARY_LABELS-ordered,
# or None if budget pressure forced a full bail.
#
# Output: ``our_probs_test`` of shape (n_windows, N_CLASSES) in PRIMARY_LABELS
# order, where ``n_windows = len(test_paths) * N_WINDOWS``. Order matches
# ``meta_test["row_id"]`` because we iterate the SAME ``test_paths`` list and
# emit windows in the SAME 5-second order, then permute model classes
# (manifest.primary_labels) onto sample_sub.columns (PRIMARY_LABELS) via
# LABEL_PERM_OUR.
#
# Failure mode: if anything below raises or HYBRID_OV_OK is False, we set
# our_probs_test = None and the blend in the post-proc cell becomes a no-op,
# so we degrade gracefully back to the open-source 0.924 baseline.
import time, json
from pathlib import Path
import numpy as np

# ---- Performance knobs ----
BATCH_FILES = 4                # OV forward batch = BATCH_FILES * 12 windows
PROBE_BATCHES = 2              # probe this many batches for ETA
WALL_BUDGET_S = 80 * 60        # 10-min cushion under 90-min Kaggle cap
HARD_ABORT_WALL_S = 84 * 60    # mid-loop abort threshold

# Adaptive fold ladder: if 5 folds won't fit, drop to 3, then 1; only bail
# (HYBRID_W → no-op) if even 1 fold can't fit. Order within each level
# picks the most-trusted folds first (S7 val_auc descending: 3, 2, 1, 4, 0).
FOLD_PRIORITY = [3, 2, 1, 4, 0]
FOLD_LADDER = [5, 3, 1]

try:
    _WALL_T0 = _WALL_START
except NameError:
    _WALL_T0 = time.time()

our_probs_test = None
HYBRID_BUNDLE = None

if not HYBRID_OV_OK:
    print("HYBRID skipped: OV not loaded.")
else:
    _BUNDLES = [
        Path("/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y1"),
        Path("/kaggle/input/birdclef-2026-bundle-y1"),
        Path("/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y"),
        Path("/kaggle/input/birdclef-2026-bundle-y"),
    ]
    HYBRID_BUNDLE = next((p for p in _BUNDLES if (p / "manifest.json").exists()), None)
    print("HYBRID bundle:", HYBRID_BUNDLE)
    _wall_now = time.time() - _WALL_T0
    if _wall_now > WALL_BUDGET_S * 0.7:
        # Already burned too much (Perch probably took longer than estimated).
        print(f"HYBRID early-skip: wall={_wall_now/60:.1f}min already > 70% of budget; "
              f"degrading to 0.924 baseline.")
        HYBRID_BUNDLE = None

if HYBRID_OV_OK and HYBRID_BUNDLE is not None:
    with open(HYBRID_BUNDLE / "manifest.json") as _f:
        _MAN = json.load(_f)
    _OUR_LABELS = [str(x) for x in _MAN["primary_labels"]]
    _MEL = _MAN["mel"]
    _idx = {l: i for i, l in enumerate(_OUR_LABELS)}
    try:
        LABEL_PERM_OUR = np.array([_idx[l] for l in PRIMARY_LABELS], dtype=np.int64)
        print("HYBRID label perm: identity?",
              np.array_equal(LABEL_PERM_OUR, np.arange(N_CLASSES)),
              "  N=", len(LABEL_PERM_OUR))
    except KeyError as _e:
        print("HYBRID label mismatch — sample_sub col", _e, "not in our manifest. Skipping.")
        LABEL_PERM_OUR = None

    if LABEL_PERM_OUR is not None:
        # ----- mel front-end: must mirror stages/s2_prepare_mel.py + s9 export shape -----
        def _build_mel_fb(sr, n_fft, n_mels, f_min, f_max):
            def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
            def mel_to_hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
            m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
            f_pts = mel_to_hz(m_pts)
            bin_pts = np.floor((n_fft + 1) * f_pts / sr).astype(int)
            fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
            for m in range(1, n_mels + 1):
                l, c, r = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
                if c == l: c = l + 1
                if r == c: r = c + 1
                fb[m-1, l:c] = (np.arange(l, c) - l) / (c - l)
                fb[m-1, c:r] = (r - np.arange(c, r)) / (r - c)
            return fb

        def _mel_from_wave(y, fb, n_fft, hop, win_length, db_lo, db_hi, power):
            window = np.hanning(win_length).astype(np.float32)
            pad = n_fft // 2
            y = np.pad(y, (pad, pad), mode="reflect")
            n_frames = 1 + (len(y) - win_length) // hop
            frames = np.lib.stride_tricks.sliding_window_view(y, win_length)[::hop][:n_frames] * window
            spec = np.fft.rfft(frames, n=n_fft).astype(np.complex64)
            mag = np.abs(spec) ** power
            mel = np.maximum(mag @ fb.T, 1e-10)
            db = np.clip(10.0 * np.log10(mel), db_lo, db_hi)
            return ((db - db_lo) / (db_hi - db_lo)).T.astype(np.float32)

        OUR_FB = _build_mel_fb(_MEL["sr"], _MEL["n_fft"], _MEL["n_mels"], _MEL["f_min"], _MEL["f_max"])
        OUR_TARGET_T = int(np.ceil(_MEL["sr"] * _MEL["window_seconds"] / _MEL["hop"])) + 1

        # ----- compile ALL available folds upfront (cheap, ~5-10s total) -----
        import openvino as ov
        _core = ov.Core()
        _core.set_property("CPU", {"INFERENCE_NUM_THREADS": 4})

        _by_fold = {int(f.get("fold", -1)): f for f in _MAN["folds"]
                    if not f.get("skipped") and "ir_xml" in f}
        ALL_FOLD_OV = {}   # fold_idx -> compiled_model
        for _i in FOLD_PRIORITY:
            _fmeta = _by_fold.get(_i)
            if _fmeta is None: continue
            _xml = HYBRID_BUNDLE / Path(_fmeta["ir_xml"]).name
            if not _xml.exists():
                _alt = HYBRID_BUNDLE / "export" / _xml.name
                if _alt.exists(): _xml = _alt
            if not _xml.exists():
                print("HYBRID skip fold", _i, "— xml missing"); continue
            try:
                ALL_FOLD_OV[_i] = _core.compile_model(str(_xml), "CPU")
            except Exception as _e:
                print("HYBRID compile fail fold", _i, ":", _e)
        _avail_folds = [i for i in FOLD_PRIORITY if i in ALL_FOLD_OV]
        print(f"HYBRID compiled folds {_avail_folds} (out of {FOLD_PRIORITY})")

        if _avail_folds:
            import soundfile as sf
            _SR = _MEL["sr"]; _NW = _MEL["windows_per_file"]; _WS = _MEL["window_seconds"]
            _WANT = _SR * _WS * _NW

            def _decode_60s(p):
                y, _sr = sf.read(str(p), dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < _WANT: y = np.pad(y, (0, _WANT - len(y)))
                elif len(y) > _WANT: y = y[:_WANT]
                return y

            def _mel_for_one(p):
                y = _decode_60s(p)
                w = y.reshape(_NW, _SR * _WS)
                mels = np.stack([
                    _mel_from_wave(w[i], OUR_FB, _MEL["n_fft"], _MEL["hop"],
                                   _MEL["win_length"], _MEL["db_lo"], _MEL["db_hi"], _MEL["power"])
                    for i in range(_NW)
                ], axis=0)
                if mels.shape[-1] < OUR_TARGET_T:
                    mels = np.pad(mels, ((0,0),(0,0),(0, OUR_TARGET_T - mels.shape[-1])), mode="edge")
                elif mels.shape[-1] > OUR_TARGET_T:
                    mels = mels[..., :OUR_TARGET_T]
                return mels[:, None, :, :].astype(np.float32)  # (12, 1, 128, T)

            def _our_infer_batch(paths, fold_subset):
                """Decode + mel-spec all paths, then ONE forward per fold on the
                stacked (N*12, 1, 128, T) tensor. Returns (N*12, 234) probs avg."""
                mels_per_file = [_mel_for_one(p) for p in paths]
                x = np.concatenate(mels_per_file, axis=0)
                acc = None
                for _f in fold_subset:
                    _c = ALL_FOLD_OV[_f]
                    _o = _c(x)[_c.outputs[0]]
                    _p = 1.0 / (1.0 + np.exp(-_o))
                    acc = _p if acc is None else acc + _p
                return acc / len(fold_subset)  # (N*12, 234)

            _n_total = len(test_paths)

            # ---- Adaptive probe ladder: pick the largest fold count that fits ----
            chosen_folds = None
            per_file_s = None
            _probe_files_n = min(BATCH_FILES * PROBE_BATCHES, _n_total)
            _probe_paths = list(test_paths[:_probe_files_n])

            for _n_folds in FOLD_LADDER:
                if _n_folds > len(_avail_folds): continue
                _subset = _avail_folds[:_n_folds]
                _t_probe = time.time()
                # Probe in real batches (so per-file s reflects batch amortisation)
                for _b0 in range(0, _probe_files_n, BATCH_FILES):
                    _bp = _probe_paths[_b0:_b0 + BATCH_FILES]
                    try: _ = _our_infer_batch(_bp, _subset)
                    except Exception as _e:
                        print(f"HYBRID probe-{_n_folds}f fail:", type(_e).__name__, _e)
                        _ = None
                _probe_dt = time.time() - _t_probe
                _pf = _probe_dt / max(1, _probe_files_n)
                _wall = time.time() - _WALL_T0
                _proj = _wall + _pf * (_n_total - _probe_files_n)
                print(f"HYBRID probe folds={_n_folds} files={_probe_files_n} "
                      f"in {_probe_dt:.1f}s ({_pf:.2f}s/file)  wall={_wall/60:.1f}min "
                      f"projected={_proj/60:.1f}min budget={WALL_BUDGET_S/60:.0f}min")
                if _proj <= WALL_BUDGET_S:
                    chosen_folds = _subset
                    per_file_s = _pf
                    print(f"HYBRID CHOSEN: folds={chosen_folds} (max from ladder that fits)")
                    break
                else:
                    print(f"HYBRID dropping from {_n_folds} folds — over budget")

            if chosen_folds is None:
                print("HYBRID BAIL: even 1 fold cannot fit. Falling back to 0.924 baseline.")
            else:
                # ---- Full batched run with mid-loop wall guard ----
                _per_file_probs = []
                # Probe outputs already computed above but discarded; redo the
                # probe range with the chosen fold subset to keep its results.
                for _b0 in range(0, _probe_files_n, BATCH_FILES):
                    _bp = _probe_paths[_b0:_b0 + BATCH_FILES]
                    try:
                        _out = _our_infer_batch(_bp, chosen_folds)  # (N*12, 234)
                        _per_file_probs.append(_out.reshape(len(_bp), _NW, -1)[:, :, LABEL_PERM_OUR])
                    except Exception as _e:
                        print("HYBRID re-probe fail:", _e)
                        for _ in _bp:
                            _per_file_probs.append(np.zeros((1, _NW, N_CLASSES), dtype=np.float32))

                _aborted = False
                for _b0 in range(_probe_files_n, _n_total, BATCH_FILES):
                    _wall = time.time() - _WALL_T0
                    if _wall > HARD_ABORT_WALL_S:
                        print(f"HYBRID HARD ABORT @ file {_b0}/{_n_total}, wall={_wall/60:.1f}min")
                        _aborted = True
                        break
                    _bp = list(test_paths[_b0:_b0 + BATCH_FILES])
                    try:
                        _out = _our_infer_batch(_bp, chosen_folds)
                        _per_file_probs.append(_out.reshape(len(_bp), _NW, -1)[:, :, LABEL_PERM_OUR])
                    except Exception as _e:
                        print("HYBRID infer batch fail:", type(_e).__name__, _e)
                        _per_file_probs.append(np.zeros((len(_bp), _NW, N_CLASSES), dtype=np.float32))
                    if ((_b0 // BATCH_FILES) + 1) % 25 == 0:
                        _w = (time.time() - _WALL_T0) / 60
                        _done = _b0 + len(_bp)
                        _eta = (_n_total - _done) * per_file_s / 60
                        print(f"HYBRID OV {_done}/{_n_total} files (folds={len(chosen_folds)}) "
                              f"wall={_w:.1f}min eta_remaining={_eta:.1f}min")

                if not _aborted:
                    # stack: list of (n_b, 12, 234) -> (sum_n_b * 12, 234)
                    our_probs_test = np.concatenate(
                        [a.reshape(-1, N_CLASSES) for a in _per_file_probs], axis=0
                    ).astype(np.float32)
                    print(f"HYBRID our_probs_test shape: {our_probs_test.shape}  "
                          f"range: {our_probs_test.min():.4f}..{our_probs_test.max():.4f}  "
                          f"folds_used={len(chosen_folds)}  "
                          f"total_wall: {(time.time()-_WALL_T0)/60:.1f}min")
'''


# Snippet inserted into cell 50 right BEFORE `# --- Build submission ---`
# (i.e. after ALL of their post-proc has finished). Two design changes vs v1:
#   * blend AT THE VERY END so the open-source notebook's per-class threshold,
#     rank-aware, file-level scale, delta smooth chain runs untouched on their
#     own probabilities (it was tuned for their distribution). v1 blended right
#     after sigmoid which detuned that chain and cost us ~0.004 PB.
#   * rank-average instead of probability-average. Kaggle macro AUC is rank-
#     based so calibration mismatch between the two models doesn't matter; only
#     within-column ordering does. Rank-blend preserves both models' ranking
#     information optimally.
BLEND_SNIPPET = '''

# === HYBRID Phase-1 tuning knobs (edit these in Kaggle UI to A/B) =========
# All three knobs are independent. Effective weight per (row, class) is:
#   w_class = PER_TAXON_W[class_taxon] if PER_TAXON_W else HYBRID_W
# Then blend method = BLEND_MODE picks how their probs vs ours combine.
HYBRID_W = 0.30
# Set PER_TAXON_W = None to use the scalar above; set to a dict to use
# per-taxon weights (good when our model is stronger on one taxon than
# another). Class taxa come from CLASS_NAME_MAP set in cell 11.
# Example tuned guess:
#   PER_TAXON_W = {"Aves": 0.25, "Amphibia": 0.35, "Insecta": 0.40,
#                  "Mammalia": 0.30, "Reptilia": 0.30}
PER_TAXON_W = None
# 'rank_arith'  : per-class column ranks, weighted arithmetic mean (default)
# 'rank_geo'    : per-class column ranks, weighted GEOMETRIC mean
#                 (sharper on tails; helps when one model is confident wrong)
# 'prob_arith'  : the v2 probability-space arithmetic mean (loses to rank;
#                 only useful if you want to roll back to v2 behaviour)
BLEND_MODE = 'rank_arith'
# ==========================================================================

try:
    if our_probs_test is not None and our_probs_test.shape == probs.shape:
        # Build per-class weight vector w_vec of shape (N_CLASSES,)
        if PER_TAXON_W is not None:
            try:
                w_vec = np.array([
                    float(PER_TAXON_W.get(CLASS_NAME_MAP.get(lbl, "Aves"), HYBRID_W))
                    for lbl in PRIMARY_LABELS
                ], dtype=np.float32)
                _w_summary = {k: float(v) for k, v in PER_TAXON_W.items()}
            except Exception as _e:
                print(f"HYBRID PER_TAXON_W failed ({_e}); falling back to scalar HYBRID_W={HYBRID_W}")
                w_vec = np.full(N_CLASSES, HYBRID_W, dtype=np.float32)
                _w_summary = {"scalar": HYBRID_W}
        else:
            w_vec = np.full(N_CLASSES, HYBRID_W, dtype=np.float32)
            _w_summary = {"scalar": HYBRID_W}
        # Broadcast to (1, N_CLASSES) for elementwise blend.
        w_ours = w_vec[None, :]
        w_them = 1.0 - w_ours

        if BLEND_MODE in ('rank_arith', 'rank_geo'):
            from scipy.stats import rankdata as _rankdata
            _n = probs.shape[0]
            _their_r = _rankdata(probs, axis=0).astype(np.float32) / _n
            _our_r   = _rankdata(our_probs_test.astype(np.float32), axis=0).astype(np.float32) / _n
            if BLEND_MODE == 'rank_arith':
                _blended = w_them * _their_r + w_ours * _our_r
            else:  # rank_geo
                _eps = 1e-9
                _blended = np.exp(w_them * np.log(_their_r + _eps) + w_ours * np.log(_our_r + _eps))
        elif BLEND_MODE == 'prob_arith':
            _blended = w_them * probs + w_ours * our_probs_test.astype(np.float32)
        else:
            print(f"HYBRID unknown BLEND_MODE={BLEND_MODE!r}; defaulting to rank_arith.")
            from scipy.stats import rankdata as _rankdata
            _n = probs.shape[0]
            _their_r = _rankdata(probs, axis=0).astype(np.float32) / _n
            _our_r   = _rankdata(our_probs_test.astype(np.float32), axis=0).astype(np.float32) / _n
            _blended = w_them * _their_r + w_ours * _our_r

        probs = np.clip(_blended, 0.0, 1.0).astype(np.float32)
        print(f"HYBRID blended  mode={BLEND_MODE!r}  weights={_w_summary}  "
              f"after their full post-proc chain.")
    else:
        print(f"HYBRID skipped at blend: our_probs_test="
              f"{None if our_probs_test is None else our_probs_test.shape}, "
              f"theirs={probs.shape} -- pure 0.924 baseline output stands.")
except NameError:
    print("HYBRID skipped at blend: our_probs_test not defined (cell 46b errored).")
'''


def main() -> None:
    nb = json.loads(SOURCE_NB.read_text())

    # Sanity: original cell 50 must still contain the line we anchor against.
    # We insert AFTER the post-proc chain (right before submission build) so
    # the open-source notebook's tuned post-proc runs on its own probs only.
    cell_50 = nb["cells"][50]
    src_50 = "".join(cell_50["source"]) if isinstance(cell_50["source"], list) else cell_50["source"]
    anchor = "# --- Build submission ---"
    if anchor not in src_50:
        raise SystemExit(f"Sanity fail: anchor {anchor!r} not in cell 50 of source notebook.")

    # Insert OUR_INFER cell *after* original index 46 (so original 46 stays at 46).
    nb["cells"].insert(47, _new_code_cell(CELL_OUR_INFER))
    # Insert OV_INSTALL cell *after* original index 1 (so it becomes index 2).
    nb["cells"].insert(2, _new_code_cell(CELL_OV_INSTALL))

    # Modify the blend in what was originally cell 50 (now shifted by +2 → cell 52).
    # Insert BEFORE the anchor so our blend runs after all post-proc finishes.
    target_idx = 52
    cell_target = nb["cells"][target_idx]
    src = "".join(cell_target["source"]) if isinstance(cell_target["source"], list) else cell_target["source"]
    if anchor not in src:
        raise SystemExit(f"Anchor missing after shift; expected at cell {target_idx}.")
    new_src = src.replace(anchor, BLEND_SNIPPET + anchor, 1)
    cell_target["source"] = new_src.splitlines(keepends=True)
    cell_target["outputs"] = []
    cell_target["execution_count"] = None

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT_NB}  ({len(nb['cells'])} cells)")


def _new_code_cell(src: str) -> dict:
    lines = src.lstrip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


if __name__ == "__main__":
    main()
