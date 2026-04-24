"""One-shot rewrite of kaggle_submit/submit_v5.ipynb.

Fixes:
  A. Cell 0 now installs ALL wheels in the bundle's wheels/ dir
     (not just ``openvino-*.whl``). The old glob missed
     ``openvino_telemetry-*.whl`` — a hard runtime dep of openvino —
     and that was the most likely root cause of the PB=0.500
     "submission succeeded in 3 min with all-zero placeholder" result.
  B. Cell 1 restores the legacy dry-run fallback: when
     ``len(test_paths) == 0`` (which is *always* the case in the
     editor Save Version commit under Code Competition), iterate over
     ``train_soundscapes`` so the commit log actually exercises Cells
     3-7 end-to-end. Without this, any bug in the real inference path
     is invisible until hidden-rerun scoring (which has no log).
  C. Cell 7 uses an adaptive ETA loop: probe the first N files with
     single-shot inference, measure real wall-time, then pick the
     largest TTA set (1/2/3 shifts) that fits inside the 86-min budget.
     We also checkpoint submission.csv every 50 files, so a timeout
     midway still leaves a valid partial CSV in /kaggle/working.
  D. Cell 6 drops the dead ``topn_smooth`` identity function (the
     multiplication by zero made it a no-op regardless).
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "kaggle_submit" / "submit_v5.ipynb"


CELL_0 = r'''# Cell 0 — install every wheel in the bundle's wheels/ dir.
#
# Bug-fix (2026-04-23): the previous version only pip-installed
# ``openvino-*.whl``, which silently skipped ``openvino_telemetry-*.whl``.
# openvino 2024.x imports openvino_telemetry at top level, so the later
# ``import openvino`` in Cell 4 would raise ModuleNotFoundError, trip
# ABORT=True, and leave the all-zero placeholder CSV as the submission
# (PB = 0.500 exactly). We now install ALL wheels in the dir.
#
# Never raise here — Cell 1 writes a placeholder submission first so
# Kaggle cannot report "Notebook Threw Exception".
import glob, subprocess, sys, os

_WHEEL_DIR_CANDIDATES = [
    '/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y1/wheels',
    '/kaggle/input/birdclef-2026-bundle-y1/wheels',
    '/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y/wheels',
    '/kaggle/input/birdclef-2026-bundle-y/wheels',
    '/kaggle/input/datasets/li147852xu/birdclef-2026/wheels',
    '/kaggle/input/birdclef-2026/wheels',
    '/kaggle/input/birdclef-2026-bundle/wheels',
]

WHEEL_DIR = next((d for d in _WHEEL_DIR_CANDIDATES if os.path.isdir(d)), None)
OV_OK = False
if WHEEL_DIR is None:
    print('WARN: no wheels/ dir found under any candidate — Cell 4 will try the preinstalled openvino if any.')
else:
    all_whl = sorted(glob.glob(f'{WHEEL_DIR}/*.whl'))
    # Install order: numpy first (ABI base), telemetry, openvino last.
    def _sort_key(p: str) -> int:
        name = os.path.basename(p).lower()
        if name.startswith('numpy'): return 0
        if 'telemetry' in name: return 1
        if name.startswith('openvino'): return 2
        return 3
    all_whl.sort(key=_sort_key)
    print('wheels found:', [os.path.basename(w) for w in all_whl])
    for whl in all_whl:
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', whl],
                check=True,
            )
            print('installed', os.path.basename(whl))
        except subprocess.CalledProcessError as e:
            print('wheel install FAILED for', os.path.basename(whl), ':', e)

try:
    import openvino as ov  # noqa: F401
    print('openvino import OK ->', ov.__version__)
    OV_OK = True
except Exception as e:
    print('openvino import FAILED:', type(e).__name__, e)
'''


CELL_1 = r'''# Cell 1 — paths, sample_sub, test enumeration (with dry-run) + placeholder CSV.
#
# Bug-fix (2026-04-23): legacy's PB-0.883 notebook fell back to
# train_soundscapes[:20] when test_soundscapes was empty. Under
# Kaggle's Code Competition, editor Save Version commit ALWAYS sees
# len(test_soundscapes)==0 (test files only mount during hidden
# scoring rerun). Without the dry-run fallback, the commit log never
# shows a real forward pass, so every downstream bug is invisible
# until scoring — where there is no log. We restore the fallback.
import time, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

_WALL_START = time.time()
INPUT_ROOT = Path('/kaggle/input')

# ----- Diagnostic: print /kaggle/input tree (2 levels) -----
print('=== /kaggle/input layout ===')
for p in sorted(INPUT_ROOT.glob('*')):
    print(' ', p, '(dir)' if p.is_dir() else '')
    if p.is_dir():
        for p2 in sorted(p.glob('*'))[:8]:
            print('    ', p2.name)
print()

# ----- Resolve competition root -----
BASE_CANDIDATES = [
    Path('/kaggle/input/competitions/birdclef-2026'),
    Path('/kaggle/input/birdclef-2026'),
]
BASE = next((p for p in BASE_CANDIDATES if (p / 'sample_submission.csv').exists()), None)
if BASE is None:
    _hits = list(INPUT_ROOT.rglob('sample_submission.csv'))
    if _hits:
        BASE = _hits[0].parent
if BASE is None:
    raise FileNotFoundError('No sample_submission.csv found anywhere under /kaggle/input')
print('BASE:', BASE)

# ----- sample_submission + PRIMARY_LABELS (legacy-correct source of truth) -----
sample_sub = pd.read_csv(BASE / 'sample_submission.csv')
SAMPLE_COLS = list(sample_sub.columns)
PRIMARY_LABELS = SAMPLE_COLS[1:]
N_CLASSES = len(PRIMARY_LABELS)
print(f'sample_sub shape: {sample_sub.shape} | n_classes: {N_CLASSES} | first labels: {PRIMARY_LABELS[:3]}')

# ----- test_paths resolution (legacy -> rglob -> stem-scan) -----
test_paths = sorted((BASE / 'test_soundscapes').glob('*.ogg'))
print(f'test_soundscapes glob under BASE: {len(test_paths)} files')
if not test_paths and len(sample_sub) >= 100:
    # In real scoring the primary glob under BASE always matches. This
    # rglob fallback is only useful if Kaggle ever relocates the test
    # files; skip it in the editor stub case where it just wastes ~75s
    # walking /kaggle/input for nothing.
    for tc in INPUT_ROOT.rglob('test_soundscapes'):
        if tc.is_dir():
            hits = sorted(tc.glob('*.ogg'))
            if hits:
                test_paths = hits
                print(f'FALLBACK test_soundscapes at {tc}: {len(test_paths)} files')
                break
# Only attempt the expensive full-tree stem-scan when sample_sub looks like
# a real scoring mount (>= 100 rows). In editor Save Version the stub is
# 3 rows and the scan just wastes ~90s rglobbing /kaggle/input for nothing.
if not test_paths and len(sample_sub) >= 100:
    import re
    stems_expected, seen = [], set()
    for rid in sample_sub['row_id'].astype(str):
        m = re.match(r'^(.+)_(\d+)$', rid)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); stems_expected.append(m.group(1))
    print(f'{len(stems_expected)} expected stems from sample_sub; scanning /kaggle/input ...')
    audio_map = {}
    for ext in ('ogg', 'flac', 'wav'):
        for p in INPUT_ROOT.rglob(f'*.{ext}'):
            if p.stem not in audio_map:
                audio_map[p.stem] = p
    test_paths = [audio_map[s] for s in stems_expected if s in audio_map]
    print(f'FALLBACK stem-scan matched {len(test_paths)}/{len(stems_expected)}')
elif not test_paths:
    print(f'sample_sub is stub-sized ({len(sample_sub)} rows); skipping stem-scan, going straight to DRY_RUN.')

# ----- Editor dry-run fallback (legacy style) -----
DRY_RUN = False
if not test_paths:
    # We are almost certainly inside the editor Save Version commit.
    # Exercise the full pipeline on train_soundscapes so the commit log
    # shows real ETA / forward-pass behaviour rather than a 3-min no-op.
    train_dir = BASE / 'train_soundscapes'
    dry_paths = sorted(train_dir.glob('*.ogg'))[:20]
    if dry_paths:
        test_paths = dry_paths
        DRY_RUN = True
        print(f'DRY-RUN: hidden test not mounted. Using first {len(test_paths)} train_soundscapes files.')
    else:
        print('WARN: no train_soundscapes either — ABORT path will keep placeholder.')
print(f'>>> test_paths count: {len(test_paths)}  dry_run={DRY_RUN}')

# ----- Bundle discovery -----
BUNDLE_CANDIDATES = [
    Path('/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y1'),
    Path('/kaggle/input/birdclef-2026-bundle-y1'),
    Path('/kaggle/input/datasets/tiantanghuaxiao/birdclef-2026-bundle-y'),
    Path('/kaggle/input/birdclef-2026-bundle-y'),
]
BUNDLE = next((p for p in BUNDLE_CANDIDATES if (p / 'manifest.json').exists()), None)
if BUNDLE is None:
    _mf = list(INPUT_ROOT.rglob('manifest.json'))
    if _mf:
        BUNDLE = _mf[0].parent
print('BUNDLE:', BUNDLE)

# ----- Placeholder submission.csv (so no-op paths still yield PB=0.500 not a crash) -----
OUT = Path('/kaggle/working/submission.csv')
placeholder = sample_sub.copy()
placeholder.loc[:, PRIMARY_LABELS] = 0.0
placeholder.to_csv(OUT, index=False)
print(f'>>> placeholder submission.csv written ({len(placeholder)} rows)')
'''


CELL_2 = r'''# Cell 2 — manifest, label perm, per-taxon temp/smooth parameters.
#
# =========================================================================
# DIAGNOSTIC VARIANT SELECTOR — edit this string, re-Save Version, re-submit
# to probe which part of the pipeline is holding PB back. Each variant
# changes exactly one lever; compare Public LB scores to locate the bug.
#
#   'baseline'         — 5-fold S7 ensemble, equal weight, adaptive TTA,
#                        rank-aware power 0.5. This is the config that
#                        scored PB 0.866.
#   'II_drop_fold1'    — drop fold 1 (S5 val 0.5856, suspected outlier).
#                        If PB goes UP, fold 1 was dragging the ensemble.
#   'III_val_weighted' — weight folds by their S7 val-set row counts
#                        (78/192/992/120/96). Gives fold 2 (val 992,
#                        honest-hardest val) the largest vote.
#   'IV_force_3tta'    — skip ETA probe, force [0, -1.5, +1.5] shifts.
#                        If PB goes UP, the adaptive probe was picking
#                        too few TTA shifts.
#   'V_power1'         — rank-aware amplify power 1.0 (full file-max
#                        crush) instead of 0.5. Some 2025 solutions used
#                        this; may help or hurt depending on class
#                        sparsity in test.
# =========================================================================
VARIANT = 'baseline'

_VARIANTS = {
    'baseline':         dict(drop_folds=[],  fold_weights=None,                   force_tta=None,          power=0.5),
    'II_drop_fold1':    dict(drop_folds=[1], fold_weights=None,                   force_tta=None,          power=0.5),
    'III_val_weighted': dict(drop_folds=[],  fold_weights={0:78, 1:192, 2:992, 3:120, 4:96}, force_tta=None, power=0.5),
    'IV_force_3tta':    dict(drop_folds=[],  fold_weights=None,                   force_tta=[0.0,-1.5,1.5], power=0.5),
    'V_power1':         dict(drop_folds=[],  fold_weights=None,                   force_tta=None,          power=1.0),
}
assert VARIANT in _VARIANTS, f'unknown VARIANT {VARIANT!r}; valid: {list(_VARIANTS)}'
_V = _VARIANTS[VARIANT]
DROP_FOLDS = set(_V['drop_folds'])
FOLD_WEIGHTS_SPEC = _V['fold_weights']           # None or {fold_idx: weight}
FORCE_TTA = _V['force_tta']                      # None or list of shift_s
RANK_AWARE_POWER = _V['power']
print(f'>>> VARIANT={VARIANT}  drop_folds={sorted(DROP_FOLDS)}  '
      f'weights={FOLD_WEIGHTS_SPEC}  force_tta={FORCE_TTA}  power={RANK_AWARE_POWER}')

MANIFEST = None
MEL_CFG = None
LABEL_PERM = None
ABORT = (BUNDLE is None) or (len(test_paths) == 0) or (not OV_OK)

if BUNDLE is not None:
    with open(BUNDLE / 'manifest.json') as f:
        MANIFEST = json.load(f)
    MODEL_LABELS = [str(x) for x in MANIFEST['primary_labels']]
    MEL_CFG = MANIFEST['mel']
    _idx = {lbl: i for i, lbl in enumerate(MODEL_LABELS)}
    try:
        LABEL_PERM = np.array([_idx[lbl] for lbl in PRIMARY_LABELS], dtype=np.int64)
        print('label perm sane:', len(LABEL_PERM), '==', N_CLASSES,
              '| identity?', np.array_equal(LABEL_PERM, np.arange(N_CLASSES)))
    except KeyError as e:
        print(f'LABEL MISMATCH: sample_sub col {e} not in manifest.primary_labels — ABORT')
        ABORT = True

# Per-taxon smoothing / temperature, indexed by PRIMARY_LABELS order.
# Values are the same as in legacy PB 0.883 (Aves gets the heaviest
# temporal smoothing, Insecta the lightest — Insecta calls are short
# & sharp, smoothing hurts them).
if not ABORT:
    try:
        tax = pd.read_csv(BASE / 'taxonomy.csv').set_index('primary_label')
        class_names = np.array(
            [tax.loc[str(l), 'class_name'] if str(l) in tax.index else 'Aves'
             for l in PRIMARY_LABELS], dtype=object)
    except Exception as e:
        print('taxonomy read failed, defaulting all to Aves:', e)
        class_names = np.array(['Aves'] * N_CLASSES, dtype=object)
    TEMP_BY_CLASS = {'Aves': 1.10, 'Amphibia': 1.00, 'Insecta': 0.95, 'Mammalia': 1.00, 'Reptilia': 1.00}
    ALPHA_BY_CLASS = {'Aves': 0.35, 'Amphibia': 0.30, 'Insecta': 0.15, 'Mammalia': 0.20, 'Reptilia': 0.20}
    temperature = np.array([TEMP_BY_CLASS.get(c, 1.0) for c in class_names], dtype=np.float32)
    smooth_alpha = np.array([ALPHA_BY_CLASS.get(c, 0.3) for c in class_names], dtype=np.float32)
    uniq, cnt = np.unique(class_names, return_counts=True)
    print('class mix:', dict(zip(uniq.tolist(), cnt.tolist())))
else:
    temperature = smooth_alpha = None
    print('ABORT=True; placeholder submission.csv stands.')
'''


CELL_3 = r'''# Cell 3 — numpy mel front-end (must match stages/s2_prepare_mel.py + common/datasets.normalize_mel).
# Training pipeline: wave -> log-mel(dB) -> clip to [db_lo, db_hi] -> rescale to [0,1].
# We replicate exactly that; the model expects float32 in [0,1], shape (1, n_mels, T).
def build_mel_fb(sr, n_fft, n_mels, f_min, f_max):
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

def mel_from_wave(y, fb, n_fft, hop, win_length, db_lo, db_hi, power):
    window = np.hanning(win_length).astype(np.float32)
    pad = n_fft // 2
    y = np.pad(y, (pad, pad), mode='reflect')
    n_frames = 1 + (len(y) - win_length) // hop
    frames = np.lib.stride_tricks.sliding_window_view(y, win_length)[::hop][:n_frames] * window
    spec = np.fft.rfft(frames, n=n_fft).astype(np.complex64)
    mag = np.abs(spec) ** power
    mel = np.maximum(mag @ fb.T, 1e-10)
    db = np.clip(10.0 * np.log10(mel), db_lo, db_hi)
    scaled = (db - db_lo) / (db_hi - db_lo)
    return scaled.T.astype(np.float32)

if not ABORT:
    MEL_FB = build_mel_fb(MEL_CFG['sr'], MEL_CFG['n_fft'],
                          MEL_CFG['n_mels'], MEL_CFG['f_min'], MEL_CFG['f_max'])
else:
    MEL_FB = None

def decode_60s(path):
    y, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    want = MEL_CFG['sr'] * MEL_CFG['window_seconds'] * MEL_CFG['windows_per_file']
    if len(y) < want: y = np.pad(y, (0, want - len(y)))
    elif len(y) > want: y = y[:want]
    return y

def window_mels(y, shift_samples=0):
    """Return (n_windows, n_mels, T_target) dB-normalised mel in [0,1].

    T_target uses s9's export formula (``ceil(sr*win_s/hop) + 1``), which
    locks the exported IR's time dim. The numpy frontend's native output
    is 1 frame shorter (`313` vs `314` for 5s @ 32kHz, hop=512), because
    it uses `1 + (padded_len - win_length) // hop`. We pad (edge-replicate)
    the tail to match the IR. This reproduces exactly what s9 used as
    the dummy input during ONNX export.
    """
    if shift_samples:
        y = np.roll(y, shift_samples)
    win = MEL_CFG['sr'] * MEL_CFG['window_seconds']
    w = y.reshape(MEL_CFG['windows_per_file'], win)
    mels = np.stack([
        mel_from_wave(w[i], MEL_FB, MEL_CFG['n_fft'], MEL_CFG['hop'],
                      MEL_CFG['win_length'], MEL_CFG['db_lo'], MEL_CFG['db_hi'],
                      MEL_CFG['power'])
        for i in range(MEL_CFG['windows_per_file'])
    ], axis=0)
    target_T = int(np.ceil(win / MEL_CFG['hop'])) + 1
    cur_T = mels.shape[-1]
    if cur_T < target_T:
        mels = np.pad(mels, ((0, 0), (0, 0), (0, target_T - cur_T)), mode='edge')
    elif cur_T > target_T:
        mels = mels[..., :target_T]
    return mels
'''


CELL_4 = r'''# Cell 4 — load OpenVINO fold models (never raise; flip ABORT on failure).
# Respects the VARIANT selector from Cell 2:
#   - DROP_FOLDS: skip those fold indices entirely
#   - FOLD_WEIGHTS_SPEC: if given, build FOLD_WEIGHTS (normalised) in the
#     order the models were actually loaded, so Cell 5 can do a weighted
#     average instead of an equal one.
FOLD_MODELS = []
FOLD_IDS = []         # fold index matching each entry of FOLD_MODELS
if not ABORT:
    try:
        import openvino as ov
        core = ov.Core()
        # Kaggle CPU notebook VM = 4 vCPUs. Pin threads to match.
        core.set_property('CPU', {'INFERENCE_NUM_THREADS': 4})
        for fmeta in MANIFEST['folds']:
            if fmeta.get('skipped'): continue
            if 'ir_xml' not in fmeta: continue
            fid = fmeta.get('fold')
            if fid in DROP_FOLDS:
                print(f'SKIP fold {fid}: dropped by VARIANT={VARIANT}')
                continue
            xml = BUNDLE / Path(fmeta['ir_xml']).name
            if not xml.exists():
                alt = BUNDLE / 'export' / xml.name
                if alt.exists(): xml = alt
            if not xml.exists():
                print('SKIP fold', fid, ':', xml, 'not found')
                continue
            try:
                comp = core.compile_model(str(xml), 'CPU')
                FOLD_MODELS.append(comp)
                FOLD_IDS.append(fid)
                in_shape = comp.inputs[0].partial_shape
                out_shape = comp.outputs[0].partial_shape
                print(f'compiled fold {fid} -> in={in_shape}, out={out_shape}')
            except Exception as e:
                print('compile FAILED for', xml, ':', e)
    except Exception as e:
        print('OV import/setup FAILED:', type(e).__name__, e)

print(f'loaded {len(FOLD_MODELS)} fold models: {FOLD_IDS}')
if not FOLD_MODELS:
    print('WARN: no fold models loaded. Placeholder submission.csv will remain.')
    ABORT = True

# Build per-fold weights (normalised to sum to 1) for the weighted ensemble.
if not ABORT:
    if FOLD_WEIGHTS_SPEC is None:
        FOLD_WEIGHTS = np.ones(len(FOLD_MODELS), dtype=np.float32) / len(FOLD_MODELS)
    else:
        raw = np.array([float(FOLD_WEIGHTS_SPEC.get(fid, 0.0)) for fid in FOLD_IDS],
                       dtype=np.float32)
        if raw.sum() <= 0:
            print(f'WARN: FOLD_WEIGHTS_SPEC sums to 0 for loaded folds, falling back to equal.')
            FOLD_WEIGHTS = np.ones(len(FOLD_MODELS), dtype=np.float32) / len(FOLD_MODELS)
        else:
            FOLD_WEIGHTS = raw / raw.sum()
    print('FOLD_WEIGHTS (normalised):', list(zip(FOLD_IDS, [round(float(w), 4) for w in FOLD_WEIGHTS])))
else:
    FOLD_WEIGHTS = np.ones(max(len(FOLD_MODELS), 1), dtype=np.float32) / max(len(FOLD_MODELS), 1)
'''


CELL_5 = r'''# Cell 5 — per-file inference: TTA list (shift 0 always first), fold averaging, label permutation.
# Model input: (12, 1, n_mels, T) batch; model output: (12, 234) logits.
# We sigmoid, then apply FOLD_WEIGHTS (set by Cell 4 from the VARIANT),
# then average across TTAs.
def _fold_avg_probs(mels_batched):
    """mels_batched: (12, 1, n_mels, T) -> (12, 234) sigmoid probs in MODEL_LABELS order.

    Per-fold sigmoid outputs are combined with FOLD_WEIGHTS (sums to 1),
    so an equal-weight ensemble reduces to ``mean``.
    """
    x = mels_batched.astype(np.float32)
    acc = None
    for i, comp in enumerate(FOLD_MODELS):
        out = comp(x)[comp.outputs[0]]  # (12, 234) logits
        p = 1.0 / (1.0 + np.exp(-out))
        contrib = p * float(FOLD_WEIGHTS[i])
        acc = contrib if acc is None else acc + contrib
    return acc.astype(np.float32)  # already sums to 1 of weights

def infer_file(path, tta_shifts_s):
    """Return (12, N_CLASSES) probs permuted into PRIMARY_LABELS order, or None on failure."""
    try:
        y = decode_60s(path)
        probs_ttas = []
        for shift_s in tta_shifts_s:
            shift_samples = int(shift_s * MEL_CFG['sr'])
            mels = window_mels(y, shift_samples=shift_samples)[:, None, :, :]
            probs_ttas.append(_fold_avg_probs(mels))
        probs_model = np.stack(probs_ttas, axis=0).mean(axis=0)
        return probs_model[:, LABEL_PERM]
    except Exception as e:
        print(f'infer_file FAILED for {path.name}: {type(e).__name__}: {e}')
        return None
'''


CELL_6 = r'''# Cell 6 — post-processing: neighbour smoothing, per-taxon temp, TopN=1 file-max amplification.
# (Dead code ``topn_smooth`` removed — the old *0 term made it an identity.)
def smooth_neighbours(p):
    """Per-class alpha blend with the mean of adjacent windows (skips the two edges)."""
    p2 = p.copy()
    for i in range(1, p.shape[0] - 1):
        neighbour_mean = 0.5 * (p[i-1] + p[i+1])
        p2[i] = (1.0 - smooth_alpha) * p[i] + smooth_alpha * neighbour_mean
    return p2

def apply_temperature(p):
    """Per-class temperature scaling in logit space."""
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    return 1.0 / (1.0 + np.exp(-logit / temperature))

def rank_aware_amplify(p, power=0.5):
    """2025 Rank-3 'rank-aware' scaling (PB 0.883 legacy).

    Scale each window by (file_max ** power). power=1.0 is the full
    ``p * file_max`` crush (what v5 originally had); power=0.5 is the
    legacy-proven compromise — silent-file windows get pushed toward 0
    but not annihilated. power=0.0 would be a no-op.
    """
    file_max = p.max(axis=0, keepdims=True)
    return p * (file_max ** power)
'''


CELL_7 = r'''# Cell 7 — adaptive ETA loop with incremental checkpointing.
#
# Strategy (fix Bug C): the 5-fold OV × 3-TTA pessimistic estimate
# (saga §6.2) can overshoot the 90-min hard cap. Instead of always
# running full TTA, we:
#   1. probe the first PROBE_N files with a single-shot [0.0] TTA to
#      learn the real per-file cost,
#   2. pick the largest TTA set that fits in the remaining budget,
#   3. run the rest with the chosen TTA, checkpointing submission.csv
#      every CHECKPOINT_EVERY files (so a timeout mid-loop still
#      leaves a valid CSV better than the all-zero placeholder).
TIME_BUDGET_SEC = 86 * 60  # 4 min headroom below the 90-min hard cap
PROBE_N = 8                # calibrate ETA on first 8 files
CHECKPOINT_EVERY = 50      # write submission.csv atomically this often
TTA_CANDIDATES = [
    ([0.0, -1.5, 1.5], '3-TTA'),
    ([0.0,  1.5],      '2-TTA'),
    ([0.0],            '1-TTA'),
]

def _write_submission(probs_by_row_id, zero_row):
    pred_matrix = np.stack([probs_by_row_id.get(rid, zero_row) for rid in sample_sub['row_id']])
    sub = pd.DataFrame(pred_matrix, columns=PRIMARY_LABELS)
    sub.insert(0, 'row_id', sample_sub['row_id'].values)
    sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].astype(np.float32)
    tmp = OUT.with_suffix('.csv.tmp')
    sub.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    return sub

# RANK_AWARE_POWER comes from the VARIANT dict in Cell 2.

def _post_process(probs):
    probs = smooth_neighbours(probs)
    probs = apply_temperature(probs)
    probs = rank_aware_amplify(probs, power=RANK_AWARE_POWER)
    return np.clip(probs, 0.0, 1.0).astype(np.float32)

if ABORT:
    print('ABORT=True — keeping placeholder submission.csv, skipping inference.')
else:
    N_WINDOWS = MEL_CFG['windows_per_file']
    WINDOW_SEC = MEL_CFG['window_seconds']
    zero_row = np.zeros(N_CLASSES, dtype=np.float32)
    probs_by_row_id = {}
    n_total = len(test_paths)

    # ---- Phase 1: probe ETA with 1-TTA on first PROBE_N files ----
    # Only count successful forward passes when measuring cost. If every
    # probe file fails (e.g. shape mismatch, unreadable audio), we keep
    # ``chosen_tta = [0.0]`` and skip the budget-based upgrade so that a
    # fast-but-failing probe can't trick us into picking 3-TTA.
    probe_n = min(PROBE_N, n_total)
    t_phase1 = time.time()
    good_probe = 0
    for i in range(probe_n):
        path = test_paths[i]
        probs = infer_file(path, [0.0])
        if probs is None:
            continue
        good_probe += 1
        probs = _post_process(probs)
        stem = path.stem
        for k in range(N_WINDOWS):
            probs_by_row_id[f'{stem}_{(k + 1) * WINDOW_SEC}'] = probs[k]
    dt_phase1 = time.time() - t_phase1
    if good_probe == 0:
        t_per_file_1tta = float('inf')
        print(f'PHASE1 probe: {probe_n}/{probe_n} FAILED ({dt_phase1:.1f}s total). Cannot measure ETA.')
    else:
        t_per_file_1tta = dt_phase1 / good_probe
        print(f'PHASE1 probe: {good_probe}/{probe_n} succeeded in {dt_phase1:.1f}s, '
              f'{t_per_file_1tta:.2f} s/file @ 1-TTA (successful only)')

    # ---- Decide TTA for remaining files ----
    wall_used = time.time() - _WALL_START
    time_left = TIME_BUDGET_SEC - wall_used
    remaining = n_total - probe_n
    if FORCE_TTA is not None:
        # Variant IV etc. — caller forces a specific TTA schedule, skip
        # budget check (but emergency brake at 88 min still applies).
        chosen_tta = list(FORCE_TTA)
        tta_name = f'{len(chosen_tta)}-TTA (VARIANT override)'
        if good_probe > 0:
            est_cost = t_per_file_1tta * len(chosen_tta) * remaining / 60
            print(f'VARIANT override: forced {tta_name}; estimated phase-2 wall = {est_cost:.1f} min')
    elif remaining <= 0:
        chosen_tta, tta_name = [0.0], '1-TTA (no remaining)'
    elif good_probe == 0:
        # Don't trust the ETA; fall back to the cheapest option.
        chosen_tta, tta_name = [0.0], '1-TTA (probe all-failed)'
    else:
        per_file_budget = time_left / remaining * 0.92  # 8% safety buffer
        print(f'budget after probe: time_left={time_left/60:.1f} min, remaining={remaining},'
              f' per_file_budget={per_file_budget:.2f}s  (1-TTA cost={t_per_file_1tta:.2f}s)')
        chosen_tta, tta_name = [0.0], '1-TTA (fallback)'
        for tta_list, name in TTA_CANDIDATES:
            cost_per_file = t_per_file_1tta * len(tta_list)
            if cost_per_file <= per_file_budget:
                chosen_tta, tta_name = tta_list, name
                break
    print(f'CHOSEN: {tta_name}  shifts={chosen_tta}')

    # ---- Phase 2: run remaining files with chosen TTA + incremental checkpointing ----
    t_phase2 = time.time()
    for i in range(probe_n, n_total):
        path = test_paths[i]
        probs = infer_file(path, chosen_tta)
        if probs is None:
            continue
        probs = _post_process(probs)
        stem = path.stem
        for k in range(N_WINDOWS):
            probs_by_row_id[f'{stem}_{(k + 1) * WINDOW_SEC}'] = probs[k]

        done = i + 1
        if done % CHECKPOINT_EVERY == 0 or done == n_total:
            _write_submission(probs_by_row_id, zero_row)
            wall = (time.time() - _WALL_START) / 60
            eta = (time.time() - t_phase2) / max(done - probe_n, 1) * (n_total - done) / 60
            print(f'[{done}/{n_total}]  wall={wall:.1f} min  eta_rem={eta:.1f} min  rows={len(probs_by_row_id)}')

        # Emergency brake: if we have burned past 88 min, stop early
        # (we will still write a partial CSV via the checkpoint above).
        if time.time() - _WALL_START > 88 * 60:
            print(f'EMERGENCY BRAKE @ file {done}/{n_total} — wall exceeded 88 min, stopping.')
            _write_submission(probs_by_row_id, zero_row)
            break

    # ---- Final full write + assertions ----
    submission = _write_submission(probs_by_row_id, zero_row)
    assert submission.columns.tolist() == ['row_id'] + PRIMARY_LABELS, \
        f'column drift: {submission.columns.tolist()[:5]}'
    assert not submission.isna().any().any(), 'NaN in submission'
    assert len(submission) == len(sample_sub), \
        f'row drift: {len(submission)} vs {len(sample_sub)}'

    hit_rows = sum(1 for rid in sample_sub['row_id'] if rid in probs_by_row_id)
    wall_total = (time.time() - _WALL_START) / 60
    print(f'>>> submission.csv FINAL: shape={submission.shape}, hit_rows={hit_rows}/{len(sample_sub)},'
          f' dry_run={DRY_RUN}, wall={wall_total:.1f} min')

# Final readback sanity (regardless of ABORT path)
_final = pd.read_csv(OUT)
assert list(_final.columns) == SAMPLE_COLS
assert len(_final) == len(sample_sub)
nonzero = (_final[PRIMARY_LABELS].values > 0).any(axis=1).sum()
print(f'final submission.csv shape={_final.shape}  nonzero_rows={nonzero}/{len(_final)}')
'''


NEW_SOURCES = [CELL_0, CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7]


def _as_source_lines(src: str) -> list[str]:
    """Return ipynb-style source list (each line keeps its trailing \\n except last)."""
    lines = src.splitlines(keepends=True)
    return lines


def main() -> None:
    with NB_PATH.open("r") as f:
        nb = json.load(f)
    assert len(nb["cells"]) == 8, f"expected 8 cells, got {len(nb['cells'])}"
    for i, src in enumerate(NEW_SOURCES):
        cell = nb["cells"][i]
        assert cell["cell_type"] == "code", f"cell {i} is not code"
        cell["source"] = _as_source_lines(src)
        cell["outputs"] = []
        cell["execution_count"] = None
    with NB_PATH.open("w") as f:
        json.dump(nb, f, indent=1)
    print(f"rewrote {NB_PATH}  ({len(NEW_SOURCES)} cells)")


if __name__ == "__main__":
    main()
