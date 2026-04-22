#!/usr/bin/env python3
"""Generate ``kaggle_submit/submit_v5.ipynb`` from the cell templates below.

Why: hand-authoring long .ipynb JSON is bug-prone; this file is the single
source of truth, and we can regenerate after each Plan Y iteration.

Run: ``python scripts/build_submit_notebook.py``
Output: ``kaggle_submit/submit_v5.ipynb`` (overwritten)
"""

from __future__ import annotations

import json
from pathlib import Path


# --- individual cells (each a self-contained string) -----------------------

CELL_SETUP = r"""
# Cell 0 — offline install of OpenVINO (Kaggle runtime has Internet=OFF)
import glob, subprocess, sys, os
_wheel_candidates = [
    '/kaggle/input/datasets/li147852xu/birdclef-2026/wheels',
    '/kaggle/input/birdclef-2026/wheels',
    '/kaggle/input/birdclef-2026-bundle/wheels',
]
_installed = False
for _d in _wheel_candidates:
    if not os.path.isdir(_d): continue
    whls = sorted(glob.glob(f'{_d}/openvino-*.whl'))
    if whls:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', whls[0]], check=True)
        _installed = True
        print(f'installed {whls[0]}')
        break
if not _installed:
    print('WARN: openvino wheel not found; trying default PyPI (likely offline)')
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openvino==2024.4.0'], check=True)
    except Exception as e:
        print(f'openvino install failed: {e}')
        raise
"""

CELL_IMPORTS = r"""
# Cell 1 — imports, paths, timer
import time, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

t_start = time.time()
TIME_BUDGET_S = 85 * 60     # leave 5 min headroom under the 90-min cap

INPUT_ROOT = Path('/kaggle/input')
COMP = INPUT_ROOT / 'birdclef-2026'
BUNDLE_CANDIDATES = [
    INPUT_ROOT / 'datasets/li147852xu/birdclef-2026',
    INPUT_ROOT / 'birdclef-2026-bundle',
    INPUT_ROOT / 'birdclef-2026',
]
BUNDLE = next((p for p in BUNDLE_CANDIDATES if (p / 'manifest.json').exists()), None)
if BUNDLE is None:
    raise FileNotFoundError(
        f'No OV bundle (manifest.json) found under {BUNDLE_CANDIDATES}'
    )
print('bundle:', BUNDLE)

TEST_DIR = COMP / 'test_soundscapes'
OUT = Path('/kaggle/working/submission.csv')
"""

CELL_MANIFEST = r"""
# Cell 2 — manifest + per-taxon temperature + neighbour smoothing config
with open(BUNDLE / 'manifest.json') as f:
    MANIFEST = json.load(f)
PRIMARY_LABELS = MANIFEST['primary_labels']       # list of str, length 234
N_CLASSES = len(PRIMARY_LABELS)
MEL_CFG = MANIFEST['mel']
print('n_classes:', N_CLASSES, ' folds available:',
      sum(1 for f in MANIFEST['folds'] if 'ir_xml' in f))

# Read taxonomy so we can apply per-taxon temperature & smoothing.
tax = pd.read_csv(COMP / 'taxonomy.csv').set_index('primary_label')
class_names = np.array(
    [tax.loc[str(l), 'class_name'] if str(l) in tax.index else 'Aves'
     for l in PRIMARY_LABELS], dtype=object
)

# Temperature per taxon class (T>1 softens, T<1 sharpens).
TEMP_BY_CLASS = {
    'Aves':     1.10,
    'Amphibia': 1.00,
    'Insecta':  0.95,   # event-style, sharpen
    'Mammalia': 1.00,
    'Reptilia': 1.00,
}
# Neighbour-window smoothing weight per taxon (texture = higher).
ALPHA_BY_CLASS = {
    'Aves':     0.35,
    'Amphibia': 0.30,
    'Insecta':  0.15,   # point-events; less smoothing
    'Mammalia': 0.20,
    'Reptilia': 0.20,
}
temperature = np.array([TEMP_BY_CLASS.get(c, 1.0) for c in class_names], dtype=np.float32)
smooth_alpha = np.array([ALPHA_BY_CLASS.get(c, 0.3) for c in class_names], dtype=np.float32)
"""

CELL_MEL = r"""
# Cell 3 — numpy mel front-end (must match the one in stages/s2_prepare_mel.py)
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
    scaled = (db - db_lo) / (db_hi - db_lo)   # [0, 1]
    return scaled.T.astype(np.float32)        # (n_mels, T)

MEL_FB = build_mel_fb(MEL_CFG['sr'], MEL_CFG['n_fft'],
                      MEL_CFG['n_mels'], MEL_CFG['f_min'], MEL_CFG['f_max'])

def decode_60s(path):
    y, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    want = MEL_CFG['sr'] * 60
    if len(y) < want:
        y = np.pad(y, (0, want - len(y)))
    elif len(y) > want:
        y = y[:want]
    return y

def window_mels(y, shift_samples=0):
    '''Return (12, n_mels, T) dB-normalised mel in [0,1], with optional wave-level shift.'''
    if shift_samples:
        y = np.roll(y, shift_samples)
    win = MEL_CFG['sr'] * MEL_CFG['window_seconds']
    w = y.reshape(MEL_CFG['windows_per_file'], win)
    out = np.stack([
        mel_from_wave(w[i], MEL_FB, MEL_CFG['n_fft'], MEL_CFG['hop'],
                      MEL_CFG['win_length'], MEL_CFG['db_lo'], MEL_CFG['db_hi'],
                      MEL_CFG['power'])
        for i in range(MEL_CFG['windows_per_file'])
    ], axis=0)
    return out
"""

CELL_OV = r"""
# Cell 4 — load OpenVINO fold models (CPU plugin)
import openvino as ov
core = ov.Core()
core.set_property('CPU', {'INFERENCE_NUM_THREADS': 4})   # Kaggle gives 4 vCPUs

FOLD_MODELS = []
for fmeta in MANIFEST['folds']:
    if 'ir_xml' not in fmeta: continue
    xml = BUNDLE / Path(fmeta['ir_xml']).name
    if not xml.exists():
        # model may live under <bundle>/export/; try that
        alt = BUNDLE / 'export' / xml.name
        if alt.exists(): xml = alt
    if not xml.exists():
        print(f'SKIP fold {fmeta[\"fold\"]}: {xml} not found')
        continue
    comp = core.compile_model(str(xml), 'CPU')
    FOLD_MODELS.append(comp)
print(f'loaded {len(FOLD_MODELS)} fold models')
if not FOLD_MODELS:
    raise RuntimeError('no OpenVINO fold models loaded')
"""

CELL_INFER = r"""
# Cell 5 — per-file inference loop with TTA + fold averaging
TTA_SHIFTS_S = [0.0, -1.5, 1.5]     # wave-level circular shifts for TTA

def run_folds_on_mels(mels):
    '''mels: (12, n_mels, T) float32 → (12, 234) sigmoid probs, avg across folds.'''
    x = mels[:, None, :, :].astype(np.float32)   # (12, 1, n_mels, T)
    acc = None
    for comp in FOLD_MODELS:
        out = comp(x)[comp.outputs[0]]           # (12, 234) logits
        p = 1.0 / (1.0 + np.exp(-out))
        acc = p if acc is None else acc + p
    return (acc / len(FOLD_MODELS)).astype(np.float32)

def infer_file(path):
    y = decode_60s(path)
    probs_ttas = []
    for shift_s in TTA_SHIFTS_S:
        shift_samples = int(shift_s * MEL_CFG['sr'])
        mels = window_mels(y, shift_samples=shift_samples)
        probs_ttas.append(run_folds_on_mels(mels))
    probs = np.stack(probs_ttas, axis=0).mean(axis=0)  # (12, 234)
    return probs
"""

CELL_POSTPROC = r"""
# Cell 6 — post-processing: neighbour-window smoothing, per-taxon temp, TopN=1
def smooth_neighbours(p):
    '''In-place neighbour-window smoothing per class with per-class alpha.'''
    # p: (12, 234).  smooth_alpha: (234,)
    p2 = p.copy()
    for i in range(1, p.shape[0] - 1):
        neighbour_mean = 0.5 * (p[i-1] + p[i+1])
        p2[i] = (1.0 - smooth_alpha) * p[i] + smooth_alpha * neighbour_mean
    return p2

def apply_temperature(p):
    '''Per-class temperature scaling in logit space.'''
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    logit_t = logit / temperature
    return 1.0 / (1.0 + np.exp(-logit_t))

def topn_smooth(p, n=1):
    '''File-level multiplicative TopN smoothing (2025 2nd place magic).'''
    # max per class across 12 windows, multiplied back in to each window
    file_max = p.max(axis=0, keepdims=True)
    return np.minimum(1.0, p * (1.0 + (file_max - p) * 0))  # identity for n=1 baseline
    # Note: the documented TopN=1 variant is simply per-class amplification:
    #   p *= file_max
    # We keep that below:

def topn_amplify(p):
    file_max = p.max(axis=0, keepdims=True)
    return p * file_max
"""

CELL_LOOP = r"""
# Cell 7 — main loop over test files + write submission.csv
test_files = sorted(TEST_DIR.glob('*.ogg'))
print(f'{len(test_files)} test files')

rows = []
for i, path in enumerate(test_files):
    elapsed = time.time() - t_start
    if elapsed > TIME_BUDGET_S:
        # Hard abort: fill remaining files with zeros so shapes stay right.
        print(f'TIME_BUDGET hit at file {i}/{len(test_files)}; filling zeros')
        for p2 in test_files[i:]:
            stem = p2.stem
            for k in range(1, MEL_CFG['windows_per_file'] + 1):
                rows.append({'row_id': f'{stem}_{k * MEL_CFG[\"window_seconds\"]}',
                             **{lbl: 0.0 for lbl in PRIMARY_LABELS}})
        break

    probs = infer_file(path)                     # (12, 234)
    probs = smooth_neighbours(probs)
    probs = apply_temperature(probs)
    probs = topn_amplify(probs)
    probs = np.clip(probs, 0.0, 1.0)

    stem = path.stem
    for k in range(probs.shape[0]):
        row = {'row_id': f'{stem}_{(k + 1) * MEL_CFG[\"window_seconds\"]}'}
        row.update({lbl: float(probs[k, j]) for j, lbl in enumerate(PRIMARY_LABELS)})
        rows.append(row)

    if (i + 1) % 50 == 0:
        dt = (time.time() - t_start) / 60
        eta = dt / (i + 1) * (len(test_files) - i - 1)
        print(f'[{i+1}/{len(test_files)}]  elapsed={dt:.1f}min  eta={eta:.1f}min')

sub = pd.DataFrame(rows, columns=['row_id'] + PRIMARY_LABELS)
sub.to_csv(OUT, index=False)
print(f'wrote {OUT}  rows={len(sub)}  elapsed={(time.time()-t_start)/60:.1f} min')
"""


# --- build notebook --------------------------------------------------------

def _code_cell(src: str) -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def build() -> dict:
    cells = [
        _code_cell(CELL_SETUP),
        _code_cell(CELL_IMPORTS),
        _code_cell(CELL_MANIFEST),
        _code_cell(CELL_MEL),
        _code_cell(CELL_OV),
        _code_cell(CELL_INFER),
        _code_cell(CELL_POSTPROC),
        _code_cell(CELL_LOOP),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "kaggle_submit" / "submit_v5.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
