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
OUT_NB = REPO / "kaggle_submit" / "submit_v6_hybrid.ipynb"


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
import glob, subprocess, sys, os

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
CELL_OUR_INFER = r'''# Cell 46b — Plan Y.1 5-fold OpenVINO ensemble on the same test_paths.
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

        # ----- compile fold IRs -----
        import openvino as ov
        _core = ov.Core()
        _core.set_property("CPU", {"INFERENCE_NUM_THREADS": 4})
        FOLD_OV = []
        for _fmeta in _MAN["folds"]:
            if _fmeta.get("skipped") or "ir_xml" not in _fmeta: continue
            _xml = HYBRID_BUNDLE / Path(_fmeta["ir_xml"]).name
            if not _xml.exists():
                _alt = HYBRID_BUNDLE / "export" / _xml.name
                if _alt.exists(): _xml = _alt
            if not _xml.exists():
                print("HYBRID skip fold", _fmeta.get("fold"), "— xml missing")
                continue
            try:
                FOLD_OV.append(_core.compile_model(str(_xml), "CPU"))
            except Exception as _e:
                print("HYBRID compile fail", _xml, ":", _e)
        print(f"HYBRID compiled {len(FOLD_OV)} fold models")

        if FOLD_OV:
            import soundfile as sf
            _SR = _MEL["sr"]; _NW = _MEL["windows_per_file"]; _WS = _MEL["window_seconds"]
            _WANT = _SR * _WS * _NW

            def _decode_60s(p):
                y, _sr = sf.read(str(p), dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < _WANT: y = np.pad(y, (0, _WANT - len(y)))
                elif len(y) > _WANT: y = y[:_WANT]
                return y

            def _our_infer_one(p):
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
                x = mels[:, None, :, :].astype(np.float32)
                acc = None
                for _c in FOLD_OV:
                    _o = _c(x)[_c.outputs[0]]
                    _p = 1.0 / (1.0 + np.exp(-_o))
                    acc = _p if acc is None else acc + _p
                return (acc / len(FOLD_OV))

            _t0 = time.time()
            _per_file = []
            for _i, _p in enumerate(test_paths):
                try:
                    _pp = _our_infer_one(_p)
                except Exception as _e:
                    print("HYBRID our_infer fail @", _p.name, ":", type(_e).__name__, _e)
                    _pp = np.zeros((_NW, len(_OUR_LABELS)), dtype=np.float32)
                _per_file.append(_pp[:, LABEL_PERM_OUR])
                if (_i + 1) % 50 == 0:
                    _wall = (time.time() - _t0) / 60
                    _eta = _wall / (_i + 1) * (len(test_paths) - _i - 1)
                    print(f"HYBRID our infer {_i+1}/{len(test_paths)}  wall={_wall:.1f}min  eta={_eta:.1f}min")
            our_probs_test = np.concatenate(_per_file, axis=0).astype(np.float32)
            print(f"HYBRID our_probs_test shape: {our_probs_test.shape}  "
                  f"range: {our_probs_test.min():.4f}..{our_probs_test.max():.4f}  "
                  f"wall: {(time.time()-_t0)/60:.1f}min")
'''


# Snippet inserted into cell 50 right after `probs = sigmoid(scaled_scores)`
BLEND_SNIPPET = '''

# --- HYBRID: late fusion with our 5-fold OpenVINO ensemble (Plan Y.1) ---
# We blend in probability space (after sigmoid, before their rank-aware /
# delta-smooth / per-class threshold pipeline). If hybrid prep failed,
# `our_probs_test` is None and this is a no-op — we keep the 0.924 baseline.
HYBRID_W = 0.4   # weight for our CNN; their stack gets 1 - HYBRID_W
try:
    if our_probs_test is not None and our_probs_test.shape == probs.shape:
        probs = (1.0 - HYBRID_W) * probs + HYBRID_W * our_probs_test.astype(np.float32)
        probs = np.clip(probs, 0.0, 1.0)
        print(f"HYBRID blended probs (w_ours={HYBRID_W}, w_theirs={1-HYBRID_W})")
    else:
        print(f"HYBRID skipped at blend: our_probs_test = {None if our_probs_test is None else our_probs_test.shape}, "
              f"theirs = {probs.shape}")
except NameError:
    print("HYBRID skipped at blend: our_probs_test not defined (cell 46b errored).")
'''


def main() -> None:
    nb = json.loads(SOURCE_NB.read_text())

    # Sanity: original cell 50 must still contain the line we anchor against.
    cell_50 = nb["cells"][50]
    src_50 = "".join(cell_50["source"]) if isinstance(cell_50["source"], list) else cell_50["source"]
    anchor = "probs = sigmoid(scaled_scores)"
    if anchor not in src_50:
        raise SystemExit(f"Sanity fail: anchor {anchor!r} not in cell 50 of source notebook.")

    # Insert OUR_INFER cell *after* original index 46 (so original 46 stays at 46).
    nb["cells"].insert(47, _new_code_cell(CELL_OUR_INFER))
    # Insert OV_INSTALL cell *after* original index 1 (so it becomes index 2).
    nb["cells"].insert(2, _new_code_cell(CELL_OV_INSTALL))

    # Modify the blend in what was originally cell 50 (now shifted by +2 → cell 52).
    target_idx = 52
    cell_target = nb["cells"][target_idx]
    src = "".join(cell_target["source"]) if isinstance(cell_target["source"], list) else cell_target["source"]
    if anchor not in src:
        raise SystemExit(f"Anchor missing after shift; expected at cell {target_idx}.")
    new_src = src.replace(anchor, anchor + BLEND_SNIPPET, 1)
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
