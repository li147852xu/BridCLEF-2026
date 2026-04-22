"""S2 — mel-spectrogram cache (resumable, per-file).

Input:
    - competition audio under ``${comp_dir}/train_audio/*/*.ogg``
    - competition soundscapes under ``${comp_dir}/train_soundscapes/*.ogg``

Output (per input file, one shard):
    ``${mel_cache}/<source>/<stem>.npz``  containing:
        mel_u8     : uint8  (n_windows, n_mels, n_frames)
        meta       : dict   with sr, n_fft, hop, win_length, n_mels, f_min, f_max,
                            db_clip, file_seconds, window_seconds, src
        label      : optional (only for train_audio) primary_label str

Design:
    * Strictly per-file: we write each .npz atomically so a crash mid-run only
      loses at most the current file. Every re-run simply skips finished files.
    * Single-process worker pool with threads (soundfile releases GIL on decode;
      torchaudio.MelSpec is pure numpy-level vectorised). If you need more, add
      ``--num-workers`` override via the env var.
    * Quantisation: log-mel (dB) is clipped to ``db_clip`` then mapped to uint8
      [0, 255]. Reconstruction: ``db = lo + (mel_u8 / 255.0) * (hi - lo)``.
      Saves ~4× vs fp32 and is plenty of resolution for training (we recover
      float32 on the fly in S5).
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common.audio import read_soundscape, windowize
from common.cloud_paths import CloudCfg
from stages._common import (
    JsonlLogger,
    atomic_write_bytes,
    require_free_disk,
)

log = logging.getLogger("bridclef.s2")


# --------------------------------------------------------------------------
# Mel front-end (pure numpy to avoid torch import cost on small boxes)
# --------------------------------------------------------------------------

def _build_mel_fb(sr: int, n_fft: int, n_mels: int,
                  f_min: float, f_max: float) -> np.ndarray:
    """Slaney-style HTK mel filterbank — matches torchaudio default."""
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    m_lo, m_hi = hz_to_mel(f_min), hz_to_mel(f_max)
    m_pts = np.linspace(m_lo, m_hi, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bin_pts = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def _mel_uint8(y: np.ndarray, mel_fb: np.ndarray, *,
               n_fft: int, hop: int, win_length: int,
               db_lo: float, db_hi: float, power: float) -> np.ndarray:
    """Compute (n_mels, T) uint8 log-mel for a 1-D signal ``y``."""
    # Hann window STFT via rfft; equivalent to torchaudio.Spectrogram(center=True).
    window = np.hanning(win_length).astype(np.float32)
    # pad reflect so frames start at t=0
    pad = n_fft // 2
    y = np.pad(y, (pad, pad), mode="reflect")
    # frame
    n_frames = 1 + (len(y) - win_length) // hop
    frames = np.lib.stride_tricks.sliding_window_view(y, win_length)[::hop]
    frames = frames[:n_frames] * window
    spec = np.fft.rfft(frames, n=n_fft).astype(np.complex64)
    mag = np.abs(spec) ** power  # (T, F)
    mel = mag @ mel_fb.T  # (T, n_mels)
    mel = np.maximum(mel, 1e-10).astype(np.float32)
    db = 10.0 * np.log10(mel)
    db = np.clip(db, db_lo, db_hi)
    scaled = (db - db_lo) / (db_hi - db_lo)  # [0, 1]
    u8 = np.clip(scaled * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return u8.T  # (n_mels, T)


# --------------------------------------------------------------------------
# Per-file worker
# --------------------------------------------------------------------------

def _process_file(ogg_path: Path, out_path: Path, mel_fb: np.ndarray,
                  audio_cfg: dict, mel_cfg: dict,
                  label: str | None) -> tuple[Path, str]:
    """Return (out_path, status). status is 'wrote' or 'skip'."""
    if out_path.exists():
        return out_path, "skip"

    y = read_soundscape(ogg_path, target_sr=audio_cfg["sample_rate"],
                        file_seconds=audio_cfg["file_seconds"])
    wins = windowize(y, target_sr=audio_cfg["sample_rate"],
                     window_seconds=audio_cfg["window_seconds"],
                     windows_per_file=audio_cfg["windows_per_file"])
    # (n_windows, n_mels, T)
    db_lo, db_hi = mel_cfg["db_clip"]
    mels = np.stack([
        _mel_uint8(w, mel_fb,
                   n_fft=mel_cfg["n_fft"],
                   hop=mel_cfg["hop_length"],
                   win_length=mel_cfg["win_length"],
                   db_lo=db_lo, db_hi=db_hi,
                   power=mel_cfg["power"])
        for w in wins
    ], axis=0)

    meta = {
        "sr": audio_cfg["sample_rate"],
        "n_fft": mel_cfg["n_fft"],
        "hop": mel_cfg["hop_length"],
        "win_length": mel_cfg["win_length"],
        "n_mels": mel_cfg["n_mels"],
        "f_min": mel_cfg["f_min"],
        "f_max": mel_cfg["f_max"],
        "power": mel_cfg["power"],
        "db_clip": list(mel_cfg["db_clip"]),
        "file_seconds": audio_cfg["file_seconds"],
        "window_seconds": audio_cfg["window_seconds"],
        "src": ogg_path.name,
    }
    save_kwargs = {"mel_u8": mels, "meta": np.array(meta, dtype=object)}
    if label is not None:
        save_kwargs["label"] = np.array(label)

    # Serialize to bytes first so atomic_write_bytes can do the tmp+rename.
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **save_kwargs)
    atomic_write_bytes(out_path, buf.getvalue())
    return out_path, "wrote"


# --------------------------------------------------------------------------
# Stage entry
# --------------------------------------------------------------------------

def run(cfg: CloudCfg, args: argparse.Namespace) -> int:  # noqa: ARG001
    s2 = cfg.raw["s2"]
    audio_cfg = cfg.raw["audio"]
    mel_cfg = cfg.raw["mel"]

    # ~5 GB per source is plenty headroom; abort early if the disk is tight.
    require_free_disk(cfg.work_root, need_gb=15.0)

    mel_fb = _build_mel_fb(
        sr=audio_cfg["sample_rate"],
        n_fft=mel_cfg["n_fft"],
        n_mels=mel_cfg["n_mels"],
        f_min=mel_cfg["f_min"],
        f_max=mel_cfg["f_max"],
    )
    jl = JsonlLogger(cfg.stage_log("S2"))

    total_done = 0
    total_wrote = 0
    for src in s2["sources"]:
        name = src["name"]
        glob = src["glob"]
        out_dir = cfg.mel_cache / name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(cfg.comp_dir.glob(glob))
        if not files:
            log.warning("S2 source %s: no files matched glob %s", name, glob)
            continue

        # train_audio: label = parent directory name (= primary_label)
        def label_for(p: Path) -> str | None:
            return p.parent.name if name == "train_audio" else None

        log.info("S2 source=%s files=%d -> %s", name, len(files), out_dir)

        def task(p: Path):
            out = out_dir / (p.stem + ".npz")
            return _process_file(p, out, mel_fb, audio_cfg, mel_cfg, label_for(p))

        nw = int(s2.get("num_workers", 4))
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futures = {ex.submit(task, p): p for p in files}
            for i, fut in enumerate(as_completed(futures)):
                p = futures[fut]
                try:
                    out, status = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.error("S2: %s FAILED: %s", p.name, e)
                    jl.log(source=name, file=p.name, status="error", error=str(e))
                    continue
                total_done += 1
                if status == "wrote":
                    total_wrote += 1
                if (i + 1) % 200 == 0:
                    log.info("S2 %s: %d / %d (wrote=%d)",
                             name, i + 1, len(files), total_wrote)
        jl.log(source=name, total=len(files), wrote=total_wrote)

    log.info("S2 done: %d processed (%d newly written)",
             total_done, total_wrote)
    return 0


if __name__ == "__main__":
    # Stand-alone invocation for debugging.
    from common.cloud_paths import load_cloud_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_cloud_config(a.config)
    cfg.mkdirs()
    sys.exit(run(cfg, argparse.Namespace()))
