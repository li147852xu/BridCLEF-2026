"""Stage 3.1 recovery: export the 3 mel-CNN .pt files to fp16 ONNX and score the
708 labeled rows → cnn_probs_labeled.npy, locally on CPU.

Why this exists: the Kaggle notebook finished all 3 seeds successfully (best_val_loss
0.137-0.140, best_epoch 8/14/14, ~7 MB each) but Cell 9 (onnx export) crashed with
`ModuleNotFoundError: onnxscript` under torch 2.10 on Kaggle's image. The .pt files
survived in /kaggle/working Output, so we just have to finish the last two cells
locally.

Architecture / pre-processing is hand-copied from kaggle_gpu/train_mel_cnn.ipynb
so ONNX graphs are bit-identical to what would have been exported on Kaggle.

Inputs (expected locations, overridable via --cnn-dir):
  artifacts/cnn_bundle/mel_cnn_seed{20260101,20260215,20260322}.pt
  artifacts/cnn_distill/teacher_cache_distill.pkl   (labeled_cache_idx + meta_row_id)
  data/birdclef-2026/train_soundscapes/*.ogg        (10658 files)

Outputs (into --cnn-dir):
  mel_cnn_seed*.onnx           (fp32, ~7 MB each)
  mel_cnn_seed*_fp16.onnx      (~3.5 MB each; this is what submit.ipynb loads)
  cnn_probs_labeled.npy        ((708, 234) float32, ensemble mean sigmoid)
  cnn_manifest.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torchvision.models import mobilenet_v3_small


# --------------------------- Config (MUST match notebook) ------------------


@dataclass
class CFG:
    sample_rate: int   = 32000
    window_sec : float = 5.0
    window_len : int   = 160000
    n_mels     : int   = 128
    n_fft      : int   = 1024
    hop_length : int   = 320
    f_min      : int   = 40
    f_max      : int   = 15000
    top_db     : float = 80.0
    # SpecAug params; kept here only because MelFrontEnd reads them, but
    # training=False at export/scoring so they have no effect.
    freq_mask_w: int   = 16
    time_mask_w: int   = 40
    spec_aug_n : int   = 2
    seeds: tuple       = (20260101, 20260215, 20260322)


# --------------------------- Model (MUST match notebook) -------------------


class MelFrontEnd(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min, f_max=cfg.f_max,
            n_mels=cfg.n_mels, power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=cfg.top_db)
        self.freq_mask_w = cfg.freq_mask_w
        self.time_mask_w = cfg.time_mask_w
        self.spec_aug_n  = cfg.spec_aug_n

    def forward(self, wav: torch.Tensor, training: bool = False) -> torch.Tensor:
        # Notebook wraps this in `autocast('cuda', enabled=False)`; on CPU that's
        # a no-op and we're always in fp32 here anyway, so we skip the context.
        wav_f32 = wav.float()
        x = self.mel(wav_f32)                          # (B, n_mels, T)
        x = torch.clamp(x, min=1e-6)
        x = self.to_db(x)
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sd = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
        x = (x - mu) / sd
        x = x.unsqueeze(1)                             # (B, 1, n_mels, T)
        # training=False path only; SpecAugment never applied.
        return x


class MelCNN(nn.Module):
    def __init__(self, n_classes: int, cfg: CFG):
        super().__init__()
        self.front = MelFrontEnd(cfg)
        backbone = mobilenet_v3_small(weights=None)
        old = backbone.features[0][0]
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=old.bias is not None)
        nn.init.kaiming_normal_(new.weight, mode='fan_out', nonlinearity='relu')
        backbone.features[0][0] = new
        in_feat = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_feat, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_classes),
        )
        self.backbone = backbone

    def forward(self, wav: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self.front(wav, training=training)
        return self.backbone(x)


class ExportWrapper(nn.Module):
    """Same wrapper used in the notebook: raw waveform -> sigmoid probabilities."""

    def __init__(self, base: MelCNN):
        super().__init__()
        self.base = base

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        logits = self.base(wav, training=False)
        return torch.sigmoid(logits)


# --------------------------- Audio I/O -------------------------------------


def load_window(path: Path, end_sec: int, sr: int, window_len: int) -> np.ndarray:
    """Read a 5s window ending at `end_sec * sr` samples. Returns float32[window_len]."""
    end_sample = int(end_sec) * sr
    start_sample = end_sample - window_len
    if start_sample < 0:
        start_sample, end_sample = 0, window_len
    try:
        audio, got_sr = sf.read(str(path), start=start_sample, stop=end_sample,
                                dtype='float32', always_2d=False)
        if got_sr != sr:
            return np.zeros(window_len, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.shape[0] < window_len:
            audio = np.concatenate([audio, np.zeros(window_len - audio.shape[0],
                                                    dtype=np.float32)])
        elif audio.shape[0] > window_len:
            audio = audio[:window_len]
        return audio
    except Exception:
        return np.zeros(window_len, dtype=np.float32)


def parse_row_id(rid: str) -> tuple[str, int]:
    stem, end = rid.rsplit('_', 1)
    return stem, int(end)


# --------------------------- Main steps ------------------------------------


def export_onnx(pt_path: Path, cfg: CFG, n_classes: int, out_fp32: Path,
                out_fp16: Path) -> None:
    """Export a single .pt to fp32 ONNX + fp16 ONNX."""
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    model = MelCNN(n_classes, cfg)
    model.load_state_dict(sd)
    model.eval()
    wrap = ExportWrapper(model).eval()

    dummy = torch.randn(1, cfg.window_len)

    # We MUST use the new dynamo-based exporter (via onnxscript) because
    # torchaudio.MelSpectrogram calls torch.stft internally and the legacy
    # torchscript exporter can't handle STFT's complex output types. onnxscript
    # is installed via `uv pip install onnxscript`.
    exported = torch.onnx.export(
        wrap, (dummy,),
        input_names=['wav'], output_names=['prob'],
        dynamic_shapes={'wav': {0: torch.export.Dim('batch')}},
        opset_version=17,
        dynamo=True,
    )
    if exported is not None:
        exported.save(out_fp32.as_posix())
    mb32 = out_fp32.stat().st_size / (1024 * 1024)
    print(f'  {out_fp32.name}  {mb32:.2f} MB (fp32)')

    # fp16 conversion
    import onnx as _onnx
    from onnxconverter_common import float16 as _float16
    m = _onnx.load(out_fp32.as_posix())
    m16 = _float16.convert_float_to_float16(m, keep_io_types=True)
    _onnx.save(m16, out_fp16.as_posix())
    mb16 = out_fp16.stat().st_size / (1024 * 1024)
    print(f'  {out_fp16.name}  {mb16:.2f} MB (fp16)')


@torch.no_grad()
def score_labeled_rows(cnn_dir: Path, cfg: CFG, *,
                       distill_path: Path, soundscapes_dir: Path,
                       batch_size: int, device: torch.device,
                       n_classes: int) -> tuple[np.ndarray, list[str]]:
    """Load 708 labeled windows, run 3-seed ensemble, return (708, K) probs."""
    print(f'Loading distill cache from {distill_path}')
    with open(distill_path, 'rb') as f:
        D = pickle.load(f)
    meta_row_id = list(D['meta_row_id'])
    primary_labels = list(D['primary_labels'])
    labeled_cache_idx = np.asarray(D['labeled_cache_idx'], dtype=np.int64)
    print(f'  labeled rows    : {len(labeled_cache_idx)}')
    print(f'  primary_labels  : {len(primary_labels)} (expect {n_classes})')
    assert len(primary_labels) == n_classes, 'class count mismatch'

    # Load the 708 waveforms once (shared across seeds).
    print(f'Loading 708 audio windows from {soundscapes_dir}')
    t_load = time.time()
    wavs = np.zeros((len(labeled_cache_idx), cfg.window_len), dtype=np.float32)
    missing = 0
    for i, idx in enumerate(labeled_cache_idx):
        stem, end_sec = parse_row_id(meta_row_id[int(idx)])
        path = soundscapes_dir / f'{stem}.ogg'
        if not path.exists():
            missing += 1
            continue
        wavs[i] = load_window(path, end_sec, cfg.sample_rate, cfg.window_len)
    print(f'  loaded 708 windows in {time.time() - t_load:.1f}s '
          f'({missing} missing → zeros)')

    P_sum = np.zeros((len(labeled_cache_idx), n_classes), dtype=np.float32)
    for seed in cfg.seeds:
        pt_path = cnn_dir / f'mel_cnn_seed{seed}.pt'
        print(f'Scoring with {pt_path.name}')
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model = MelCNN(n_classes, cfg).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        t0 = time.time()
        for start in range(0, len(labeled_cache_idx), batch_size):
            chunk = torch.from_numpy(wavs[start:start + batch_size]).to(device)
            logits = model(chunk, training=False)
            P_sum[start:start + chunk.size(0)] += torch.sigmoid(logits).cpu().numpy()
        print(f'  scored 708 rows in {time.time() - t0:.1f}s  '
              f'(best_val_loss={ckpt.get("best_val_loss", "?"):.4f} '
              f'best_auc={ckpt.get("best_auc", "?"):.4f} '
              f'ep={ckpt.get("best_epoch", "?")})')
        del model

    return (P_sum / len(cfg.seeds)).astype(np.float32), primary_labels


# --------------------------- CLI -------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cnn-dir', type=Path,
                    default=Path('artifacts/cnn_bundle'),
                    help='where the 3 .pt files live, and where ONNX/npy go')
    ap.add_argument('--distill', type=Path,
                    default=Path('artifacts/cnn_distill/teacher_cache_distill.pkl'))
    ap.add_argument('--soundscapes', type=Path,
                    default=Path('data/birdclef-2026/train_soundscapes'))
    ap.add_argument('--batch', type=int, default=16,
                    help='inference batch size (CPU: 8-16 is reasonable)')
    ap.add_argument('--skip-onnx', action='store_true',
                    help='only produce cnn_probs_labeled.npy')
    ap.add_argument('--skip-score', action='store_true',
                    help='only produce ONNX files')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFG()

    cnn_dir: Path = args.cnn_dir
    cnn_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'cnn_dir     : {cnn_dir.resolve()}')
    print(f'distill     : {args.distill}  (exists={args.distill.exists()})')
    print(f'soundscapes : {args.soundscapes}  (exists={args.soundscapes.exists()})')
    print(f'device      : {device}')
    print(f'cfg         : {asdict(cfg)}')

    # Verify all 3 .pt are present and have the expected schema.
    pts = [cnn_dir / f'mel_cnn_seed{seed}.pt' for seed in cfg.seeds]
    for p in pts:
        assert p.exists(), f'missing {p}  —  download it from the Kaggle Output panel'
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        assert 'state_dict' in ckpt and ckpt['state_dict'] is not None, \
            f'{p} has no state_dict'
        sz = sum(v.numel() for v in ckpt['state_dict'].values()) * 4 / 1024 ** 2
        print(f'  {p.name}  {sz:.1f} MB  ep={ckpt.get("best_epoch", "?")}  '
              f'val_loss={ckpt.get("best_val_loss", "?"):.4f}')

    # Infer K from state_dict
    n_classes = ckpt['state_dict']['backbone.classifier.3.weight'].shape[0]
    print(f'n_classes   : {n_classes}')

    # -------- ONNX export --------
    if not args.skip_onnx:
        print('\n== ONNX export ==')
        for seed in cfg.seeds:
            pt = cnn_dir / f'mel_cnn_seed{seed}.pt'
            fp32 = cnn_dir / f'mel_cnn_seed{seed}.onnx'
            fp16 = cnn_dir / f'mel_cnn_seed{seed}_fp16.onnx'
            export_onnx(pt, cfg, n_classes, fp32, fp16)

    # -------- cnn_probs_labeled.npy --------
    manifest = {
        'seeds'          : list(cfg.seeds),
        'n_classes'      : int(n_classes),
        'sample_rate'    : int(cfg.sample_rate),
        'window_sec'     : float(cfg.window_sec),
        'window_len'     : int(cfg.window_len),
        'onnx'           : [f'mel_cnn_seed{s}_fp16.onnx' for s in cfg.seeds],
        'per_seed'       : [],
    }
    for seed in cfg.seeds:
        ckpt = torch.load(cnn_dir / f'mel_cnn_seed{seed}.pt',
                          map_location='cpu', weights_only=False)
        manifest['per_seed'].append({
            'seed'         : int(seed),
            'best_epoch'   : int(ckpt.get('best_epoch', -1)),
            'best_val_loss': float(ckpt.get('best_val_loss', float('nan'))),
            'best_auc'     : float(ckpt.get('best_auc', float('nan'))),
        })

    if not args.skip_score:
        print('\n== Scoring 708 labeled rows (ensemble mean) ==')
        P, primary_labels = score_labeled_rows(
            cnn_dir, cfg,
            distill_path=args.distill,
            soundscapes_dir=args.soundscapes,
            batch_size=args.batch,
            device=device,
            n_classes=n_classes,
        )
        np.save(cnn_dir / 'cnn_probs_labeled.npy', P)
        manifest['primary_labels'] = primary_labels
        print(f'  saved cnn_probs_labeled.npy  shape {P.shape}  '
              f'min={P.min():.4f} max={P.max():.4f} mean={P.mean():.4f}')

    with open(cnn_dir / 'cnn_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'  saved cnn_manifest.json')

    print('\nDONE. Files in', cnn_dir)
    for p in sorted(cnn_dir.iterdir()):
        print(f'  {p.stat().st_size / 1024**2:7.2f} MB  {p.name}')


if __name__ == '__main__':
    main()
