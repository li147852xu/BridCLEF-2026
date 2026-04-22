"""Perch v2 wrapper.

Loads the ``perch_v2_cpu`` SavedModel once per process and exposes a batched
generator that yields ``(meta_df, scores, embeddings)`` per batch of files.

``scores`` is already projected to the competition's ``primary_labels`` space,
with genus-level proxies applied for unmapped Amphibia/Insecta/Aves.
``embeddings`` is the raw 1536-dim Perch output (before any PCA).

Both training and Kaggle inference import this module; no retraining here.
"""

from __future__ import annotations

# macOS import-order workaround: pandas 2.x eagerly imports pyarrow, whose
# statically-linked abseil clashes with TF 2.20's abseil and causes Perch's
# StableHLO executor to deadlock. Importing tensorflow here (before pandas /
# pyarrow) lets TF own abseil symbol resolution. Safe on Linux/Kaggle.
import os as _os
import sys as _sys
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# On macOS (Perch v2 + TF 2.20 + Darwin arm64) we disable CUDA to avoid the
# XlaCallModule deadlock; on Linux we leave the env alone so cloud GPU Perch
# works when the caller exports CUDA_VISIBLE_DEVICES=0.
if _sys.platform == "darwin":
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
try:
    import tensorflow as _tf  # noqa: F401
except Exception:
    _tf = None

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common.audio import read_soundscape, windowize
from common.filenames import parse_soundscape_filename, row_ids_for_file
from common.taxonomy import LabelMapping


@dataclass
class PerchEngine:
    """Lightweight wrapper around Perch v2.

    ``engine.infer(...)`` returns a generator of ``(meta_df, scores, embeddings)``
    batches. Keep the engine instance alive across multiple batches to avoid
    reloading the SavedModel.
    """

    model_dir: Path
    sample_rate: int = 32000
    window_seconds: int = 5
    file_seconds: int = 60
    windows_per_file: int = 12
    embedding_dim: int = 1536
    _infer_fn = None
    _model = None

    def __post_init__(self) -> None:
        import os
        import sys
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        if sys.platform == "darwin":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import tensorflow as tf

        # TF 2.20 on Darwin arm64 deadlocks inside ProcessFunctionLibraryRuntime
        # when inter_op > 1 for Perch's XlaCallModule. Serialize inter-op so
        # the executor actually completes. On Linux we leave TF threading
        # alone so CUDA Perch can use all available parallelism.
        if sys.platform == "darwin":
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(
                    max(4, (os.cpu_count() or 8))
                )
            except RuntimeError:
                pass

        tf.experimental.numpy.experimental_enable_numpy_behavior()
        self._model = tf.saved_model.load(str(self.model_dir))
        self._infer_fn = self._model.signatures["serving_default"]

    # ------------------------------------------------------------------ public

    @property
    def window_samples(self) -> int:
        return self.sample_rate * self.window_seconds

    def infer(
        self,
        paths: Sequence[str | Path],
        mapping: LabelMapping,
        *,
        batch_files: int = 32,
        proxy_reduce: str = "max",
        verbose: bool = True,
        gc_every: int = 4,
    ) -> Iterator[tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        """Yield ``(meta_df, scores, embeddings)`` per batch.

        ``scores`` shape: ``(batch_n * windows_per_file, n_classes)``.
        ``embeddings`` shape: ``(batch_n * windows_per_file, embedding_dim)``.
        """
        import tensorflow as tf

        paths = [Path(p) for p in paths]
        n_files = len(paths)
        n_classes = len(mapping.primary_labels)

        iterator = range(0, n_files, batch_files)
        if verbose:
            iterator = tqdm(
                iterator,
                total=(n_files + batch_files - 1) // batch_files,
                desc="Perch batches",
            )

        batch_idx = 0
        for start in iterator:
            batch_paths = paths[start : start + batch_files]
            batch_n = len(batch_paths)
            n_rows = batch_n * self.windows_per_file

            x = np.empty((n_rows, self.window_samples), dtype=np.float32)
            row_ids = np.empty(n_rows, dtype=object)
            filenames = np.empty(n_rows, dtype=object)
            sites = np.empty(n_rows, dtype=object)
            hours = np.empty(n_rows, dtype=np.int16)
            months = np.empty(n_rows, dtype=np.int16)

            write_row = 0
            for path in batch_paths:
                y = read_soundscape(
                    path,
                    target_sr=self.sample_rate,
                    file_seconds=self.file_seconds,
                )
                x[write_row : write_row + self.windows_per_file] = windowize(
                    y,
                    target_sr=self.sample_rate,
                    window_seconds=self.window_seconds,
                    windows_per_file=self.windows_per_file,
                )

                meta = parse_soundscape_filename(path.name)
                stem = path.stem
                ids = row_ids_for_file(stem, self.windows_per_file, self.window_seconds)
                row_ids[write_row : write_row + self.windows_per_file] = ids
                filenames[write_row : write_row + self.windows_per_file] = path.name
                sites[write_row : write_row + self.windows_per_file] = meta["site"]
                hours[write_row : write_row + self.windows_per_file] = int(meta["hour_utc"])
                months[write_row : write_row + self.windows_per_file] = int(meta["month"])
                write_row += self.windows_per_file

            outputs = self._infer_fn(inputs=tf.convert_to_tensor(x))
            logits = outputs["label"].numpy().astype(np.float32, copy=False)
            emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
            # release TF tensor handles for unused outputs (spectrogram/spatial_embedding)
            # immediately to keep resident memory bounded on macOS.
            outputs.clear()
            del outputs

            scores = np.zeros((n_rows, n_classes), dtype=np.float32)
            scores[:, mapping.mapped_pos] = logits[:, mapping.mapped_bc_indices]

            for pos, bc_idx_arr in mapping.proxy_pos_to_bc.items():
                sub = logits[:, bc_idx_arr]
                proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
                scores[:, pos] = proxy_score.astype(np.float32)

            meta_df = pd.DataFrame(
                {
                    "row_id": row_ids,
                    "filename": filenames,
                    "site": sites,
                    "hour_utc": hours,
                    "month": months,
                }
            )

            yield meta_df, scores, emb

            del x, logits, emb, scores, meta_df
            batch_idx += 1
            if gc_every and (batch_idx % gc_every == 0):
                gc.collect()
