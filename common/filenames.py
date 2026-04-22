"""Parsing of BirdCLEF+ 2026 soundscape filenames.

Expected schema: ``BC2026_{Train|Test}_{file_id}_{site}_{YYYYMMDD}_{HHMMSS}.ogg``

Example: ``BC2026_Train_0001_S01_20240915_053000.ogg``
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


@dataclass(frozen=True)
class SoundscapeMeta:
    file_id: str | None
    site: str | None
    date: pd.Timestamp | pd.NaT
    time_utc: str | None
    hour_utc: int
    month: int


def parse_soundscape_filename(name: str) -> dict:
    """Parse an ``.ogg`` filename into ``{file_id, site, date, time_utc, hour_utc, month}``.

    Returns a dict with ``hour_utc = -1`` and ``month = -1`` for unparseable names,
    which matches the legacy notebook behavior. ``site`` becomes ``None``.
    """
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }


def parse_many(names) -> pd.DataFrame:
    """Vectorized parse → DataFrame aligned with input order."""
    rows = [parse_soundscape_filename(str(n)) for n in names]
    return pd.DataFrame(rows)


def row_ids_for_file(stem: str, windows_per_file: int, window_seconds: int) -> list[str]:
    """Build Kaggle row_id strings for a single soundscape file.

    ``row_id = f"{stem}_{end_sec}"`` where end_sec goes
    ``[window_seconds, 2*window_seconds, ..., windows_per_file*window_seconds]``.
    """
    return [f"{stem}_{(i + 1) * window_seconds}" for i in range(windows_per_file)]
