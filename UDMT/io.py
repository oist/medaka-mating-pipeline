from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRACK_COLUMNS = ("frame", "track_id", "x", "y")
IDENTITY_COLUMNS = (
    "identity",
    "track_label",
    "assigned_identity",
    "id",
    "final_fish_id",
)


class TrackTableError(ValueError):
    """Raised when a CSV-like table cannot be used as a track table."""


def read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def require_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise TrackTableError(f"{label} missing columns: {', '.join(sorted(missing))}")


def normalize_tracks(df: pd.DataFrame, *, label: str = "tracks") -> pd.DataFrame:
    """Return a copy with canonical frame/track/x/y dtypes.

    `track_id` is kept as-is because segmentation exports may use stable string
    labels in parallel with numeric IDs. `frame` must be integer-like.
    """

    require_columns(df, TRACK_COLUMNS, label=label)
    out = df.copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    if out["frame"].isna().any():
        raise TrackTableError(f"{label} has non-numeric frame values")
    out["frame"] = out["frame"].astype(int)
    for col in ("x", "y"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


def identity_keys(df: pd.DataFrame, *, fallback: str = "track_id") -> pd.Series:
    """Return the best stable identity key available in a track/candidate table."""

    key = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for col in IDENTITY_COLUMNS:
        if col not in df.columns:
            continue
        value = df[col].fillna("").astype(str).str.strip()
        value = value.mask(value.isin(["", "nan", "None", "unknown", "unassigned"]))
        key = key.fillna(value)
    if fallback in df.columns:
        key = key.fillna(df[fallback].astype(str))
    return key.fillna("").astype(str)


def finite_xy_mask(df: pd.DataFrame) -> np.ndarray:
    xy = df[["x", "y"]].to_numpy(dtype=float)
    return np.isfinite(xy).all(axis=1)


def write_jsonable_summary(summary: dict, path: str | Path) -> None:
    import json

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
