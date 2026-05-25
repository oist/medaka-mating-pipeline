from __future__ import annotations

"""
Loader for manual BORIS behavior annotations.

BORIS exports one CSV per observation; each row is either a POINT event or a
START/STOP edge of a STATE event. The repo expects one file per experiment
named `<folder_name>.csv` (e.g. `20251220_E_1_1_FCMC.csv`) inside the
`boris_events_dir` configured in `configs/paths.local.toml`.

Key column from BORIS we rely on:
    Image index    — top-cam frame number for the event. This lets us pin
                     events directly to frames without re-deriving from time
                     and FPS.

Subject vocabulary used in these annotations:
    f, m1, m2                  — a single fish doing something on its own
    m1tf, m2tf                 — m1 / m2 targeting the female
    m1tm2, m2tm1               — male-male targeting
    m1m2                       — mutual (both males engaged in the bout)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


BORIS_FILENAME_RE = re.compile(r"^(?P<folder>\d{8}_[A-Za-z]+_\d+_\d+_\w+)\.csv$")

# Columns we expose after normalisation, in this order.
NORMALISED_COLUMNS = [
    "folder_name",
    "subject",
    "behavior",
    "category",
    "behavior_type",   # POINT / START / STOP
    "time_sec",
    "frame_index",     # int; -1 if BORIS left it as NA
    "media_file",
]


@dataclass(frozen=True)
class BorisSummary:
    """Per-experiment digest used to enrich the catalog."""

    csv_path: Path
    n_events: int
    n_unique_behaviors: int
    n_unique_subjects: int
    first_frame: int | None
    last_frame: int | None
    duration_sec: float | None
    behaviors: dict[str, int]   # behavior -> event count
    subjects: dict[str, int]    # subject  -> event count


def find_boris_csv(folder_name: str, boris_dir: str | Path | None) -> Path | None:
    """Return the BORIS CSV for an experiment folder, or None if absent."""
    if boris_dir is None:
        return None
    path = Path(boris_dir) / f"{folder_name}.csv"
    return path if path.exists() else None


def list_boris_files(boris_dir: str | Path) -> list[Path]:
    """List per-experiment BORIS CSVs (skips desktop.ini and friends)."""
    root = Path(boris_dir)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and BORIS_FILENAME_RE.match(p.name)
    )


def load_boris_csv(path: str | Path) -> pd.DataFrame:
    """Read one BORIS CSV and return a normalised long-form DataFrame."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # BORIS leaves Image index as "NA" when not applicable. Coerce to int with -1 sentinel.
    image_index = pd.to_numeric(df.get("Image index"), errors="coerce")
    time_sec = pd.to_numeric(df.get("Time"), errors="coerce")

    out = pd.DataFrame({
        "folder_name": df.get("Observation id", "").astype(str),
        "subject": df.get("Subject", "").astype(str),
        "behavior": df.get("Behavior", "").astype(str),
        "category": df.get("Behavioral category", "").astype(str),
        "behavior_type": df.get("Behavior type", "").astype(str),
        "time_sec": time_sec,
        "frame_index": image_index.fillna(-1).astype("int64"),
        "media_file": df.get("Media file name", "").astype(str),
    })
    return out[NORMALISED_COLUMNS]


def summarise_boris(csv_path: str | Path, df: pd.DataFrame | None = None) -> BorisSummary:
    """Build a BorisSummary for catalog use."""
    path = Path(csv_path)
    if df is None:
        df = load_boris_csv(path)

    valid_frames = df.loc[df["frame_index"] >= 0, "frame_index"]
    valid_time = df["time_sec"].dropna()

    behaviors = df["behavior"].value_counts().to_dict()
    subjects = df["subject"].value_counts().to_dict()

    return BorisSummary(
        csv_path=path,
        n_events=len(df),
        n_unique_behaviors=int(df["behavior"].nunique()),
        n_unique_subjects=int(df["subject"].nunique()),
        first_frame=int(valid_frames.min()) if not valid_frames.empty else None,
        last_frame=int(valid_frames.max()) if not valid_frames.empty else None,
        duration_sec=float(valid_time.max() - valid_time.min()) if len(valid_time) >= 2 else None,
        behaviors=behaviors,
        subjects=subjects,
    )


def state_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair START/STOP rows into (subject, behavior, start_frame, stop_frame) intervals.
    Useful for downstream snippet extraction and metric computation.
    Unpaired edges are skipped with no warning — fix the BORIS file if you care.
    """
    states = df[df["behavior_type"].isin(["START", "STOP"])].copy()
    rows: list[dict] = []
    open_edges: dict[tuple[str, str], dict] = {}
    for _, row in states.iterrows():
        key = (row["subject"], row["behavior"])
        if row["behavior_type"] == "START":
            open_edges[key] = row.to_dict()
        else:  # STOP
            start = open_edges.pop(key, None)
            if start is None:
                continue
            rows.append({
                "subject": row["subject"],
                "behavior": row["behavior"],
                "category": row["category"],
                "start_frame": start["frame_index"],
                "stop_frame": row["frame_index"],
                "start_time_sec": start["time_sec"],
                "stop_time_sec": row["time_sec"],
            })
    return pd.DataFrame(rows)


def iter_event_frames(
    df: pd.DataFrame,
    behaviors: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Filter the events to those usable for pinning a single frame.
    Returns rows with a non-negative frame_index, optionally narrowed by
    behavior / subject. POINT events and START edges of states are both kept;
    callers that want only POINTs should filter further on behavior_type.
    """
    out = df[df["frame_index"] >= 0].copy()
    if behaviors is not None:
        wanted = set(behaviors)
        out = out[out["behavior"].isin(wanted)]
    if subjects is not None:
        wanted = set(subjects)
        out = out[out["subject"].isin(wanted)]
    return out.reset_index(drop=True)
