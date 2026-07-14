from __future__ import annotations

"""Common data model and label-blind metrics for identity recovery benchmarks.

The reference trajectories are used in two deliberately separate ways:

* as an **unordered set of fish locations** for localization and event selection;
* as provisional labelled trajectories for a clearly marked geometry-proxy score.

Only the independent human audit can be interpreted as biological identity ground
truth.  Keeping the two score families separate prevents a tracker that drifts off
the fish from receiving credit merely because it continues to emit three labels.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


TRACK_COLUMNS = ["frame", "track_id", "x", "y"]
REFERENCE_COLUMNS = ["frame", "fish_id", "x", "y"]


@dataclass(frozen=True)
class CrossingEvent:
    event_id: str
    start_frame: int
    end_frame: int
    center_frame: int
    fish_a: str
    fish_b: str
    min_distance_px: float
    n_close_frames: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_columns(table: pd.DataFrame, columns: list[str], *, source: str) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {', '.join(sorted(missing))}")


def load_reference(path: str | Path) -> pd.DataFrame:
    """Load reprojection reference CSVs with either fish/gx/gy or fish_id/x/y."""
    table = pd.read_csv(path)
    renames = {}
    if "fish" in table:
        renames["fish"] = "fish_id"
    if "gx" in table:
        renames["gx"] = "x"
    if "gy" in table:
        renames["gy"] = "y"
    table = table.rename(columns=renames)
    _require_columns(table, REFERENCE_COLUMNS, source=str(path))
    out = table[REFERENCE_COLUMNS].copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="raise").astype(int)
    out["fish_id"] = out["fish_id"].astype(str)
    out[["x", "y"]] = out[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return out[np.isfinite(out[["x", "y"]]).all(axis=1)].reset_index(drop=True)


def load_idtracker_tidy(path: str | Path, *, frame_offset: int) -> pd.DataFrame:
    """Load idtracker.ai v6 ``trajectories_tidy.csv`` output."""
    table = pd.read_csv(path)
    _require_columns(table, ["frame", "individual", "x", "y"], source=str(path))
    out = table[["frame", "individual", "x", "y"]].rename(
        columns={"individual": "track_id"}
    )
    out["frame"] = pd.to_numeric(out["frame"], errors="raise").astype(int) + int(frame_offset)
    out["track_id"] = out["track_id"].astype(str)
    out[["x", "y"]] = out[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return out[np.isfinite(out[["x", "y"]]).all(axis=1)].reset_index(drop=True)


def load_udmt_npy(
    path: str | Path, *, frame_offset: int, resize_factor: float
) -> pd.DataFrame:
    """Load native UDMT ``(identity, frame, xy)`` centers in original pixels."""
    if resize_factor <= 0:
        raise ValueError("resize_factor must be positive")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"expected UDMT shape (tracks, frames, 2), got {array.shape}")
    track, rel_frame = np.indices(array.shape[:2])
    out = pd.DataFrame(
        {
            "frame": rel_frame.ravel() + int(frame_offset),
            "track_id": track.ravel().astype(str),
            "x": array[:, :, 0].ravel() / float(resize_factor),
            "y": array[:, :, 1].ravel() / float(resize_factor),
        }
    )
    return out[np.isfinite(out[["x", "y"]]).all(axis=1)].reset_index(drop=True)


def load_tracks_csv(path: str | Path, *, frame_offset: int = 0) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "individual" in table and "track_id" not in table:
        table = table.rename(columns={"individual": "track_id"})
    _require_columns(table, TRACK_COLUMNS, source=str(path))
    out = table[TRACK_COLUMNS].copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="raise").astype(int) + int(frame_offset)
    out["track_id"] = out["track_id"].astype(str)
    out[["x", "y"]] = out[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return out[np.isfinite(out[["x", "y"]]).all(axis=1)].reset_index(drop=True)


def derive_crossing_events(
    reference: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
    distance_threshold_px: float = 35.0,
    merge_gap_frames: int = 8,
    min_close_frames: int = 2,
    required_context_frames: int = 83,
) -> list[CrossingEvent]:
    """Select all close-approach episodes from unordered reference locations.

    Labels are used only to keep pair-specific close runs separate.  Selection is
    exhaustive inside the locked interval and does not inspect method outputs.
    Events too near either interval boundary are excluded *a priori* because a
    before/after continuity judgment would be impossible.
    """
    ref = reference[
        (reference["frame"] >= int(start_frame))
        & (reference["frame"] <= int(end_frame))
    ]
    close_rows: list[dict[str, object]] = []
    for frame, group in ref.groupby("frame", sort=True):
        group = group.sort_values("fish_id")
        ids = group["fish_id"].astype(str).to_numpy()
        xy = group[["x", "y"]].to_numpy(dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                distance = float(np.linalg.norm(xy[left] - xy[right]))
                if distance <= float(distance_threshold_px):
                    close_rows.append(
                        {
                            "frame": int(frame),
                            "fish_a": ids[left],
                            "fish_b": ids[right],
                            "distance": distance,
                        }
                    )
    if not close_rows:
        return []

    close = pd.DataFrame(close_rows).sort_values(["fish_a", "fish_b", "frame"])
    candidates: list[dict[str, object]] = []
    for (fish_a, fish_b), group in close.groupby(["fish_a", "fish_b"], sort=True):
        rows = group.sort_values("frame").reset_index(drop=True)
        run = (rows["frame"].diff().fillna(merge_gap_frames + 1) > merge_gap_frames).cumsum()
        for _, episode in rows.groupby(run):
            if len(episode) < int(min_close_frames):
                continue
            start = int(episode["frame"].min())
            end = int(episode["frame"].max())
            if start - int(required_context_frames) < int(start_frame):
                continue
            if end + int(required_context_frames) > int(end_frame):
                continue
            min_idx = episode["distance"].idxmin()
            candidates.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "center_frame": int(episode.loc[min_idx, "frame"]),
                    "fish_a": str(fish_a),
                    "fish_b": str(fish_b),
                    "min_distance_px": float(episode["distance"].min()),
                    "n_close_frames": int(len(episode)),
                }
            )
    candidates.sort(key=lambda row: (int(row["center_frame"]), str(row["fish_a"])))
    return [
        CrossingEvent(event_id=f"crossing_{index:02d}", **row)
        for index, row in enumerate(candidates, start=1)
    ]


def _frame_assignments(
    tracks: pd.DataFrame, reference: pd.DataFrame, *, gate_px: float
) -> pd.DataFrame:
    """One-to-one per-frame Hungarian matches, independent of identity labels."""
    rows: list[dict[str, object]] = []
    reference_by_frame = {int(f): g for f, g in reference.groupby("frame")}
    for frame, method_frame in tracks.groupby("frame", sort=True):
        ref_frame = reference_by_frame.get(int(frame))
        if ref_frame is None or ref_frame.empty or method_frame.empty:
            continue
        method_frame = method_frame.drop_duplicates("track_id", keep="first")
        method_xy = method_frame[["x", "y"]].to_numpy(dtype=float)
        ref_xy = ref_frame[["x", "y"]].to_numpy(dtype=float)
        distances = np.linalg.norm(method_xy[:, None, :] - ref_xy[None, :, :], axis=2)
        method_idx, ref_idx = linear_sum_assignment(distances)
        for left, right in zip(method_idx, ref_idx):
            distance = float(distances[left, right])
            if distance > float(gate_px):
                continue
            rows.append(
                {
                    "frame": int(frame),
                    "track_id": str(method_frame.iloc[left]["track_id"]),
                    "fish_id": str(ref_frame.iloc[right]["fish_id"]),
                    "distance_px": distance,
                }
            )
    return pd.DataFrame(rows, columns=["frame", "track_id", "fish_id", "distance_px"])


def localization_metrics(
    tracks: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
    gate_px: float,
) -> dict[str, object]:
    """Identity-blind fish localization metrics at a declared pixel gate."""
    method = tracks[(tracks["frame"] >= start_frame) & (tracks["frame"] <= end_frame)]
    ref = reference[(reference["frame"] >= start_frame) & (reference["frame"] <= end_frame)]
    assignments = _frame_assignments(method, ref, gate_px=gate_px)
    n_expected = int(len(ref))
    n_emitted = int(len(method))
    n_matched = int(len(assignments))
    expected_per_frame = ref.groupby("frame")["fish_id"].nunique()
    matched_per_frame = assignments.groupby("frame")["fish_id"].nunique()
    full = matched_per_frame.reindex(expected_per_frame.index, fill_value=0) >= expected_per_frame
    distances = assignments["distance_px"].to_numpy(dtype=float)
    return {
        "gate_px": float(gate_px),
        "n_expected_positions": n_expected,
        "n_emitted_positions": n_emitted,
        "n_matched_positions": n_matched,
        "position_recall": n_matched / n_expected if n_expected else float("nan"),
        "position_precision": n_matched / n_emitted if n_emitted else float("nan"),
        "all_fish_frame_coverage": float(full.mean()) if len(full) else float("nan"),
        "median_error_px": float(np.median(distances)) if len(distances) else float("nan"),
        "p95_error_px": float(np.quantile(distances, 0.95)) if len(distances) else float("nan"),
    }


def geometry_proxy_identity_metrics(
    tracks: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
    gate_px: float = 80.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Identity metrics against provisional geometry labels, never human GT."""
    method = tracks[(tracks["frame"] >= start_frame) & (tracks["frame"] <= end_frame)]
    ref = reference[(reference["frame"] >= start_frame) & (reference["frame"] <= end_frame)]
    assignments = _frame_assignments(method, ref, gate_px=gate_px)
    track_ids = sorted(method["track_id"].astype(str).unique())
    fish_ids = sorted(ref["fish_id"].astype(str).unique())
    counts = pd.crosstab(assignments["track_id"], assignments["fish_id"]).reindex(
        index=track_ids, columns=fish_ids, fill_value=0
    )
    mapping: dict[str, str] = {}
    if len(track_ids) and len(fish_ids):
        row, col = linear_sum_assignment(-counts.to_numpy(dtype=float))
        mapping = {track_ids[left]: fish_ids[right] for left, right in zip(row, col)}
    matched = assignments.copy()
    matched["mapped_fish_id"] = matched["track_id"].map(mapping)
    matched["proxy_correct"] = matched["mapped_fish_id"] == matched["fish_id"]
    correct = int(matched["proxy_correct"].sum())
    denominator = int(len(method) + len(ref))
    idf1_like = 2 * correct / denominator if denominator else float("nan")

    switch_count = 0
    track_purities: dict[str, float] = {}
    for track_id, group in matched.sort_values("frame").groupby("track_id"):
        values = group["fish_id"].astype(str).to_numpy()
        frames = group["frame"].to_numpy(dtype=int)
        adjacent = np.diff(frames) == 1
        switch_count += int(np.sum((values[1:] != values[:-1]) & adjacent))
        value_counts = group["fish_id"].value_counts()
        track_purities[str(track_id)] = (
            float(value_counts.iloc[0] / value_counts.sum()) if len(value_counts) else float("nan")
        )
    report = {
        "warning": "geometry-proxy labels are not independent biological ground truth",
        "gate_px": float(gate_px),
        "global_track_to_proxy_fish": mapping,
        "idf1_like": float(idf1_like),
        "proxy_correct_positions": correct,
        "proxy_switches_adjacent_frames": int(switch_count),
        "track_purity": track_purities,
    }
    return report, matched


def crossing_proxy_metrics(
    matched: pd.DataFrame,
    events: list[CrossingEvent],
    mapping: dict[str, str],
    *,
    before_after_frames: int = 40,
    exclusion_frames: int = 5,
) -> pd.DataFrame:
    """Measure whether each globally mapped track localizes to the same proxy fish
    before and after an event.  This is a diagnostic; human audit remains primary.
    """
    inverse = {fish: track for track, fish in mapping.items()}
    rows: list[dict[str, object]] = []
    for event in events:
        for fish_id in (event.fish_a, event.fish_b):
            track_id = inverse.get(str(fish_id))
            before = matched[
                (matched["track_id"] == track_id)
                & (matched["frame"] >= event.start_frame - before_after_frames)
                & (matched["frame"] < event.start_frame - exclusion_frames)
            ]
            after = matched[
                (matched["track_id"] == track_id)
                & (matched["frame"] > event.end_frame + exclusion_frames)
                & (matched["frame"] <= event.end_frame + before_after_frames)
            ]
            before_same = float((before["fish_id"] == str(fish_id)).mean()) if len(before) else np.nan
            after_same = float((after["fish_id"] == str(fish_id)).mean()) if len(after) else np.nan
            passed = bool(before_same >= 0.5 and after_same >= 0.5) if np.isfinite(before_same) and np.isfinite(after_same) else False
            rows.append(
                {
                    "event_id": event.event_id,
                    "fish_id": str(fish_id),
                    "track_id": track_id,
                    "before_same_fraction": before_same,
                    "after_same_fraction": after_same,
                    "proxy_continuity_pass": passed,
                }
            )
    return pd.DataFrame(rows)

