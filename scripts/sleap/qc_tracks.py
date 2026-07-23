from __future__ import annotations

"""
Multi-view SLEAP tracking QC: missing data versus occlusion.

This script reads three synchronized `.slp` files (top / side1 / side2) and
classifies missing detections frame by frame:

- `true_missing`: all views are below the expected individual count
- `*_occlusion`: one view is below the expected count while at least one other
  view remains complete
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import sleap_io as sio  # type: ignore[import]
except ImportError as e:  # noqa: BLE001
    raise SystemExit(
        "Could not import sleap-io. Install the project environment first, "
        "for example with:\n"
        "  uv sync --extra sleap"
    ) from e


def load_slp_data(slp_path: Path) -> pd.DataFrame:
    """Load instance-level data from a `.slp` file into a DataFrame."""
    print(f"Loading SLEAP file: {slp_path}")

    try:
        labels = sio.load_slp(slp_path)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load SLEAP file: {slp_path}\n"
            f"Error: {type(error).__name__}: {error}"
        ) from error

    rows: list[dict] = []

    # Iterate over all labeled frames with instances.
    for lf in labels.labeled_frames:  # type: ignore[attr-defined]
        frame_idx = lf.frame_idx

        for inst in lf.instances:
            # Extract track name.
            track_name = None
            if getattr(inst, "track", None) is not None:
                track_name = getattr(inst.track, "name", None) or getattr(
                    inst.track, "track_id", None
                )
            if track_name is None:
                track_name = f"untracked_{id(inst)}"

            # Count keypoints for the instance.
            node_count = 0
            points_data = inst.points

            # Support multiple object layouts when counting nodes.
            if isinstance(points_data, dict) or (
                hasattr(points_data, "items")
                and callable(getattr(points_data, "items"))
            ):
                try:
                    node_count = len(points_data)
                except Exception:  # noqa: BLE001
                    node_count = 0
            else:
                try:
                    import numpy as _np

                    if hasattr(inst, "numpy"):
                        arr = inst.numpy()  # type: ignore[attr-defined]
                    else:
                        arr = _np.array(points_data)
                    if hasattr(arr, "shape") and arr.size > 0:
                        if arr.ndim >= 2:
                            node_count = arr.shape[0]
                        else:
                            node_count = 1
                except Exception:  # noqa: BLE001
                    node_count = 0

            rows.append(
                dict(
                    frame=int(frame_idx),
                    track=str(track_name),
                    node_count=int(node_count),
                )
            )

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        raise SystemExit("Error: No tracking data found in .slp file.")

    return df


def summarize_instance_counts(df: pd.DataFrame, expected_count: int) -> pd.DataFrame:
    """Summarize one view into frame-wise counts and missing flags."""
    frame_counts_raw = df.groupby("frame")["track"].nunique()
    frame_min = int(df["frame"].min())
    frame_max = int(df["frame"].max())
    full_index = range(frame_min, frame_max + 1)
    frame_counts = frame_counts_raw.reindex(full_index, fill_value=0)

    out = pd.DataFrame(
        {
            "frame": list(full_index),
            "instance_count": frame_counts.values.astype(int),
        }
    )
    out["missing_degree"] = np.clip(expected_count - out["instance_count"], 0, None)
    out["is_missing"] = out["missing_degree"] > 0
    return out


def _normalize_event_label(label: str) -> str:
    """Normalize event names into safe column-name fragments."""
    label = str(label)
    label = re.sub(r"\s+|-+", "_", label)
    label = re.sub(r"[^\w]", "_", label)
    return label or "event"


def load_events_csv(
    events_path: Path,
    time_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """
    Load behavioral events from CSV and normalize them to `start_s, end_s, label`.

    Supported formats:
    1. interval events using Start/Stop columns
    2. point events using one Time column
    3. BORIS START/STOP/POINT style event tables
    """
    df = pd.read_csv(events_path)

    # Identify the label column first.
    label_source_col: str | None = None
    if label_col is not None and label_col in df.columns:
        label_source_col = label_col
    else:
        for c in ["Behavior", "behavior", "event", "label"]:
            if c in df.columns:
                label_source_col = c
                break

    # Prefer an explicit interval-event format first.
    candidate_start = [c for c in [start_col, "Start (s)", "Start(s)", "start_s", "start", "Start"] if c is not None]
    candidate_end = [c for c in [end_col, "Stop (s)", "Stop(s)", "end_s", "end", "Stop"] if c is not None]

    start_used = next((c for c in candidate_start if c in df.columns), None)
    end_used = next((c for c in candidate_end if c in df.columns), None)

    if start_used is not None and end_used is not None:
        events = pd.DataFrame(
            {
                "start_s": df[start_used].astype(float),
                "end_s": df[end_used].astype(float),
            }
        )
        # Interval-event format: copy labels directly.
        if label_source_col is not None:
            events["label"] = df[label_source_col].astype(str).values
        else:
            events["label"] = "event"
    else:
        # Otherwise try Time + START/STOP/POINT style input.
        candidate_time = [
            c
            for c in [
                time_col,
                "Time (s)",
                "Time(s)",
                "Time",
                "time_s",
                "time",
                "Image index",
                "Image index (frames)",
                "Frame",
            ]
            if c is not None
        ]
        time_used = next((c for c in candidate_time if c in df.columns), None)
        if time_used is None:
                raise SystemExit(
                    "Could not find a time column in the behavioral event CSV.\n"
                    "Use --events-time-col or --events-start-col/--events-end-col "
                    "to match the BORIS export format."
                )

        # Auto-detect the event-type column (START/STOP/POINT).
        type_col: str | None = None
        for c in df.columns:
            vals = df[c].astype(str).str.upper().str.strip()
            if vals.isin({"START", "STOP", "POINT"}).mean() > 0.5:
                type_col = c
                break

        rows: list[dict] = []

        if type_col is not None:
            # BORIS START/STOP mode: group by label and pair in time order.
            if label_source_col is not None and label_source_col in df.columns:
                groups = df.groupby(label_source_col, sort=False)
            else:
                # Without a label column, treat all events as one class.
                groups = [(None, df)]

            for label_value, g in groups:
                g_sorted = g.sort_values(time_used)
                current_start: float | None = None
                for _, row in g_sorted.iterrows():
                    t = float(row[time_used])
                    etype = str(row[type_col]).upper().strip()
                    label_str = (
                        str(label_value)
                        if label_value is not None
                        else "event"
                    )
                    if etype == "START":
                        current_start = t
                    elif etype == "STOP" and current_start is not None:
                        rows.append(
                            {
                                "start_s": current_start,
                                "end_s": t,
                                "label": label_str,
                            }
                        )
                        current_start = None
                    elif etype == "POINT":
                        rows.append(
                            {
                                "start_s": t,
                                "end_s": t,
                                "label": label_str,
                            }
                        )
                # Ignore an unmatched trailing START rather than creating a long interval.
        else:
            # Time-only input becomes point events.
            label_values: pd.Series | None = None
            if label_source_col is not None and label_source_col in df.columns:
                label_values = df[label_source_col].astype(str)

            for _, row in df.iterrows():
                t = float(row[time_used])
                label_str = (
                    str(row[label_source_col])
                    if label_source_col is not None and label_source_col in row
                    else "event"
                )
                rows.append({"start_s": t, "end_s": t, "label": label_str})

        events = pd.DataFrame(rows)
        if events.empty:
            raise SystemExit(
                "No valid events could be constructed from the behavioral event CSV.\n"
                "Check whether the file contains Time + START/STOP/POINT or Start/Stop columns."
            )

    # Drop invalid rows.
    events = events.dropna(subset=["start_s", "end_s"])
    events = events[events["end_s"] >= events["start_s"]]
    events = events.sort_values("start_s").reset_index(drop=True)
    return events


def extract_clip_start_time(filename: str | Path) -> float | None:
    """Extract the clip start time in seconds from a filename."""
    filename_str = str(filename)
    
    pattern = r'(\d+\.\d+)-(\d+\.\d+)'
    match = re.search(pattern, filename_str)
    
    if match:
        start_time = float(match.group(1))
        return start_time
    
    return None


def find_side_slp_files(top_slp_path: Path) -> tuple[Path | None, Path | None]:
    """Find the matching side-view `.slp` files for a top-view `.slp` file."""
    top_str = str(top_slp_path)
    top_path = Path(top_slp_path)
    
    # Method 1: replace PLAYER1 and _top together.
    if "PLAYER1" in top_str and "_top" in top_str:
        side1_str = top_str.replace("PLAYER1", "PLAYER2").replace("_top", "_side1")
        side2_str = top_str.replace("PLAYER1", "PLAYER3").replace("_top", "_side2")
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        
        if side1_path.exists() and side2_path.exists():
            return side1_path, side2_path
    
    # Method 2: replace _top.slp directly.
    if "_top.slp" in top_str:
        side1_str = top_str.replace("_top.slp", "_side1.slp")
        side2_str = top_str.replace("_top.slp", "_side2.slp")
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        
        if side1_path.exists() and side2_path.exists():
            return side1_path, side2_path
    
    # Method 3: replace PLAYER1 only.
    if "PLAYER1" in top_str:
        side1_str = top_str.replace("PLAYER1", "PLAYER2")
        side2_str = top_str.replace("PLAYER1", "PLAYER3")
        
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        
        side1_exists = side1_path.exists()
        side2_exists = side2_path.exists()
        
        if side1_exists and side2_exists:
            return side1_path, side2_path
        elif side1_exists or side2_exists:
            return (
                side1_path if side1_exists else None,
                side2_path if side2_exists else None,
            )
    
    # Method 4: search for side1/side2 files in the same directory.
    parent_dir = top_path.parent
    top_stem = top_path.stem
    
    base_stem = top_stem.replace("_top", "")
    
    candidates = [
        (parent_dir / f"{base_stem}_side1.slp", parent_dir / f"{base_stem}_side2.slp"),
        (parent_dir / f"{top_stem}_side1.slp", parent_dir / f"{top_stem}_side2.slp"),
    ]
    
    for side1_candidate, side2_candidate in candidates:
        if side1_candidate.exists() and side2_candidate.exists():
            return side1_candidate, side2_candidate
    
    return None, None


def classify_multiview_missing(
    slp_top: Path,
    slp_side1: Path,
    slp_side2: Path,
    expected_count: int = 3,
) -> pd.DataFrame:
    """
    Classify missing data versus likely occlusion across the three views.

    For each frame:
    - if top/side1/side2 are all below `expected_count`, mark `true_missing`
    - if one view is below `expected_count` while another stays complete, mark
      that view as occluded
    """
    # Load the three views separately.
    df_top = load_slp_data(slp_top)
    df_a = load_slp_data(slp_side1)
    df_b = load_slp_data(slp_side2)

    # Summarize counts frame by frame.
    fr_top = summarize_instance_counts(df_top, expected_count)
    fr_a = summarize_instance_counts(df_a, expected_count)
    fr_b = summarize_instance_counts(df_b, expected_count)

    # Align the frame range to the shared overlap.
    frame_min = max(fr_top["frame"].min(), fr_a["frame"].min(), fr_b["frame"].min())
    frame_max = min(fr_top["frame"].max(), fr_a["frame"].max(), fr_b["frame"].max())

    def _clip_to_range(fr: pd.DataFrame) -> pd.DataFrame:
        return fr[(fr["frame"] >= frame_min) & (fr["frame"] <= frame_max)]

    fr_top = _clip_to_range(fr_top).set_index("frame")
    fr_a = _clip_to_range(fr_a).set_index("frame")
    fr_b = _clip_to_range(fr_b).set_index("frame")

    # Merge into one frame-wise table.
    merged = pd.DataFrame(index=range(frame_min, frame_max + 1))
    merged["top_count"] = fr_top["instance_count"]
    merged["side1_count"] = fr_a["instance_count"]
    merged["side2_count"] = fr_b["instance_count"]

    # Per-view missing flags.
    merged["top_is_missing"] = merged["top_count"] < expected_count
    merged["side1_is_missing"] = merged["side1_count"] < expected_count
    merged["side2_is_missing"] = merged["side2_count"] < expected_count

    # Frame-level "true missing": all three views are missing.
    merged["true_missing"] = (
        merged["top_is_missing"]
        & merged["side1_is_missing"]
        & merged["side2_is_missing"]
    )

    # Per-view occlusion: one view is missing while at least one other is complete.
    def view_occlusion(this_missing, other1_count, other2_count):
        other_ok = (other1_count >= expected_count) | (other2_count >= expected_count)
        return this_missing & other_ok

    merged["top_occlusion"] = view_occlusion(
        merged["top_is_missing"], merged["side1_count"], merged["side2_count"]
    )
    merged["side1_occlusion"] = view_occlusion(
        merged["side1_is_missing"], merged["top_count"], merged["side2_count"]
    )
    merged["side2_occlusion"] = view_occlusion(
        merged["side2_is_missing"], merged["top_count"], merged["side1_count"]
    )

    # Restore the index as a regular column for downstream analysis.
    merged = merged.reset_index().rename(columns={"index": "frame"})
    return merged


def generate_multiview_visualizations(
    df: pd.DataFrame,
    expected_count: int,
    output_dir: Path,
    fps: float | None = None,
    events: pd.DataFrame | None = None,
    events_margin_s: float = 0.0,
    output_filename: str | None = None,
    time_offset_s: float = 0.0,
    title: str | None = None,
) -> None:
    """
    Generate the multiview QC overview image.

    The figure contains:
    - a per-view missing / occlusion heatmap
    - optional event overlays from BORIS annotations
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared time axis for all views.
    frames = df["frame"].to_numpy()
    true_missing = df["true_missing"].to_numpy()

    # Category layout:
    # - first three rows: per-view status
    # - fourth row: overall missing status
    views = ["top", "side1", "side2", "all_missing"]

    # Build a `(n_views, n_frames)` category matrix.
    n_frames = len(df)
    n_views = len(views)
    category = np.zeros((n_views, n_frames), dtype=int)

    for v_idx, view in enumerate(views):
        if view == "all_missing":
            top_count = df["top_count"].to_numpy()
            side1_count = df["side1_count"].to_numpy()
            side2_count = df["side2_count"].to_numpy()
            all_fully_missing = (top_count == 0) & (side1_count == 0) & (side2_count == 0)
            all_have_data = (top_count >= expected_count) & (side1_count >= expected_count) & (side2_count >= expected_count)
            has_missing = (top_count < expected_count) | (side1_count < expected_count) | (side2_count < expected_count)
            partial_missing = has_missing & (~all_fully_missing) & (~all_have_data)

            cat = np.zeros(n_frames, dtype=int)
            cat[all_fully_missing] = 1
            cat[all_have_data] = 4
            cat[partial_missing] = 3
        else:
            count_col = f"{view}_count"
            counts = df[count_col].to_numpy()
            has_data = counts >= expected_count
            missing_count = expected_count - counts
            missing_count = np.clip(missing_count, 0, expected_count)

            cat = np.zeros(n_frames, dtype=int)
            cat[has_data] = 2

        category[v_idx, :] = cat

    # Color mapping.
    cmap = {
        0: "#FFFFFF",
        1: "#F39C12",
        2: "#1F77B4",
        3: "#87CEEB",
        4: "#2E7D32",
    }

    # Gray levels for partial missing counts in the first three rows.
    missing_gray_map = {
        1: "#E8E8E8",
        2: "#B0B0B0",
        3: "#707070",
    }

    color_array = np.ones((n_views, n_frames, 3), dtype=float)

    # Fill the first three rows with either "has data" blue or gray missing levels.
    for v_idx, view in enumerate(views):
        if view != "all_missing":
            count_col = f"{view}_count"
            counts = df[count_col].to_numpy()
            missing_count = np.clip(expected_count - counts, 0, expected_count)
            has_data = counts >= expected_count

            for c in range(3):
                color_array[v_idx, :, c][has_data] = int(cmap[2][1 + c * 2:3 + c * 2], 16) / 255.0
                for miss_level in [1, 2, 3]:
                    if miss_level == 3:
                        mask_miss = (missing_count >= 3) & (~has_data)
                    else:
                        mask_miss = (missing_count == miss_level) & (~has_data)
                    if mask_miss.any():
                        gray_hex = missing_gray_map[miss_level]
                        color_array[v_idx, :, c][mask_miss] = int(gray_hex[1 + c * 2:3 + c * 2], 16) / 255.0

    # Fill the fourth row for overall missing state.
    all_missing_idx = views.index("all_missing")
    for code, hex_color in cmap.items():
        if code == 0:
            continue
        rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        mask = category[all_missing_idx, :] == code
        for c in range(3):
            color_array[all_missing_idx, :, c][mask] = rgb[c]

    # Layout: state heatmap plus optional event rows.
    n_event_labels = 0
    event_labels: list[str] = []

    # Keep only event labels that overlap the current clip time range.
    if (
        events is not None
        and len(events) > 0
        and fps is not None
        and fps > 0
    ):
        frame0_tmp = float(frames.min())
        times_tmp = (frames - frame0_tmp) / float(fps) + time_offset_s
        t_min = float(times_tmp.min())
        t_max = float(times_tmp.max())

        labels_all = sorted(str(x) for x in events.get("label", ["event"]).unique())
        kept_labels: list[str] = []
        for lb in labels_all:
            g = events[events["label"].astype(str) == lb]
            has_overlap = False
            for _, row in g.iterrows():
                start_true = float(row["start_s"])
                end_true = float(row["end_s"])
                margin = float(events_margin_s)
                start_ext = start_true - margin
                end_ext = end_true + margin
                if (end_ext >= t_min) and (start_ext <= t_max):
                    has_overlap = True
                    break
            if has_overlap:
                kept_labels.append(lb)

        event_labels = kept_labels
        n_event_labels = len(event_labels)

    if n_event_labels > 0:
        fig, (ax_state, ax_evt) = plt.subplots(
            2,
            1,
            figsize=(18, 5.0),
            sharex=False,
            gridspec_kw={"height_ratios": [4, max(1, n_event_labels)]},
        )
    else:
        fig, ax_state = plt.subplots(1, 1, figsize=(18, 3.5))
        ax_evt = None

    im = ax_state.imshow(color_array, aspect="auto", interpolation="nearest")
    ax_state.set_yticks(list(range(n_views)))
    ax_state.set_yticklabels(views, fontsize=10)

    # Use the original frame numbers on the x-axis.
    idx_ticks = np.linspace(0, n_frames - 1, num=min(6, n_frames), dtype=int)
    ax_state.set_xticks(idx_ticks)
    frame_labels = [f"{int(frames[i])}" for i in idx_ticks]
    ax_state.set_xticklabels(frame_labels, fontsize=9)
    ax_state.set_xlabel("Frame", fontsize=12, fontweight="bold")
    ax_state.tick_params(axis='x', labelsize=9, labelrotation=0, bottom=True, labelbottom=True)
    plt.setp(ax_state.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Set title.
    if title is not None:
        ax_state.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax_state.set_title(
            "Per-view Missing / Occlusion Classification", fontsize=14, fontweight="bold"
        )

    # Add event overlays below the state heatmap if needed.
    if (
        ax_evt is not None
        and fps is not None
        and fps > 0
        and n_event_labels > 0
    ):
        frame0 = float(frames.min())
        times = (frames - frame0) / float(fps) + time_offset_s
        margin = float(events_margin_s)
        
        clip_start_time = float(times.min())
        clip_end_time = float(times.max())

        if margin > 0:
            print(f"  Event margin: {margin:.2f} seconds")
            print(f"  Clip time range: {clip_start_time:.2f} - {clip_end_time:.2f} seconds")

        n_evt_rows = len(event_labels)
        for row_idx, label in enumerate(event_labels):
            y0 = row_idx / max(1, n_evt_rows)
            y1 = (row_idx + 1) / max(1, n_evt_rows)

            if label == "__ALL_MISSING__":
                mask_all = df["true_missing"].to_numpy()
                if not mask_all.any():
                    continue
                idx_all = np.where(mask_all)[0]
                splits = np.split(
                    idx_all,
                    np.where(np.diff(idx_all) != 1)[0] + 1,
                )
                for seg in splits:
                    if len(seg) == 0:
                        continue
                    start_idx = seg[0]
                    end_idx = seg[-1]
                    start_true = times[start_idx]
                    end_true = times[end_idx]
                    start_ext = start_true - margin
                    end_ext = end_true + margin

                    start_true_clipped = max(start_true, clip_start_time)
                    end_true_clipped = min(end_true, clip_end_time)
                    start_ext_clipped = max(start_ext, clip_start_time)
                    end_ext_clipped = min(end_ext, clip_end_time)

                    mask_ext = (times >= start_ext_clipped) & (times <= end_ext_clipped)
                    mask_true = (times >= start_true_clipped) & (times <= end_true_clipped)

                    if mask_ext.any():
                        idx_ext = np.where(mask_ext)[0]
                        x0_ext = idx_ext[0] - 0.5
                        x1_ext = idx_ext[-1] + 0.5
                        ax_evt.axvspan(
                            x0_ext,
                            x1_ext,
                            color="#B0BEC5",
                            alpha=0.4,
                            ymin=y0,
                            ymax=y1,
                        )
                    if mask_true.any():
                        idx_true = np.where(mask_true)[0]
                        x0_true = idx_true[0] - 0.5
                        x1_true = idx_true[-1] + 0.5
                        ax_evt.axvspan(
                            x0_true,
                            x1_true,
                            color="#455A64",
                            alpha=0.8,
                            ymin=y0,
                            ymax=y1,
                        )
            else:
                if events is None or len(events) == 0:
                    continue
                g = events[events["label"].astype(str) == label]
                if g.empty:
                    continue

                # Merge overlapping intervals for cleaner display.
                event_intervals = []
                for _, row in g.iterrows():
                    start_true = float(row["start_s"])
                    end_true = float(row["end_s"])
                    start_ext = start_true - margin
                    end_ext = end_true + margin

                    start_true_clipped = max(start_true, clip_start_time)
                    end_true_clipped = min(end_true, clip_end_time)
                    start_ext_clipped = max(start_ext, clip_start_time)
                    end_ext_clipped = min(end_ext, clip_end_time)

                    if start_true_clipped <= end_true_clipped:
                        event_intervals.append({
                            'core': (start_true_clipped, end_true_clipped),
                            'margin': (start_ext_clipped, end_ext_clipped)
                        })

                if event_intervals:
                    core_intervals = sorted([iv['core'] for iv in event_intervals])
                    merged_cores = []
                    for start, end in core_intervals:
                        if merged_cores and start <= merged_cores[-1][1]:
                            merged_cores[-1] = (merged_cores[-1][0], max(merged_cores[-1][1], end))
                        else:
                            merged_cores.append((start, end))

                    margin_intervals = sorted([iv['margin'] for iv in event_intervals])
                    merged_margins = []
                    for start, end in margin_intervals:
                        if merged_margins and start <= merged_margins[-1][1]:
                            merged_margins[-1] = (merged_margins[-1][0], max(merged_margins[-1][1], end))
                        else:
                            merged_margins.append((start, end))

                    for start_ext, end_ext in merged_margins:
                        mask_ext = (times >= start_ext) & (times <= end_ext)
                        if mask_ext.any():
                            idx_ext = np.where(mask_ext)[0]
                            x0_ext = idx_ext[0] - 0.5
                            x1_ext = idx_ext[-1] + 0.5
                            ax_evt.axvspan(
                                x0_ext,
                                x1_ext,
                                color="#F5B7A6",
                                alpha=0.5,
                                ymin=y0,
                                ymax=y1,
                                edgecolor='none',
                                linewidth=0,
                            )

                    for start_true, end_true in merged_cores:
                        mask_true = (times >= start_true) & (times <= end_true)
                        if mask_true.any():
                            idx_true = np.where(mask_true)[0]
                            x0_true = idx_true[0] - 0.5
                            x1_true = idx_true[-1] + 0.5
                            ax_evt.axvspan(
                                x0_true,
                                x1_true,
                                color="#C73E1D",
                                alpha=0.5,
                                ymin=y0,
                                ymax=y1,
                                edgecolor='none',
                                linewidth=0,
                            )

        # Use global time in seconds on the event x-axis.
        if fps is not None and fps > 0:
            frame0_evt = float(frames.min())
            times_evt = (frames - frame0_evt) / float(fps) + time_offset_s
            idx_ticks_evt = np.linspace(0, n_frames - 1, num=min(6, n_frames), dtype=int)
            ax_evt.set_xticks(idx_ticks_evt)
            ax_evt.set_xticklabels([f"{times_evt[i]:.1f}" for i in idx_ticks_evt], fontsize=9)
            ax_evt.set_xlabel("Time (s, global)", fontsize=12, fontweight="bold")
            ax_evt.set_xlim(-0.5, n_frames - 0.5)
        else:
            ax_evt.set_xlabel("Frame index (global)", fontsize=12, fontweight="bold")
            ax_evt.set_xlim(-0.5, n_frames - 0.5)

        yticks = [(i + 0.5) / max(1, n_evt_rows) for i in range(n_evt_rows)]
        ylabels = [lb for lb in event_labels]
        ax_evt.set_yticks(yticks)
        ax_evt.set_yticklabels(ylabels, fontsize=9)
        ax_evt.set_ylabel("Events", fontsize=11, fontweight="bold")
        ax_evt.set_ylim(0, 1)

    # Build the legend.
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=cmap[2], label="Has all individuals (top/side1/side2 rows: this view has all individuals)"),
        Patch(color=missing_gray_map[1], label="Missing 1 (top/side1/side2 rows: 1 individual missing)"),
        Patch(color=missing_gray_map[2], label="Missing 2 (top/side1/side2 rows: 2 individuals missing)"),
        Patch(color=missing_gray_map[3], label="Missing 3+ (top/side1/side2 rows: 3+ individuals missing)"),
        Patch(color=cmap[1], label="All fully missing (4th row: all 3 views count=0)"),
        Patch(color=cmap[3], label="Partial missing (4th row: has missing but not all count=0)"),
        Patch(color=cmap[4], label="All views have all individuals (4th row: all 3 views count>=expected)"),
    ]
    if events is not None and fps is not None and fps > 0 and len(events) > 0:
        legend_patches.extend(
            [
                Patch(color="#C73E1D", alpha=0.5, label="Event (core)"),
                Patch(color="#F5B7A6", alpha=0.5, label="Event margin"),
            ]
        )

    ax_state.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        fontsize=8,
    )

    # Tighten the layout.
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    if output_filename is None:
        output_filename = "multiview_missing_occlusion_overview.png"
    fig.savefig(
        output_dir / output_filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _compute_in_event_mask(
    df: pd.DataFrame,
    events: pd.DataFrame | None,
    fps: float | None,
    time_offset_s: float = 0.0,
) -> np.ndarray:
    """Return a boolean mask indicating whether each frame is inside a core event interval."""
    frames = df["frame"].to_numpy()
    in_event = np.zeros_like(frames, dtype=bool)

    if events is None or len(events) == 0:
        return in_event

    # With fps, assume events are in seconds; otherwise treat them as frames.
    if fps is not None and fps > 0:
        frame0 = float(frames.min())
        times = (frames - frame0) / float(fps) + time_offset_s
        for _, row in events.iterrows():
            start_true = float(row["start_s"])
            end_true = float(row["end_s"])
            in_event |= (times >= start_true) & (times <= end_true)
    else:
        for _, row in events.iterrows():
            start_true = float(row["start_s"])
            end_true = float(row["end_s"])
            in_event |= (frames >= start_true) & (frames <= end_true)

    return in_event


def annotate_events_on_frames(
    df: pd.DataFrame,
    events: pd.DataFrame | None,
    fps: float | None,
    time_offset_s: float = 0.0,
) -> pd.DataFrame:
    """Add `events` and `event_name` columns to the per-frame table."""
    df = df.copy()
    frames = df["frame"].to_numpy()

    df["events"] = False
    df["event_name"] = ""

    if events is None or len(events) == 0:
        return df

    in_event = _compute_in_event_mask(df, events, fps, time_offset_s)
    df["events"] = in_event

    names = np.array([""] * len(df), dtype=object)

    label_to_mask: dict[str, np.ndarray] = {}

    if fps is not None and fps > 0:
        frame0 = float(frames.min())
        coord = (frames - frame0) / float(fps) + time_offset_s
    else:
        coord = frames

    for _, row in events.iterrows():
        start_true = float(row["start_s"])
        end_true = float(row["end_s"])
        raw_label = str(row.get("label", "event"))
        label = raw_label
        col_key = _normalize_event_label(raw_label)

        mask = (coord >= start_true) & (coord <= end_true)
        idxs = np.where(mask)[0]
        for idx in idxs:
            if names[idx]:
                names[idx] = f"{names[idx]};{label}"
            else:
                names[idx] = label

        if col_key not in label_to_mask:
            label_to_mask[col_key] = np.zeros(len(df), dtype=bool)
        label_to_mask[col_key] |= mask

    df["event_name"] = names

    for col_key, mask in label_to_mask.items():
        col_name = f"event_{col_key}"
        df[col_name] = mask

    return df


def generate_multiview_summary_table(
    df: pd.DataFrame,
    events: pd.DataFrame | None,
    fps: float | None,
    output_dir: Path,
    output_filename: str | None = None,
    time_offset_s: float = 0.0,
) -> None:
    """Write a summary CSV describing overall and event-window missingness."""
    rows: list[dict] = []

    total_frames = len(df)
    rows.append({"Metric": "total_frames", "Value": total_frames})

    total_true_missing = int(df["true_missing"].sum())
    rows.append({"Metric": "total_true_missing_frames", "Value": total_true_missing})

    for view in ["top", "side1", "side2"]:
        miss = int(df[f"{view}_is_missing"].sum())
        rows.append({"Metric": f"{view}_missing_frames", "Value": miss})

    if events is not None and len(events) > 0:
        in_event = _compute_in_event_mask(df, events, fps, time_offset_s)
        rows.append({"Metric": "event_frames", "Value": int(in_event.sum())})

        event_all_missing = int((df["true_missing"].to_numpy() & in_event).sum())
        rows.append({"Metric": "event_all_views_missing_frames", "Value": event_all_missing})

        miss_counts = (
            df[["top_is_missing", "side1_is_missing", "side2_is_missing"]]
            .to_numpy(dtype=int)
            .sum(axis=1)
        )
        event_single_view_missing = int(((miss_counts == 1) & in_event).sum())
        rows.append(
            {"Metric": "event_single_view_missing_frames", "Value": event_single_view_missing}
        )

    summary_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        output_filename = "multiview_summary.csv"
    summary_df.to_csv(output_dir / output_filename, index=False)


def process_single_trial(
    slp_top: Path,
    slp_side1: Path,
    slp_side2: Path,
    expected_count: int,
    output_dir: Path | None,
    events_df: pd.DataFrame | None,
    fps: float | None,
    events_margin_s: float,
    trial_prefix: str | None = None,
) -> None:
    """Run QC analysis for a single trial."""
    print(f"\n{'='*60}")
    print(f"Processing: {slp_top.name}")
    print(f"{'='*60}")

    if not slp_top.exists():
        raise SystemExit(f"Input file not found: {slp_top}")
    if not slp_side1.exists():
        raise SystemExit(f"Input file not found: {slp_side1}")
    if not slp_side2.exists():
        raise SystemExit(f"Input file not found: {slp_side2}")

    if output_dir is None:
        output_dir = slp_top.parent / f"{slp_top.stem}_multiview_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_offset_s = extract_clip_start_time(slp_top) or 0.0
    if time_offset_s > 0:
        print(f"Detected clip start time from filename: {time_offset_s:.3f} s")

    print("Classifying multi-view missing vs occlusion...")
    result = classify_multiview_missing(
        slp_top=slp_top,
        slp_side1=slp_side1,
        slp_side2=slp_side2,
        expected_count=expected_count,
    )

    # Convert frame indices into absolute time relative to the original long video.
    if fps is not None and fps > 0:
        frame0 = float(result["frame"].min())
        result["time_s"] = (result["frame"].to_numpy() - frame0) / float(fps) + time_offset_s
    else:
        result["time_s"] = np.nan

    if events_df is not None:
        result = annotate_events_on_frames(
            df=result,
            events=events_df,
            fps=fps,
            time_offset_s=time_offset_s,
        )
    else:
        result["events"] = False
        result["event_name"] = ""

    if trial_prefix:
        out_csv = output_dir / f"{trial_prefix}_multiview_missing_occlusion.csv"
    else:
        out_csv = output_dir / "multiview_missing_occlusion.csv"
    result.to_csv(out_csv, index=False)

    n_frames = len(result)
    n_true_missing = int(result["true_missing"].sum())
    print(f"Total frames (after alignment): {n_frames}")
    print(f"Frames with true_missing (all views missing): {n_true_missing}")

    for view in ["top", "side1", "side2"]:
        miss = int(result[f"{view}_is_missing"].sum())
        occ = int(result[f"{view}_occlusion"].sum())
        print(
            f"{view}: missing frames={miss}, occlusion-like frames={occ} "
            f"(missing but at least one other view OK)"
        )

    title_base = slp_top.stem
    if title_base.endswith("_top"):
        title_base = title_base[:-4]
    title_base = re.sub(r'_PLAYER[123]_', '_', title_base)
    title_base = re.sub(r'_[a-z0-9]+_(?=[A-Z])', '_', title_base)
    title = title_base

    if trial_prefix:
        vis_filename = f"{trial_prefix}_multiview_missing_occlusion_overview.png"
    else:
        vis_filename = None
    
    generate_multiview_visualizations(
        df=result,
        expected_count=expected_count,
        output_dir=output_dir,
        fps=fps,
        events=events_df,
        events_margin_s=events_margin_s,
        output_filename=vis_filename,
        time_offset_s=time_offset_s,
        title=title,
    )

    if trial_prefix:
        summary_filename = f"{trial_prefix}_multiview_summary.csv"
    else:
        summary_filename = None
    
    generate_multiview_summary_table(
        df=result,
        events=events_df,
        fps=fps,
        output_dir=output_dir,
        output_filename=summary_filename,
        time_offset_s=time_offset_s,
    )

    print(f"\n✓ Multi-view QC complete! Results saved to: {out_csv}")
    if summary_filename:
        print(f"  - Summary table: {output_dir / summary_filename}")
    else:
        print(f"  - Summary table: {output_dir / 'multiview_summary.csv'}")
    if vis_filename:
        print(f"  - Missing / occlusion overview: {output_dir / vis_filename}")
    else:
        print(
            f"  - Missing / occlusion overview: "
            f"{output_dir / 'multiview_missing_occlusion_overview.png'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-view SLEAP tracking QC (missing vs occlusion).\n"
            "Two modes are supported:\n"
            "1. batch mode: use --top-slp multiple times and auto-find side1/side2 files\n"
            "2. single-file mode: use --top/--side1/--side2 for one trial"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-slp",
        action="append",
        default=None,
        help="Top-view .slp path. Repeat this flag to process multiple files; side1/side2 files are inferred automatically.",
    )
    parser.add_argument(
        "--top",
        default=None,
        help="Top-view .slp path for single-file mode",
    )
    parser.add_argument(
        "--side1",
        default=None,
        help="Side1 .slp path for single-file mode",
    )
    parser.add_argument(
        "--side2",
        default=None,
        help="Side2 .slp path for single-file mode",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=3,
        help="Expected number of individuals per frame (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <stem>_multiview_qc next to the top .slp file)",
    )
    parser.add_argument(
        "--events-csv",
        default=None,
        help=(
            "Optional BORIS event CSV. If provided, event intervals and margins "
            "are highlighted in the visualization."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video frame rate in Hz. Needed for frame-to-time conversion and event alignment.",
    )
    parser.add_argument(
        "--events-time-col",
        default=None,
        help="Optional point-event time column name, for example 'Time (s)' or 'Image index'.",
    )
    parser.add_argument(
        "--events-start-col",
        default=None,
        help="Optional interval-event start column name, for example 'Start (s)'.",
    )
    parser.add_argument(
        "--events-end-col",
        default=None,
        help="Optional interval-event end column name, for example 'Stop (s)'.",
    )
    parser.add_argument(
        "--events-label-col",
        default=None,
        help="Optional behavior-label column name, for example 'Behavior'.",
    )
    parser.add_argument(
        "--events-margin-s",
        type=float,
        default=0.0,
        help=(
            "Optional event margin in seconds. For example, 1.0 expands each "
            "event interval to [start_s-1, end_s+1]."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Short alias for --events-margin-s",
    )
    parser.add_argument(
        "--skip-missing-side",
        action="store_true",
        help="In batch mode, skip clips whose side files are missing",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "In batch mode, write all outputs into one directory using trial-specific "
            "filenames instead of creating one subdirectory per trial."
        ),
    )
    args = parser.parse_args()

    use_batch_mode = args.top_slp is not None and len(args.top_slp) > 0
    use_single_mode = args.top is not None

    if not use_batch_mode and not use_single_mode:
        raise SystemExit(
            "Choose a processing mode:\n"
            "  - batch mode: use --top-slp one or more times\n"
            "  - single-file mode: use --top/--side1/--side2"
        )

    if use_batch_mode and use_single_mode:
        raise SystemExit(
            "Do not mix batch mode (--top-slp) and single-file mode (--top/--side1/--side2). "
            "Choose one mode only."
        )

    # Load behavioral events if provided.
    events_df: pd.DataFrame | None = None
    if args.events_csv:
        events_path = Path(args.events_csv)
        if not events_path.exists():
            raise SystemExit(f"Events CSV not found: {events_path}")
        if args.fps is None or args.fps <= 0:
            raise SystemExit("When using --events-csv, you must also provide a positive --fps value.")

        print(f"Loading behavioural events from: {events_path}")
        events_df = load_events_csv(
            events_path,
            time_col=args.events_time_col,
            start_col=args.events_start_col,
            end_col=args.events_end_col,
            label_col=args.events_label_col,
        )
        print(f"Loaded {len(events_df)} events.")

    fps = float(args.fps) if args.fps is not None and args.fps > 0 else None
    
    # Support the short alias --margin.
    if args.margin is not None:
        events_margin_s = float(args.margin)
    else:
        events_margin_s = float(args.events_margin_s or 0.0)

    # Batch mode
    if use_batch_mode:
        top_slp_files = [Path(t).resolve() for t in args.top_slp]

        print("\nStarting batch QC...")
        print(f"  Number of top .slp files: {len(top_slp_files)}")
        if args.flat_output:
            if args.output_dir:
                print("  Output mode: flat")
                print(f"  Output directory: {args.output_dir}")
            else:
                print("  Output mode: flat")
                print(f"  Output directory: {top_slp_files[0].parent / 'multiview_qc_batch'}")
        else:
            print("  Output mode: one subdirectory per trial")
            if args.output_dir:
                print(f"  Base output directory: {args.output_dir}")
            else:
                print("  Output directory: next to each trial's .slp file")
        print(f"  Expected individual count: {args.expected_count}")
        if events_df is not None:
            print(f"  Behavioral events: {len(events_df)}")
        if fps is not None:
            print(f"  Video frame rate: {fps} Hz")

        for i, top_slp in enumerate(top_slp_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing top .slp {i}/{len(top_slp_files)}: {top_slp.name}")
            print(f"{'='*60}")

            if not top_slp.exists():
                raise SystemExit(f"Input file not found: {top_slp}")

            # Auto-find the matching side files.
            side1_slp, side2_slp = find_side_slp_files(top_slp)

            if side1_slp is None or side2_slp is None:
                error_msg = (
                    f"Matching side file(s) not found:\n"
                    f"  Top: {top_slp}\n"
                )
                if side1_slp is None:
                    error_msg += f"  Side1: missing (expected: {str(top_slp).replace('PLAYER1', 'PLAYER2').replace('_top', '_side1')})\n"
                if side2_slp is None:
                    error_msg += f"  Side2: missing (expected: {str(top_slp).replace('PLAYER1', 'PLAYER3').replace('_top', '_side2')})\n"
                
                if args.skip_missing_side:
                    print("Skipping clip because side file(s) are missing")
                    continue
                else:
                    raise SystemExit(error_msg)

            # Resolve output directory and optional filename prefix.
            if args.flat_output:
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                else:
                    output_dir = top_slp_files[0].parent / "multiview_qc_batch"
                output_dir.mkdir(parents=True, exist_ok=True)
                trial_prefix = top_slp.stem.replace("_top", "")
            else:
                if args.output_dir:
                    output_dir = Path(args.output_dir) / top_slp.stem
                else:
                    output_dir = top_slp.parent / f"{top_slp.stem}_multiview_qc"
                trial_prefix = None

            process_single_trial(
                slp_top=top_slp,
                slp_side1=side1_slp,
                slp_side2=side2_slp,
                expected_count=args.expected_count,
                output_dir=output_dir,
                events_df=events_df,
                fps=fps,
                events_margin_s=events_margin_s,
                trial_prefix=trial_prefix,
            )

        print(f"\n{'='*60}")
        print("✓ All files processed!")
        if args.flat_output:
            if args.output_dir:
                print(f"  Output directory: {args.output_dir} (flat mode)")
            else:
                print(f"  Output directory: {top_slp_files[0].parent / 'multiview_qc_batch'} (flat mode)")
        else:
            if args.output_dir:
                print(f"  Base output directory: {args.output_dir} (one subdirectory per trial)")
            else:
                print("  Output directory: next to each trial's .slp file")
        print(f"{'='*60}")

    # Single-file mode
    else:
        slp_top = Path(args.top).resolve()
        slp_a = Path(args.side1).resolve() if args.side1 else None
        slp_b = Path(args.side2).resolve() if args.side2 else None

        if slp_a is None or slp_b is None:
            raise SystemExit("Single-file mode requires --top, --side1, and --side2.")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = slp_top.parent / f"{slp_top.stem}_multiview_qc"

        process_single_trial(
            slp_top=slp_top,
            slp_side1=slp_a,
            slp_side2=slp_b,
            expected_count=args.expected_count,
            output_dir=output_dir,
            events_df=events_df,
            fps=fps,
            events_margin_s=float(args.events_margin_s or 0.0),
        )


if __name__ == "__main__":
    main()
