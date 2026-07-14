from __future__ import annotations

"""Render the exact geometry-contrastive positive supervision on source video.

``Pxx`` labels are connected components of the explicit positive-pair graph and
therefore represent the precise transitive "same" claims available to training.
``G0/G1/G2`` labels are frame-local geometry slots used for synchronized
cross-camera pairing; they are deliberately not presented as temporal identities.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


VIEW_ORDER = ("top_cam", "side1_cam", "side2_cam")
INK = "#25313B"
MUTED = "#65727D"
GREY = "#A7B0B7"
EVENT = "#D66A3A"
WHITE = "#FFFFFF"
PALETTE = (
    "#2F6B8A",
    "#D9A441",
    "#A95783",
    "#4F8065",
    "#D66A3A",
    "#665C9E",
    "#6F8F9D",
    "#9A7442",
    "#8D6A78",
    "#527A73",
    "#A45B42",
    "#6D6D6D",
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def _find_table(root: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Expected {stem}.parquet or {stem}.csv under {root}")


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    text = color.lstrip("#")
    red, green, blue = (int(text[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _mode_int(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if numeric.empty:
        return 999
    modes = numeric.mode()
    return int(modes.iloc[0]) if not modes.empty else int(numeric.iloc[0])


def _positive_components(
    samples: pd.DataFrame,
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    min_temporal_delta: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach deterministic P labels and return component/event audit tables."""
    required_samples = {
        "sample_id",
        "frame",
        "view",
        "window",
        "split",
        "local_fragment_id",
        "geometry_source_idx_local",
    }
    missing = required_samples - set(samples)
    if missing:
        raise ValueError(f"samples missing columns: {sorted(missing)}")
    for name, table in (("positive_pairs", positives), ("negative_pairs", negatives)):
        pair_missing = {"sample_a_id", "sample_b_id"} - set(table)
        if pair_missing:
            raise ValueError(f"{name} missing columns: {sorted(pair_missing)}")

    work = samples.copy()
    work["local_fragment_id"] = work.local_fragment_id.astype(str)
    by_sample = work.set_index("sample_id")
    fragment_for = by_sample.local_fragment_id.to_dict()
    window_for = by_sample.window.astype(str).to_dict()
    split_for = by_sample.split.astype(str).to_dict()

    pos = positives.copy()
    if "pair_type" in pos and "pair_frame_delta" in pos:
        temporal = pos.pair_type.astype(str).eq("same_local_fragment")
        delta = pd.to_numeric(pos.pair_frame_delta, errors="coerce")
        pos = pos[~temporal | delta.ge(min_temporal_delta)].copy()
    pos["fragment_a"] = pos.sample_a_id.map(fragment_for)
    pos["fragment_b"] = pos.sample_b_id.map(fragment_for)
    pos["window"] = pos.sample_a_id.map(window_for)
    pos["split"] = pos.sample_a_id.map(split_for)
    if pos[["fragment_a", "fragment_b", "window", "split"]].isna().any().any():
        raise ValueError("Positive pairs reference unknown samples")

    neg = negatives.copy()
    neg["fragment_a"] = neg.sample_a_id.map(fragment_for)
    neg["fragment_b"] = neg.sample_b_id.map(fragment_for)
    neg["window"] = neg.sample_a_id.map(window_for)
    neg["split"] = neg.sample_a_id.map(split_for)
    if neg[["fragment_a", "fragment_b", "window", "split"]].isna().any().any():
        raise ValueError("Negative pairs reference unknown samples")

    label_for_fragment: dict[str, str] = {}
    component_records: list[dict[str, Any]] = []
    component_index = 0
    contradictions_total = 0
    window_records: list[dict[str, Any]] = []

    window_groups = list(work.groupby(["split", "window"], sort=False))
    window_groups.sort(key=lambda item: int(item[1].frame.min()))
    for (split, window), group in window_groups:
        fragments = sorted(group.local_fragment_id.unique().tolist())
        parent = {fragment: fragment for fragment in fragments}

        def find(fragment: str) -> str:
            root = fragment
            while parent[root] != root:
                root = parent[root]
            while parent[fragment] != fragment:
                next_fragment = parent[fragment]
                parent[fragment] = root
                fragment = next_fragment
            return root

        def union(left: str, right: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                if root_left < root_right:
                    parent[root_right] = root_left
                else:
                    parent[root_left] = root_right

        scope_pos = pos[pos.split.eq(str(split)) & pos.window.eq(str(window))]
        for left, right in scope_pos[["fragment_a", "fragment_b"]].itertuples(
            index=False, name=None
        ):
            union(str(left), str(right))

        grouped_fragments: dict[str, set[str]] = defaultdict(set)
        for fragment in fragments:
            grouped_fragments[find(fragment)].add(fragment)

        sortable: list[tuple[tuple[int, int, int, str], set[str], pd.DataFrame]] = []
        for fragment_set in grouped_fragments.values():
            rows = group[group.local_fragment_id.isin(fragment_set)]
            start = int(rows.frame.min())
            first = rows[rows.frame.eq(start)]
            first_slot = _mode_int(first.geometry_source_idx_local)
            key = (start, first_slot, -len(rows), min(fragment_set))
            sortable.append((key, fragment_set, rows))
        sortable.sort(key=lambda item: item[0])

        local_component_for_fragment: dict[str, str] = {}
        for _key, fragment_set, rows in sortable:
            component_index += 1
            p_label = f"P{component_index:02d}"
            color = PALETTE[(component_index - 1) % len(PALETTE)]
            for fragment in fragment_set:
                label_for_fragment[fragment] = p_label
                local_component_for_fragment[fragment] = p_label
            component_records.append(
                {
                    "p_label": p_label,
                    "color_hex": color,
                    "split": str(split),
                    "window": str(window),
                    "start_frame": int(rows.frame.min()),
                    "end_frame": int(rows.frame.max()),
                    "n_samples": int(len(rows)),
                    "n_local_fragments": int(len(fragment_set)),
                    "views": "|".join(sorted(rows.view.astype(str).unique())),
                    "geometry_slots_seen": "|".join(
                        map(
                            str,
                            sorted(
                                pd.to_numeric(
                                    rows.geometry_source_idx_local, errors="coerce"
                                )
                                .dropna()
                                .astype(int)
                                .unique()
                            ),
                        )
                    ),
                    "first_geometry_slot": _mode_int(
                        rows[rows.frame.eq(rows.frame.min())].geometry_source_idx_local
                    ),
                }
            )

        scope_neg = neg[neg.split.eq(str(split)) & neg.window.eq(str(window))]
        contradictions = 0
        for left, right in scope_neg[["fragment_a", "fragment_b"]].itertuples(
            index=False, name=None
        ):
            if local_component_for_fragment[str(left)] == local_component_for_fragment[str(right)]:
                contradictions += 1
        contradictions_total += contradictions
        window_records.append(
            {
                "split": str(split),
                "window": str(window),
                "start_frame": int(group.frame.min()),
                "end_frame": int(group.frame.max()),
                "n_samples": int(len(group)),
                "n_local_fragments": int(len(fragments)),
                "n_positive_components": int(len(sortable)),
                "n_negative_pairs": int(len(scope_neg)),
                "transitive_positive_negative_contradictions": int(contradictions),
            }
        )

    work["p_label"] = work.local_fragment_id.map(label_for_fragment)
    if work.p_label.isna().any():
        raise RuntimeError("Some samples did not receive a positive-component label")

    components = pd.DataFrame(component_records)
    events: list[dict[str, Any]] = []
    for row in components.itertuples(index=False):
        events.append(
            {
                "frame": int(row.start_frame),
                "event": "component_start",
                "p_label": str(row.p_label),
                "window": str(row.window),
            }
        )
        events.append(
            {
                "frame": int(row.end_frame),
                "event": "component_end",
                "p_label": str(row.p_label),
                "window": str(row.window),
            }
        )
    event_table = pd.DataFrame(events).sort_values(["frame", "event", "p_label"])
    audit = {
        "schema_version": "geometry_supervision_components_v1",
        "component_semantics": (
            "P labels are connected components of retained positive-pair supervision; "
            "they are not independently verified biological identities."
        ),
        "geometry_semantics": (
            "G labels are frame-local geometry slots used only for synchronized cross-camera grouping."
        ),
        "min_temporal_pair_delta": int(min_temporal_delta),
        "n_components": int(len(components)),
        "transitive_positive_negative_contradictions": int(contradictions_total),
        "windows": window_records,
    }
    return work, components, event_table, audit


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
    shadow: bool = False,
) -> None:
    if shadow:
        cv2.putText(
            image,
            text,
            (origin[0] + 1, origin[1] + 1),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _open_sources(
    paths: dict[str, Path], start_frame: int
) -> tuple[dict[str, cv2.VideoCapture], dict[str, dict[str, Any]]]:
    captures: dict[str, cv2.VideoCapture] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for view in VIEW_ORDER:
        capture = cv2.VideoCapture(str(paths[view]))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {view}: {paths[view]}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        captures[view] = capture
        metadata[view] = {
            "path": str(paths[view]),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    return captures, metadata


def _release(captures: Iterable[cv2.VideoCapture]) -> None:
    for capture in captures:
        capture.release()


def _draw_label(
    canvas: np.ndarray,
    *,
    point: tuple[int, int],
    panel_left: int,
    panel_right: int,
    panel_top: int,
    panel_bottom: int,
    p_label: str,
    geometry_slot: int,
    color: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> None:
    cv2.circle(canvas, point, 10, color, 3, cv2.LINE_AA)
    text = f"{p_label} / G{geometry_slot}"
    scale = 0.43
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, scale, 2
    )
    candidates = (
        (point[0] + 10, point[1] - 10),
        (point[0] - width - 10, point[1] - 10),
        (point[0] + 10, point[1] + height + 14),
        (point[0] - width - 10, point[1] + height + 14),
        (point[0] + 10, point[1] - height - 18),
        (point[0] - width - 10, point[1] - height - 18),
    )

    def clamp(candidate: tuple[int, int]) -> tuple[int, int]:
        x, y = candidate
        x = min(max(panel_left + 6, x), panel_right - width - 6)
        y = min(max(panel_top + height + 6, y), panel_bottom - baseline - 6)
        return int(x), int(y)

    def box(candidate: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = clamp(candidate)
        return x - 3, y - height - 3, x + width + 3, y + baseline + 3

    def overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
        width_overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]))
        height_overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
        return width_overlap * height_overlap

    scored = []
    for index, candidate in enumerate(candidates):
        candidate_box = box(candidate)
        score = sum(overlap(candidate_box, prior) for prior in occupied)
        scored.append((score, index, clamp(candidate), candidate_box))
    _score, _index, (x, y), chosen_box = min(scored)
    occupied.append(chosen_box)
    _put_text(canvas, text, (x, y), color=color, scale=scale, thickness=2, shadow=True)


def _compose_frame(
    frames: dict[str, np.ndarray],
    frame_number: int,
    sample_lookup: dict[tuple[int, str], list[dict[str, Any]]],
    color_for_label: dict[str, str],
    events_by_frame: dict[int, list[str]],
    *,
    source_fps: float,
    sequence_start: int,
    title: str,
) -> np.ndarray:
    panel_size = 400
    header = 60
    footer = 170
    canvas = np.full((header + panel_size + footer, panel_size * 3, 3), 255, dtype=np.uint8)
    _put_text(
        canvas,
        title,
        (16, 30),
        color=_hex_to_bgr(INK),
        scale=0.72,
        thickness=2,
    )
    _put_text(
        canvas,
        "Exact training supervision; no temporal smoothing or identity repair",
        (16, 51),
        color=_hex_to_bgr(MUTED),
        scale=0.43,
    )

    active_labels: set[str] = set()
    window_names: set[str] = set()
    accepted_counts: list[str] = []
    for column, view in enumerate(VIEW_ORDER):
        source = frames[view]
        height, width = source.shape[:2]
        panel = cv2.resize(source, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
        left = column * panel_size
        right = left + panel_size
        canvas[header : header + panel_size, left:right] = panel
        _put_text(
            canvas,
            view.replace("_cam", ""),
            (left + 10, header + 24),
            color=_hex_to_bgr(WHITE),
            scale=0.60,
            thickness=2,
            shadow=True,
        )
        rows = sample_lookup.get((frame_number, view), [])
        accepted_counts.append(f"{view.replace('_cam', '')}:{len(rows)}")
        occupied: list[tuple[int, int, int, int]] = []
        for row in rows:
            x = float(row["sleap_x"])
            y = float(row["sleap_y"])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            p_label = str(row["p_label"])
            active_labels.add(p_label)
            window_names.add(str(row["window"]))
            point = (
                left + int(np.clip(x / max(1, width) * panel_size, 0, panel_size - 1)),
                header + int(np.clip(y / max(1, height) * panel_size, 0, panel_size - 1)),
            )
            _draw_label(
                canvas,
                point=point,
                panel_left=left,
                panel_right=right,
                panel_top=header,
                panel_bottom=header + panel_size,
                p_label=p_label,
                geometry_slot=int(row["geometry_source_idx_local"]),
                color=_hex_to_bgr(color_for_label[p_label]),
                occupied=occupied,
            )
        cv2.rectangle(canvas, (left, header), (right - 1, header + panel_size - 1), _hex_to_bgr(GREY), 1)

    footer_y = header + panel_size + 27
    _put_text(
        canvas,
        "Pxx = positive-pair component (same-over-time claim); Gx = frame-local geometry slot (not identity).",
        (16, footer_y),
        color=_hex_to_bgr(INK),
        scale=0.45,
        thickness=1,
    )
    window_text = ",".join(sorted(window_names)) if window_names else "no strict sample"
    _put_text(
        canvas,
        f"source frame {frame_number} | {(frame_number - sequence_start) / source_fps:5.2f} s from clip start | window {window_text}",
        (16, footer_y + 27),
        color=_hex_to_bgr(MUTED),
        scale=0.45,
    )
    _put_text(
        canvas,
        f"active labels: {','.join(sorted(active_labels)) or 'none'} | accepted strict crops: {' '.join(accepted_counts)}",
        (16, footer_y + 54),
        color=_hex_to_bgr(MUTED),
        scale=0.45,
    )
    events = events_by_frame.get(frame_number, [])
    event_text = "EVENT: " + "; ".join(events) if events else "Watch whether each P label remains on one physical fish."
    _put_text(
        canvas,
        event_text,
        (16, footer_y + 88),
        color=_hex_to_bgr(EVENT if events else INK),
        scale=0.49,
        thickness=2 if events else 1,
    )
    return canvas


def _render_video(
    *,
    output: Path,
    poster: Path,
    video_paths: dict[str, Path],
    labeled_samples: pd.DataFrame,
    components: pd.DataFrame,
    events: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    output_fps: float,
    poster_frame: int,
    title: str,
) -> dict[str, Any]:
    captures, metadata = _open_sources(video_paths, start_frame)
    source_fps = float(np.median([metadata[view]["fps"] for view in VIEW_ORDER]))
    color_for_label = components.set_index("p_label").color_hex.astype(str).to_dict()
    sample_scope = labeled_samples[
        labeled_samples.frame.between(start_frame, end_frame, inclusive="both")
    ]
    sample_lookup = {
        (int(frame), str(view)): rows.to_dict("records")
        for (frame, view), rows in sample_scope.groupby(["frame", "view"])
    }
    events_by_frame: dict[int, list[str]] = defaultdict(list)
    for row in events[events.frame.between(start_frame, end_frame, inclusive="both")].itertuples(index=False):
        verb = "starts" if str(row.event) == "component_start" else "ends"
        events_by_frame[int(row.frame)].append(f"{row.p_label} {verb}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (1200, 630),
    )
    if not writer.isOpened():
        _release(captures.values())
        raise RuntimeError(f"Could not create {output}")
    written = 0
    poster_written = False
    try:
        for frame_number in range(start_frame, end_frame + 1):
            frames: dict[str, np.ndarray] = {}
            for view, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read {view} frame {frame_number}")
                frames[view] = frame
            canvas = _compose_frame(
                frames,
                frame_number,
                sample_lookup,
                color_for_label,
                events_by_frame,
                source_fps=source_fps,
                sequence_start=start_frame,
                title=title,
            )
            writer.write(canvas)
            written += 1
            if not poster_written and frame_number >= poster_frame:
                cv2.imwrite(str(poster), canvas)
                poster_written = True
    finally:
        writer.release()
        _release(captures.values())
    return {
        "output": str(output),
        "poster": str(poster),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "source_fps": source_fps,
        "output_fps": float(output_fps),
        "output_frames": int(written),
        "playback_seconds": float(written / output_fps),
        "slowdown_vs_real_time": float(source_fps / output_fps),
        "video_sources": metadata,
    }


def render_audit(
    *,
    dataset_dir: Path,
    output_dir: Path,
    video_paths: dict[str, Path],
    full_start: int,
    full_end: int,
    event_start: int,
    event_end: int,
    output_fps: float,
) -> dict[str, Any]:
    samples_path = _find_table(dataset_dir, "samples")
    positives_path = _find_table(dataset_dir, "positive_pairs")
    negatives_path = _find_table(dataset_dir, "negative_pairs")
    samples = _read_table(samples_path)
    positives = _read_table(positives_path)
    negatives = _read_table(negatives_path)
    labeled, components, events, component_audit = _positive_components(
        samples, positives, negatives
    )
    scope = labeled[labeled.frame.between(full_start, full_end, inclusive="both")]
    scope_labels = sorted(scope.p_label.unique())
    scope_components = components[components.p_label.isin(scope_labels)].copy()
    scope_events = events[events.p_label.isin(scope_labels)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    scope_components.to_csv(output_dir / "SUPERVISION_COMPONENTS.csv", index=False)
    scope_events.to_csv(output_dir / "SUPERVISION_EVENTS.csv", index=False)
    review = scope_components.copy()
    review["review_same_fish_over_time"] = ""
    review["first_wrong_frame"] = ""
    review["review_notes"] = ""
    review.to_csv(output_dir / "REVIEW_TEMPLATE.csv", index=False)

    full = _render_video(
        output=output_dir / "AUDIT_1_HELDOUT_EXACT_SUPERVISION.mp4",
        poster=output_dir / "AUDIT_1_POSTER.png",
        video_paths=video_paths,
        labeled_samples=scope,
        components=scope_components,
        events=scope_events,
        start_frame=full_start,
        end_frame=full_end,
        output_fps=output_fps,
        poster_frame=78547,
        title="Held-out audit: exact positive-pair supervision",
    )
    event_clip = _render_video(
        output=output_dir / "AUDIT_2_FRAGMENT_BREAK_EVENT.mp4",
        poster=output_dir / "AUDIT_2_POSTER.png",
        video_paths=video_paths,
        labeled_samples=scope,
        components=scope_components,
        events=scope_events,
        start_frame=event_start,
        end_frame=event_end,
        output_fps=output_fps,
        poster_frame=78364,
        title="Difficult interval: supervision fragments and restarts",
    )
    manifest = {
        "schema_version": "geometry_supervision_audit_video_v1",
        "identity_warning": (
            "P labels reproduce positive-pair components; they are review targets, not verified biological IDs. "
            "G labels are frame-local and may permute."
        ),
        "inputs": {
            "samples": str(samples_path),
            "positive_pairs": str(positives_path),
            "negative_pairs": str(negatives_path),
        },
        "component_audit": component_audit,
        "scope_components": scope_components.to_dict("records"),
        "videos": [full, event_clip],
        "review_files": [
            "SUPERVISION_COMPONENTS.csv",
            "SUPERVISION_EVENTS.csv",
            "REVIEW_TEMPLATE.csv",
        ],
    }
    (output_dir / "AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--view-video",
        nargs=2,
        action="append",
        metavar=("VIEW", "PATH"),
        required=True,
    )
    parser.add_argument("--full-start", type=int, default=78000)
    parser.add_argument("--full-end", type=int, default=78999)
    parser.add_argument("--event-start", type=int, default=78240)
    parser.add_argument("--event-end", type=int, default=78410)
    parser.add_argument("--output-fps", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = {str(view): Path(path) for view, path in args.view_video}
    missing = set(VIEW_ORDER) - set(paths)
    if missing:
        raise ValueError(f"Missing view videos: {sorted(missing)}")
    result = render_audit(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        video_paths=paths,
        full_start=args.full_start,
        full_end=args.full_end,
        event_start=args.event_start,
        event_end=args.event_end,
        output_fps=args.output_fps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
