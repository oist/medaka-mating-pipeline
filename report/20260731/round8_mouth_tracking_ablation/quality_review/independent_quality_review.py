from __future__ import annotations

"""Independent read-only QC helpers for the Round8 mouth-tracking study.

This script deliberately reads the source SLP and reconstruction artifacts
directly. It does not import the executor's study implementation.
"""

import argparse
import csv
import hashlib
import json
import math
import mmap
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VIEWS = ("top_cam", "side1_cam", "side2_cam")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_frame_objects(
    path: Path, frames: Iterable[int]
) -> dict[str, dict[str, Any]]:
    """Extract selected ``frames`` objects without loading all JSON."""
    output: dict[str, dict[str, Any]] = {}
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            frames_marker = mapped.find(b'"frames"')
            if frames_marker < 0:
                raise ValueError(f"no frames object in {path}")
            frames_colon = mapped.find(b":", frames_marker + len(b'"frames"'))
            search_start = mapped.find(b"{", frames_colon + 1)
            if frames_colon < 0 or search_start < 0:
                raise ValueError(f"malformed frames object in {path}")
            for frame in frames:
                pattern = f'"{int(frame)}"'.encode("ascii")
                found = mapped.find(pattern, search_start)
                if found < 0:
                    raise KeyError(f"frame {frame} not found in {path}")
                colon = mapped.find(b":", found + len(pattern))
                object_start = mapped.find(b"{", colon + 1)
                if colon < 0 or object_start < 0:
                    raise ValueError(f"malformed frame {frame} in {path}")
                depth = 0
                in_string = False
                escaped = False
                object_end = None
                for index in range(object_start, len(mapped)):
                    character = mapped[index]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif character == ord("\\"):
                            escaped = True
                        elif character == ord('"'):
                            in_string = False
                    else:
                        if character == ord('"'):
                            in_string = True
                        elif character == ord("{"):
                            depth += 1
                        elif character == ord("}"):
                            depth -= 1
                            if depth == 0:
                                object_end = index + 1
                                break
                if object_end is None:
                    raise ValueError(f"unterminated frame {frame} in {path}")
                output[str(frame)] = json.loads(mapped[object_start:object_end])
    return output


def _instance_rows(labels: Any, frame_start: int, frame_end: int) -> list[dict[str, Any]]:
    if not labels.skeletons:
        raise ValueError("SLP has no skeleton")
    node_names = [node.name for node in labels.skeletons[0].nodes]
    mouth_index = node_names.index("mouth")
    tail_index = node_names.index("tail")
    rows: list[dict[str, Any]] = []
    for labeled_frame in labels.labeled_frames:
        frame = int(labeled_frame.frame_idx)
        if frame < frame_start or frame > frame_end:
            continue
        for instance_index, instance in enumerate(labeled_frame.instances):
            points = np.asarray(instance.numpy(), dtype=float)
            structured_points = np.asarray(getattr(instance, "points", ()))
            structured_names = structured_points.dtype.names or ()
            node_scores = (
                np.asarray(structured_points["score"], dtype=float).reshape(-1)
                if "score" in structured_names
                else np.full(len(points), np.nan, dtype=float)
            )
            mouth = points[mouth_index] if mouth_index < len(points) else np.full(2, np.nan)
            tail = points[tail_index] if tail_index < len(points) else np.full(2, np.nan)
            finite = np.isfinite(points).all(axis=1)
            centroid = (
                np.mean(points[finite], axis=0) if finite.any() else np.full(2, np.nan)
            )
            score = getattr(instance, "score", np.nan)
            rows.append(
                {
                    "frame": frame,
                    "instance_index": instance_index,
                    "instance_score": float(score) if score is not None else math.nan,
                    "mouth_score": (
                        float(node_scores[mouth_index])
                        if mouth_index < len(node_scores)
                        else math.nan
                    ),
                    "tail_score": (
                        float(node_scores[tail_index])
                        if tail_index < len(node_scores)
                        else math.nan
                    ),
                    "mouth_x": float(mouth[0]),
                    "mouth_y": float(mouth[1]),
                    "tail_x": float(tail[0]),
                    "tail_y": float(tail[1]),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "finite_mouth": bool(np.isfinite(mouth).all()),
                    "finite_tail": bool(np.isfinite(tail).all()),
                }
            )
    return rows


def audit_raw_event(
    slp_root: Path, output_dir: Path, frame_start: int, frame_end: int
) -> dict[str, Any]:
    import sleap_io as sio

    raw_rows: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        source = slp_root / f"{view}.frames_0-87149.slp"
        labels = sio.load_slp(str(source))
        raw_rows[view] = _instance_rows(labels, frame_start, frame_end)
        source_files[view] = {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "sleap_io": getattr(sio, "__version__", "unknown"),
        }

    flat_path = output_dir / "critical_event_raw_instances.csv"
    fieldnames = ["view"] + list(raw_rows[VIEWS[0]][0].keys())
    with flat_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for view in VIEWS:
            for row in raw_rows[view]:
                writer.writerow({"view": view, **row})

    top_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in raw_rows["top_cam"]:
        top_by_frame.setdefault(int(row["frame"]), []).append(row)
    before = top_by_frame[34547]
    after = top_by_frame[34548]
    missing_tail_after = [row for row in after if not row["finite_tail"]]
    if len(missing_tail_after) != 1:
        raise AssertionError(
            f"expected one top instance losing its tail at 34548; got {len(missing_tail_after)}"
        )
    target_after = missing_tail_after[0]
    after_mouth = np.array([target_after["mouth_x"], target_after["mouth_y"]])
    before_distances = [
        float(
            np.linalg.norm(
                np.array([row["mouth_x"], row["mouth_y"]], dtype=float) - after_mouth
            )
        )
        for row in before
    ]
    target_before = before[int(np.argmin(before_distances))]
    mouth_before = np.array([target_before["mouth_x"], target_before["mouth_y"]])
    centroid_before = np.array(
        [target_before["centroid_x"], target_before["centroid_y"]]
    )
    centroid_after = np.array(
        [target_after["centroid_x"], target_after["centroid_y"]]
    )

    close_pair_count: dict[str, dict[str, int]] = {}
    for view in ("side1_cam", "side2_cam"):
        close_pair_count[view] = {}
        for frame in (34547, 34548):
            rows = [row for row in raw_rows[view] if int(row["frame"]) == frame]
            # The close pair is at the lower image region; the third fish is
            # spatially distant near the upper region in both side views.
            close_rows = [row for row in rows if float(row["mouth_y"]) > 700.0]
            close_pair_count[view][str(frame)] = len(close_rows)

    result = {
        "source_files": source_files,
        "event_frames": [frame_start, frame_end],
        "reported_values": {
            "top_mouth_displacement_px": 0.236,
            "top_visible_node_centroid_displacement_px": 56.209,
        },
        "independent_values": {
            "top_target_before_instance": int(target_before["instance_index"]),
            "top_target_after_instance": int(target_after["instance_index"]),
            "top_mouth_before_xy": mouth_before,
            "top_mouth_after_xy": after_mouth,
            "top_centroid_before_xy": centroid_before,
            "top_centroid_after_xy": centroid_after,
            "top_mouth_displacement_px": float(
                np.linalg.norm(after_mouth - mouth_before)
            ),
            "top_visible_node_centroid_displacement_px": float(
                np.linalg.norm(centroid_after - centroid_before)
            ),
            "side_view_close_pair_pose_count": close_pair_count,
            "total_instance_count": {
                view: {
                    str(frame): sum(
                        int(row["frame"]) == frame for row in raw_rows[view]
                    )
                    for frame in (34547, 34548)
                }
                for view in VIEWS
            },
        },
        "method": {
            "top_correspondence": (
                "unique frame-34548 finite-mouth instance with missing tail, "
                "matched backward by nearest raw mouth coordinate"
            ),
            "centroid": "mean of all finite nodes in the same raw SLEAP instance",
            "side_close_pair_region": "mouth_y > 700 px; third fish is spatially distant",
        },
    }
    result_path = output_dir / "critical_event_raw_audit.json"
    result_path.write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=True), encoding="utf-8"
    )
    return result


def audit_prelink(
    prelink_json: Path,
    linked_json: Path | None,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    frames = range(frame_start, frame_end + 1)
    result: dict[str, Any] = {
        "pre_link_source": {
            "path": str(prelink_json),
            "size_bytes": prelink_json.stat().st_size,
        },
        "pre_link_frames": extract_frame_objects(prelink_json, frames),
    }
    before = result["pre_link_frames"]["34547"]
    after = result["pre_link_frames"]["34548"]
    close_labels = ("b", "c")
    distance_matrix = np.array(
        [
            [
                np.linalg.norm(
                    np.asarray(before["groups"][old_label], dtype=float)
                    - np.asarray(after["groups"][new_label], dtype=float)
                )
                for new_label in close_labels
            ]
            for old_label in close_labels
        ],
        dtype=float,
    )
    gate_mm = 8.0
    result["gate_audit_34547_to_34548"] = {
        "row_previous_labels": list(close_labels),
        "column_current_labels": list(close_labels),
        "euclidean_distance_mm": distance_matrix.tolist(),
        "feasible_at_8mm": (distance_matrix <= gate_mm).tolist(),
        "straight_cost_mm": float(distance_matrix[0, 0] + distance_matrix[1, 1]),
        "crossed_cost_mm": float(distance_matrix[0, 1] + distance_matrix[1, 0]),
        "straight_feasible": bool(
            distance_matrix[0, 0] <= gate_mm
            and distance_matrix[1, 1] <= gate_mm
        ),
        "crossed_feasible": bool(
            distance_matrix[0, 1] <= gate_mm
            and distance_matrix[1, 0] <= gate_mm
        ),
        "top_detection_slots_before": {
            label: int(before["perm_cam0"][index + 1])
            for index, label in enumerate(close_labels)
        },
        "top_detection_slots_after": {
            label: int(after["perm_cam0"][index + 1])
            for index, label in enumerate(close_labels)
        },
        "interpretation": (
            "The same top detection slots define the straight continuation. "
            "Its c-to-c 3D step exceeds 8 mm; both crossed steps are below 8 mm."
        ),
    }
    if linked_json is not None:
        result["linked_source"] = {
            "path": str(linked_json),
            "size_bytes": linked_json.stat().st_size,
        }
        result["linked_frames"] = extract_frame_objects(linked_json, frames)
    output = output_dir / "critical_event_existing_geometry.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slp-root", type=Path, required=True)
    parser.add_argument("--prelink-json", type=Path, required=True)
    parser.add_argument("--linked-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=34540)
    parser.add_argument("--frame-end", type=int, default=34565)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = audit_raw_event(
        args.slp_root, args.output_dir, args.frame_start, args.frame_end
    )
    geometry = audit_prelink(
        args.prelink_json,
        args.linked_json,
        args.output_dir,
        34547,
        34555,
    )
    summary = {
        "raw_event": raw,
        "geometry_frame_keys": sorted(geometry["pre_link_frames"]),
    }
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
