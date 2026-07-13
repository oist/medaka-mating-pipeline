from __future__ import annotations

"""Build conservative mouth/tail-aware 2D tracklets and sweep ambiguity gates."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.tracking.compare_anchor_tracklets import seg_seg_dist


@dataclass(frozen=True)
class GateConfig:
    name: str
    min_score: float
    max_length_robust_z: float
    close_segment_bl: float
    max_endpoint_prediction_bl: float
    max_body_turn_deg: float
    min_assignment_margin_bl: float
    max_match_cost_bl: float


PRESETS = {
    "permissive": GateConfig("permissive", 0.25, 5.0, 0.05, 0.50, 75.0, 0.04, 1.20),
    "balanced": GateConfig("balanced", 0.35, 4.0, 0.10, 0.35, 50.0, 0.08, 0.90),
    "strict": GateConfig("strict", 0.50, 3.0, 0.18, 0.25, 35.0, 0.14, 0.65),
}


@dataclass(frozen=True)
class Detection:
    frame: int
    raw_index: int
    mouth: np.ndarray
    tail: np.ndarray
    centroid: np.ndarray
    body_vector: np.ndarray
    body_length: float
    score: float


@dataclass
class ActiveTrack:
    tracklet_id: int
    last_frame: int
    last_centroid: np.ndarray
    centroid_velocity: np.ndarray
    last_mouth: np.ndarray
    mouth_velocity: np.ndarray
    last_tail: np.ndarray
    tail_velocity: np.ndarray
    last_vector: np.ndarray
    last_length: float


def _finite(point: np.ndarray) -> bool:
    return point.shape == (2,) and bool(np.isfinite(point).all())


def load_detections(
    slp_path: Path, *, frame_start: int, frame_end: int
) -> dict[int, list[Detection]]:
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path))
    if not labels.skeletons:
        raise RuntimeError(f"No skeleton in {slp_path}")
    names = [node.name for node in labels.skeletons[0].nodes]
    if "mouth" not in names or "tail" not in names:
        raise RuntimeError(f"Expected mouth/tail skeleton, found {names}")
    mouth_index = names.index("mouth")
    tail_index = names.index("tail")
    output = {frame: [] for frame in range(frame_start, frame_end + 1)}
    for labeled_frame in labels.labeled_frames:
        frame = int(labeled_frame.frame_idx)
        if frame < frame_start or frame > frame_end:
            continue
        rows: list[Detection] = []
        for raw_index, instance in enumerate(labeled_frame.instances):
            points = np.asarray(instance.numpy(), dtype=float)[:, :2]
            mouth = points[mouth_index].copy()
            tail = points[tail_index].copy()
            if not (_finite(mouth) and _finite(tail)):
                continue
            vector = mouth - tail
            length = float(np.linalg.norm(vector))
            score = getattr(instance, "score", None)
            score = float(score) if score is not None and np.isfinite(score) else 0.0
            rows.append(
                Detection(
                    frame=frame,
                    raw_index=raw_index,
                    mouth=mouth,
                    tail=tail,
                    centroid=(mouth + tail) / 2.0,
                    body_vector=vector,
                    body_length=length,
                    score=score,
                )
            )
        output[frame] = rows
    return output


def robust_length_reference(detections: dict[int, list[Detection]]) -> dict[str, float]:
    lengths = np.asarray(
        [det.body_length for rows in detections.values() for det in rows if det.score >= 0.6],
        dtype=float,
    )
    lengths = lengths[np.isfinite(lengths) & (lengths > 1e-6)]
    if lengths.size == 0:
        raise RuntimeError("No valid body lengths")
    log_lengths = np.log(lengths)
    median_log = float(np.median(log_lengths))
    mad_log = float(np.median(np.abs(log_lengths - median_log)))
    mad_log = max(mad_log, 0.02)
    return {
        "body_length_px": float(np.exp(median_log)),
        "median_log_length": median_log,
        "mad_log_length": mad_log,
        "reference_count": int(lengths.size),
    }


def _length_z(det: Detection, reference: dict[str, float]) -> float:
    return float(
        0.67448975
        * (np.log(max(det.body_length, 1e-6)) - reference["median_log_length"])
        / reference["mad_log_length"]
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _close_detection_indices(
    detections: list[Detection], *, threshold_px: float
) -> tuple[set[int], float]:
    close: set[int] = set()
    minimum = np.inf
    for left in range(len(detections)):
        for right in range(left + 1, len(detections)):
            distance = float(
                seg_seg_dist(
                    detections[left].mouth,
                    detections[left].tail,
                    detections[right].mouth,
                    detections[right].tail,
                )
            )
            minimum = min(minimum, distance)
            if distance < threshold_px:
                close.update((left, right))
    return close, minimum


def _match_diagnostics(
    track: ActiveTrack,
    detection: Detection,
    *,
    frame: int,
    body_length_px: float,
) -> dict[str, float]:
    gap = max(1, frame - track.last_frame)
    predicted_centroid = track.last_centroid + track.centroid_velocity * gap
    predicted_mouth = track.last_mouth + track.mouth_velocity * gap
    predicted_tail = track.last_tail + track.tail_velocity * gap
    centroid_residual = float(np.linalg.norm(detection.centroid - predicted_centroid)) / body_length_px
    mouth_residual = float(np.linalg.norm(detection.mouth - predicted_mouth)) / body_length_px
    tail_residual = float(np.linalg.norm(detection.tail - predicted_tail)) / body_length_px
    endpoint_residual = 0.5 * (mouth_residual + tail_residual)
    body_turn_deg = _angle_deg(track.last_vector, detection.body_vector)
    length_log_change = abs(float(np.log(max(detection.body_length, 1e-6) / max(track.last_length, 1e-6))))
    score_penalty = max(0.0, 0.8 - min(1.0, detection.score))
    cost = (
        0.55 * centroid_residual
        + 0.75 * endpoint_residual
        + 0.35 * np.radians(body_turn_deg)
        + 0.20 * length_log_change
        + 0.35 * score_penalty
    )
    return {
        "cost_bl": float(cost),
        "centroid_prediction_bl": centroid_residual,
        "endpoint_prediction_bl": endpoint_residual,
        "body_turn_deg": body_turn_deg,
        "length_log_change": length_log_change,
    }


def _new_track(tracklet_id: int, detection: Detection) -> ActiveTrack:
    zero = np.zeros(2, dtype=float)
    return ActiveTrack(
        tracklet_id=tracklet_id,
        last_frame=detection.frame,
        last_centroid=detection.centroid.copy(),
        centroid_velocity=zero.copy(),
        last_mouth=detection.mouth.copy(),
        mouth_velocity=zero.copy(),
        last_tail=detection.tail.copy(),
        tail_velocity=zero.copy(),
        last_vector=detection.body_vector.copy(),
        last_length=detection.body_length,
    )


def _update_track(track: ActiveTrack, detection: Detection) -> None:
    gap = max(1, detection.frame - track.last_frame)
    centroid_velocity = (detection.centroid - track.last_centroid) / gap
    mouth_velocity = (detection.mouth - track.last_mouth) / gap
    tail_velocity = (detection.tail - track.last_tail) / gap
    track.centroid_velocity = 0.4 * track.centroid_velocity + 0.6 * centroid_velocity
    track.mouth_velocity = 0.4 * track.mouth_velocity + 0.6 * mouth_velocity
    track.tail_velocity = 0.4 * track.tail_velocity + 0.6 * tail_velocity
    track.last_frame = detection.frame
    track.last_centroid = detection.centroid.copy()
    track.last_mouth = detection.mouth.copy()
    track.last_tail = detection.tail.copy()
    track.last_vector = detection.body_vector.copy()
    track.last_length = detection.body_length


def build_node_aware_tracklets(
    detections: dict[int, list[Detection]],
    *,
    config: GateConfig,
    reference: dict[str, float] | None = None,
    num_fish: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reference = reference or robust_length_reference(detections)
    body_length_px = float(reference["body_length_px"])
    active: dict[int, ActiveTrack] = {}
    observations: list[dict[str, object]] = []
    frame_qc: list[dict[str, object]] = []
    next_tracklet_id = 0
    break_counts: dict[str, int] = {}

    def record(tracklet_id: int, detection: Detection, status: str, diag: dict[str, float] | None = None) -> None:
        row = {
            "frame": detection.frame,
            "tracklet_id": tracklet_id,
            "raw_detection_index": detection.raw_index,
            "x": detection.centroid[0],
            "y": detection.centroid[1],
            "mouth_x": detection.mouth[0],
            "mouth_y": detection.mouth[1],
            "tail_x": detection.tail[0],
            "tail_y": detection.tail[1],
            "body_length_px": detection.body_length,
            "score": detection.score,
            "observation_status": status,
        }
        row.update(diag or {})
        observations.append(row)

    for frame in sorted(detections):
        raw = detections[frame]
        valid = [
            det
            for det in raw
            if det.score >= config.min_score
            and abs(_length_z(det, reference)) <= config.max_length_robust_z
        ]
        invalid_score = sum(det.score < config.min_score for det in raw)
        invalid_length = sum(
            det.score >= config.min_score
            and abs(_length_z(det, reference)) > config.max_length_robust_z
            for det in raw
        )
        close_indices, minimum_segment_distance = _close_detection_indices(
            valid, threshold_px=config.close_segment_bl * body_length_px
        )
        active_before = len(active)
        assigned_detection_indices: set[int] = set()
        matched_track_ids: set[int] = set()
        breaks_this_frame: list[str] = []

        track_ids = list(active)
        diagnostics: list[list[dict[str, float]]] = []
        cost = np.empty((len(track_ids), len(valid)), dtype=float)
        for row_index, tracklet_id in enumerate(track_ids):
            row_diag = []
            for column_index, detection in enumerate(valid):
                diag = _match_diagnostics(
                    active[tracklet_id], detection, frame=frame, body_length_px=body_length_px
                )
                row_diag.append(diag)
                cost[row_index, column_index] = diag["cost_bl"]
            diagnostics.append(row_diag)

        pairs: list[tuple[int, int]] = []
        if track_ids and valid:
            row_index, column_index = linear_sum_assignment(cost)
            pairs = [(int(left), int(right)) for left, right in zip(row_index, column_index)]

        tracks_to_close: dict[int, str] = {}
        for row_index, column_index in pairs:
            tracklet_id = track_ids[row_index]
            detection = valid[column_index]
            diag = diagnostics[row_index][column_index]
            alternatives = np.delete(cost[row_index], column_index)
            margin = float(np.min(alternatives) - cost[row_index, column_index]) if alternatives.size else np.inf
            diag = {**diag, "assignment_margin_bl": margin}
            reason = ""
            if column_index in close_indices:
                reason = "close_body_segments"
            elif diag["endpoint_prediction_bl"] > config.max_endpoint_prediction_bl:
                reason = "endpoint_prediction_jump"
            elif diag["body_turn_deg"] > config.max_body_turn_deg:
                reason = "body_vector_turn"
            elif diag["cost_bl"] > config.max_match_cost_bl:
                reason = "high_match_cost"
            elif margin < config.min_assignment_margin_bl:
                reason = "assignment_margin_ambiguous"
            if reason:
                tracks_to_close[tracklet_id] = reason
                assigned_detection_indices.add(column_index)
                breaks_this_frame.append(reason)
                continue
            _update_track(active[tracklet_id], detection)
            record(tracklet_id, detection, "extended", diag)
            assigned_detection_indices.add(column_index)
            matched_track_ids.add(tracklet_id)

        for tracklet_id in track_ids:
            if tracklet_id in matched_track_ids:
                continue
            reason = tracks_to_close.get(tracklet_id)
            if reason is None:
                reason = "false_negative_or_unmatched" if len(valid) < num_fish else "unmatched_extra_candidate"
                breaks_this_frame.append(reason)
            break_counts[reason] = break_counts.get(reason, 0) + 1
            active.pop(tracklet_id, None)

        # Start fresh fragments only from individually clear, unused detections.
        seed_candidates = [
            (index, det)
            for index, det in enumerate(valid)
            if index not in assigned_detection_indices and index not in close_indices
        ]
        seed_candidates.sort(key=lambda item: (-item[1].score, abs(_length_z(item[1], reference))))
        for column_index, detection in seed_candidates:
            if len(active) >= num_fish:
                break
            tracklet_id = next_tracklet_id
            next_tracklet_id += 1
            active[tracklet_id] = _new_track(tracklet_id, detection)
            record(tracklet_id, detection, "seeded")

        frame_qc.append(
            {
                "frame": frame,
                "raw_detection_count": len(raw),
                "valid_detection_count": len(valid),
                "invalid_score_count": invalid_score,
                "invalid_length_count": invalid_length,
                "close_detection_count": len(close_indices),
                "minimum_segment_distance_px": minimum_segment_distance,
                "active_before": active_before,
                "active_after": len(active),
                "observed_track_count": sum(row["frame"] == frame for row in observations[-num_fish:]),
                "break_reasons": "|".join(sorted(set(breaks_this_frame))),
            }
        )

    observation_df = pd.DataFrame(observations)
    qc_df = pd.DataFrame(frame_qc)
    if observation_df.empty:
        lengths = pd.Series(dtype=float)
    else:
        lengths = observation_df.groupby("tracklet_id").size()
    total_frames = len(detections)
    stats: dict[str, object] = {
        "preset": config.name,
        "total_frames": total_frames,
        "body_length_px": body_length_px,
        "n_fragments": int(len(lengths)),
        "observation_rows": int(len(observation_df)),
        "coverage_frac_of_fish_frames": float(len(observation_df) / max(1, total_frames * num_fish)),
        "full_three_observed_frames": int((observation_df.groupby("frame").size() == num_fish).sum()) if not observation_df.empty else 0,
        "full_three_observed_fraction": float((observation_df.groupby("frame").size() == num_fish).sum() / max(1, total_frames)) if not observation_df.empty else 0.0,
        "fragment_length_median": float(lengths.median()) if not lengths.empty else 0.0,
        "fragment_length_mean": float(lengths.mean()) if not lengths.empty else 0.0,
        "fragment_length_max": int(lengths.max()) if not lengths.empty else 0,
        "fragments_ge_30": int((lengths >= 30).sum()),
        "fragments_ge_100": int((lengths >= 100).sum()),
        "fragments_ge_300": int((lengths >= 300).sum()),
        "fragments_ge_1000": int((lengths >= 1000).sum()),
        "frames_raw_count_lt_3": int((qc_df["raw_detection_count"] < num_fish).sum()),
        "frames_raw_count_gt_3": int((qc_df["raw_detection_count"] > num_fish).sum()),
        "frames_invalid_score": int((qc_df["invalid_score_count"] > 0).sum()),
        "frames_invalid_length": int((qc_df["invalid_length_count"] > 0).sum()),
        "frames_close_segments": int((qc_df["close_detection_count"] > 0).sum()),
        "break_counts": break_counts,
        "config": asdict(config),
        "length_reference": reference,
    }
    return observation_df, qc_df, stats


def _plot_sweep(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = list(summary["view"].drop_duplicates())
    colors = {"permissive": "#6B9AC4", "balanced": "#D17A22", "strict": "#7C8B3B"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for preset in ("permissive", "balanced", "strict"):
        sub = summary[summary["preset"].eq(preset)].set_index("view").reindex(views)
        axes[0].plot(views, sub["coverage_frac_of_fish_frames"], marker="o", lw=1.8, color=colors[preset], label=preset)
        axes[1].plot(views, sub["n_fragments"], marker="o", lw=1.8, color=colors[preset], label=preset)
    axes[0].set_ylabel("Fish-frame coverage")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("Concrete tracklet fragments")
    for ax in axes:
        ax.grid(axis="y", color="#D9DDE3", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    fig.suptitle("Node-aware concrete-tracklet threshold sweep", x=0.06, y=0.98, ha="left", fontsize=15)
    fig.text(0.06, 0.93, "Round7 frames 10000–14999; mouth/tail prediction, segment proximity, assignment margin, and temporal endpoint consistency", color="#4C5663", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.90))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slp", action="append", required=True, help="view=path; repeat for each camera")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-start", type=int, default=10000)
    parser.add_argument("--frame-end", type=int, default=14999)
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = {}
    for value in args.slp:
        view, path = value.split("=", 1)
        specifications[view] = Path(path)
    presets = args.preset or ["permissive", "balanced", "strict"]
    summary_rows: list[dict[str, object]] = []
    for view, slp_path in specifications.items():
        print(f"[load] {view}: {slp_path}")
        detections = load_detections(slp_path, frame_start=args.frame_start, frame_end=args.frame_end)
        reference = robust_length_reference(detections)
        for preset in presets:
            observations, qc, stats = build_node_aware_tracklets(
                detections, config=PRESETS[preset], reference=reference
            )
            prefix = args.output_dir / f"{view}.{preset}"
            Path(f"{prefix}.tracklets.csv").write_text(
                observations.to_csv(index=False), encoding="utf-8"
            )
            if observations.empty:
                substantial = observations.copy()
            else:
                lengths = observations.groupby("tracklet_id").size()
                substantial = observations[
                    observations["tracklet_id"].isin(lengths[lengths >= 30].index)
                ]
            Path(f"{prefix}.tracklets_ge30.csv").write_text(
                substantial.to_csv(index=False), encoding="utf-8"
            )
            Path(f"{prefix}.frame_qc.csv").write_text(
                qc.to_csv(index=False), encoding="utf-8"
            )
            Path(f"{prefix}.summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            summary_rows.append({"view": view, **{key: value for key, value in stats.items() if not isinstance(value, dict)}})
            print(f"  {preset}: fragments={stats['n_fragments']} coverage={stats['coverage_frac_of_fish_frames']:.3f} median={stats['fragment_length_median']:.1f}")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "node_aware_threshold_sweep.csv", index=False)
    _plot_sweep(summary, args.output_dir / "node_aware_threshold_sweep.png")
    print(f"[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
