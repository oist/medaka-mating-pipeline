from __future__ import annotations

"""Compare several top-camera-only identity trackers against current IDs.

The trackers use only SLEAP mouth/tail predictions after a single identity seed
frame.  Current tracked IDs are used only to name the three tracks at that seed
and to evaluate the resulting trajectories; they are not consulted afterward.
"""

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.tracking.plot_identity_window_diagnostics import selected_2d_positions


FISH_IDS = ("ID_1", "ID_2", "ID_3")
COLORS = {"ID_1": "#3568A8", "ID_2": "#D17A22", "ID_3": "#7C8B3B"}
MARKERS = {"ID_1": "o", "ID_2": "s", "ID_3": "^"}


@dataclass(frozen=True)
class Detection:
    index: int
    mouth: np.ndarray
    tail: np.ndarray
    midpoint: np.ndarray
    body_vector: np.ndarray
    body_length: float
    score: float


@dataclass
class TrackState:
    last_frame: int
    last_anchor: np.ndarray
    velocity: np.ndarray
    last_vector: np.ndarray
    body_length: float
    last_mouth: np.ndarray
    last_tail: np.ndarray

    def copy(self) -> "TrackState":
        return TrackState(
            last_frame=self.last_frame,
            last_anchor=self.last_anchor.copy(),
            velocity=self.velocity.copy(),
            last_vector=self.last_vector.copy(),
            body_length=self.body_length,
            last_mouth=self.last_mouth.copy(),
            last_tail=self.last_tail.copy(),
        )


@dataclass
class BeamHypothesis:
    cost: float
    tracks: list[TrackState]
    history: list[tuple[int | None, int | None, int | None]]


def _finite_point(point: np.ndarray) -> bool:
    return point.shape == (2,) and bool(np.isfinite(point).all())


def load_sleap_detections(
    slp_path: Path, *, frame_start: int, frame_end: int
) -> dict[int, list[Detection]]:
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path))
    if not labels.skeletons:
        raise RuntimeError(f"No skeleton in {slp_path}")
    node_names = [node.name for node in labels.skeletons[0].nodes]
    if "mouth" not in node_names or "tail" not in node_names:
        raise RuntimeError(f"Expected mouth and tail nodes, found {node_names}")
    mouth_index = node_names.index("mouth")
    tail_index = node_names.index("tail")

    by_frame: dict[int, list[Detection]] = {
        frame: [] for frame in range(frame_start, frame_end + 1)
    }
    for labeled_frame in labels.labeled_frames:
        frame = int(labeled_frame.frame_idx)
        if frame < frame_start or frame > frame_end:
            continue
        detections: list[Detection] = []
        for raw_index, instance in enumerate(labeled_frame.instances):
            points = np.asarray(instance.numpy(), dtype=float)[:, :2]
            mouth = points[mouth_index].copy()
            tail = points[tail_index].copy()
            visible = points[np.isfinite(points).all(axis=1)]
            if visible.size == 0:
                continue
            midpoint = visible.mean(axis=0)
            if _finite_point(mouth) and _finite_point(tail):
                body_vector = mouth - tail
                body_length = float(np.linalg.norm(body_vector))
            else:
                body_vector = np.full(2, np.nan)
                body_length = np.nan
            score = getattr(instance, "score", None)
            score = float(score) if score is not None and np.isfinite(score) else 0.0
            detections.append(
                Detection(
                    index=raw_index,
                    mouth=mouth,
                    tail=tail,
                    midpoint=midpoint,
                    body_vector=body_vector,
                    body_length=body_length,
                    score=score,
                )
            )
        by_frame[frame] = detections
    return by_frame


def _anchor(detection: Detection, kind: str) -> np.ndarray:
    if kind == "mouth" and _finite_point(detection.mouth):
        return detection.mouth
    return detection.midpoint


def _angle(a: np.ndarray, b: np.ndarray, *, axial: bool = False) -> float:
    if not _finite_point(a) or not _finite_point(b):
        return 0.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))
    value = float(np.arccos(cosine))
    return min(value, np.pi - value) if axial else value


def seed_tracks(
    detections: dict[int, list[Detection]],
    current_top: pd.DataFrame,
    *,
    requested_frame: int,
    anchor_kind: str,
) -> tuple[int, list[TrackState], tuple[int, int, int]]:
    current_lookup = {
        (int(row.frame), str(row.fish_id)): np.asarray([row.x, row.y], dtype=float)
        for row in current_top.itertuples(index=False)
    }
    for frame in range(requested_frame, max(detections) + 1):
        frame_detections = detections.get(frame, [])
        if len(frame_detections) < 3:
            continue
        if not all((frame, fish_id) in current_lookup for fish_id in FISH_IDS):
            continue
        reference = np.asarray([current_lookup[(frame, fish_id)] for fish_id in FISH_IDS])
        candidate = np.asarray([det.midpoint for det in frame_detections])
        cost = np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        if len(rows) != 3 or float(cost[rows, cols].max()) > 5.0:
            continue
        assigned = [-1, -1, -1]
        states: list[TrackState] = []
        for fish_index in range(3):
            det_index = int(cols[np.where(rows == fish_index)[0][0]])
            assigned[fish_index] = det_index
            detection = frame_detections[det_index]
            states.append(
                TrackState(
                    last_frame=frame,
                    last_anchor=_anchor(detection, anchor_kind).copy(),
                    velocity=np.zeros(2, dtype=float),
                    last_vector=detection.body_vector.copy(),
                    body_length=detection.body_length,
                    last_mouth=detection.mouth.copy(),
                    last_tail=detection.tail.copy(),
                )
            )
        return frame, states, tuple(assigned)
    raise RuntimeError("Could not find a clean three-fish seed frame")


def _update_state(state: TrackState, detection: Detection, frame: int, anchor_kind: str) -> None:
    anchor = _anchor(detection, anchor_kind)
    gap = max(1, frame - state.last_frame)
    observed_velocity = (anchor - state.last_anchor) / float(gap)
    state.velocity = 0.45 * state.velocity + 0.55 * observed_velocity
    state.last_anchor = anchor.copy()
    state.last_frame = frame
    if _finite_point(detection.body_vector):
        state.last_vector = detection.body_vector.copy()
    if np.isfinite(detection.body_length):
        state.body_length = detection.body_length
    state.last_mouth = detection.mouth.copy()
    state.last_tail = detection.tail.copy()


def _online_cost(
    state: TrackState,
    detection: Detection,
    *,
    frame: int,
    anchor_kind: str,
    vector_weight: float,
    axial_vector: bool,
) -> float:
    gap = max(1, frame - state.last_frame)
    anchor = _anchor(detection, anchor_kind)
    predicted = state.last_anchor + state.velocity * float(gap)
    position_error = float(np.linalg.norm(anchor - predicted))
    observed_velocity = (anchor - state.last_anchor) / float(gap)
    acceleration = float(np.linalg.norm(observed_velocity - state.velocity))
    velocity_turn = _angle(state.velocity, observed_velocity)
    body_turn = _angle(state.last_vector, detection.body_vector, axial=axial_vector)
    length_change = (
        abs(float(detection.body_length - state.body_length))
        if np.isfinite(detection.body_length) and np.isfinite(state.body_length)
        else 0.0
    )
    score_penalty = max(0.0, 0.8 - min(1.0, detection.score)) * 25.0
    return (
        position_error
        + 1.8 * acceleration
        + 2.5 * velocity_turn
        + vector_weight * body_turn
        + 0.08 * length_change
        + score_penalty
    )


def track_online_hungarian(
    detections: dict[int, list[Detection]],
    seed_frame: int,
    seed_states: list[TrackState],
    seed_assignment: tuple[int, int, int],
    *,
    anchor_kind: str,
    vector_weight: float = 0.0,
    axial_vector: bool = False,
    max_prediction_error: float = 80.0,
) -> dict[int, tuple[int | None, int | None, int | None]]:
    states = [state.copy() for state in seed_states]
    history: dict[int, tuple[int | None, int | None, int | None]] = {
        seed_frame: seed_assignment
    }
    for frame in range(seed_frame + 1, max(detections) + 1):
        frame_detections = detections.get(frame, [])
        assignment: list[int | None] = [None, None, None]
        if frame_detections:
            cost = np.asarray(
                [
                    [
                        _online_cost(
                            state,
                            detection,
                            frame=frame,
                            anchor_kind=anchor_kind,
                            vector_weight=vector_weight,
                            axial_vector=axial_vector,
                        )
                        for detection in frame_detections
                    ]
                    for state in states
                ],
                dtype=float,
            )
            rows, cols = linear_sum_assignment(cost)
            for fish_index, detection_index in zip(rows, cols):
                gap = max(1, frame - states[int(fish_index)].last_frame)
                anchor = _anchor(frame_detections[int(detection_index)], anchor_kind)
                predicted = states[int(fish_index)].last_anchor + states[int(fish_index)].velocity * gap
                if float(np.linalg.norm(anchor - predicted)) > max_prediction_error * gap:
                    continue
                assignment[int(fish_index)] = int(detection_index)
                _update_state(
                    states[int(fish_index)],
                    frame_detections[int(detection_index)],
                    frame,
                    anchor_kind,
                )
        history[frame] = tuple(assignment)
    return history


def _assignments_for_detection_count(count: int) -> list[tuple[int | None, int | None, int | None]]:
    if count <= 0:
        return [(None, None, None)]
    assigned_count = min(3, count)
    output: list[tuple[int | None, int | None, int | None]] = []
    for fish_subset in itertools.combinations(range(3), assigned_count):
        for detection_perm in itertools.permutations(range(count), assigned_count):
            assignment: list[int | None] = [None, None, None]
            for fish_index, detection_index in zip(fish_subset, detection_perm):
                assignment[fish_index] = detection_index
            output.append(tuple(assignment))
    return output


def track_vector_beam(
    detections: dict[int, list[Detection]],
    seed_frame: int,
    seed_states: list[TrackState],
    seed_assignment: tuple[int, int, int],
    *,
    axial_vector: bool,
    beam_width: int = 256,
) -> dict[int, tuple[int | None, int | None, int | None]]:
    beam = [
        BeamHypothesis(
            cost=0.0,
            tracks=[state.copy() for state in seed_states],
            history=[seed_assignment],
        )
    ]
    for frame in range(seed_frame + 1, max(detections) + 1):
        frame_detections = detections.get(frame, [])
        choices = _assignments_for_detection_count(len(frame_detections))
        expanded: list[BeamHypothesis] = []
        for hypothesis in beam:
            for assignment in choices:
                tracks = [track.copy() for track in hypothesis.tracks]
                transition_cost = 0.0
                for fish_index, detection_index in enumerate(assignment):
                    if detection_index is None:
                        transition_cost += 9.0
                        continue
                    detection = frame_detections[detection_index]
                    transition_cost += _online_cost(
                        tracks[fish_index],
                        detection,
                        frame=frame,
                        anchor_kind="midpoint",
                        vector_weight=11.0,
                        axial_vector=axial_vector,
                    )
                    _update_state(tracks[fish_index], detection, frame, "midpoint")
                expanded.append(
                    BeamHypothesis(
                        cost=hypothesis.cost + transition_cost,
                        tracks=tracks,
                        history=hypothesis.history + [assignment],
                    )
                )
        expanded.sort(key=lambda hypothesis: hypothesis.cost)
        beam = expanded[:beam_width]
    best = beam[0]
    return {
        frame: assignment
        for frame, assignment in zip(range(seed_frame, max(detections) + 1), best.history)
    }


def history_to_rows(
    method: str,
    history: dict[int, tuple[int | None, int | None, int | None]],
    detections: dict[int, list[Detection]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame, assignment in history.items():
        for fish_index, detection_index in enumerate(assignment):
            if detection_index is None:
                continue
            detection = detections[frame][detection_index]
            rows.append(
                {
                    "method": method,
                    "frame": frame,
                    "fish_id": FISH_IDS[fish_index],
                    "detection_index": detection.index,
                    "mouth_x": detection.mouth[0],
                    "mouth_y": detection.mouth[1],
                    "tail_x": detection.tail[0],
                    "tail_y": detection.tail[1],
                    "x": detection.midpoint[0],
                    "y": detection.midpoint[1],
                    "body_vector_x": detection.body_vector[0],
                    "body_vector_y": detection.body_vector[1],
                    "body_length_px": detection.body_length,
                    "score": detection.score,
                }
            )
    return rows


def add_current_comparison(method_rows: pd.DataFrame, current_top: pd.DataFrame) -> pd.DataFrame:
    current = current_top[["frame", "fish_id", "x", "y", "tracklet_node"]].rename(
        columns={"x": "current_x", "y": "current_y"}
    )
    compared = method_rows.merge(current, on=["frame", "fish_id"], how="left")
    compared["distance_to_current_px"] = np.hypot(
        compared["x"] - compared["current_x"],
        compared["y"] - compared["current_y"],
    )
    compared["same_detection_as_current"] = compared["distance_to_current_px"] <= 1e-3
    return compared


def plot_trajectory_comparison(
    compared: pd.DataFrame,
    current_top: pd.DataFrame,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(compared["method"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    current_window = current_top[current_top["frame"].between(frame_start, frame_end)]
    all_x = list(current_window["x"])
    all_y = list(current_window["y"])
    for ax, method in zip(axes.ravel(), methods):
        method_data = compared[
            compared["method"].eq(method) & compared["frame"].between(frame_start, frame_end)
        ]
        all_x.extend(method_data["x"].tolist())
        all_y.extend(method_data["y"].tolist())
        for fish_id in FISH_IDS:
            current_id = current_window[current_window["fish_id"].eq(fish_id)].sort_values("frame")
            method_id = method_data[method_data["fish_id"].eq(fish_id)].sort_values("frame")
            ax.plot(
                current_id["x"], current_id["y"], color=COLORS[fish_id], alpha=0.32,
                lw=2.0, ls="--",
            )
            ax.plot(
                method_id["x"], method_id["y"], color=COLORS[fish_id], lw=1.7,
                marker=MARKERS[fish_id], ms=3.2, label=fish_id,
            )
            for event_frame in (11265, 11275, 11277, 11285, 11288):
                event = method_id[method_id["frame"].eq(event_frame)]
                if not event.empty:
                    row = event.iloc[0]
                    ax.annotate(
                        str(event_frame), (row.x, row.y), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=COLORS[fish_id],
                    )
        ax.set_title(method.replace("_", " "), loc="left", fontweight="bold")
        ax.grid(color="#D9DDE3", lw=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()
    if all_x and all_y:
        x_min, x_max = min(all_x) - 8, max(all_x) + 8
        y_min, y_max = min(all_y) - 8, max(all_y) + 8
        for ax in axes.ravel():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel("Top-camera x (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-camera y (px)")
    fig.suptitle("Top-only SLEAP trajectories vs current tracked IDs", x=0.07, y=0.985, ha="left", fontsize=16)
    fig.text(
        0.07, 0.958,
        f"Frames {frame_start}–{frame_end}; solid = top-only method, faint dashed = current best with the same ID label",
        ha="left", color="#4C5663", fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.925))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_disagreement(
    compared: pd.DataFrame,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(compared["method"].drop_duplicates())
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 11), sharex=True)
    for ax, method in zip(axes, methods):
        method_data = compared[
            compared["method"].eq(method) & compared["frame"].between(frame_start, frame_end)
        ]
        for fish_id in FISH_IDS:
            sub = method_data[method_data["fish_id"].eq(fish_id)]
            ax.plot(
                sub["frame"], sub["distance_to_current_px"], color=COLORS[fish_id],
                marker=MARKERS[fish_id], ms=3.5, lw=1.4, label=fish_id,
            )
        ax.axhline(5, color="#555555", lw=0.8, ls=(0, (4, 3)))
        ax.set_ylabel("Δ px")
        ax.set_title(method.replace("_", " "), loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#D9DDE3", lw=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Source frame")
    axes[-1].set_xlim(frame_start - 0.5, frame_end + 0.5)
    axes[-1].set_xticks(np.arange(frame_start, frame_end + 1, 2))
    fig.suptitle("Distance from each top-only trajectory to the current same-ID position", x=0.08, y=0.985, ha="left", fontsize=16)
    fig.text(
        0.08, 0.957,
        f"Frames {frame_start}–{frame_end}; values near zero mean identical raw SLEAP detection assignments",
        ha="left", color="#4C5663", fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.925))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_permutation_timeline(
    compared: pd.DataFrame,
    current_top: pd.DataFrame,
    *,
    frame_start: int,
    frame_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_window = current_top[current_top["frame"].between(frame_start, frame_end)]
    for method in compared["method"].drop_duplicates():
        method_window = compared[
            compared["method"].eq(method) & compared["frame"].between(frame_start, frame_end)
        ]
        for frame in range(frame_start, frame_end + 1):
            observed = method_window[method_window["frame"].eq(frame)].set_index("fish_id")
            current = current_window[current_window["frame"].eq(frame)].set_index("fish_id")
            if not all(fish_id in observed.index and fish_id in current.index for fish_id in FISH_IDS):
                state = "incomplete"
                permutation = ""
            else:
                method_points = observed.loc[list(FISH_IDS), ["x", "y"]].to_numpy(dtype=float)
                current_points = current.loc[list(FISH_IDS), ["x", "y"]].to_numpy(dtype=float)
                cost = np.linalg.norm(method_points[:, None, :] - current_points[None, :, :], axis=2)
                row_index, col_index = linear_sum_assignment(cost)
                mapping = [-1, -1, -1]
                for left, right in zip(row_index, col_index):
                    mapping[int(left)] = int(right)
                permutation = ",".join(FISH_IDS[index] for index in mapping)
                if mapping == [0, 1, 2]:
                    state = "same"
                elif mapping == [1, 0, 2]:
                    state = "ID_1_ID_2_swapped"
                else:
                    state = "other_permutation"
            rows.append(
                {"method": method, "frame": frame, "state": state, "permutation": permutation}
            )
    return pd.DataFrame(rows)


def plot_assignment_timeline(
    timeline: pd.DataFrame,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    methods = list(timeline["method"].drop_duplicates())
    state_code = {"incomplete": 0, "same": 1, "ID_1_ID_2_swapped": 2, "other_permutation": 3}
    matrix = np.zeros((len(methods), frame_end - frame_start + 1), dtype=int)
    for method_index, method in enumerate(methods):
        sub = timeline[timeline["method"].eq(method)].set_index("frame")
        for frame in range(frame_start, frame_end + 1):
            matrix[method_index, frame - frame_start] = state_code[str(sub.loc[frame, "state"])]

    colors = ["#E5E7EB", "#6B9AC4", "#D17A22", "#9B5F9B"]
    labels = ["incomplete", "same as current", "ID_1 ↔ ID_2 vs current", "other permutation"]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5)
    ax.set_yticks(np.arange(len(methods)), [method.replace("_", " ") for method in methods])
    ticks = np.arange(frame_start, frame_end + 1, 2)
    ax.set_xticks(ticks - frame_start, ticks)
    ax.set_xlabel("Source frame")
    ax.spines[:].set_visible(False)
    ax.set_title("Top-only assignment permutation relative to current tracked IDs", loc="left", fontsize=15, pad=34)
    ax.text(
        0.0, 1.08,
        "Orange means the method assigns the two raw detections to ID_1 and ID_2 in the opposite order from the current pipeline",
        transform=ax.transAxes, color="#4C5663", fontsize=10,
    )
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, labels)],
        frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.22),
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.94))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_summary(
    compared: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    frame_start: int,
    frame_end: int,
    seed_frame: int,
) -> dict[str, object]:
    window = compared[compared["frame"].between(frame_start, frame_end)]
    methods: dict[str, object] = {}
    for method, sub in window.groupby("method", sort=False):
        finite = sub.dropna(subset=["distance_to_current_px"])
        methods[method] = {
            "rows": int(len(sub)),
            "same_detection_fraction": float(finite["same_detection_as_current"].mean()),
            "median_distance_to_current_px": float(finite["distance_to_current_px"].median()),
            "max_distance_to_current_px": float(finite["distance_to_current_px"].max()),
            "disagreement_frames": sorted(
                int(value)
                for value in finite.loc[~finite["same_detection_as_current"], "frame"].unique()
            ),
            "by_fish_id": {
                fish_id: {
                    "rows": int(len(fish_rows)),
                    "same_detection_fraction": float(fish_rows["same_detection_as_current"].mean()),
                    "max_distance_to_current_px": float(fish_rows["distance_to_current_px"].max()),
                }
                for fish_id, fish_rows in finite.groupby("fish_id", sort=True)
            },
            "permutation_state_frames": {
                state: sorted(int(value) for value in state_rows["frame"].tolist())
                for state, state_rows in timeline[timeline["method"].eq(method)].groupby("state", sort=True)
            },
        }

    method_names = list(window["method"].drop_duplicates())
    pairwise: dict[str, object] = {}
    for left, right in itertools.combinations(method_names, 2):
        a = window[window["method"].eq(left)][["frame", "fish_id", "detection_index"]]
        b = window[window["method"].eq(right)][["frame", "fish_id", "detection_index"]]
        merged = a.merge(b, on=["frame", "fish_id"], suffixes=("_left", "_right"))
        pairwise[f"{left}__vs__{right}"] = {
            "comparable_rows": int(len(merged)),
            "same_detection_fraction": float(
                (merged["detection_index_left"] == merged["detection_index_right"]).mean()
            ),
        }
    return {"seed_frame": seed_frame, "methods": methods, "pairwise": pairwise}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slp", required=True, type=Path)
    parser.add_argument("--best-json", required=True, type=Path)
    parser.add_argument("--dispositions-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context-start", type=int, default=11220)
    parser.add_argument("--frame-start", type=int, default=11260)
    parser.add_argument("--frame-end", type=int, default=11288)
    parser.add_argument("--context-end", type=int, default=11310)
    parser.add_argument("--beam-width", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections = load_sleap_detections(
        args.slp, frame_start=args.context_start, frame_end=args.context_end
    )
    current = selected_2d_positions(args.best_json, args.dispositions_csv)
    current_top = current[current["view"].eq("top_cam")].copy()

    midpoint_seed_frame, midpoint_states, seed_assignment = seed_tracks(
        detections,
        current_top,
        requested_frame=args.context_start,
        anchor_kind="midpoint",
    )
    mouth_seed_frame, mouth_states, mouth_seed_assignment = seed_tracks(
        detections,
        current_top,
        requested_frame=midpoint_seed_frame,
        anchor_kind="mouth",
    )
    if mouth_seed_frame != midpoint_seed_frame:
        raise RuntimeError("Mouth and midpoint methods did not share a seed frame")

    histories = {
        "mouth_cv_hungarian": track_online_hungarian(
            detections,
            mouth_seed_frame,
            mouth_states,
            mouth_seed_assignment,
            anchor_kind="mouth",
        ),
        "midpoint_cv_hungarian": track_online_hungarian(
            detections,
            midpoint_seed_frame,
            midpoint_states,
            seed_assignment,
            anchor_kind="midpoint",
        ),
        "body_vector_directed_beam": track_vector_beam(
            detections,
            midpoint_seed_frame,
            midpoint_states,
            seed_assignment,
            axial_vector=False,
            beam_width=args.beam_width,
        ),
        "body_vector_axial_beam": track_vector_beam(
            detections,
            midpoint_seed_frame,
            midpoint_states,
            seed_assignment,
            axial_vector=True,
            beam_width=args.beam_width,
        ),
    }
    rows: list[dict[str, object]] = []
    for method, history in histories.items():
        rows.extend(history_to_rows(method, history, detections))
    method_rows = pd.DataFrame(rows)
    compared = add_current_comparison(method_rows, current_top)
    compared.to_csv(args.output_dir / "top_only_method_comparison.csv", index=False)
    timeline = build_permutation_timeline(
        compared,
        current_top,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    timeline.to_csv(args.output_dir / "top_only_assignment_timeline.csv", index=False)

    plot_trajectory_comparison(
        compared,
        current_top,
        args.output_dir / "top_only_trajectory_comparison.png",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    plot_disagreement(
        compared,
        args.output_dir / "top_only_distance_from_current.png",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    plot_assignment_timeline(
        timeline,
        args.output_dir / "top_only_assignment_timeline.png",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    summary = build_summary(
        compared,
        timeline,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        seed_frame=midpoint_seed_frame,
    )
    summary["configuration"] = {
        "context_start": args.context_start,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "context_end": args.context_end,
        "beam_width": args.beam_width,
        "identity_seed_note": (
            "Current IDs were used only to name raw detections on the seed frame; "
            "all subsequent assignments use top-camera SLEAP nodes only."
        ),
    }
    (args.output_dir / "top_only_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] seed_frame={midpoint_seed_frame} wrote {args.output_dir}")


if __name__ == "__main__":
    main()
