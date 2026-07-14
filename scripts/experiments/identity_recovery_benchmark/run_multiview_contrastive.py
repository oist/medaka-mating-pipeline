from __future__ import annotations

"""Run a frozen multi-view appearance linker on locked crossing episodes.

This method never trains identity classes.  A geometry-supervised contrastive
encoder was fit on disjoint historical frames.  At each evaluation crossing it:

1. associates detections to the three same-frame 3D candidates in each camera;
2. pools pre/post appearance prototypes with equal weight per available camera;
3. solves one one-to-one pre/post permutation; and
4. propagates the resulting anonymous identities through the top-view detections.

The 3D slot name is not treated as a biological identity.  It is only a
same-frame cross-camera association key and may be permuted after every crossing.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .common import _frame_assignments, load_reference, load_tracks_csv


@dataclass(frozen=True)
class Episode:
    episode_id: str
    start_frame: int
    end_frame: int
    center_frame: int
    event_ids: tuple[str, ...]


def merge_events(events: pd.DataFrame, *, maximum_gap_frames: int) -> list[Episode]:
    rows = events.sort_values("start_frame").to_dict("records")
    if not rows:
        return []
    groups: list[list[dict]] = [[rows[0]]]
    for row in rows[1:]:
        if int(row["start_frame"]) - int(groups[-1][-1]["end_frame"]) <= maximum_gap_frames:
            groups[-1].append(row)
        else:
            groups.append([row])
    episodes = []
    for index, group in enumerate(groups, start=1):
        start = min(int(row["start_frame"]) for row in group)
        end = max(int(row["end_frame"]) for row in group)
        episodes.append(
            Episode(
                episode_id=f"episode_{index:02d}",
                start_frame=start,
                end_frame=end,
                center_frame=(start + end) // 2,
                event_ids=tuple(str(row["event_id"]) for row in group),
            )
        )
    return episodes


def solve_episode_permutation(
    pre_prototypes: dict[str, np.ndarray],
    post_prototypes: dict[str, np.ndarray],
) -> tuple[dict[str, str], np.ndarray, list[str], list[str]]:
    """Return pre-slot -> post-slot Hungarian assignment by cosine similarity."""
    pre_slots = sorted(pre_prototypes)
    post_slots = sorted(post_prototypes)
    if not pre_slots or len(pre_slots) != len(post_slots):
        return {}, np.empty((len(pre_slots), len(post_slots))), pre_slots, post_slots
    pre = np.stack([pre_prototypes[key] for key in pre_slots])
    post = np.stack([post_prototypes[key] for key in post_slots])
    pre /= np.maximum(np.linalg.norm(pre, axis=1, keepdims=True), 1e-12)
    post /= np.maximum(np.linalg.norm(post, axis=1, keepdims=True), 1e-12)
    similarities = pre @ post.T
    left, right = linear_sum_assignment(-similarities)
    mapping = {pre_slots[i]: post_slots[j] for i, j in zip(left, right)}
    return mapping, similarities, pre_slots, post_slots


def _crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    half = size // 2
    center_x, center_y = int(round(x)), int(round(y))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT)
    shifted_x, shifted_y = center_x + half, center_y + half
    crop = padded[shifted_y - half : shifted_y + half, shifted_x - half : shifted_x + half]
    if crop.shape[:2] != (size, size):
        raise RuntimeError(f"unexpected crop shape {crop.shape}")
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _load_embedder(checkpoint_path: Path, device: str):
    import torch

    from sam2_id_experiment.src.id_model.geometry_contrastive import GeometryContrastiveEmbedder
    from sam2_id_experiment.src.id_model.model import SmallMedakaEmbeddingNet
    from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model_state/state_dict")
    architecture = str(checkpoint.get("architecture", checkpoint.get("arch", "")))
    classifier_state = any(key.startswith("embedder.") for key in state)
    if architecture == "resnet18_pair_embedding" or any(
        key.startswith("features.") or key.startswith("embedder.features.") for key in state
    ):
        model = GeometryContrastiveEmbedder(
            embedding_dim=int(checkpoint.get("embedding_dim", 128)),
            pretrained=False,
            train_backbone=bool(checkpoint.get("train_backbone", False)),
            normalize=bool(checkpoint.get("normalize_embeddings", True)),
        ).to(device)
        if classifier_state:
            state = {
                key.removeprefix("embedder."): value
                for key, value in state.items()
                if key.startswith("embedder.")
            }
        architecture = "resnet18_pair_embedding"
    elif architecture.startswith("resnet18") or any("net.layer4" in key for key in state):
        model = ResNet18Embedder(
            embedding_dim=int(checkpoint.get("embedding_dim", 128)), pretrained=False
        ).to(device)
        if classifier_state:
            state = {
                key.removeprefix("embedder."): value
                for key, value in state.items()
                if key.startswith("embedder.")
            }
        architecture = "resnet18_embedder"
    else:
        model = SmallMedakaEmbeddingNet(
            embedding_dim=int(checkpoint.get("embedding_dim", 64))
        ).to(device)
        if classifier_state:
            state = {
                key.removeprefix("embedder."): value
                for key, value in state.items()
                if key.startswith("embedder.")
            }
        architecture = "small_medaka_embedding_net"
    model.load_state_dict(state)
    model.eval()
    return model, architecture


def _infer(crops: np.ndarray, checkpoint: Path, *, device: str, batch_size: int):
    import torch

    model, architecture = _load_embedder(checkpoint, device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            array = crops[start : start + batch_size].astype(np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(device)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs, axis=0), architecture


def _matched_detection_rows(
    tracks: pd.DataFrame, reference: pd.DataFrame, *, gate_px: float
) -> pd.DataFrame:
    matches = _frame_assignments(tracks, reference, gate_px=gate_px)
    return matches.merge(
        tracks[["frame", "track_id", "x", "y"]],
        on=["frame", "track_id"],
        how="left",
        validate="one_to_one",
    )


def _sample_rows(
    matched_by_view: dict[str, pd.DataFrame],
    episodes: list[Episode],
    *,
    prototype_frames: int,
    exclusion_frames: int,
    sample_every: int,
) -> pd.DataFrame:
    rows = []
    for episode in episodes:
        intervals = {
            "pre": (
                episode.start_frame - prototype_frames,
                episode.start_frame - exclusion_frames - 1,
            ),
            "post": (
                episode.end_frame + exclusion_frames + 1,
                episode.end_frame + prototype_frames,
            ),
        }
        for view, table in matched_by_view.items():
            for phase, (start, end) in intervals.items():
                subset = table[(table["frame"] >= start) & (table["frame"] <= end)].copy()
                subset = subset[((subset["frame"] - start) % sample_every) == 0]
                for _, row in subset.iterrows():
                    rows.append(
                        {
                            "episode_id": episode.episode_id,
                            "phase": phase,
                            "view": view,
                            "frame": int(row["frame"]),
                            "geometry_slot": str(row["fish_id"]),
                            "source_track_id": str(row["track_id"]),
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                        }
                    )
    return pd.DataFrame(rows)


def _extract_crops(
    samples: pd.DataFrame,
    video_by_view: dict[str, Path],
    *,
    video_frame_zero: int,
    crop_size: int,
) -> np.ndarray:
    crops: list[np.ndarray | None] = [None] * len(samples)
    for view, view_rows in samples.groupby("view"):
        capture = cv2.VideoCapture(str(video_by_view[view]))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {video_by_view[view]}")
        by_frame = {int(frame): group for frame, group in view_rows.groupby("frame")}
        needed = sorted(by_frame)
        if not needed:
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, needed[0] - int(video_frame_zero))
        needed_set = set(needed)
        try:
            for absolute_frame in range(needed[0], needed[-1] + 1):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"read failed for {view} frame {absolute_frame}")
                if absolute_frame not in needed_set:
                    continue
                for index, row in by_frame[absolute_frame].iterrows():
                    crops[int(index)] = _crop(image, float(row["x"]), float(row["y"]), crop_size)
        finally:
            capture.release()
    if any(crop is None for crop in crops):
        raise RuntimeError("some requested crops were not extracted")
    return np.stack(crops)  # type: ignore[arg-type]


def _balanced_prototypes(samples: pd.DataFrame, embeddings: np.ndarray):
    table = samples.copy()
    table["embedding_index"] = np.arange(len(table))
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (episode_id, phase, slot, view), group in table.groupby(
        ["episode_id", "phase", "geometry_slot", "view"]
    ):
        vector = embeddings[group["embedding_index"].to_numpy(dtype=int)].mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        out.setdefault((str(episode_id), str(phase)), {}).setdefault(str(slot), []).append(vector)
    balanced = {}
    for key, slot_vectors in out.items():
        balanced[key] = {}
        for slot, vectors in slot_vectors.items():
            vector = np.stack(vectors).mean(axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            balanced[key][slot] = vector
    return balanced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        action="append",
        nargs=4,
        metavar=("NAME", "VIDEO", "TRACKS_CSV", "REFERENCE_CSV"),
        required=True,
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--video-frame-zero", type=int, default=0)
    parser.add_argument("--top-view", default="top_cam")
    parser.add_argument("--association-gate-px", type=float, default=80.0)
    parser.add_argument("--merge-event-gap-frames", type=int, default=100)
    parser.add_argument("--prototype-frames", type=int, default=80)
    parser.add_argument("--exclusion-frames", type=int, default=15)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-threshold-cosine", type=float, default=0.8697548758603217)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_by_view: dict[str, Path] = {}
    tracks_by_view: dict[str, pd.DataFrame] = {}
    reference_by_view: dict[str, pd.DataFrame] = {}
    matched_by_view: dict[str, pd.DataFrame] = {}
    for name, video, tracks, reference in args.view:
        video_by_view[name] = Path(video)
        tracks_by_view[name] = load_tracks_csv(tracks)
        reference_by_view[name] = load_reference(reference)
        matched_by_view[name] = _matched_detection_rows(
            tracks_by_view[name], reference_by_view[name], gate_px=args.association_gate_px
        )
    if args.top_view not in tracks_by_view:
        raise ValueError(f"top view {args.top_view!r} not supplied")

    events = pd.read_csv(args.events)
    episodes = merge_events(events, maximum_gap_frames=args.merge_event_gap_frames)
    samples = _sample_rows(
        matched_by_view,
        episodes,
        prototype_frames=args.prototype_frames,
        exclusion_frames=args.exclusion_frames,
        sample_every=args.sample_every,
    ).reset_index(drop=True)
    if samples.empty:
        raise ValueError("no prototype samples were available")
    crops = _extract_crops(
        samples,
        video_by_view,
        video_frame_zero=args.video_frame_zero,
        crop_size=args.crop_size,
    )
    embeddings, architecture = _infer(
        crops, args.checkpoint, device=args.device, batch_size=args.batch_size
    )
    embedding_columns = [f"embedding_{index:03d}" for index in range(embeddings.shape[1])]
    pd.concat(
        [samples, pd.DataFrame(embeddings, columns=embedding_columns)], axis=1
    ).to_parquet(args.output_dir / "prototype_crop_embeddings.parquet", index=False)
    prototypes = _balanced_prototypes(samples, embeddings)

    top_reference = reference_by_view[args.top_view]
    slots = sorted(top_reference["fish_id"].astype(str).unique())
    identity_by_slot = {slot: f"I{index}" for index, slot in enumerate(slots, start=1)}
    states: list[tuple[int, dict[str, str]]] = [(args.start_frame, dict(identity_by_slot))]
    episode_rows = []
    for episode in episodes:
        pre = prototypes.get((episode.episode_id, "pre"), {})
        post = prototypes.get((episode.episode_id, "post"), {})
        permutation, similarities, pre_slots, post_slots = solve_episode_permutation(pre, post)
        complete = len(permutation) == len(slots) and set(permutation) == set(slots)
        selected = []
        if complete:
            new_identity_by_slot: dict[str, str] = {}
            for pre_slot, post_slot in permutation.items():
                new_identity_by_slot[post_slot] = identity_by_slot[pre_slot]
                i, j = pre_slots.index(pre_slot), post_slots.index(post_slot)
                selected.append(float(similarities[i, j]))
            identity_by_slot = new_identity_by_slot
        confidence = min(selected) if selected else float("nan")
        decision_frame = episode.center_frame
        states.append((decision_frame, dict(identity_by_slot)))
        episode_rows.append(
            {
                "episode_id": episode.episode_id,
                "event_ids": "|".join(episode.event_ids),
                "start_frame": episode.start_frame,
                "end_frame": episode.end_frame,
                "decision_frame": decision_frame,
                "pre_to_post_slot_permutation": json.dumps(permutation, sort_keys=True),
                "assignment_complete": bool(complete),
                "minimum_selected_cosine": confidence,
                "passes_calibration_eer_threshold": bool(
                    np.isfinite(confidence) and confidence >= args.calibration_threshold_cosine
                ),
                "similarity_matrix": json.dumps(similarities.tolist()),
                "pre_slots": json.dumps(pre_slots),
                "post_slots": json.dumps(post_slots),
            }
        )
    episode_table = pd.DataFrame(episode_rows)
    episode_table.to_csv(args.output_dir / "episode_decisions.csv", index=False)

    top_matched = matched_by_view[args.top_view].copy()
    state_index = 0
    output_rows = []
    for _, row in top_matched.sort_values("frame").iterrows():
        frame = int(row["frame"])
        while state_index + 1 < len(states) and frame >= states[state_index + 1][0]:
            state_index += 1
        mapping = states[state_index][1]
        slot = str(row["fish_id"])
        if slot not in mapping:
            continue
        output_rows.append(
            {
                "frame": frame,
                "track_id": mapping[slot],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "source_top_track_id": str(row["track_id"]),
                "geometry_slot_local": slot,
            }
        )
    output = pd.DataFrame(output_rows)
    output = output[(output["frame"] >= args.start_frame) & (output["frame"] <= args.end_frame)]
    output.to_csv(args.output_dir / "multiview_constrained_tracks.csv", index=False)
    samples.to_csv(args.output_dir / "prototype_samples.csv", index=False)

    summary = {
        "schema_version": "multiview_constrained_contrastive_v1",
        "identity_semantics": (
            "anonymous identities initialized arbitrarily; geometry slots are same-frame "
            "cross-view keys and may be permuted at every crossing"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": architecture,
        "training_overlap_with_evaluation": "none by declared frame windows",
        "evaluation_interval": [args.start_frame, args.end_frame],
        "views": sorted(video_by_view),
        "episodes": len(episodes),
        "prototype_crops": int(len(samples)),
        "output_track_rows": int(len(output)),
        "output_frames": int(output["frame"].nunique()),
        "calibration_threshold_cosine": args.calibration_threshold_cosine,
        "confident_episodes": int(episode_table["passes_calibration_eer_threshold"].sum()),
        "caveat": (
            "shared cross-view encoder; view-specific top/side encoders are an ablation, "
            "not silently substituted into this frozen primary run"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()

