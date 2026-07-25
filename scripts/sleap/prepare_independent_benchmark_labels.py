"""Prepare a blank, event-disjoint multiview SLEAP annotation package.

The package contains synchronized raw-video frame references but no instances,
predictions, or keypoint coordinates. Round 7 labels and the complete Round 8
selection manifest are treated as exposure sources and guarded in time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths


VERSION = "independent40-v1"
DEFAULT_SEED = 20260724
DEFAULT_EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_FILES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


def stable_hex(*parts: object) -> str:
    """Return a stable SHA-256 digest for deterministic ordering."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_quotas(total: int, experiments: Iterable[str]) -> dict[str, int]:
    names = list(experiments)
    if total < len(names):
        raise ValueError("total frames must be at least the number of experiments")
    base, remainder = divmod(total, len(names))
    return {
        experiment: base + (1 if index < remainder else 0)
        for index, experiment in enumerate(names)
    }


def original_video_path(video) -> str:
    """Return the deepest source video path from an embedded SLP chain."""
    seen: set[int] = set()
    current = video
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        source = getattr(current, "source_video", None)
        if source is None:
            break
        current = source
    return str(getattr(current, "filename", "") or "")


def experiment_from_path(path: str, experiments: Iterable[str]) -> str | None:
    normalized = str(path).replace("\\", "/")
    for experiment in experiments:
        if f"/{experiment}/" in normalized:
            return experiment
    return None


def collect_round7_exposures(
    round7_dir: Path,
    experiments: Iterable[str],
) -> tuple[dict[str, set[int]], dict[str, dict[str, int | str]]]:
    """Collect every source frame present in the Round 7 top/side packages."""
    import sleap_io as sio

    experiment_names = tuple(experiments)
    exposed: dict[str, set[int]] = defaultdict(set)
    packages: dict[str, dict[str, int | str]] = {}
    for family in ("top", "side"):
        package = round7_dir / family / "labels.pkg.slp"
        if not package.is_file():
            raise FileNotFoundError(f"Round 7 label package not found: {package}")
        labels = sio.load_slp(str(package), open_videos=False)
        matched = 0
        for labeled_frame in labels.labeled_frames:
            experiment = experiment_from_path(
                original_video_path(labeled_frame.video),
                experiment_names,
            )
            if experiment is None:
                continue
            exposed[experiment].add(int(labeled_frame.frame_idx))
            matched += 1
        packages[family] = {
            "path": str(package),
            "labeled_frames": len(labels.labeled_frames),
            "matched_exposure_rows": matched,
            "size_bytes": package.stat().st_size,
        }
    return dict(exposed), packages


def add_round8_selection_exposures(
    exposed: dict[str, set[int]],
    selection_manifest: Path,
    experiments: Iterable[str],
) -> dict[str, int]:
    table = pd.read_csv(selection_manifest, keep_default_na=False)
    # Round 8 was finalized from the pilot only. Expansion rows were candidate
    # frames that were never corrected or included in train/validation SLPs;
    # treating them as model exposure would incorrectly eliminate the future
    # locked-test queue derived from those untouched events.
    if "annotation_batch" in table.columns:
        table = table.loc[table["annotation_batch"].astype(str).eq("pilot")]
    counts: dict[str, int] = {}
    wanted = set(experiments)
    for experiment, group in table.groupby("experiment"):
        if experiment not in wanted:
            continue
        frames = {int(frame) for frame in group["frame"]}
        exposed.setdefault(experiment, set()).update(frames)
        counts[experiment] = len(frames)
    return counts


def expanded_interval(row: dict[str, object], guard_frames: int) -> tuple[int, int]:
    start = int(row["event_start_frame"])
    end = int(row["event_end_frame"])
    return start - guard_frames, end + guard_frames


def intervals_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def interval_is_unexposed(
    row: dict[str, object],
    exposed_frames: set[int],
    guard_frames: int,
) -> bool:
    start, end = expanded_interval(row, guard_frames)
    return not any(start <= frame <= end for frame in exposed_frames)


def select_future_queue_events(
    queue_path: Path,
    *,
    experiment: str,
    count: int,
    exposed_frames: set[int],
    guard_frames: int,
    seed: int,
) -> list[dict[str, object]]:
    """Select one representative from deterministic, non-overlapping events."""
    table = pd.read_csv(queue_path, keep_default_na=False)
    if "future_lock_eligible" in table:
        eligible = (
            table["future_lock_eligible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        table = table.loc[eligible]
    table = table.loc[
        table["experiment"].eq(experiment) & table["view"].eq("top_cam")
    ].copy()

    event_rows: list[dict[str, object]] = []
    for event_id, group in table.groupby("event_id", sort=True):
        start = int(group.iloc[0]["event_start_frame"])
        end = int(group.iloc[0]["event_end_frame"])
        midpoint = (start + end) / 2.0
        candidates = sorted(
            group.to_dict("records"),
            key=lambda row: (
                not (start <= int(row["frame_idx"]) <= end),
                abs(int(row["frame_idx"]) - midpoint),
                stable_hex(seed, event_id, row["frame_idx"]),
            ),
        )
        source = candidates[0]
        row = {
            "experiment": experiment,
            "frame_idx": int(source["frame_idx"]),
            "event_id": str(event_id),
            "event_start_frame": start,
            "event_end_frame": end,
            "difficulty_hint": str(source.get("difficulty", "unknown_until_annotation")),
            "source_category": str(source.get("source_category", "")),
            "selection_source": "future_lock_eligible_queue",
        }
        if interval_is_unexposed(row, exposed_frames, guard_frames):
            event_rows.append(row)

    event_rows.sort(key=lambda row: stable_hex(seed, experiment, row["event_id"]))
    selected: list[dict[str, object]] = []
    selected_intervals: list[tuple[int, int]] = []
    for row in event_rows:
        interval = expanded_interval(row, guard_frames)
        if any(intervals_overlap(interval, other) for other in selected_intervals):
            continue
        selected.append(row)
        selected_intervals.append(interval)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} non-overlapping eligible events found for "
            f"{experiment}; requested {count}"
        )
    return sorted(selected, key=lambda row: int(row["frame_idx"]))


def select_uniform_unexposed_frames(
    *,
    experiment: str,
    count: int,
    exposed_frames: set[int],
    total_frames: int,
    guard_frames: int,
    seed: int,
) -> list[dict[str, object]]:
    """Hash-select model-independent point events outside all exposure guards."""
    candidates: list[dict[str, object]] = []
    margin = max(guard_frames + 1, 1000)
    for frame in range(margin, total_frames - margin, guard_frames):
        row = {
            "experiment": experiment,
            "frame_idx": frame,
            "event_id": f"independent_point_{frame}",
            "event_start_frame": frame,
            "event_end_frame": frame,
            "difficulty_hint": "unknown_until_annotation",
            "source_category": "model_independent_uniform_time",
            "selection_source": "stable_hash_uniform_unexposed",
        }
        if interval_is_unexposed(row, exposed_frames, guard_frames):
            candidates.append(row)

    candidates.sort(
        key=lambda row: stable_hex(seed, experiment, row["frame_idx"])
    )
    selected: list[dict[str, object]] = []
    selected_intervals: list[tuple[int, int]] = []
    for row in candidates:
        interval = expanded_interval(row, guard_frames)
        if any(intervals_overlap(interval, other) for other in selected_intervals):
            continue
        selected.append(row)
        selected_intervals.append(interval)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} unexposed point events found for {experiment}; "
            f"requested {count}"
        )
    return sorted(selected, key=lambda row: int(row["frame_idx"]))


def nearest_exposure_distance(frame: int, exposed: set[int]) -> int | None:
    if not exposed:
        return None
    return min(abs(frame - value) for value in exposed)


def build_blank_package(
    manifest: pd.DataFrame,
    *,
    raw_data_dir: Path,
    output_path: Path,
    provenance: dict[str, object],
) -> dict[str, int]:
    """Write 3 empty labeled frames per synchronized source frame."""
    import sleap_io as sio

    skeleton = sio.Skeleton(
        nodes=["mouth", "tail"],
        edges=[("mouth", "tail")],
        name="medaka_mouth_tail",
    )
    videos = {}
    labeled_frames = []
    suggestions = []
    for row in manifest.itertuples(index=False):
        for view in VIEWS:
            video_path = (
                raw_data_dir
                / str(row.experiment)
                / view
                / VIEW_FILES[view]
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"raw video not found: {video_path}")
            video = videos.get(str(video_path))
            if video is None:
                video = sio.load_video(str(video_path))
                videos[str(video_path)] = video
            labeled_frame = sio.LabeledFrame(
                video=video,
                frame_idx=int(row.frame_idx),
                instances=[],
            )
            labeled_frames.append(labeled_frame)
            suggestions.append(
                sio.SuggestionFrame(video=video, frame_idx=int(row.frame_idx))
            )

    labels = sio.Labels(
        labeled_frames=labeled_frames,
        videos=list(videos.values()),
        skeletons=[skeleton],
        suggestions=suggestions,
        provenance=provenance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(output_path), embed=False)

    reloaded = sio.load_slp(str(output_path), open_videos=False)
    instance_count = sum(len(frame.instances) for frame in reloaded.labeled_frames)
    expected_frames = len(manifest) * len(VIEWS)
    if len(reloaded.labeled_frames) != expected_frames or instance_count != 0:
        raise RuntimeError(
            "blank-package verification failed: "
            f"{len(reloaded.labeled_frames)} frames, {instance_count} instances; "
            f"expected {expected_frames} frames and zero instances"
        )
    return {
        "source_frames": len(manifest),
        "view_frames": len(reloaded.labeled_frames),
        "instances": instance_count,
        "videos": len(reloaded.videos),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--guard-frames", type=int, default=166)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total-frames", type=int, default=348600)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--round7-dir", type=Path)
    parser.add_argument(
        "--round8-selection",
        type=Path,
        default=Path("report/20260721/round8_active_learning/selection_360.csv"),
    )
    parser.add_argument(
        "--future-queue",
        type=Path,
        default=Path(
            "report/20260724/sleap_topdown_optimization/"
            "future_locked_test_annotation_queue_eligible_v1.csv"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    local = load_local_paths(args.paths_config)
    if local is None or local.medaka_data_dir is None or local.sleap_models_dir is None:
        raise SystemExit(
            f"{args.paths_config} must define medaka_data_dir and sleap_models_dir"
        )
    round7_dir = args.round7_dir or (
        local.sleap_models_dir / "retrain_20260704"
    )
    experiments = tuple(args.experiments)
    quotas = balanced_quotas(args.num_frames, experiments)

    exposed, round7_packages = collect_round7_exposures(
        round7_dir,
        experiments,
    )
    round8_counts = add_round8_selection_exposures(
        exposed,
        args.round8_selection,
        experiments,
    )

    selected: list[dict[str, object]] = []
    for experiment in experiments:
        if experiment == "20251225_K_1_2_FCMC":
            rows = select_future_queue_events(
                args.future_queue,
                experiment=experiment,
                count=quotas[experiment],
                exposed_frames=exposed.get(experiment, set()),
                guard_frames=args.guard_frames,
                seed=args.seed,
            )
        else:
            rows = select_uniform_unexposed_frames(
                experiment=experiment,
                count=quotas[experiment],
                exposed_frames=exposed.get(experiment, set()),
                total_frames=args.total_frames,
                guard_frames=args.guard_frames,
                seed=args.seed,
            )
        for row in rows:
            row["nearest_exposed_frame_distance"] = nearest_exposure_distance(
                int(row["frame_idx"]),
                exposed.get(experiment, set()),
            )
            row["guard_frames"] = args.guard_frames
            row["split_role"] = "locked_test_candidate_pending_annotation"
            row["annotation_source"] = "blank_manual_independent"
            row["frame_key"] = "li1_" + stable_hex(
                VERSION,
                experiment,
                row["frame_idx"],
            )[:20]
        selected.extend(rows)

    selected.sort(key=lambda row: (str(row["experiment"]), int(row["frame_idx"])))
    manifest = pd.DataFrame(selected)
    if len(manifest) != args.num_frames or manifest["frame_key"].duplicated().any():
        raise RuntimeError("selection count/key uniqueness verification failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "independent_annotation_40_manifest.csv"
    seed_path = args.output_dir / "independent_annotation_40_blank.pkg.slp"
    provenance_path = args.output_dir / "independent_annotation_40_provenance.json"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")

    provenance: dict[str, object] = {
        "version": VERSION,
        "selection_seed": args.seed,
        "guard_frames": args.guard_frames,
        "selection_rule": (
            "K_1_2: one stable-hash-ranked future-lock-eligible frame per "
            "non-overlapping event; K_1_1/K_1_3: stable-hash-ranked uniform "
            "point events; all expanded event intervals exclude every Round 7 "
            "label and every Round 8 selection frame"
        ),
        "quotas": quotas,
        "round7_packages": round7_packages,
        "round8_selection": {
            "path": str(args.round8_selection.resolve()),
            "sha256": sha256_file(args.round8_selection),
            "unique_exposure_frames_by_experiment": round8_counts,
        },
        "future_queue": {
            "path": str(args.future_queue.resolve()),
            "sha256": sha256_file(args.future_queue),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "contains_model_predictions": False,
        "contains_keypoint_coordinates": False,
    }
    package_summary = build_blank_package(
        manifest,
        raw_data_dir=local.medaka_data_dir,
        output_path=seed_path,
        provenance=provenance,
    )
    provenance["blank_package"] = {
        "path": str(seed_path.resolve()),
        "sha256": sha256_file(seed_path),
        **package_summary,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "manifest": str(manifest_path),
            "blank_package": str(seed_path),
            "provenance": str(provenance_path),
            "quotas": quotas,
            **package_summary,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
