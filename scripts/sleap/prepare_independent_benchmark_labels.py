"""Prepare or finalize an event-disjoint multiview SLEAP annotation package.

The package contains synchronized raw-video frame references but no instances,
predictions, or keypoint coordinates. Round 7 labels and the Round 8
pilot/model-training rows are treated as exposure sources and guarded in time.
Expansion candidates that were never corrected or used for model training are
not treated as model exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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
LOCKED_SOURCE_FRAMES = 40
LOCKED_VIEW_FRAMES = LOCKED_SOURCE_FRAMES * len(VIEWS)
LOCKED_FISH_PER_VIEW = 3
LOCKED_LEDGER_COLUMNS = (
    "frame_key",
    "source_frame_order",
    "experiment",
    "frame_idx",
    "view",
    "fish_index",
    "mouth_x",
    "mouth_y",
    "tail_x",
    "tail_y",
)


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


def reserve_new_output_dir(path: Path) -> None:
    """Create a new output directory, refusing every pre-existing target."""
    if path.exists():
        raise FileExistsError(
            f"refusing existing output directory: {path}. "
            "Choose a new versioned directory so prior packages are immutable."
        )
    path.mkdir(parents=True, exist_ok=False)


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


def view_from_path(path: str) -> str | None:
    """Return the canonical camera view encoded in a raw-video path."""
    normalized = str(path).replace("\\", "/").lower()
    for view, filename in VIEW_FILES.items():
        if f"/{view}/" in normalized or normalized.endswith(f"/{filename}"):
            return view
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
    """Add only Round 8 pilot rows that could have influenced the model.

    Expansion rows remained uncorrected selection candidates and were not
    included in the finalized Round 8 training/validation packages. Treating
    them as model exposure would erase the untouched candidate pool.
    """
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


def validate_locked_annotation_labels(
    labels: Any,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate the exact locked selection and return its coordinate ledger."""
    required = {"frame_key", "experiment", "frame_idx"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            "locked manifest is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if len(manifest) != LOCKED_SOURCE_FRAMES:
        raise ValueError(
            f"locked manifest must contain {LOCKED_SOURCE_FRAMES} source frames, "
            f"found {len(manifest)}"
        )
    if manifest["frame_key"].astype(str).duplicated().any():
        raise ValueError("locked manifest contains duplicate frame_key values")
    source_keys = list(
        zip(
            manifest["experiment"].astype(str),
            manifest["frame_idx"].astype(int),
        )
    )
    if len(set(source_keys)) != LOCKED_SOURCE_FRAMES:
        raise ValueError(
            "locked manifest must contain 40 unique (experiment, frame_idx) rows"
        )

    if len(getattr(labels, "skeletons", [])) != 1:
        raise ValueError("locked package must contain exactly one skeleton")
    skeleton = labels.skeletons[0]
    node_names = [
        str(getattr(node, "name", node))
        for node in getattr(skeleton, "nodes", [])
    ]
    if "mouth" not in node_names or "tail" not in node_names:
        raise ValueError(
            "locked package skeleton must contain mouth and tail nodes"
        )
    mouth_index = node_names.index("mouth")
    tail_index = node_names.index("tail")

    package_frames: dict[tuple[str, int, str], Any] = {}
    experiment_names = tuple(manifest["experiment"].astype(str).unique())
    for labeled_frame in labels.labeled_frames:
        source_path = original_video_path(labeled_frame.video)
        experiment = experiment_from_path(source_path, experiment_names)
        view = view_from_path(source_path)
        if experiment is None or view is None:
            raise ValueError(
                "could not identify experiment/view for locked package video: "
                f"{source_path}"
            )
        key = (experiment, int(labeled_frame.frame_idx), view)
        if key in package_frames:
            raise ValueError(f"locked package contains duplicate view frame: {key}")
        package_frames[key] = labeled_frame

    expected_keys = {
        (experiment, int(frame_idx), view)
        for experiment, frame_idx in source_keys
        for view in VIEWS
    }
    actual_keys = set(package_frames)
    missing_keys = sorted(expected_keys.difference(actual_keys))
    extra_keys = sorted(actual_keys.difference(expected_keys))
    if missing_keys or extra_keys:
        raise ValueError(
            "locked package view-frame selection mismatch: "
            f"missing={missing_keys[:5]} ({len(missing_keys)} total), "
            f"extra={extra_keys[:5]} ({len(extra_keys)} total)"
        )
    if len(actual_keys) != LOCKED_VIEW_FRAMES:
        raise ValueError(
            f"locked package must contain {LOCKED_VIEW_FRAMES} view frames, "
            f"found {len(actual_keys)}"
        )

    ledger_rows: list[dict[str, object]] = []
    for source_order, row in enumerate(manifest.itertuples(index=False)):
        experiment = str(row.experiment)
        frame_idx = int(row.frame_idx)
        frame_key = str(row.frame_key)
        for view in VIEWS:
            labeled_frame = package_frames[(experiment, frame_idx, view)]
            instances = list(labeled_frame.instances)
            if len(instances) != LOCKED_FISH_PER_VIEW:
                raise ValueError(
                    "locked package requires exactly "
                    f"{LOCKED_FISH_PER_VIEW} fish per view; "
                    f"{experiment} frame {frame_idx} {view} has {len(instances)}"
                )
            for fish_index, instance in enumerate(instances):
                points = np.asarray(instance.numpy(), dtype=float)
                if (
                    points.ndim != 2
                    or points.shape[0] <= max(mouth_index, tail_index)
                    or points.shape[1] < 2
                ):
                    raise ValueError(
                        "invalid mouth/tail point array for "
                        f"{experiment} frame {frame_idx} {view} fish {fish_index}"
                    )
                mouth = points[mouth_index, :2]
                tail = points[tail_index, :2]
                if not np.isfinite(mouth).all() or not np.isfinite(tail).all():
                    raise ValueError(
                        "non-finite mouth/tail coordinate for "
                        f"{experiment} frame {frame_idx} {view} fish {fish_index}"
                    )
                ledger_rows.append(
                    {
                        "frame_key": frame_key,
                        "source_frame_order": source_order,
                        "experiment": experiment,
                        "frame_idx": frame_idx,
                        "view": view,
                        "fish_index": fish_index,
                        "mouth_x": float(mouth[0]),
                        "mouth_y": float(mouth[1]),
                        "tail_x": float(tail[0]),
                        "tail_y": float(tail[1]),
                    }
                )

    ledger = pd.DataFrame(ledger_rows, columns=LOCKED_LEDGER_COLUMNS)
    expected_instances = LOCKED_VIEW_FRAMES * LOCKED_FISH_PER_VIEW
    if len(ledger) != expected_instances:
        raise RuntimeError(
            f"locked ledger has {len(ledger)} instances; "
            f"expected {expected_instances}"
        )
    return ledger, {
        "source_frames": LOCKED_SOURCE_FRAMES,
        "view_frames": LOCKED_VIEW_FRAMES,
        "fish_per_view": LOCKED_FISH_PER_VIEW,
        "instances": expected_instances,
        "finite_mouths": expected_instances,
        "finite_tails": expected_instances,
    }


def finalize_locked_package(
    *,
    annotated_path: Path,
    manifest_path: Path,
    selection_provenance_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Validate and freeze a manual package without modifying the input."""
    if not annotated_path.is_file():
        raise FileNotFoundError(f"annotated package not found: {annotated_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"locked manifest not found: {manifest_path}")
    if not selection_provenance_path.is_file():
        raise FileNotFoundError(
            "blank-package provenance not found: "
            f"{selection_provenance_path}"
        )
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"locked output directory does not exist: {output_dir}"
        )

    final_path = output_dir / "independent_annotation_40_locked.pkg.slp"
    ledger_path = output_dir / "independent_annotation_40_locked_ledger.csv"
    provenance_path = (
        output_dir / "independent_annotation_40_locked_provenance.json"
    )
    outputs = (final_path, ledger_path, provenance_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite finalized locked artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    if annotated_path.resolve() == final_path.resolve():
        raise ValueError("annotated input and finalized output must differ")

    import sleap_io as sio

    selection_provenance = json.loads(
        selection_provenance_path.read_text(encoding="utf-8")
    )
    if selection_provenance.get("contains_model_predictions") is not False:
        raise ValueError(
            "blank-package provenance must explicitly record "
            "contains_model_predictions=false"
        )
    if selection_provenance.get("contains_keypoint_coordinates") is not False:
        raise ValueError(
            "blank-package provenance must explicitly record "
            "contains_keypoint_coordinates=false"
        )
    manifest_hash = sha256_file(manifest_path)
    recorded_manifest_hash = (
        selection_provenance.get("manifest", {}).get("sha256")
    )
    if recorded_manifest_hash != manifest_hash:
        raise ValueError(
            "manifest hash does not match blank-package provenance: "
            f"recorded={recorded_manifest_hash!r}, actual={manifest_hash}"
        )

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    labels = sio.load_slp(str(annotated_path), open_videos=False)
    ledger, counts = validate_locked_annotation_labels(labels, manifest)

    final_tmp = final_path.with_name(final_path.name + ".tmp")
    ledger_tmp = ledger_path.with_name(ledger_path.name + ".tmp")
    provenance_tmp = provenance_path.with_name(provenance_path.name + ".tmp")
    stale = [path for path in (final_tmp, ledger_tmp, provenance_tmp) if path.exists()]
    if stale:
        raise FileExistsError(
            "refusing stale finalization temporary artifacts: "
            + ", ".join(str(path) for path in stale)
        )

    try:
        ledger_tmp.write_text(
            ledger.to_csv(index=False, lineterminator="\n"),
            encoding="utf-8",
        )
        shutil.copy2(annotated_path, final_tmp)
        final_tmp.replace(final_path)
        ledger_tmp.replace(ledger_path)

        frozen: dict[str, object] = {
            "schema_version": "independent40-locked-final-v1",
            "status": "locked_manual_annotations",
            "selection_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_hash,
            },
            "blank_package_provenance": {
                "path": str(selection_provenance_path.resolve()),
                "sha256": sha256_file(selection_provenance_path),
            },
            "annotated_input": {
                "path": str(annotated_path.resolve()),
                "sha256": sha256_file(annotated_path),
            },
            "locked_package": {
                "path": str(final_path.resolve()),
                "sha256": sha256_file(final_path),
            },
            "selection_ledger": {
                "path": str(ledger_path.resolve()),
                "sha256": sha256_file(ledger_path),
                "columns": list(LOCKED_LEDGER_COLUMNS),
            },
            "counts": counts,
            "validation": {
                "exact_manifest_selection": True,
                "exactly_three_fish_per_view": True,
                "all_mouth_tail_coordinates_finite": True,
                "contains_model_predictions": False,
            },
            "source_package_provenance": dict(
                getattr(labels, "provenance", {}) or {}
            ),
        }
        provenance_tmp.write_text(
            json.dumps(frozen, indent=2) + "\n",
            encoding="utf-8",
        )
        provenance_tmp.replace(provenance_path)
    finally:
        for temporary in (final_tmp, ledger_tmp, provenance_tmp):
            temporary.unlink(missing_ok=True)

    return {
        "locked_package": str(final_path),
        "selection_ledger": str(ledger_path),
        "provenance": str(provenance_path),
        **counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--medaka-data-dir",
        type=Path,
        help="Override paths.medaka_data_dir (for example with an OIST UNC path).",
    )
    parser.add_argument(
        "--finalize-annotations",
        type=Path,
        help=(
            "Validate this manually annotated package and freeze a separate "
            "locked package, coordinate ledger, and hash provenance in "
            "--output-dir. This never modifies the annotated input."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Manifest for --finalize-annotations. Defaults to "
            "<output-dir>/independent_annotation_40_manifest.csv."
        ),
    )
    parser.add_argument(
        "--selection-provenance",
        type=Path,
        help=(
            "Blank-package provenance for --finalize-annotations. Defaults to "
            "<output-dir>/independent_annotation_40_provenance.json."
        ),
    )
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
        help=(
            "Round 8 selection table. Only pilot/model-training rows are "
            "treated as exposure; uncorrected expansion candidates are not."
        ),
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
    if args.finalize_annotations is not None:
        manifest_path = (
            args.manifest
            if args.manifest is not None
            else args.output_dir / "independent_annotation_40_manifest.csv"
        )
        selection_provenance_path = (
            args.selection_provenance
            if args.selection_provenance is not None
            else args.output_dir / "independent_annotation_40_provenance.json"
        )
        result = finalize_locked_package(
            annotated_path=args.finalize_annotations,
            manifest_path=manifest_path,
            selection_provenance_path=selection_provenance_path,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.manifest is not None or args.selection_provenance is not None:
        raise SystemExit(
            "--manifest and --selection-provenance are only valid with "
            "--finalize-annotations"
        )

    if args.output_dir.exists():
        raise SystemExit(
            f"refusing existing output directory: {args.output_dir}. "
            "Choose a new versioned directory so prior packages are immutable."
        )

    local = load_local_paths(args.paths_config)
    medaka_data_dir = args.medaka_data_dir or (
        local.medaka_data_dir if local is not None else None
    )
    if medaka_data_dir is None:
        raise SystemExit(
            "medaka_data_dir is required; pass --medaka-data-dir or define it "
            f"in {args.paths_config}"
        )
    if args.round7_dir is not None:
        round7_dir = args.round7_dir
    elif local is not None and local.sleap_models_dir is not None:
        round7_dir = local.sleap_models_dir / "retrain_20260704"
    else:
        raise SystemExit(
            "Round 7 root is required; pass --round7-dir or define "
            f"sleap_models_dir in {args.paths_config}"
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

    try:
        reserve_new_output_dir(args.output_dir)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
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
            "label and every Round 8 pilot/model-training exposure frame. "
            "Uncorrected expansion candidates were not model exposure."
        ),
        "quotas": quotas,
        "round7_packages": round7_packages,
        "round8_selection": {
            "path": str(args.round8_selection.resolve()),
            "sha256": sha256_file(args.round8_selection),
            "exposure_filter": "annotation_batch == pilot",
            "exposure_role": "pilot/model-training exposure only",
            "excluded_role": (
                "uncorrected expansion candidates; not used as model exposure"
            ),
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
        raw_data_dir=medaka_data_dir,
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
