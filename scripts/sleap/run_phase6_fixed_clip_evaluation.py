from __future__ import annotations

"""Prepare or run the mixed-model Phase-6 fixed-clip promotion gate.

The top-camera model is held fixed while side-view model pairs are compared on
the same inclusive frame windows. The launcher calls a small checkpoint-only
torch wrapper because ``run_round_inference`` assumes one round root/tag for
every view and the installed legacy ``sleap_nn.cli track`` command does not
expose independent stage-specific thresholds.

By default this preregisters the known unresolved contact event
``95000:99000`` and the non-event control ``151050:155050``. Use
``--prepare-only`` to write the complete command/provenance manifest without
launching GPU inference.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths


EXPERIMENT = "20251225_K_1_2_FCMC"
WORKFLOW_SUBDIR = "phase6_fixed_clips"
DEFAULT_REPORT_DIR = Path(
    "report/20260725/sleap_side_optimization_phase6_fixed_clips"
)
DEFAULT_WINDOWS = (
    "unresolved_contact=95000:99000",
    "non_event_control=151050:155050",
)
VIEW_VIDEO_NAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


@dataclass(frozen=True)
class ClipWindow:
    name: str
    start: int
    end: int

    @property
    def frames(self) -> str:
        return f"{self.start}-{self.end}"


@dataclass(frozen=True)
class ModelPair:
    name: str
    centroid: Path
    centered_instance: Path
    centroid_threshold: float = 0.2
    keypoint_threshold: float = 0.2


@dataclass(frozen=True)
class Phase6Paths:
    data_root: Path
    sleap_output_root: Path
    reconstruction_output_root: Path
    report_dir: Path
    calibration_model_dir: Path
    paths_config: Path


@dataclass(frozen=True)
class PlannedStep:
    name: str
    kind: str
    expected_output: Path
    command: tuple[str, ...] = ()
    copy_source: Path | None = None
    provenance_output: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "expected_output": str(self.expected_output),
            "command": list(self.command),
            "copy_source": (
                str(self.copy_source) if self.copy_source is not None else None
            ),
            "provenance_output": (
                str(self.provenance_output)
                if self.provenance_output is not None
                else None
            ),
        }


def parse_window(value: str) -> ClipWindow:
    """Parse ``NAME=START:END`` with inclusive non-negative bounds."""
    name, separator, bounds = value.partition("=")
    start_text, bounds_separator, end_text = bounds.partition(":")
    if (
        not separator
        or not bounds_separator
        or not name.strip()
        or not start_text.strip()
        or not end_text.strip()
    ):
        raise ValueError(
            f"window must be NAME=START:END, got {value!r}"
        )
    safe_name = name.strip()
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in safe_name
    ):
        raise ValueError(
            f"window name must contain only letters, digits, '-' or '_': {safe_name!r}"
        )
    start, end = int(start_text), int(end_text)
    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive frame bounds: {start}:{end}")
    return ClipWindow(safe_name, start, end)


def parse_model_pair(value: str) -> tuple[str, Path, Path]:
    """Parse ``NAME=CENTROID_DIR,CENTERED_INSTANCE_DIR``."""
    name, separator, raw_paths = value.partition("=")
    parts = [part.strip() for part in raw_paths.split(",") if part.strip()]
    if not separator or not name.strip() or len(parts) != 2:
        raise ValueError(
            "--candidate must be NAME=CENTROID_DIR,CENTERED_INSTANCE_DIR"
        )
    return name.strip(), Path(parts[0]), Path(parts[1])


def parse_candidate_threshold(value: str) -> tuple[str, float, float]:
    """Parse ``NAME=CENTROID_THRESHOLD,KEYPOINT_THRESHOLD``."""
    name, separator, raw_values = value.partition("=")
    parts = [part.strip() for part in raw_values.split(",") if part.strip()]
    if not separator or not name.strip() or len(parts) != 2:
        raise ValueError(
            "--candidate-threshold must be NAME=CENTROID,KEYPOINT"
        )
    centroid, keypoint = map(float, parts)
    for threshold in (centroid, keypoint):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be between 0 and 1, got {threshold}"
            )
    return name.strip(), centroid, keypoint


def default_model_pairs(model_root: Path) -> tuple[ModelPair, list[ModelPair]]:
    """Return the fixed Round-8 top pair and the three side comparators."""
    round8 = model_root / "retrain_20260722_round8"
    sideopt = model_root / "retrain_20260724_sideopt"
    top = ModelPair(
        "round8_top_fixed",
        round8
        / "top"
        / "runs"
        / "round8_260723_strictmouth.centroid_n1382",
        round8
        / "top"
        / "runs"
        / "round8_260723_strictmouth.centered_instance_n1382",
    )
    control_instance = (
        sideopt / "side" / "runs" / "side_centered_instance_control_p40"
    )
    sides = [
        ModelPair(
            "round8_side_baseline",
            round8
            / "side"
            / "runs"
            / "round8_260723_strictmouth.centroid_n2764",
            round8
            / "side"
            / "runs"
            / "round8_260723_strictmouth.centered_instance_n2764",
        ),
        ModelPair(
            "sigma20_control",
            sideopt
            / "side"
            / "runs"
            / "side_centroid_sigma20_strictframe_p40",
            control_instance,
        ),
        ModelPair(
            "sigma15_control",
            sideopt
            / "side"
            / "runs"
            / "side_centroid_sigma15_strictframe_p40",
            control_instance,
        ),
    ]
    return top, sides


def build_inference_command(
    *,
    python: str,
    video: Path,
    output_slp: Path,
    pair: ModelPair,
    window: ClipWindow,
    batch_size: int,
    max_instances: int = 6,
    device: str = "cuda",
) -> tuple[str, ...]:
    """Build one explicit checkpoint-only torch inference command."""
    return (
        python,
        "-m",
        "scripts.sleap.run_fixed_clip_topdown_inference",
        "--video",
        str(video),
        "--centroid-model",
        str(pair.centroid),
        "--centered-instance-model",
        str(pair.centered_instance),
        "--output-slp",
        str(output_slp),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--max-instances",
        str(max_instances),
        "--start-frame",
        str(window.start),
        "--end-frame",
        str(window.end),
        "--centroid-threshold",
        str(pair.centroid_threshold),
        "--keypoint-threshold",
        str(pair.keypoint_threshold),
    )


def build_top_track_command(
    *,
    python: str,
    experiment: str,
    slp_path: Path,
    output_csv: Path,
    window: ClipWindow,
    paths_config: Path,
) -> tuple[str, ...]:
    """Build the production visible-node-centroid top tracking command."""
    return (
        python,
        "-m",
        "scripts.tracking.track_simple_nn",
        experiment,
        "--paths-config",
        str(paths_config),
        "--view",
        "top_cam",
        "--slp",
        str(slp_path),
        "--output",
        str(output_csv),
        "--start-frame",
        str(window.start),
        "--end-frame",
        str(window.end),
        "--max-distance",
        "100",
        "--max-tracks",
        "3",
        "--max-gap",
        "500",
    )


def build_reconstruction_command(
    *,
    python: str,
    experiment: str,
    sleap_subdir: Path,
    top_tracks_csv: Path,
    output_dir: Path,
    output_tag: str,
    window: ClipWindow,
    paths_config: Path,
    sleap_output_root: Path,
    calibration_model_dir: Path,
) -> tuple[str, ...]:
    """Build the canonical six-candidate refractive reconstruction command."""
    return (
        python,
        "-m",
        "scripts.reconstruction_3d.reconstruct_from_slp",
        experiment,
        "--paths-config",
        str(paths_config),
        "--sleap-output-dir",
        str(sleap_output_root),
        "--model-folder",
        str(calibration_model_dir),
        "--calibration-backend",
        "auto",
        "--sleap-subdir",
        str(sleap_subdir),
        "--num-candidates",
        "6",
        "--use-visible-node-centroid",
        "--start-frame",
        str(window.start),
        "--end-frame",
        str(window.end),
        "--xyz-bounds",
        "-20",
        "220",
        "--no-shift-to-positive",
        "--max-jump-mm",
        "8",
        "--max-jump-hard-cap-mm",
        "80",
        "--max-gap-frames",
        "100",
        "--num-individuals",
        "3",
        "--bind-top-id",
        "--top-tracks-csv",
        str(top_tracks_csv),
        "--bind-min-frames",
        "500",
        "--output-dir",
        str(output_dir),
        "--output-tag",
        output_tag,
    )


def build_qc_command(
    *,
    python: str,
    candidates: Iterable[ModelPair],
    reconstruction_dirs: dict[str, Path],
    tags: dict[str, str],
    output_dir: Path,
    window: ClipWindow,
) -> tuple[str, ...]:
    command = [
        python,
        "-m",
        "scripts.reconstruction_3d.compare_fixed_clip_candidates",
        "--window",
        f"{window.name}={window.start}:{window.end}",
        "--baseline",
        "round8_side_baseline",
        "--output-json",
        str(output_dir / "fixed_clip_qc.json"),
        "--output-csv",
        str(output_dir / "fixed_clip_qc.csv"),
        "--output-md",
        str(output_dir / "FIXED_CLIP_QC.md"),
    ]
    for candidate in candidates:
        reconstruction_dir = reconstruction_dirs[candidate.name]
        tag = tags[candidate.name]
        command.extend(
            [
                "--candidate",
                (
                    f"{candidate.name}="
                    f"{reconstruction_dir / f'{tag}_best_pre_link.json'},"
                    f"{reconstruction_dir / f'{tag}_3d.csv'},"
                    f"{reconstruction_dir / f'{tag}_3d_identified.csv'}"
                ),
            ]
        )
    return tuple(command)


def build_render_command(
    *,
    python: str,
    experiment: str,
    slp_dir: Path,
    best_json: Path,
    calibration_model_dir: Path,
    video_dir: Path,
    output_mp4: Path,
    triangulation_npz: Path,
    identified_csv: Path,
    window: ClipWindow,
) -> tuple[str, ...]:
    """Build the restartable identified 4-panel visual-QC command."""
    return (
        python,
        "-m",
        "scripts.visualization.render_4panel_video",
        experiment,
        "--slp-dir",
        str(slp_dir),
        "--best-json",
        str(best_json),
        "--model-folder",
        str(calibration_model_dir),
        "--video-dir",
        str(video_dir),
        "--triangulation-npz",
        str(triangulation_npz),
        "--out",
        str(output_mp4),
        "--start-frame",
        str(window.start),
        "--end-frame",
        str(window.end),
        "--frame-step",
        "5",
        "--identified-csv",
        str(identified_csv),
        "--display-id-col",
        "final_fish_id",
        "--show-unassigned-sleap",
        "--xyz-lim",
        "-20",
        "220",
    )


def build_plan(
    *,
    experiment: str,
    windows: list[ClipWindow],
    top_pair: ModelPair,
    side_pairs: list[ModelPair],
    paths: Phase6Paths,
    python: str,
    batch_size: int,
    device: str,
) -> list[PlannedStep]:
    """Build an ordered, restartable execution plan."""
    steps: list[PlannedStep] = []
    experiment_dir = paths.data_root / experiment
    for window in windows:
        shared_relative = Path(WORKFLOW_SUBDIR) / window.name / "_top_common"
        shared_dir = paths.sleap_output_root / experiment / shared_relative
        shared_top_slp = shared_dir / "top_cam.slp"
        shared_top_tracks = shared_dir / "top_cam.tracks.csv"
        steps.append(
            PlannedStep(
                name=f"{window.name}:top_inference",
                kind="command",
                expected_output=shared_top_slp,
                command=build_inference_command(
                    python=python,
                    video=(
                        experiment_dir
                        / "top_cam"
                        / VIEW_VIDEO_NAMES["top_cam"]
                    ),
                    output_slp=shared_top_slp,
                    pair=top_pair,
                    window=window,
                    batch_size=batch_size,
                    device=device,
                ),
                provenance_output=shared_top_slp.with_suffix(
                    ".inference_provenance.json"
                ),
            )
        )
        steps.append(
            PlannedStep(
                name=f"{window.name}:top_track",
                kind="command",
                expected_output=shared_top_tracks,
                command=build_top_track_command(
                    python=python,
                    experiment=experiment,
                    slp_path=shared_top_slp,
                    output_csv=shared_top_tracks,
                    window=window,
                    paths_config=paths.paths_config,
                ),
            )
        )

        reconstruction_dirs: dict[str, Path] = {}
        tags: dict[str, str] = {}
        for candidate in side_pairs:
            relative = Path(WORKFLOW_SUBDIR) / window.name / candidate.name
            slp_dir = paths.sleap_output_root / experiment / relative
            for view in ("side1_cam", "side2_cam"):
                output_slp = slp_dir / f"{view}.slp"
                steps.append(
                    PlannedStep(
                        name=f"{window.name}:{candidate.name}:{view}_inference",
                        kind="command",
                        expected_output=output_slp,
                        command=build_inference_command(
                            python=python,
                            video=(
                                experiment_dir
                                / view
                                / VIEW_VIDEO_NAMES[view]
                            ),
                            output_slp=output_slp,
                            pair=candidate,
                            window=window,
                            batch_size=batch_size,
                            device=device,
                        ),
                        provenance_output=output_slp.with_suffix(
                            ".inference_provenance.json"
                        ),
                    )
                )
            staged_top = slp_dir / "top_cam.slp"
            steps.append(
                PlannedStep(
                    name=f"{window.name}:{candidate.name}:stage_top",
                    kind="copy",
                    expected_output=staged_top,
                    copy_source=shared_top_slp,
                )
            )
            tag = f"phase6_{window.name}_{candidate.name}"
            reconstruction_dir = (
                paths.reconstruction_output_root
                / experiment
                / WORKFLOW_SUBDIR
                / window.name
                / candidate.name
            )
            identified_csv = reconstruction_dir / f"{tag}_3d_identified.csv"
            steps.append(
                PlannedStep(
                    name=f"{window.name}:{candidate.name}:reconstruct",
                    kind="command",
                    expected_output=identified_csv,
                    command=build_reconstruction_command(
                        python=python,
                        experiment=experiment,
                        sleap_subdir=relative,
                        top_tracks_csv=shared_top_tracks,
                        output_dir=reconstruction_dir,
                        output_tag=tag,
                        window=window,
                        paths_config=paths.paths_config,
                        sleap_output_root=paths.sleap_output_root,
                        calibration_model_dir=paths.calibration_model_dir,
                    ),
                )
            )
            reconstruction_dirs[candidate.name] = reconstruction_dir
            tags[candidate.name] = tag
            render_dir = paths.report_dir / window.name / "renders"
            steps.append(
                PlannedStep(
                    name=f"{window.name}:{candidate.name}:render_4panel",
                    kind="command",
                    expected_output=(
                        render_dir / f"{candidate.name}_4panel_step5.mp4"
                    ),
                    command=build_render_command(
                        python=python,
                        experiment=experiment,
                        slp_dir=slp_dir,
                        best_json=reconstruction_dir / f"{tag}_best.json",
                        calibration_model_dir=paths.calibration_model_dir,
                        video_dir=experiment_dir,
                        output_mp4=(
                            render_dir
                            / f"{candidate.name}_4panel_step5.mp4"
                        ),
                        triangulation_npz=(
                            reconstruction_dir
                            / f"{tag}_4panel_xyz_step5.npz"
                        ),
                        identified_csv=identified_csv,
                        window=window,
                    ),
                    provenance_output=(
                        reconstruction_dir
                        / f"{tag}_reconstruction_provenance.json"
                    ),
                )
            )

        qc_dir = paths.report_dir / window.name
        steps.append(
            PlannedStep(
                name=f"{window.name}:qc",
                kind="command",
                expected_output=qc_dir / "fixed_clip_qc.json",
                command=build_qc_command(
                    python=python,
                    candidates=side_pairs,
                    reconstruction_dirs=reconstruction_dirs,
                    tags=tags,
                    output_dir=qc_dir,
                    window=window,
                ),
            )
        )
    return steps


def _path_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    checkpoint = path / "best.ckpt"
    config = path / "training_config.yaml"
    for key, artifact in (("checkpoint", checkpoint), ("config", config)):
        if artifact.is_file():
            stat = artifact.stat()
            record[key] = {
                "path": str(artifact),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(artifact),
            }
        else:
            record[key] = None
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(
    *,
    experiment: str,
    paths: Phase6Paths,
    top_pair: ModelPair,
    side_pairs: list[ModelPair],
) -> None:
    for pair in (top_pair, *side_pairs):
        for model_dir in (pair.centroid, pair.centered_instance):
            if not (model_dir / "best.ckpt").is_file():
                raise FileNotFoundError(
                    f"model checkpoint not found: {model_dir / 'best.ckpt'}"
                )
            if not (model_dir / "training_config.yaml").is_file():
                raise FileNotFoundError(
                    "model training config not found: "
                    f"{model_dir / 'training_config.yaml'}"
                )
    experiment_dir = paths.data_root / experiment
    for view, filename in VIEW_VIDEO_NAMES.items():
        video = experiment_dir / view / filename
        if not video.is_file():
            raise FileNotFoundError(f"video not found: {video}")


def _write_status(
    path: Path,
    *,
    state: str,
    active: str | None,
    completed: list[str],
    failed: list[dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "state": state,
                "active": active,
                "completed": completed,
                "failed": failed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _execute_step(
    step: PlannedStep,
    *,
    repo_root: Path,
    log_dir: Path,
    force: bool,
) -> int:
    complete = step.expected_output.exists() and (
        step.provenance_output is None
        or step.provenance_output.exists()
    )
    if complete and not force:
        print(f"[skip complete] {step.name}: {step.expected_output}", flush=True)
        return 0
    step.expected_output.parent.mkdir(parents=True, exist_ok=True)
    if step.kind == "copy":
        if step.copy_source is None or not step.copy_source.is_file():
            raise FileNotFoundError(
                f"copy source missing for {step.name}: {step.copy_source}"
            )
        shutil.copy2(step.copy_source, step.expected_output)
        print(
            f"[staged] {step.copy_source} -> {step.expected_output}",
            flush=True,
        )
        return 0
    if step.kind != "command" or not step.command:
        raise ValueError(f"invalid planned step: {step}")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.name.replace(':', '__')}.log"
    print(f"\n--- {step.name} ---", flush=True)
    print(" ".join(step.command), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        result = subprocess.run(
            list(step.command),
            cwd=str(repo_root),
            env={
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    missing_outputs = [
        path
        for path in (step.expected_output, step.provenance_output)
        if path is not None and not path.exists()
    ]
    if result.returncode == 0 and missing_outputs:
        print(
            "[error] command succeeded but required output is missing: "
            + ", ".join(str(path) for path in missing_outputs),
            flush=True,
        )
        return 2
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="NAME=START:END",
        help="Inclusive clip; repeat. Defaults to the preregistered event/control.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=CENTROID_DIR,CENTERED_INSTANCE_DIR",
        help="Override the default Round-8/sigma20/sigma15 side pairs.",
    )
    parser.add_argument(
        "--candidate-threshold",
        action="append",
        default=[],
        metavar="NAME=CENTROID,KEYPOINT",
    )
    parser.add_argument("--top-centroid", type=Path)
    parser.add_argument("--top-centered-instance", type=Path)
    parser.add_argument("--top-centroid-threshold", type=float, default=0.2)
    parser.add_argument("--top-keypoint-threshold", type=float, default=0.2)
    parser.add_argument("--side-centroid-threshold", type=float, default=0.2)
    parser.add_argument("--side-keypoint-threshold", type=float, default=0.2)
    parser.add_argument(
        "--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sleap-output-root", type=Path)
    parser.add_argument("--reconstruction-output-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-input-checks", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args(argv)

    try:
        windows = [
            parse_window(value)
            for value in (args.window or DEFAULT_WINDOWS)
        ]
    except ValueError as exc:
        parser.error(str(exc))
    if len({window.name for window in windows}) != len(windows):
        parser.error("window names must be unique")
    for threshold in (
        args.top_centroid_threshold,
        args.top_keypoint_threshold,
        args.side_centroid_threshold,
        args.side_keypoint_threshold,
    ):
        if not 0.0 <= threshold <= 1.0:
            parser.error(f"threshold must be between 0 and 1: {threshold}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    local = load_local_paths(args.paths_config)

    def resolved(
        explicit: Path | None,
        configured: Path | None,
        option: str,
    ) -> Path:
        value = explicit or configured
        if value is None:
            parser.error(f"{option} is required or must be set in paths config")
        return Path(value)

    data_root = resolved(
        args.data_root,
        (
            local.medaka_data_dir or local.raw_data_dir
            if local is not None
            else None
        ),
        "--data-root",
    )
    sleap_output_root = resolved(
        args.sleap_output_root,
        local.sleap_output_dir if local is not None else None,
        "--sleap-output-root",
    )
    reconstruction_output_root = resolved(
        args.reconstruction_output_root,
        local.reconstruction_output_dir if local is not None else None,
        "--reconstruction-output-root",
    )
    model_root = resolved(
        args.model_root,
        local.sleap_models_dir if local is not None else None,
        "--model-root",
    )
    calibration_model_dir = resolved(
        args.calibration_model_dir,
        local.calibration_model_dir if local is not None else None,
        "--calibration-model-dir",
    )
    paths = Phase6Paths(
        data_root=data_root,
        sleap_output_root=sleap_output_root,
        reconstruction_output_root=reconstruction_output_root,
        report_dir=args.report_dir.resolve(),
        calibration_model_dir=calibration_model_dir,
        paths_config=args.paths_config.resolve(),
    )

    default_top, default_sides = default_model_pairs(model_root)
    if bool(args.top_centroid) != bool(args.top_centered_instance):
        parser.error(
            "--top-centroid and --top-centered-instance must be passed together"
        )
    top_pair = ModelPair(
        default_top.name,
        args.top_centroid or default_top.centroid,
        args.top_centered_instance or default_top.centered_instance,
        args.top_centroid_threshold,
        args.top_keypoint_threshold,
    )
    if args.candidate:
        parsed_candidates: list[tuple[str, Path, Path]] = []
        try:
            parsed_candidates = [
                parse_model_pair(value) for value in args.candidate
            ]
        except ValueError as exc:
            parser.error(str(exc))
        side_pairs = [
            ModelPair(
                name,
                centroid,
                centered_instance,
                args.side_centroid_threshold,
                args.side_keypoint_threshold,
            )
            for name, centroid, centered_instance in parsed_candidates
        ]
    else:
        side_pairs = [
            ModelPair(
                pair.name,
                pair.centroid,
                pair.centered_instance,
                args.side_centroid_threshold,
                args.side_keypoint_threshold,
            )
            for pair in default_sides
        ]
    if len({pair.name for pair in side_pairs}) != len(side_pairs):
        parser.error("candidate names must be unique")

    threshold_overrides: dict[str, tuple[float, float]] = {}
    try:
        for value in args.candidate_threshold:
            name, centroid, keypoint = parse_candidate_threshold(value)
            if name in threshold_overrides:
                raise ValueError(f"duplicate candidate threshold: {name}")
            threshold_overrides[name] = (centroid, keypoint)
    except ValueError as exc:
        parser.error(str(exc))
    unknown = set(threshold_overrides) - {pair.name for pair in side_pairs}
    if unknown:
        parser.error(
            "threshold override names unknown candidates: "
            + ", ".join(sorted(unknown))
        )
    side_pairs = [
        ModelPair(
            pair.name,
            pair.centroid,
            pair.centered_instance,
            *threshold_overrides.get(
                pair.name,
                (pair.centroid_threshold, pair.keypoint_threshold),
            ),
        )
        for pair in side_pairs
    ]

    if not args.skip_input_checks:
        try:
            _validate_inputs(
                experiment=args.experiment,
                paths=paths,
                top_pair=top_pair,
                side_pairs=side_pairs,
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))

    steps = build_plan(
        experiment=args.experiment,
        windows=windows,
        top_pair=top_pair,
        side_pairs=side_pairs,
        paths=paths,
        python=args.python,
        batch_size=args.batch_size,
        device=args.device,
    )
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        repo_root
        / "scripts"
        / "sleap"
        / "run_fixed_clip_topdown_inference.py",
        repo_root
        / "scripts"
        / "reconstruction_3d"
        / "compare_fixed_clip_candidates.py",
        repo_root
        / "scripts"
        / "reconstruction_3d"
        / "reconstruct_from_slp.py",
        repo_root
        / "scripts"
        / "visualization"
        / "render_4panel_video.py",
    ]
    manifest = {
        "schema_version": "phase6_fixed_clip_evaluation_v1",
        "workflow": "mixed_top_side_sleap_to_3d_fixed_clip_gate",
        "scientific_status": (
            "engineering_and_known_failure_gate_not_independent_promotion"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": args.experiment,
        "source_artifacts": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in source_paths
        ],
        "windows": [asdict(window) for window in windows],
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "top_pair": {
            **asdict(top_pair),
            "centroid": str(top_pair.centroid),
            "centered_instance": str(top_pair.centered_instance),
            "centroid_artifacts": _path_record(top_pair.centroid),
            "centered_instance_artifacts": _path_record(
                top_pair.centered_instance
            ),
        },
        "side_pairs": [
            {
                **asdict(pair),
                "centroid": str(pair.centroid),
                "centered_instance": str(pair.centered_instance),
                "centroid_artifacts": _path_record(pair.centroid),
                "centered_instance_artifacts": _path_record(
                    pair.centered_instance
                ),
            }
            for pair in side_pairs
        ],
        "inference": {
            "mode": "checkpoint-only torch API wrapper",
            "api": "sleap_nn.inference.predict",
            "device": args.device,
            "batch_size": args.batch_size,
            "max_instances": 6,
            "integral_refinement": "integral",
            "tensorrt_prohibited_on_this_machine": True,
            "per_slp_provenance_suffix": ".inference_provenance.json",
        },
        "visual_qc": {
            "renderer": "scripts.visualization.render_4panel_video",
            "frame_step": 5,
            "display_id_column": "final_fish_id",
            "show_unassigned_sleap": True,
            "one_video_per_candidate_window": True,
        },
        "steps": [step.to_json() for step in steps],
        "limitations": [
            "The two default windows are known-failure/control diagnostics, not an independent locked test.",
            "Female ground truth identifies only one fish and cannot score male-to-male swaps.",
            "Scientific promotion still requires independent event-disjoint manual labels.",
        ],
    }
    manifest_path = paths.report_dir / "phase6_execution_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        print(
            f"Prepared {len(steps)} Phase-6 steps in {manifest_path}",
            flush=True,
        )
        return 0

    status_path = paths.report_dir / "phase6_status.json"
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    for step in steps:
        _write_status(
            status_path,
            state="running",
            active=step.name,
            completed=completed,
            failed=failed,
        )
        try:
            return_code = _execute_step(
                step,
                repo_root=repo_root,
                log_dir=paths.report_dir / "logs",
                force=args.force,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[error] {step.name}: {exc}", flush=True)
            return_code = 2
        if return_code == 0:
            completed.append(step.name)
            continue
        failed.append(
            {"step": step.name, "return_code": int(return_code)}
        )
        if not args.continue_on_fail:
            _write_status(
                status_path,
                state="failed",
                active=None,
                completed=completed,
                failed=failed,
            )
            return return_code

    state = "complete" if not failed else "complete_with_failures"
    _write_status(
        status_path,
        state=state,
        active=None,
        completed=completed,
        failed=failed,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
