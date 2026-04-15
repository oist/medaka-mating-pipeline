from __future__ import annotations

"""
Medaka experiment-level metadata: folder-name parsing, campy metadata.csv
reading, and JSON I/O for a per-experiment manifest.

Folder name convention:
    <yyyymmdd>_<group>_<trial>_<day>_<paradigm>
    e.g. 20251220_D_1_1_FCMC

    yyyymmdd = experiment START date (NOT the recording date; days 2 and 3
               keep the day-1 prefix to link trials together)
    group    = group id, e.g. 'D'
    trial    = nth trial within the group
    day      = day of recording (1, 2, or 3)
    paradigm = experiment paradigm, e.g. 'FCMC'
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path


EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<ymd>\d{8})_(?P<group>[A-Za-z]+)_(?P<trial>\d+)_(?P<day>\d+)_(?P<paradigm>\w+)$"
)

VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEO_BASENAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


@dataclass
class ExperimentName:
    """Structured view of a folder name like `20251220_D_1_1_FCMC`."""

    folder_name: str
    start_date: str   # ISO-8601, e.g. "2025-12-20"
    group: str
    trial: int
    day: int
    paradigm: str


def parse_experiment_name(folder_name: str) -> ExperimentName:
    """Parse a folder name. Raises ValueError on malformed input."""
    m = EXPERIMENT_NAME_RE.match(folder_name)
    if m is None:
        raise ValueError(
            f"Folder name {folder_name!r} does not match "
            "yyyymmdd_<group>_<trial>_<day>_<paradigm>"
        )
    ymd = m["ymd"]
    try:
        d = date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
    except ValueError as exc:
        raise ValueError(f"Folder name {folder_name!r} has invalid date: {exc}") from exc
    return ExperimentName(
        folder_name=folder_name,
        start_date=d.isoformat(),
        group=m["group"],
        trial=int(m["trial"]),
        day=int(m["day"]),
        paradigm=m["paradigm"],
    )


def read_metadata_csv(path: str | Path) -> dict[str, str]:
    """
    Read the campy metadata.csv (a two-column `"key","value"` file, one pair
    per line) into a flat dict of strings.
    """
    result: dict[str, str] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^"([^"]*)","(.*)"$', line)
            if m:
                result[m.group(1)] = m.group(2)
    return result


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class ViewRecording:
    """One camera view's recording metadata, derived from metadata.csv."""

    camera_name: str | None = None
    video_path: str | None = None
    total_frames: int | None = None
    frame_rate: float | None = None
    rec_time_sec: float | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    codec: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    camera_serial: str | None = None
    exposure_us: int | None = None


def build_view_recording(cam_dir: str | Path) -> ViewRecording:
    """
    Build a ViewRecording from a single `<cam>_cam/` folder containing
    `metadata.csv` and the mp4.
    """
    cam_dir = Path(cam_dir)
    meta_path = cam_dir / "metadata.csv"
    meta: dict[str, str] = {}
    if meta_path.exists():
        meta = read_metadata_csv(meta_path)

    video_basename = meta.get("videoFilename")
    video_path = cam_dir / video_basename if video_basename else None
    if video_path is None or not video_path.exists():
        # fall back to conventional names (cam0_top.mp4 etc.)
        for candidate in cam_dir.glob("cam*.mp4"):
            video_path = candidate
            break

    return ViewRecording(
        camera_name=meta.get("cameraName") or meta.get("cameraNames"),
        video_path=str(video_path) if video_path else None,
        total_frames=_int_or_none(meta.get("totalFrames")),
        frame_rate=_float_or_none(meta.get("frameRate")),
        rec_time_sec=_float_or_none(meta.get("totalTime") or meta.get("recTimeInSec")),
        frame_width=_int_or_none(meta.get("frameWidth")),
        frame_height=_int_or_none(meta.get("frameHeight")),
        codec=meta.get("codec"),
        camera_make=meta.get("cameraMake"),
        camera_model=meta.get("cameraModel"),
        camera_serial=meta.get("cameraSerialNo"),
        exposure_us=_int_or_none(meta.get("cameraExposureTimeInUs")),
    )


@dataclass
class ExperimentMetadata:
    """All metadata we can derive without running inference."""

    folder_name: str
    experiment: ExperimentName
    raw_data_dir: str | None = None
    views: dict[str, ViewRecording] = field(default_factory=dict)
    animal: dict | None = None    # filled in manually / by later tooling

    def to_dict(self) -> dict:
        return {
            "folder_name": self.folder_name,
            "experiment": asdict(self.experiment),
            "raw_data_dir": self.raw_data_dir,
            "views": {k: asdict(v) for k, v in self.views.items()},
            "animal": self.animal,
        }


def build_experiment_metadata(
    experiment_dir: str | Path,
) -> ExperimentMetadata:
    """
    Build an ExperimentMetadata from the raw-recording experiment folder, e.g.
    `/bucket/StephensU/medaka/medaka_202512/20251220_D_1_1_FCMC`.
    """
    experiment_dir = Path(experiment_dir)
    exp_name = parse_experiment_name(experiment_dir.name)

    views: dict[str, ViewRecording] = {}
    for view in VIEWS:
        cam_dir = experiment_dir / view
        if cam_dir.is_dir():
            views[view] = build_view_recording(cam_dir)

    return ExperimentMetadata(
        folder_name=experiment_dir.name,
        experiment=exp_name,
        raw_data_dir=str(experiment_dir),
        views=views,
    )


def save_metadata_json(metadata: ExperimentMetadata | dict, path: str | Path) -> None:
    payload = metadata.to_dict() if isinstance(metadata, ExperimentMetadata) else metadata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_metadata_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


MANIFEST_FILENAME = "experiment_metadata.json"


def manifest_path_for(experiment_dir: str | Path) -> Path:
    """Canonical manifest path for a raw recording folder."""
    return Path(experiment_dir) / MANIFEST_FILENAME


def write_experiment_manifest(
    experiment_dir: str | Path,
    force: bool = False,
    sleap_section: dict | None = None,
) -> dict:
    """
    Write (or update) the per-experiment manifest inside a raw recording folder.

    Behaviour:
        - If the manifest already exists, it is loaded and the recording-side
          fields (`folder_name`, `experiment`, `raw_data_dir`, `views`) are
          refreshed from the folder name and each camera's metadata.csv.
          Hand-authored fields such as `animal` or anything under custom keys
          are preserved.
        - If `sleap_section` is provided, it replaces the `sleap` key.
        - If `force=True`, the manifest is rebuilt from scratch (existing
          custom keys are dropped).

    Returns the final manifest dict.
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")

    fresh = build_experiment_metadata(experiment_dir).to_dict()
    manifest_path = manifest_path_for(experiment_dir)

    if force or not manifest_path.exists():
        payload = fresh
    else:
        payload = load_metadata_json(manifest_path)
        # Refresh the recording-side fields in case metadata.csv changed or a
        # view was added. Keep anything the user appended (animal, notes, ...).
        for key in ("folder_name", "experiment", "raw_data_dir", "views"):
            payload[key] = fresh[key]

    if sleap_section is not None:
        payload["sleap"] = sleap_section

    save_metadata_json(payload, manifest_path)
    return payload


def _cli_parser() -> "argparse.ArgumentParser":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Generate or refresh the per-experiment manifest "
            "(experiment_metadata.json) in each raw recording folder."
        )
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Experiment folder name, e.g. 20251220_D_1_1_FCMC. "
             "Omit with --all to process every matching folder.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Iterate every folder under medaka_data_dir that matches --pattern.",
    )
    parser.add_argument(
        "--pattern",
        default=r"^\d{8}_[A-Za-z]+_\d+_\d+_\w+$",
        help="Regex used with --all to filter folder names (default: FCMC-style).",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Explicit experiment folder path (overrides lookup via paths.local.toml).",
    )
    parser.add_argument(
        "--medaka-data-dir",
        default=None,
        help="Override medaka_data_dir from paths.local.toml.",
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild from scratch instead of refreshing recording fields.",
    )
    return parser


def _cli_main(argv: list[str] | None = None) -> None:
    import re
    import sys

    # Local imports so `from scripts.common.experiment_metadata import *` stays light.
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from scripts.common.local_paths import choose_path, load_local_paths

    parser = _cli_parser()
    args = parser.parse_args(argv)

    if args.dir:
        payload = write_experiment_manifest(args.dir, force=args.force)
        print(f"Wrote {manifest_path_for(args.dir)}")
        print(f"  group={payload['experiment']['group']} "
              f"trial={payload['experiment']['trial']} "
              f"day={payload['experiment']['day']}")
        return

    local = load_local_paths(args.paths_config)
    medaka_data_dir = choose_path(
        args.medaka_data_dir,
        local.medaka_data_dir if local else None,
    )
    if medaka_data_dir is None:
        parser.error(
            "medaka_data_dir not set. Pass --medaka-data-dir, --dir, or configure "
            "medaka_data_dir in configs/paths.local.toml."
        )
    root = Path(medaka_data_dir)

    if args.all:
        regex = re.compile(args.pattern)
        targets = [p for p in sorted(root.iterdir()) if p.is_dir() and regex.match(p.name)]
        print(f"Matched {len(targets)} experiments under {root}")
        for exp_dir in targets:
            try:
                write_experiment_manifest(exp_dir, force=args.force)
                print(f"  ok  {exp_dir.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  err {exp_dir.name}: {exc}")
        return

    if not args.experiment:
        parser.error("Pass an experiment folder name, --dir, or --all.")
    exp_dir = root / args.experiment
    write_experiment_manifest(exp_dir, force=args.force)
    print(f"Wrote {manifest_path_for(exp_dir)}")


if __name__ == "__main__":
    _cli_main()
