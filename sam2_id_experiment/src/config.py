from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.common.local_paths import choose_path, load_local_paths


VIEW_VIDEO_BASENAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    video_path: Path
    output_dir: Path
    n_animals_expected: int
    experiment_name: str | None = None
    view: str | None = None


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [key for key in ("output_dir",) if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    experiment_name = raw.get("experiment_name")
    view = raw.get("view")
    video_path = _resolve_video_path(raw, path.parent)
    output_dir = _resolve_output_dir(raw, experiment_name, view)

    return ExperimentConfig(
        path=path,
        raw=raw,
        video_path=video_path,
        output_dir=output_dir,
        n_animals_expected=int(raw.get("n_animals_expected", 0)),
        experiment_name=str(experiment_name) if experiment_name else None,
        view=str(view) if view else None,
    )


def ensure_output_dir(config: ExperimentConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config.output_dir


def config_section(config: ExperimentConfig, name: str) -> dict[str, Any]:
    value = config.raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return dict(value)


def _resolve_video_path(raw: dict[str, Any], config_dir: Path) -> Path:
    explicit = raw.get("video_path")
    if explicit:
        return Path(str(explicit))

    experiment_name = raw.get("experiment_name")
    view = raw.get("view")
    if not experiment_name or not view:
        raise ValueError(
            "Config needs either video_path or both experiment_name and view."
        )
    view = str(view)
    if view not in VIEW_VIDEO_BASENAMES:
        valid = ", ".join(sorted(VIEW_VIDEO_BASENAMES))
        raise ValueError(f"Unknown view {view!r}; expected one of: {valid}")

    paths_config = raw.get("paths_config")
    if paths_config:
        paths_config_path = Path(str(paths_config))
        if not paths_config_path.is_absolute():
            paths_config_path = config_dir.parents[1] / paths_config_path
    else:
        paths_config_path = None
    local = load_local_paths(paths_config_path)
    medaka_data_dir = choose_path(
        raw.get("medaka_data_dir"),
        local.medaka_data_dir if local else None,
    )
    if medaka_data_dir is None:
        raise ValueError(
            "experiment_name/view resolution needs medaka_data_dir in config "
            "or paths_config."
        )
    return Path(medaka_data_dir) / str(experiment_name) / view / VIEW_VIDEO_BASENAMES[view]


def _resolve_output_dir(
    raw: dict[str, Any],
    experiment_name: Any,
    view: Any,
) -> Path:
    output_dir = str(raw["output_dir"])
    if experiment_name is not None:
        output_dir = output_dir.replace("{experiment_name}", str(experiment_name))
    if view is not None:
        output_dir = output_dir.replace("{view}", str(view))
    return Path(output_dir)
