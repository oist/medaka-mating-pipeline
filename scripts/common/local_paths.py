from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = REPO_ROOT / "configs" / "paths.local.toml"


@dataclass(frozen=True)
class LocalPaths:
    """User-local project paths loaded from a gitignored TOML file."""

    config_path: Path
    data_root: Path | None = None
    raw_data_dir: Path | None = None
    sleap_output_dir: Path | None = None
    processed_dir: Path | None = None
    tracking_output_dir: Path | None = None
    calibration_model_dir: Path | None = None
    reconstruction_output_dir: Path | None = None
    runs_dir: Path | None = None

    def as_dict(self) -> dict[str, Path | None]:
        return {
            "data_root": self.data_root,
            "raw_data_dir": self.raw_data_dir,
            "sleap_output_dir": self.sleap_output_dir,
            "processed_dir": self.processed_dir,
            "tracking_output_dir": self.tracking_output_dir,
            "calibration_model_dir": self.calibration_model_dir,
            "reconstruction_output_dir": self.reconstruction_output_dir,
            "runs_dir": self.runs_dir,
        }


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_local_paths(config_path: str | Path | None = None) -> LocalPaths | None:
    """
    Load gitignored local path settings from a TOML file.

    Missing config files return None so CLI callers can keep working with
    explicit command-line paths only.
    """
    path = _resolve_path(str(config_path)) if config_path else DEFAULT_PATHS_CONFIG
    if path is None or not path.exists():
        return None

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"{path} must define a [paths] table")

    return LocalPaths(
        config_path=path,
        data_root=_resolve_path(paths.get("data_root")),
        raw_data_dir=_resolve_path(paths.get("raw_data_dir")),
        sleap_output_dir=_resolve_path(paths.get("sleap_output_dir")),
        processed_dir=_resolve_path(paths.get("processed_dir")),
        tracking_output_dir=_resolve_path(paths.get("tracking_output_dir")),
        calibration_model_dir=_resolve_path(paths.get("calibration_model_dir")),
        reconstruction_output_dir=_resolve_path(paths.get("reconstruction_output_dir")),
        runs_dir=_resolve_path(paths.get("runs_dir")),
    )


def choose_path(*values: str | Path | None) -> str | None:
    """Return the first non-empty path-like value as a string."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, Path):
            return str(value)
        if str(value).strip():
            return str(value)
    return None
