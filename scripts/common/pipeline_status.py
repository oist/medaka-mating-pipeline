from __future__ import annotations

"""
Detect on-disk pipeline-stage outputs for one medaka experiment.

The per-experiment `experiment_metadata.json` records recording + SLEAP-merge
state, but the later stages of the pipeline write to sibling output roots
(`sleap_output/`, `detectron2_output/`, `reconstruction_3d/`). This module
scans those roots for a single experiment folder name and returns a compact
`pipeline` section describing how far the experiment has progressed:

    recorded -> sleap_inferred -> detectron2 -> reconstructed_3d -> identified_3d

It is filesystem-only (no SLEAP / torch / pandas import) so `build_catalog`
can call it cheaply for every experiment, including on machines where the
heavy deps are not installed.
"""

import glob as _glob
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.common.experiment_metadata import VIEWS


# Ordered pipeline stages, coarsest first. `furthest_stage` returns the last
# one an experiment satisfies; the HTML/CSV views colour progress by this.
STAGE_ORDER = (
    "recorded",
    "sleap_inferred",
    "detectron2",
    "reconstructed_3d",
    "identified_3d",
)

# Subfolders under an sleap_output/<exp> dir that are not model-round outputs.
_NON_ROUND_SUBDIRS = {"logs", "chunks", "viz"}


def _view_stem(view: str) -> str:
    """`top_cam` -> `top`, matching the compact per-view column naming."""
    return view.replace("_cam", "")


@dataclass
class SleapInferStatus:
    """Merged/chunked `.slp` presence under `sleap_output/<exp>/`."""

    present: bool = False
    output_dir: str | None = None
    merged: dict[str, bool] = field(default_factory=dict)
    merged_all: bool = False
    n_merged_views: int = 0
    n_chunks: dict[str, int] = field(default_factory=dict)
    rounds: list[str] = field(default_factory=list)
    has_viz: bool = False


@dataclass
class Detectron2Status:
    """Segmentation-mask `.h5` presence under `detectron2_output/<exp>/`."""

    present: bool = False
    output_dir: str | None = None
    n_mask_chunks: dict[str, int] = field(default_factory=dict)
    n_views: int = 0


@dataclass
class Reconstruction3DStatus:
    """3D-track CSV presence under the reconstruction output root."""

    present: bool = False
    output_dir: str | None = None
    has_3d_csv: bool = False
    has_identified_csv: bool = False
    csv_files: list[str] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)


@dataclass
class PipelineStatus:
    """Aggregate of every non-recording stage for one experiment."""

    sleap_infer: SleapInferStatus = field(default_factory=SleapInferStatus)
    detectron2: Detectron2Status = field(default_factory=Detectron2Status)
    reconstruction_3d: Reconstruction3DStatus = field(
        default_factory=Reconstruction3DStatus
    )
    furthest_stage: str = "recorded"

    def to_dict(self) -> dict:
        return {
            "sleap_infer": asdict(self.sleap_infer),
            "detectron2": asdict(self.detectron2),
            "reconstruction_3d": asdict(self.reconstruction_3d),
            "furthest_stage": self.furthest_stage,
        }


def detect_sleap_infer(sleap_output_dir: Path | None, folder_name: str) -> SleapInferStatus:
    status = SleapInferStatus()
    if sleap_output_dir is None:
        return status
    exp_dir = Path(sleap_output_dir) / folder_name
    if not exp_dir.is_dir():
        return status

    status.output_dir = str(exp_dir)
    rounds = sorted(
        child.name
        for child in exp_dir.iterdir()
        if child.is_dir() and child.name not in _NON_ROUND_SUBDIRS
    )
    status.rounds = rounds

    for view in VIEWS:
        # Merged `.slp` lives at the top level, or (for round-tagged merges,
        # e.g. `--sleap-subdir round4_260520`) inside a round subdir.
        merged = (exp_dir / f"{view}.slp").exists() or any(
            (exp_dir / r / f"{view}.slp").exists() for r in rounds
        )
        status.merged[view] = merged
        # chunk files look like `top_cam.frames_0-87149.slp`
        status.n_chunks[view] = len(list(exp_dir.glob(f"{view}.frames_*.slp")))

    status.n_merged_views = sum(1 for v in VIEWS if status.merged.get(v))
    status.merged_all = status.n_merged_views == len(VIEWS)
    status.has_viz = any(exp_dir.glob("*.viz.mp4"))
    # Only count as inferred when there is actual evidence, so a stray/empty
    # output folder does not falsely advance the pipeline funnel.
    status.present = (
        status.n_merged_views > 0
        or any(status.n_chunks.values())
        or bool(rounds)
    )
    return status


def detect_detectron2(detectron2_output_dir: Path | None, folder_name: str) -> Detectron2Status:
    status = Detectron2Status()
    if detectron2_output_dir is None:
        return status
    exp_dir = Path(detectron2_output_dir) / folder_name
    if not exp_dir.is_dir():
        return status

    status.output_dir = str(exp_dir)
    for view in VIEWS:
        masks = list(exp_dir.glob(f"{view}.masks_*.h5"))
        status.n_mask_chunks[view] = len(masks)
    status.n_views = sum(1 for v in VIEWS if status.n_mask_chunks.get(v))
    status.present = status.n_views > 0
    return status


def detect_reconstruction_3d(
    reconstruction_output_dir: Path | None, folder_name: str
) -> Reconstruction3DStatus:
    status = Reconstruction3DStatus()
    if reconstruction_output_dir is None:
        return status
    root = Path(reconstruction_output_dir)

    # 3D CSVs may live either at the recon root (`<exp>_3d.csv`) or inside a
    # per-experiment subdir with per-config subfolders.
    exp_dir = root / folder_name
    csv_files: list[Path] = []
    # Escape the folder name so glob metacharacters in it are matched literally.
    fn_glob = _glob.escape(folder_name)
    if root.is_dir():
        csv_files.extend(root.glob(f"{fn_glob}*_3d*.csv"))
    if exp_dir.is_dir():
        status.output_dir = str(exp_dir)
        # shallow: recon-root/<exp>/*.csv and recon-root/<exp>/<config>/*.csv
        csv_files.extend(exp_dir.glob("*_3d*.csv"))
        csv_files.extend(exp_dir.glob("*/*_3d*.csv"))
        status.configs = sorted(
            child.name for child in exp_dir.iterdir() if child.is_dir()
        )

    names = sorted({p.name for p in csv_files})
    status.csv_files = names
    status.has_identified_csv = any("_identified" in n for n in names)
    status.has_3d_csv = len(names) > 0
    status.present = status.has_3d_csv or bool(status.configs)
    if status.present and status.output_dir is None:
        status.output_dir = str(root)
    return status


def _compute_furthest_stage(
    recorded: bool, status: PipelineStatus
) -> str:
    reached = "recorded" if recorded else STAGE_ORDER[0]
    if status.sleap_infer.merged_all or status.sleap_infer.present:
        reached = "sleap_inferred"
    if status.detectron2.present:
        reached = "detectron2"
    if status.reconstruction_3d.has_3d_csv:
        reached = "reconstructed_3d"
    if status.reconstruction_3d.has_identified_csv:
        reached = "identified_3d"
    return reached


def detect_pipeline_status(
    folder_name: str,
    *,
    recorded: bool = True,
    sleap_output_dir: Path | None = None,
    detectron2_output_dir: Path | None = None,
    reconstruction_output_dir: Path | None = None,
) -> PipelineStatus:
    """Scan every stage output root for one experiment and summarise progress."""
    status = PipelineStatus(
        sleap_infer=detect_sleap_infer(sleap_output_dir, folder_name),
        detectron2=detect_detectron2(detectron2_output_dir, folder_name),
        reconstruction_3d=detect_reconstruction_3d(
            reconstruction_output_dir, folder_name
        ),
    )
    status.furthest_stage = _compute_furthest_stage(recorded, status)
    return status
