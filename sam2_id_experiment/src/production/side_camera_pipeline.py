from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


SIDE_VIEWS = ("side1_cam", "side2_cam")


@dataclass(frozen=True)
class SidePilotSpec:
    experiment_name: str = "20251225_K_1_2_FCMC"
    start_frame: int = 176300
    end_frame: int = 181300
    sleap_subdir: str = "round4_260520"
    paths_config: str = "configs/paths.local.toml"


def build_side_pilot_config(
    template: Mapping[str, Any],
    *,
    view: str,
    spec: SidePilotSpec = SidePilotSpec(),
    side_anchor_node: str | None = None,
) -> dict[str, Any]:
    if view not in SIDE_VIEWS:
        raise ValueError(f"view must be one of {', '.join(SIDE_VIEWS)}")

    cfg = deepcopy(dict(template))
    cfg["experiment_name"] = spec.experiment_name
    cfg["view"] = view
    cfg["paths_config"] = spec.paths_config
    cfg["output_dir"] = f"outputs/{{experiment_name}}/{{view}}_f{spec.start_frame}_{spec.end_frame}"
    cfg["start_frame"] = int(spec.start_frame)
    cfg["end_frame"] = int(spec.end_frame)
    cfg["n_animals_expected"] = int(cfg.get("n_animals_expected", 3) or 3)

    sam2 = _section(cfg, "sam2")
    sam2["runner"] = sam2.get("runner", "sam2_video_predictor")
    sam2["prompt_mode"] = "seeded_from_3d"
    sam2["prompts"] = []
    sam2["frame_cache_dir"] = (
        f"outputs/{{experiment_name}}/{{view}}_f{spec.start_frame}_{spec.end_frame}/sam2_frame_cache"
    )

    sleap_gate = _section(cfg, "sleap_gate")
    sleap_gate["enabled"] = True
    sleap_gate["view"] = view
    sleap_gate["sleap_subdir"] = spec.sleap_subdir
    _set_optional_anchor(sleap_gate, side_anchor_node)

    hybrid = _section(cfg, "hybrid_tracking")
    hybrid["view"] = view
    hybrid["sleap_subdir"] = spec.sleap_subdir
    _set_optional_anchor(hybrid, side_anchor_node)

    qc = _section(cfg, "qc")
    qc.setdefault("expected_object_count", cfg["n_animals_expected"])

    cfg["side_pipeline"] = {
        "pilot": True,
        "source_template": "top_cam",
        "review_tool": "sam2_id_experiment/scripts/27_napari_review_tracklets.py",
        "identity_reference": "projected_3d_identity_reference.csv",
    }
    return cfg


def write_side_pilot_configs(
    *,
    template_config: str | Path,
    output_dir: str | Path,
    spec: SidePilotSpec = SidePilotSpec(),
    side_anchor_node: str | None = None,
) -> dict[str, str]:
    template_path = Path(template_config)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for view in SIDE_VIEWS:
        cfg = build_side_pilot_config(
            template,
            view=view,
            spec=spec,
            side_anchor_node=side_anchor_node,
        )
        path = out_dir / f"{spec.experiment_name}_{view}_f{spec.start_frame}_{spec.end_frame}.yaml"
        cfg["paths_config"] = _config_relative_paths_config(path, spec.paths_config)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        paths[view] = str(path)
    return paths


def build_side_pilot_commands(
    *,
    config_paths: Mapping[str, str],
    identified_3d_csv: str,
    best_pre_link_json: str,
    detectron_output_dir: str = "outputs/20251225_K_1_2_FCMC/side_detectron2_model",
) -> list[str]:
    commands: list[str] = []
    coco_dirs: list[str] = []
    for view, config in config_paths.items():
        out_dir = _config_output_dir(config)
        seed_dir = f"{out_dir}/seed_from_3d"
        reference_csv = f"{seed_dir}/projected_3d_identity_reference.csv"
        coco_dir = f"{out_dir}/coco_sam2_reviewed"
        coco_dirs.append(coco_dir)
        commands.extend([
            (
                "python sam2_id_experiment/scripts/24_seed_prompts_from_3d.py "
                f"--identified-3d-csv {identified_3d_csv} "
                f"--best-pre-link-json {best_pre_link_json} "
                f"--paths-config configs/paths.local.toml --view {view} "
                f"--output-dir {seed_dir} --reference-csv {reference_csv}"
            ),
            f"python sam2_id_experiment/scripts/01_run_sam2_segmentation.py --config {config}",
            f"python sam2_id_experiment/scripts/02_convert_masks_to_tracklets.py --config {config} --masks {out_dir}/sam2_masks.h5",
            f"python sam2_id_experiment/scripts/03_filter_clean_tracklets.py --config {config} --tracklets {out_dir}/tracklets_raw.parquet",
            f"python sam2_id_experiment/scripts/07_make_qc_overlay_video.py --config {config} --masks {out_dir}/sam2_masks.h5 --tracklets {out_dir}/tracklets_qc.parquet",
            (
                "python sam2_id_experiment/scripts/27_napari_review_tracklets.py "
                f"--config {config} --masks {out_dir}/sam2_masks.h5 "
                f"--tracklets {out_dir}/tracklets_qc.parquet --reference-points {reference_csv} "
                f"--output-csv {out_dir}/napari_tracklet_review.csv"
            ),
            (
                "python sam2_id_experiment/scripts/03_filter_clean_tracklets.py "
                f"--config {config} --tracklets {out_dir}/tracklets_raw.parquet "
                f"--review-labels {out_dir}/napari_tracklet_review.csv --output-prefix reviewed_"
            ),
            (
                "python sam2_id_experiment/scripts/11_export_coco_from_sam2.py "
                f"--config {config} --masks {out_dir}/sam2_masks.h5 "
                f"--tracklets {out_dir}/reviewed_tracklets_qc.parquet --output {coco_dir}"
            ),
        ])

    commands.append(
        "python sam2_id_experiment/scripts/15_merge_coco_exports.py "
        f"--output outputs/20251225_K_1_2_FCMC/side_coco_merged "
        + " ".join(f"--coco-dir {path}" for path in coco_dirs)
    )
    commands.append(
        "python sam2_id_experiment/scripts/12_train_detectron2_segmentation.py "
        "--train-json outputs/20251225_K_1_2_FCMC/side_coco_merged/annotations.json "
        "--train-images outputs/20251225_K_1_2_FCMC/side_coco_merged/images "
        f"--output {detectron_output_dir}"
    )
    for view, config in config_paths.items():
        out_dir = _config_output_dir(config)
        commands.extend([
            (
                "python sam2_id_experiment/scripts/13_run_detectron2_segmentation.py "
                f"--config {config} --detectron-config {detectron_output_dir}/detectron2_config.yaml "
                f"--model {detectron_output_dir}/model_final.pth "
                f"--output-masks {out_dir}/detectron2_masks.h5 --device cuda"
            ),
            (
                "python sam2_id_experiment/scripts/19_filter_masks_by_sleap_points.py "
                f"--config {config} --masks {out_dir}/detectron2_masks.h5 "
                f"--output-masks {out_dir}/sleap_filtered_detectron2_masks.h5 "
                f"--output-table {out_dir}/detectron2_mask_sleap_coincidence.parquet "
                f"--output-csv {out_dir}/detectron2_mask_sleap_coincidence.csv"
            ),
            (
                "python sam2_id_experiment/scripts/17_link_detectron2_masks_to_tracklets.py "
                f"--config {config} --masks {out_dir}/sleap_filtered_detectron2_masks.h5 "
                f"--output-masks {out_dir}/sleap_filtered_detectron2_linked_masks.h5"
            ),
            (
                "python sam2_id_experiment/scripts/02_convert_masks_to_tracklets.py "
                f"--config {config} --masks {out_dir}/sleap_filtered_detectron2_linked_masks.h5 "
                f"--output {out_dir}/detectron2_tracklets_raw.parquet"
            ),
            (
                "python sam2_id_experiment/scripts/03_filter_clean_tracklets.py "
                f"--config {config} --tracklets {out_dir}/detectron2_tracklets_raw.parquet "
                "--output-prefix detectron2_"
            ),
            (
                "python sam2_id_experiment/scripts/04_make_id_training_dataset.py "
                f"--config {config} --masks {out_dir}/sleap_filtered_detectron2_linked_masks.h5 "
                f"--tracklets {out_dir}/detectron2_tracklets_qc.parquet"
            ),
        ])
    return commands


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, dict):
        value = {}
    cfg[name] = value
    return value


def _set_optional_anchor(section: dict[str, Any], anchor_node: str | None) -> None:
    if anchor_node:
        section["anchor_node"] = str(anchor_node)
    else:
        section.pop("anchor_node", None)


def _config_relative_paths_config(config_path: Path, paths_config: str) -> str:
    path = Path(paths_config)
    if path.is_absolute() or not paths_config:
        return paths_config
    repo_root = Path.cwd()
    target = repo_root / path
    try:
        loader_base = config_path.parent.parents[1]
        return os.path.relpath(target, loader_base)
    except (IndexError, ValueError):
        return str(path)


def _config_output_dir(config_path: str) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    output = str(cfg["output_dir"])
    return (
        output
        .replace("{experiment_name}", str(cfg.get("experiment_name", "")))
        .replace("{view}", str(cfg.get("view", "")))
    )
