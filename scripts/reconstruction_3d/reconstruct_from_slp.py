from __future__ import annotations

"""
Three-view 3D reconstruction + identity tracking, driven by merged SLEAP
`.slp` files (one per view) produced by the HPC pipeline.

For each frame:
    1. Collect up to `num_individuals` predicted 2D points per view.
    2. Enumerate cross-view assignments, triangulate each hypothesis through
       the resolved public calibration backend, and score it in that backend's
       declared unit (px^2 for polynomial, mm^2 for refractive geometry).
    3. Pick the minimum-cost candidate per frame.
    4. Link the chosen 3D points across frames with a Hungarian NN pass.
    5. Write `<experiment>_3d.csv` with columns [frame, fish_id, X, Y, Z].

CLI:
    python -m scripts.reconstruction_3d.reconstruct_from_slp 20251220_D_1_1_FCMC \\
        --paths-config configs/paths.local.toml \\
        --start-frame 171800 --end-frame 176799
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
    apply_first_frame_id_only,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.load_2d_slp import VIEW_TO_CAM, load_points_by_frame
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.triangulate import compute_candidates


def _resolve_score_gate(
    scoring_cal: Calibration,
    *,
    max_residual_per_fish: float | None,
    max_residual_px2: float | None,
    max_ray_residual_mm2: float | None,
) -> tuple[float | None, str, str | None]:
    """Resolve exactly one score gate without converting between units."""
    provided = [
        ("--max-residual-per-fish", max_residual_per_fish, "px^2"),
        ("--max-residual-px2", max_residual_px2, "px^2"),
        ("--max-ray-residual-mm2", max_ray_residual_mm2, "mm^2"),
    ]
    selected = [(name, value, unit) for name, value, unit in provided if value is not None]
    if len(selected) > 1:
        raise ValueError(
            "Score gates are mutually exclusive; received "
            + ", ".join(name for name, _, _ in selected)
        )
    if not selected:
        return None, scoring_cal.score_unit, None
    name, value, unit = selected[0]
    if unit != scoring_cal.score_unit:
        raise ValueError(
            f"{name} is declared in {unit}, but resolved scoring backend "
            f"{scoring_cal.backend_name!r} reports {scoring_cal.score_unit}. "
            "Select the matching backend-specific gate; thresholds are never "
            "converted between pixels and millimetres."
        )
    return float(value), unit, name


def _json_ready(value):
    """Return a JSON-safe copy of backend provenance (including Path values)."""
    return json.loads(json.dumps(value, default=str))


def _calibration_metadata(
    geometry_cal: Calibration,
    scoring_cal: Calibration,
    *,
    score_gate: float | None,
    score_gate_unit: str,
    score_gate_option: str | None,
    scoring_profile: str,
) -> dict[str, object]:
    def score_definition(calibration: Calibration) -> str:
        profiles = calibration.provenance.get("score_profiles", {})
        profile_info = (
            profiles.get(scoring_profile) if isinstance(profiles, dict) else None
        )
        provenance_definition = (
            profile_info.get("definition")
            if isinstance(profile_info, dict)
            else None
        ) or calibration.provenance.get("score_definition")
        if provenance_definition:
            return str(provenance_definition)
        if calibration.backend_name == "refractive":
            return (
                "mean squared perpendicular distance from the triangulated XYZ "
                "to each observed refracted ray"
            )
        return "sum_squared_reprojection_error_over_3_cameras"

    return _json_ready(
        {
            "schema_version": 1,
            "camera_order": ["cam0", "cam1", "cam2"],
            "triangulation": {
                "backend": geometry_cal.backend_name,
                "score_unit": geometry_cal.score_unit,
                "score_definition": score_definition(geometry_cal),
                "provenance": geometry_cal.provenance,
                "capabilities": geometry_cal.capabilities,
            },
            "assignment_scoring": {
                "backend": scoring_cal.backend_name,
                "score_unit": scoring_cal.score_unit,
                "score_definition": score_definition(scoring_cal),
                "scoring_profile": scoring_profile,
                "provenance": scoring_cal.provenance,
                "capabilities": scoring_cal.capabilities,
            },
            "score_gate": {
                "value": score_gate,
                "unit": score_gate_unit,
                "cli_option": score_gate_option,
            },
            "missing_view_policy": {
                "observation_marker": "NaN camera row",
                "refractive": "direct intersection of the two observed refracted rays",
                "polynomial": "legacy pairwise-helper completion inside polynomial backend",
                "silent_cross_backend_fallback": False,
            },
            "projection": {
                "backend": geometry_cal.backend_name,
                "unit": "px",
                "capability": bool(geometry_cal.capabilities.get("projection", False)),
                "provenance": geometry_cal.provenance,
                "legacy_projection_fallback_configured": False,
                "silent_cross_backend_fallback": False,
            },
        }
    )


def _print_calibration_resolution(metadata: dict[str, object]) -> None:
    geometry = metadata["triangulation"]
    scoring = metadata["assignment_scoring"]
    gate = metadata["score_gate"]
    print(
        "[calibration] triangulation backend="
        f"{geometry['backend']} score_unit={geometry['score_unit']} "
        f"provenance={json.dumps(geometry['provenance'], sort_keys=True)}"
    )
    print(
        "[calibration] assignment scoring backend="
        f"{scoring['backend']} score_unit={scoring['score_unit']} "
        f"profile={scoring['scoring_profile']} "
        f"definition={scoring['score_definition']} "
        f"gate={gate['value']} {gate['unit']}"
    )


def run_reconstruction_from_slp(
    experiment_name: str,
    *,
    slp_dir: Path,
    model_folder: Path,
    calibration_backend: str = "auto",
    assignment_model_folder: Path | None = None,
    assignment_calibration_backend: str | None = None,
    output_dir: Path,
    num_individuals: int = 3,
    num_candidates: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    anchor_node: str | None = None,
    require_anchor_node: bool = False,
    xyz_bounds: Tuple[float, float] | None = None,
    max_residual_per_fish: float | None = None,
    max_residual_px2: float | None = None,
    max_ray_residual_mm2: float | None = None,
    max_jump_mm: float | None = None,
    max_jump_hard_cap_mm: float | None = None,
    max_gap_frames: int | None = None,
    motion_velocity_prediction_weight: float = 0.0,
    motion_tangent_weight: float = 0.0,
    motion_acceleration_weight: float = 0.0,
    motion_uncertainty_growth_per_frame: float = 0.25,
    max_motion_gap_frames: int | None = None,
    motion_history_frames: int = 2,
    first_frame_id_only: bool = False,
    fill_missing: bool = True,
    shift_to_positive: bool = True,
    plot: bool = True,
    output_tag: str | None = None,
    save_best_json: bool = True,
    save_best_pre_link_json: bool = True,
    tracklet_paths: dict[str, Path] | None = None,
    silhouette_link_paths: dict[str, Path] | None = None,
    appearance_csv: Path | None = None,
    temporal_geometry_annotations: bool = False,
) -> dict:
    slp_paths = {
        view: slp_dir / f"{view}.slp" for view in VIEW_TO_CAM
    }

    points_by_frame = load_points_by_frame(
        slp_paths,
        num_individuals=num_individuals,
        num_candidates=num_candidates,
        start_frame=start_frame,
        end_frame=end_frame,
        anchor_node=anchor_node,
        require_anchor_node=require_anchor_node,
    )
    if not points_by_frame:
        raise RuntimeError(
            f"No frames loaded from {slp_dir} (filters: "
            f"{start_frame}..{end_frame})."
        )

    cal = Calibration(str(model_folder), backend=calibration_backend)
    assignment_cal = (
        Calibration(
            str(assignment_model_folder),
            backend=assignment_calibration_backend or "auto",
        )
        if assignment_model_folder is not None
        else cal
    )
    score_gate, score_gate_unit, score_gate_option = _resolve_score_gate(
        assignment_cal,
        max_residual_per_fish=max_residual_per_fish,
        max_residual_px2=max_residual_px2,
        max_ray_residual_mm2=max_ray_residual_mm2,
    )
    scoring_profile = (
        "production_fast"
        if num_candidates is not None and num_candidates > num_individuals
        else "legacy_combined"
    )
    calibration_metadata = _calibration_metadata(
        cal,
        assignment_cal,
        score_gate=score_gate,
        score_gate_unit=score_gate_unit,
        score_gate_option=score_gate_option,
        scoring_profile=scoring_profile,
    )
    calibration_metadata["temporal_linking"] = {
        "algorithm": (
            "first_frame_id_only"
            if first_frame_id_only
            else "cross_frame_hungarian"
        ),
        "max_jump_mm": max_jump_mm,
        "max_jump_hard_cap_mm": max_jump_hard_cap_mm,
        "max_gap_frames": max_gap_frames,
        "motion_velocity_prediction_weight": (
            motion_velocity_prediction_weight
        ),
        "motion_tangent_weight": motion_tangent_weight,
        "motion_acceleration_weight": motion_acceleration_weight,
        "motion_uncertainty_growth_per_frame": (
            motion_uncertainty_growth_per_frame
        ),
        "max_motion_gap_frames": max_motion_gap_frames,
        "motion_history_frames": motion_history_frames,
    }
    _print_calibration_resolution(calibration_metadata)
    allow_permutations = bool(
        assignment_cal.capabilities.get("hypothesis_scoring", True)
    )

    tracklet_artifacts: dict[str, object] = {}
    source_id_hints_by_frame: dict[int, dict[int, str]] | None = None
    if num_candidates is not None and num_candidates > num_individuals:
        # Each view exposes more candidates than fish: enumerate all triplets
        # through the calibration model, drop those outside xyz_bounds, and
        # globally pick `num_individuals` mutually-exclusive triplets minimising
        # the scoring backend's declared metric. Skips select_best_all_frames
        # since the picker already produces one optimal set per frame.
        from scripts.reconstruction_3d.triangulate_extra_candidates import (
            compute_best_assignments_per_frame,
        )
        best = compute_best_assignments_per_frame(
            cal,
            points_by_frame,
            scoring_cal=assignment_cal,
            xyz_bounds=xyz_bounds,
            num_individuals=num_individuals,
            num_candidates=num_candidates,
            max_hypothesis_score=score_gate,
        )
        if tracklet_paths:
            import pandas as pd
            from scripts.reconstruction_3d.multiview_tracklet_tracking import (
                build_component_hints,
                apply_targeted_appearance_fallback,
                build_constrained_tracklet_graph,
                build_identified_2d_rows,
                build_observations,
                label_components,
                label_components_from_tracked_geometry,
                load_silhouette_link_edges,
                load_tracklet_slot_map,
                propagate_component_labels_by_exclusion,
            )
            from scripts.reconstruction_3d.tracklet_first_assignment import (
                build_detection_dispositions,
            )

            geometry_bootstrap_best = best
            missing_views = set(VIEW_TO_CAM) - set(tracklet_paths)
            if missing_views:
                raise ValueError(
                    "tracklet-first reconstruction requires all views; missing "
                    + ", ".join(sorted(missing_views))
                )
            frames = sorted(points_by_frame)
            slot_maps: dict[str, dict[int, dict[int, str]]] = {}
            quality_maps = {}
            meta_parts = []
            slot_qc = {}
            for view, tracklet_path in tracklet_paths.items():
                slot_maps[view], quality_maps[view], meta, qc = load_tracklet_slot_map(
                    view,
                    slp_paths[view],
                    Path(tracklet_path),
                    start_frame=frames[0],
                    end_frame=frames[-1],
                    anchor_node=anchor_node,
                    require_anchor_node=require_anchor_node,
                )
                meta_parts.append(meta)
                slot_qc[view] = qc
            node_meta = pd.concat(meta_parts, ignore_index=True)
            observations = build_observations(
                best,
                slot_maps,
                quality_maps,
                num_individuals=num_individuals,
                max_residual=score_gate if score_gate is not None else float("inf"),
                hardcore_mm=12.0,
            )
            silhouette_edge_parts = []
            for view, link_path in (silhouette_link_paths or {}).items():
                silhouette_edge_parts.append(
                    load_silhouette_link_edges(view, Path(link_path))
                )
            silhouette_edges = (
                pd.concat(silhouette_edge_parts, ignore_index=True)
                if silhouette_edge_parts
                else None
            )
            component, edges = build_constrained_tracklet_graph(
                observations,
                node_meta.node_id.astype(str).tolist(),
                silhouette_edges=silhouette_edges,
            )
            appearance = pd.read_csv(appearance_csv) if appearance_csv else None
            component_labels, component_sources, seed_frame = label_components(
                best,
                observations,
                component,
                num_individuals=num_individuals,
                appearance=None,
            )
            propagated, exclusion_sources, identity_domains = (
                propagate_component_labels_by_exclusion(
                    slot_maps,
                    component,
                    component_labels,
                    num_individuals=num_individuals,
                )
            )
            component_labels.update(propagated)
            component_sources.update(exclusion_sources)
            appearance_labels, appearance_confidence, appearance_decisions = (
                apply_targeted_appearance_fallback(
                    appearance,
                    component,
                    identity_domains,
                )
            )
            if appearance_labels:
                component_labels.update(appearance_labels)
                component_sources.update(
                    {component_id: "appearance_fallback" for component_id in appearance_labels}
                )
                propagated, more_sources, identity_domains = (
                    propagate_component_labels_by_exclusion(
                        slot_maps,
                        component,
                        component_labels,
                        num_individuals=num_individuals,
                    )
                )
                component_labels.update(propagated)
                component_sources.update(more_sources)
            node_identities = {
                node: component_labels[comp]
                for node, comp in component.items()
                if comp in component_labels
            }
            # Candidate slots are associated with concrete tracklets before the
            # identity-aware global re-solve.
            tracklet_nodes_by_frame: dict[int, dict[str, dict[int, str]]] = {}
            for view, by_frame in slot_maps.items():
                cam = VIEW_TO_CAM[view]
                for frame, mapping in by_frame.items():
                    tracklet_nodes_by_frame.setdefault(int(frame), {})[cam] = mapping
            best = compute_best_assignments_per_frame(
                cal,
                points_by_frame,
                scoring_cal=assignment_cal,
                xyz_bounds=xyz_bounds,
                num_individuals=num_individuals,
                num_candidates=num_candidates,
                max_hypothesis_score=score_gate,
                tracklet_nodes_by_frame=tracklet_nodes_by_frame,
                tracklet_identities=node_identities,
                motion_weight=1.0,
                max_motion_mm=max_jump_mm,
                max_motion_hard_cap_mm=max_jump_hard_cap_mm,
            )
            # Identity/motion constraints must never erase otherwise usable
            # geometry.  If the constrained alternative search cannot match
            # the bootstrap cardinality, retain that frame as geometry-only;
            # reliable per-view biological identities remain in identified_2d
            # and can reacquire on the next compatible frame.
            for frame, bootstrap_info in geometry_bootstrap_best.items():
                constrained_info = best[frame]
                bootstrap_count = sum(
                    np.isfinite(np.asarray(point, float)).all()
                    for point in bootstrap_info.get("groups", {}).values()
                )
                constrained_count = sum(
                    np.isfinite(np.asarray(point, float)).all()
                    for point in constrained_info.get("groups", {}).values()
                )
                if constrained_count >= bootstrap_count:
                    continue
                fallback = {
                    **bootstrap_info,
                    "source_id_hints": {},
                    "selection_diagnostics": {
                        **dict(constrained_info.get("selection_diagnostics", {})),
                        "selection_kind": "geometry_bootstrap_coverage_fallback",
                        "constrained_count": int(constrained_count),
                        "bootstrap_count": int(bootstrap_count),
                        "alternative_search_exhausted": True,
                    },
                }
                best[frame] = fallback
            source_id_hints_by_frame = {
                int(frame): {int(k): str(v) for k, v in info.get("source_id_hints", {}).items()}
                for frame, info in best.items()
                if info.get("source_id_hints")
            }
            final_observations = build_observations(
                best,
                slot_maps,
                quality_maps,
                num_individuals=num_individuals,
                max_residual=score_gate if score_gate is not None else float("inf"),
                hardcore_mm=12.0,
            )
            # Derive annotation-only votes from the final source grouping, not
            # the bootstrap grouping. Equivalent camera permutations can have
            # identical XYZ but different source indices; voting on the final
            # grouping keeps 2D annotations consistent with the delivered 3D.
            if temporal_geometry_annotations:
                annotation_tracked = apply_cross_frame_tracking(
                    best,
                    num_individuals=num_individuals,
                    max_jump_mm=max_jump_mm,
                    max_jump_hard_cap_mm=max_jump_hard_cap_mm,
                    max_gap_frames=max_gap_frames,
                    source_id_hints_by_frame=source_id_hints_by_frame,
                    source_hints_override_motion_gate=False,
                )
                temporal_geometry_labels, temporal_geometry_decisions = (
                    label_components_from_tracked_geometry(
                        final_observations,
                        component,
                        annotation_tracked,
                        component_labels,
                        alignment_frame=seed_frame,
                        min_frames=3,
                        min_fraction=1.0,
                    )
                )
            else:
                temporal_geometry_labels = {}
                temporal_geometry_decisions = pd.DataFrame()
            # Temporal geometry votes are output annotations only. They improve
            # coverage for disconnected 2D components but never become hard
            # triangulation/source constraints.
            temporal_geometry_labels = {
                component_id: identity
                for component_id, identity in temporal_geometry_labels.items()
                if component_id not in component_labels
            }
            output_component_labels = {
                **component_labels,
                **temporal_geometry_labels,
            }
            output_component_sources = {
                **component_sources,
                **{
                    component_id: "unanimous_temporal_geometry_vote_annotation"
                    for component_id in temporal_geometry_labels
                },
            }
            output_identity_domains = dict(identity_domains)
            output_identity_domains.update(
                {
                    component_id: {identity}
                    for component_id, identity in temporal_geometry_labels.items()
                }
            )
            output_node_identities = {
                node: output_component_labels[comp]
                for node, comp in component.items()
                if comp in output_component_labels
            }
            assignments = node_meta.copy()
            assignments["component_id"] = assignments.node_id.map(component)
            assignments["assigned_identity"] = assignments.node_id.map(output_node_identities).fillna("")
            assignments["identity_source"] = assignments.component_id.map(output_component_sources).fillna("unresolved")
            assignments["identity_domain"] = assignments.component_id.map(
                lambda value: "|".join(sorted(output_identity_domains.get(value, set())))
            )
            assignments["appearance_confidence"] = assignments.component_id.map(
                appearance_confidence
            )
            identified_2d = build_identified_2d_rows(
                {view: Path(path) for view, path in tracklet_paths.items()},
                component,
                output_component_labels,
                output_component_sources,
                final_observations,
                start_frame=frames[0],
                end_frame=frames[-1],
            )
            dispositions = pd.DataFrame(
                build_detection_dispositions(
                    points_by_frame,
                    best,
                    slot_nodes_by_frame=tracklet_nodes_by_frame,
                    node_identities=output_node_identities,
                )
            )
            _hints, conflicts = build_component_hints(
                final_observations,
                component,
                output_component_labels,
                num_individuals=num_individuals,
            )
            tracklet_artifacts = {
                "slot_qc": slot_qc,
                "seed_frame": seed_frame,
                "assignments": assignments,
                "identified_2d": identified_2d,
                "dispositions": dispositions,
                "edges": edges,
                "conflicts": conflicts,
                "exclusion_labeled_components": len(exclusion_sources),
                "geometry_bootstrap_best": geometry_bootstrap_best,
                "appearance_decisions": appearance_decisions,
                "temporal_geometry_decisions": temporal_geometry_decisions,
                "temporal_geometry_labeled_components": len(temporal_geometry_labels),
                "geometry_coverage_fallback_frames": sum(
                    info.get("selection_diagnostics", {}).get("selection_kind")
                    == "geometry_bootstrap_coverage_fallback"
                    for info in best.values()
                ),
            }
    else:
        candidates = compute_candidates(
            cal,
            points_by_frame,
            scoring_cal=assignment_cal,
            xyz_bounds=xyz_bounds,
            allow_permutations=allow_permutations,
            num_individuals=num_individuals,
        )
        best = select_best_all_frames(
            candidates,
            xyz_bounds=xyz_bounds,
            num_individuals=num_individuals,
            max_residual_per_fish=score_gate,
        )
    if first_frame_id_only:
        best_tracked = apply_first_frame_id_only(best)
    else:
        best_tracked = apply_cross_frame_tracking(
            best,
            num_individuals=num_individuals,
            max_jump_mm=max_jump_mm,
            max_jump_hard_cap_mm=max_jump_hard_cap_mm,
            max_gap_frames=max_gap_frames,
            velocity_prediction_weight=motion_velocity_prediction_weight,
            flow_weight=motion_tangent_weight,
            acceleration_weight=motion_acceleration_weight,
            motion_uncertainty_growth_per_frame=(
                motion_uncertainty_growth_per_frame
            ),
            max_motion_gap_frames=max_motion_gap_frames,
            motion_history_frames=motion_history_frames,
            source_id_hints_by_frame=source_id_hints_by_frame,
            source_hints_override_motion_gate=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = output_tag or experiment_name

    provenance_path = output_dir / f"{tag}_reconstruction_provenance.json"
    provenance_path.write_text(
        json.dumps(calibration_metadata, indent=2) + "\n", encoding="utf-8"
    )
    artifact_paths: dict[str, str] = {
        "calibration_provenance": str(provenance_path)
    }
    if tracklet_artifacts:
        bootstrap_path = output_dir / f"{tag}_geometry_bootstrap_best_pre_link.json"
        save_best_pre_link_json_to(
            tracklet_artifacts["geometry_bootstrap_best"],
            bootstrap_path,
            num_individuals=num_individuals,
            calibration_metadata=calibration_metadata,
        )
        artifact_paths["geometry_bootstrap"] = str(bootstrap_path)
        for key, suffix in (
            ("assignments", "tracklet_assignments.csv"),
            ("identified_2d", "identified_2d.csv"),
            ("dispositions", "detection_dispositions.csv"),
            ("edges", "cross_view_edges.csv"),
            ("conflicts", "candidate_conflicts.csv"),
            ("appearance_decisions", "appearance_decisions.csv"),
            ("temporal_geometry_decisions", "temporal_geometry_decisions.csv"),
        ):
            path = output_dir / f"{tag}_{suffix}"
            tracklet_artifacts[key].to_csv(path, index=False)
            artifact_paths[key] = str(path)
        summary = {
            "seed_frame": int(tracklet_artifacts["seed_frame"]),
            "slot_matching": tracklet_artifacts["slot_qc"],
            "exclusion_labeled_components": int(
                tracklet_artifacts["exclusion_labeled_components"]
            ),
            "temporal_geometry_labeled_components": int(
                tracklet_artifacts["temporal_geometry_labeled_components"]
            ),
            "disposition_rows": int(len(tracklet_artifacts["dispositions"])),
            "assigned_2d_only_rows": int(
                (tracklet_artifacts["identified_2d"].observation_status == "assigned_2d_only").sum()
            ),
            "conflicts_reoptimized": int(len(tracklet_artifacts["conflicts"])),
            "geometry_coverage_fallback_frames": int(
                tracklet_artifacts["geometry_coverage_fallback_frames"]
            ),
        }
        summary_path = output_dir / f"{tag}_tracklet_first_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifact_paths["summary"] = str(summary_path)

    pre_link_json_path: Path | None = None
    if save_best_pre_link_json:
        pre_link_json_path = output_dir / f"{tag}_best_pre_link.json"
        save_best_pre_link_json_to(
            best,
            pre_link_json_path,
            num_individuals=num_individuals,
            calibration_metadata=calibration_metadata,
        )

    best_json_path: Path | None = None
    if save_best_json:
        best_json_path = output_dir / f"{tag}_best.json"
        _save_best_tracked_json(
            best_tracked,
            best_json_path,
            num_individuals=num_individuals,
            calibration_metadata=calibration_metadata,
        )

    dfs_3d = best_by_frame_to_dfs(
        best_tracked,
        shift_to_positive=shift_to_positive,
        ensure_non_negative=False,
        num_individuals=num_individuals,
    )
    if fill_missing and points_by_frame:
        all_frames = sorted(points_by_frame.keys())
        # Respect the linker's gap policy: never bridge a gap the linker itself gave up
        # on (prevents a transient detection from anchoring a long phantom interpolation).
        dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames, max_gap_frames=max_gap_frames)
    csv_path = dfs_to_single_csv(dfs_3d, set_key=tag, output_dir=str(output_dir))

    png_path = None
    if plot:
        from scripts.reconstruction_3d.visualize_3d import run_plot
        try:
            png_path = run_plot(str(output_dir), tag, xyz_lim=xyz_bounds)
        except Exception as exc:  # noqa: BLE001
            print(f"Plot failed: {exc}")

    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "best_json_path": str(best_json_path) if best_json_path else None,
        "best_pre_link_json_path": (
            str(pre_link_json_path) if pre_link_json_path else None
        ),
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "num_frames": len(points_by_frame),
        "num_individuals": num_individuals,
        "tracklet_artifacts": artifact_paths,
        "calibration": calibration_metadata,
    }


def _src_labels(num_individuals: int) -> list[str]:
    return [chr(ord("a") + i) for i in range(num_individuals)]


def save_best_pre_link_json_to(
    best: dict,
    path: Path,
    *,
    num_individuals: int,
    calibration_metadata: dict[str, object] | None = None,
) -> None:
    """
    Per-frame pre-link artifact: the chosen min-cost candidate before any
    cross-frame ID linking. Re-linking experiments consume this so they can
    iterate without re-running triangulation.

    Format (compact JSON):
        {
          "num_individuals": 3,
          "src_labels": ["a", "b", "c"],
          "frames": {
            "<frame>": {
              "groups":  {"a":[x,y,z], "b":[x,y,z], "c":[x,y,z]},
              "errors":  {"a":<float|null>, ...},
              "perm_cam1": [..],
              "perm_cam2": [..],
              "dropped_high_residual": ["a", ...]
            }, ...
          }
        }
    """
    src = _src_labels(num_individuals)
    out: dict = {}
    nan = float("nan")
    for frame, info in best.items():
        groups = info.get("groups", {}) or {}
        errors = info.get("errors", {}) or {}
        # perm_cam0 is implicit identity in the legacy path (cam0 anchor) but
        # explicit when triangulating from a larger candidate pool than fish
        # count; persist whichever is provided so cross-view linking can map
        # back to the right .slp instance index.
        perm_cam0 = list(info.get("perm_cam0") or list(range(num_individuals)))
        out[int(frame)] = {
            "groups": {
                lb: [float(v) for v in (groups.get(lb) or [nan, nan, nan])]
                for lb in src
            },
            "errors": {
                lb: (
                    float(errors[lb])
                    if (lb in errors and errors[lb] is not None)
                    else None
                )
                for lb in src
            },
            "perm_cam0": perm_cam0,
            "perm_cam1": list(info.get("perm_cam1", []) or []),
            "perm_cam2": list(info.get("perm_cam2", []) or []),
            "recovered": [bool(r) for r in (info.get("recovered") or [])],
            "missing_views": [
                int(v)
                for v in (
                    info.get("missing_views")
                    or info.get("synthetic_views")
                    or []
                )
            ],
            "synthetic_views": [int(v) for v in (info.get("synthetic_views") or [])],
            "source_id_hints": {
                str(k): str(v) for k, v in (info.get("source_id_hints") or {}).items()
            },
            "selection_diagnostics": dict(info.get("selection_diagnostics") or {}),
            "dropped_high_residual": list(
                info.get("dropped_high_residual", []) or []
            ),
            "score_unit": info.get("score_unit"),
            "triangulation_backend": info.get("triangulation_backend"),
            "scoring_backend": info.get("scoring_backend"),
            "scoring_profile": info.get("scoring_profile"),
        }
    payload = {
        "num_individuals": num_individuals,
        "src_labels": src,
        "calibration": _json_ready(calibration_metadata or {}),
        "frames": out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        # Compact: no indent. 348k frames at ~150 B/frame ≈ 50 MB.
        json.dump(payload, fh, separators=(",", ":"))


def load_best_pre_link_json(path: Path) -> dict:
    """
    Load a pre-link JSON produced by `save_best_pre_link_json_to` into the
    same shape `select_best_all_frames` returns, so it can be fed straight
    into `apply_cross_frame_tracking` (or any alternate linker).
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict = {}
    n = int(payload.get("num_individuals", 3))
    for f_str, info in payload["frames"].items():
        out[int(f_str)] = {
            "groups": {k: list(v) for k, v in info["groups"].items()},
            "errors": {
                k: (float(v) if v is not None else None)
                for k, v in info["errors"].items()
            },
            "perm_cam0": list(info.get("perm_cam0") or list(range(n))),
            "perm_cam1": list(info["perm_cam1"]),
            "perm_cam2": list(info["perm_cam2"]),
            "recovered": [bool(v) for v in info.get("recovered", [])],
            "missing_views": [
                int(v)
                for v in info.get(
                    "missing_views", info.get("synthetic_views", [])
                )
            ],
            "synthetic_views": [int(v) for v in info.get("synthetic_views", [])],
            "source_id_hints": {
                int(k): str(v) for k, v in info.get("source_id_hints", {}).items()
            },
            "selection_diagnostics": dict(info.get("selection_diagnostics", {})),
            "dropped_high_residual": list(info.get("dropped_high_residual", [])),
            "score_unit": info.get("score_unit"),
            "triangulation_backend": info.get("triangulation_backend"),
            "scoring_backend": info.get("scoring_backend"),
            "scoring_profile": info.get("scoring_profile"),
        }
    return out


def _save_best_tracked_json(
    best_tracked: dict,
    path: Path,
    *,
    num_individuals: int,
    calibration_metadata: dict[str, object] | None = None,
) -> None:
    """
    Write a compact per-frame mapping usable by the multiview visualiser:
    - `cam0/1/2_to_id`: list of ID names (or null) for each SLEAP instance
      slot, indexed by the detection's position in the per-view sort used
      in `load_2d_slp`. Length = num_individuals.
    """
    id_labels = [f"ID_{i + 1}" for i in range(num_individuals)]
    out: dict = {}
    for frame, info in best_tracked.items():
        id_mapping = info.get("id_mapping") or []
        perm_cam0 = info.get("perm_cam0") or list(range(num_individuals))
        perm_cam1 = info.get("perm_cam1") or list(range(num_individuals))
        perm_cam2 = info.get("perm_cam2") or list(range(num_individuals))

        # detection-slot -> label index. With more candidates than fish, a
        # given .slp instance may not map to any fish (-1).
        det_indices = [d for d in (*perm_cam0, *perm_cam1, *perm_cam2) if d >= 0]
        n_slots = max([num_individuals] + [d + 1 for d in det_indices])
        cam0_label = [-1] * n_slots
        cam1_label = [-1] * n_slots
        cam2_label = [-1] * n_slots
        for lbl_idx, det_idx in enumerate(perm_cam0):
            if 0 <= det_idx < n_slots:
                cam0_label[det_idx] = lbl_idx
        for lbl_idx, det_idx in enumerate(perm_cam1):
            if 0 <= det_idx < n_slots:
                cam1_label[det_idx] = lbl_idx
        for lbl_idx, det_idx in enumerate(perm_cam2):
            if 0 <= det_idx < n_slots:
                cam2_label[det_idx] = lbl_idx

        # id_mapping[i] = label idx for ID_(i+1); invert to label -> ID_name.
        label_to_id = [None] * num_individuals
        for id_idx, src_label in enumerate(id_mapping):
            if 0 <= src_label < num_individuals:
                label_to_id[src_label] = id_labels[id_idx]

        def _view_map(cam_label_list):
            return [
                label_to_id[k] if 0 <= k < num_individuals else None
                for k in cam_label_list
            ]

        # Recovered-by-id flag: indices align with id_labels (ID_1, ID_2, ...).
        recovered_by_id = [
            bool(r) for r in (info.get("recovered") or [False] * num_individuals)
        ]
        out[int(frame)] = {
            "cam0_to_id": _view_map(cam0_label),
            "cam1_to_id": _view_map(cam1_label),
            "cam2_to_id": _view_map(cam2_label),
            "recovered": recovered_by_id,
            "missing_views": list(
                info.get("missing_views", info.get("synthetic_views", []))
            ),
            "synthetic_views": list(info.get("synthetic_views", [])),
            "id_anchor": [bool(v) for v in (info.get("id_anchor") or [])],
            "groups": {k: v for k, v in info.get("groups", {}).items()},
        }

    payload = {
        "num_individuals": num_individuals,
        "id_labels": id_labels,
        "calibration": _json_ready(calibration_metadata or {}),
        "frames": out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "3D reconstruction + identity tracking from merged SLEAP .slp files."
        )
    )
    parser.add_argument(
        "experiment",
        help="Experiment folder name, e.g. 20251220_D_1_1_FCMC.",
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--sleap-output-dir", default=None,
                        help="Overrides sleap_output_dir.")
    parser.add_argument(
        "--sleap-subdir",
        default=None,
        help=(
            "Optional subfolder inside the experiment SLEAP directory, e.g. "
            "round3_3node_full. Defaults to the experiment root."
        ),
    )
    parser.add_argument("--model-folder", default=None,
                        help="Overrides calibration_model_dir.")
    parser.add_argument(
        "--calibration-backend",
        choices=("auto", "refractive", "polynomial"),
        default="auto",
        help=(
            "Triangulation backend policy. 'auto' prefers a complete refractive "
            "camera bundle and otherwise recognizes legacy polynomial/joblib "
            "folders (default: auto)."
        ),
    )
    parser.add_argument(
        "--assignment-model-folder",
        default=None,
        help=(
            "Optional calibration bundle used only for cross-camera candidate "
            "completion and scoring. The --model-folder bundle still computes the "
            "reported 3D coordinates. Use this to freeze production assignments "
            "while testing a new triangulation model."
        ),
    )
    parser.add_argument(
        "--assignment-calibration-backend",
        choices=("auto", "refractive", "polynomial"),
        default=None,
        help=(
            "Backend policy for --assignment-model-folder. Defaults to auto; "
            "ignored when no separate assignment folder is supplied."
        ),
    )
    parser.add_argument("--output-dir", default=None,
                        help="Overrides reconstruction_output_dir.")
    parser.add_argument("--num-individuals", type=int, default=3,
                        help="Max fish slots per view (default 3).")
    parser.add_argument(
        "--num-candidates", type=int, default=None,
        help=(
            "Per-view candidate count to load when SLEAP was run with "
            "more detections per frame than fish (e.g. -n 6 to recover real "
            "fish that would otherwise be displaced by reflections). When set "
            "and > --num-individuals, triangulation switches to global "
            "mutually-exclusive triplet selection across N^3 candidate "
            "triplets (with xyz_bounds gating reflections). Default: same as "
            "--num-individuals (legacy behaviour)."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--tracklets",
        action="append",
        default=[],
        metavar="VIEW=CSV",
        help=(
            "Enable tracklet-first assignment with a concrete-tracklet CSV; "
            "repeat for top_cam, side1_cam, and side2_cam."
        ),
    )
    parser.add_argument(
        "--appearance-csv",
        type=Path,
        default=None,
        help="Optional high-confidence appearance evidence for unresolved components.",
    )
    parser.add_argument(
        "--silhouette-links",
        action="append",
        default=[],
        metavar="VIEW=CSV",
        help=(
            "Optional trusted silhouette fragment-link candidates; repeat for "
            "each view with masks. Edges are accepted only after local 3D "
            "endpoint/turn validation and graph cannot-link checks."
        ),
    )
    parser.add_argument(
        "--temporal-geometry-annotations",
        action="store_true",
        help=(
            "EXPERIMENTAL: annotate disconnected 2D components from unanimous "
            "final-geometry ID votes. These annotations never constrain 3D, "
            "but geometry IDs may still differ from biological IDs across "
            "swaps; disabled by default."
        ),
    )
    parser.add_argument(
        "--anchor-node",
        default="mouth",
        help=(
            "Use this SLEAP node for canonical 2D/3D tracking points "
            "(default: mouth)."
        ),
    )
    parser.add_argument(
        "--use-visible-node-centroid",
        action="store_true",
        help=(
            "Use the centroid of all visible nodes instead of --anchor-node. "
            "This is the round4 production choice: the mouth+tail centroid is "
            "invariant to the known round4 mouth<->tail node swaps. Leave off "
            "(mouth anchor) only for mouth-vs-centroid comparison experiments."
        ),
    )
    parser.add_argument(
        "--xyz-bounds", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="Accept only 3D points inside this cube. Default: no bound.",
    )
    parser.add_argument(
        "--max-residual-per-fish", type=float, default=None,
        help=(
            "Legacy alias for --max-residual-px2 (polynomial backend only). "
            "Points "
            "from the winning candidate whose error exceeds this are set to "
            "NaN. Default: no threshold."
        ),
    )
    parser.add_argument(
        "--require-anchor-node",
        action="store_true",
        help=(
            "Discard an instance when --anchor-node is not finite instead of "
            "falling back to its other visible nodes."
        ),
    )
    parser.add_argument(
        "--max-residual-px2",
        type=float,
        default=None,
        help=(
            "Per-fish polynomial hypothesis-score gate in px^2. Rejected if "
            "the resolved assignment backend reports another score unit."
        ),
    )
    parser.add_argument(
        "--max-ray-residual-mm2",
        type=float,
        default=None,
        help=(
            "Per-fish refractive ray-consistency score gate in mm^2. Rejected "
            "if the resolved assignment backend reports another score unit."
        ),
    )
    parser.add_argument(
        "--max-jump-mm", type=float, default=None,
        help=(
            "Per-frame velocity gate in millimetres. Hungarian matches "
            "farther than this * gap_frames are rejected. Default: no gate."
        ),
    )
    parser.add_argument(
        "--max-jump-hard-cap-mm", type=float, default=None,
        help=(
            "Absolute ceiling on the per-match distance (combined with "
            "--max-jump-mm so the effective threshold at gap=N is "
            "min(max_jump_mm*N, hard_cap)). Prevents long NaN gaps from "
            "accepting a different fish as the same ID. Default: no cap."
        ),
    )
    parser.add_argument(
        "--max-gap-frames", type=int, default=None,
        help=(
            "Retire an ID whose last valid 3D position is older than this "
            "many frames. Default: never retire."
        ),
    )
    motion = parser.add_argument_group("optional cross-frame motion costs")
    motion.add_argument(
        "--motion-velocity-prediction-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for distance from each ID's constant-velocity predicted "
            "position. Opt-in; default 0 preserves the validated linker."
        ),
    )
    motion.add_argument(
        "--motion-tangent-weight",
        type=float,
        default=0.0,
        help="Weight for motion-direction discontinuity (1-cosine; default 0).",
    )
    motion.add_argument(
        "--motion-acceleration-weight",
        type=float,
        default=0.0,
        help="Weight for acceleration discontinuity (default 0).",
    )
    motion.add_argument(
        "--motion-uncertainty-growth-per-frame",
        type=float,
        default=0.25,
        help=(
            "Attenuate motion priors across gaps as "
            "1/(1+growth*(gap-1)); default 0.25."
        ),
    )
    motion.add_argument(
        "--max-motion-gap-frames",
        type=int,
        default=None,
        help=(
            "Stop applying velocity/tangent/acceleration priors beyond this "
            "gap while retaining the normal distance gate."
        ),
    )
    motion.add_argument(
        "--motion-history-frames",
        type=int,
        default=2,
        help=(
            "Recent valid observations used for velocity/tangent estimation. "
            "Two uses the last interval; values above two use a robust median "
            "of pairwise slopes. Default 2."
        ),
    )
    parser.add_argument(
        "--no-shift-to-positive", action="store_true",
        help="Do not shift coordinates so the minimum sits at origin.",
    )
    parser.add_argument(
        "--first-frame-id-only", action="store_true",
        help="Use the first-frame ID mapping for the whole clip instead of "
             "Hungarian NN across frames.",
    )
    parser.add_argument(
        "--no-fill-missing", action="store_true",
        help="Do not linearly interpolate missing 3D frames.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the trajectory plot.",
    )
    parser.add_argument(
        "--no-save-best-json", action="store_true",
        help="Skip writing the per-frame detection->ID mapping JSON.",
    )
    parser.add_argument(
        "--no-save-best-pre-link-json", action="store_true",
        help="Skip writing the pre-link per-frame candidate JSON.",
    )
    parser.add_argument(
        "--output-tag", default=None,
        help="Suffix for output files (default: experiment name).",
    )
    parser.add_argument(
        "--bind-top-id", action="store_true",
        help="After 3D temporal linking, attach top_cam 2D track IDs via "
             "reprojection-NN (writes <tag>_3d_identified.csv). Requires "
             "--top-tracks-csv.",
    )
    parser.add_argument(
        "--top-tracks-csv", default=None, type=Path,
        help="Path to top_cam.tracks.csv produced by track_simple_nn. "
             "Required when --bind-top-id is set.",
    )
    parser.add_argument(
        "--bind-distance", type=float, default=60.0,
        help="Max projection-NN gate in px for the bind step (default 60).",
    )
    parser.add_argument(
        "--bind-min-frames", type=int, default=500,
        help="Drop top tracks shorter than this from the bind anchor pool.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    local = load_local_paths(args.paths_config)
    sleap_output_dir = choose_path(
        args.sleap_output_dir,
        local.sleap_output_dir if local else None,
    )
    model_folder = choose_path(
        args.model_folder,
        local.calibration_model_dir if local else None,
    )
    output_dir = choose_path(
        args.output_dir,
        local.reconstruction_output_dir if local else None,
        sleap_output_dir,
    )
    if sleap_output_dir is None:
        parser.error(
            "sleap_output_dir is required (paths.local.toml or --sleap-output-dir)."
        )
    if model_folder is None:
        parser.error(
            "calibration_model_dir is required "
            "(paths.local.toml or --model-folder)."
        )

    slp_dir = Path(sleap_output_dir) / args.experiment
    if args.sleap_subdir:
        slp_dir = slp_dir / args.sleap_subdir
    if not slp_dir.is_dir():
        parser.error(f"SLEAP output folder not found: {slp_dir}")

    tag = args.output_tag or args.experiment
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else "start"
        end = args.end_frame if args.end_frame is not None else "end"
        tag = f"{tag}_f{start}-{end}"

    tracklet_paths: dict[str, Path] = {}
    for spec in args.tracklets:
        if "=" not in spec:
            parser.error(f"--tracklets expects VIEW=CSV, got {spec!r}")
        view, value = spec.split("=", 1)
        if view not in VIEW_TO_CAM:
            parser.error(f"unknown tracklet view {view!r}")
        path = Path(value)
        if not path.exists():
            parser.error(f"tracklet CSV not found: {path}")
        tracklet_paths[view] = path
    if tracklet_paths and set(tracklet_paths) != set(VIEW_TO_CAM):
        parser.error("--tracklets must be supplied once for all three camera views")

    silhouette_link_paths: dict[str, Path] = {}
    for spec in args.silhouette_links:
        if "=" not in spec:
            parser.error(f"--silhouette-links expects VIEW=CSV, got {spec!r}")
        view, value = spec.split("=", 1)
        if view not in VIEW_TO_CAM:
            parser.error(f"unknown silhouette-link view {view!r}")
        path = Path(value)
        if not path.exists():
            parser.error(f"silhouette-link CSV not found: {path}")
        silhouette_link_paths[view] = path
    if silhouette_link_paths and not tracklet_paths:
        parser.error("--silhouette-links requires --tracklets")

    result = run_reconstruction_from_slp(
        experiment_name=args.experiment,
        slp_dir=slp_dir,
        model_folder=Path(model_folder),
        calibration_backend=args.calibration_backend,
        assignment_model_folder=(
            Path(args.assignment_model_folder) if args.assignment_model_folder else None
        ),
        assignment_calibration_backend=args.assignment_calibration_backend,
        output_dir=Path(output_dir),
        num_individuals=args.num_individuals,
        num_candidates=args.num_candidates,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        anchor_node=None if args.use_visible_node_centroid else args.anchor_node,
        require_anchor_node=args.require_anchor_node,
        xyz_bounds=tuple(args.xyz_bounds) if args.xyz_bounds else None,
        max_residual_per_fish=args.max_residual_per_fish,
        max_residual_px2=args.max_residual_px2,
        max_ray_residual_mm2=args.max_ray_residual_mm2,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        motion_velocity_prediction_weight=(
            args.motion_velocity_prediction_weight
        ),
        motion_tangent_weight=args.motion_tangent_weight,
        motion_acceleration_weight=args.motion_acceleration_weight,
        motion_uncertainty_growth_per_frame=(
            args.motion_uncertainty_growth_per_frame
        ),
        max_motion_gap_frames=args.max_motion_gap_frames,
        motion_history_frames=args.motion_history_frames,
        first_frame_id_only=args.first_frame_id_only,
        fill_missing=not args.no_fill_missing,
        shift_to_positive=not args.no_shift_to_positive,
        plot=not args.no_plot,
        output_tag=tag,
        save_best_json=not args.no_save_best_json,
        save_best_pre_link_json=not args.no_save_best_pre_link_json,
        tracklet_paths=tracklet_paths or None,
        silhouette_link_paths=silhouette_link_paths or None,
        appearance_csv=args.appearance_csv,
        temporal_geometry_annotations=args.temporal_geometry_annotations,
    )

    print()
    print(f"Frames processed : {result['num_frames']}")
    print(f"Num individuals  : {result['num_individuals']}")
    print(f"3D CSV           : {result['csv_path']}")
    if result["png_path"]:
        print(f"Trajectory plot  : {result['png_path']}")
    if result["best_json_path"]:
        print(f"best-tracked JSON: {result['best_json_path']}")
    if result["best_pre_link_json_path"]:
        print(f"pre-link JSON    : {result['best_pre_link_json_path']}")
    for name, path in result.get("tracklet_artifacts", {}).items():
        print(f"tracklet {name:10}: {path}")

    if args.bind_top_id:
        if args.top_tracks_csv is None:
            parser.error("--bind-top-id requires --top-tracks-csv.")
        if not args.top_tracks_csv.exists():
            parser.error(f"top tracks CSV not found: {args.top_tracks_csv}")
        from scripts.reconstruction_3d.bind_topcam_id import bind
        exp_3d_csv = Path(result["csv_path"])
        out_csv = exp_3d_csv.with_name(exp_3d_csv.stem + "_identified.csv")
        calib = Calibration(str(model_folder), backend=args.calibration_backend)
        bind(
            exp_3d_csv=exp_3d_csv,
            top_tracks_csv=args.top_tracks_csv,
            calib=calib,
            out_csv=out_csv,
            bind_distance=args.bind_distance,
            require_min_frames=args.bind_min_frames,
        )
        print(f"3D identified CSV: {out_csv}")


if __name__ == "__main__":
    main()
