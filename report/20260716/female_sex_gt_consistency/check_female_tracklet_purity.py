"""Check whether provisional tracklets mix female and non-female observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.build_female_sex_gt_candidates import (
    build_refractive_geometry_tracklets,
)

from female_track_consistency_analysis import (
    ANNOTATION_CSV,
    GEOMETRY_CSV,
    REPORT_DIR,
    TRACKLET_JUMP_GATE_MM,
)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _purity_table(
    annotations: pd.DataFrame,
    active_tracklets: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    subset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    evidence = annotations[["annotation_id", "frame", "consensus_geometry_id"]].merge(
        active_tracklets,
        on="frame",
        how="inner",
        validate="one_to_many",
    )
    evidence["tracklet_is_female"] = evidence["geometry_source_id"].eq(
        evidence["consensus_geometry_id"]
    )

    grouped = (
        evidence.groupby(["clean_tracklet_id", "geometry_source_id"], sort=True)
        .agg(
            observed_annotation_frames=("frame", "nunique"),
            first_observed_frame=("frame", "min"),
            last_observed_frame=("frame", "max"),
            female_observation_count=("tracklet_is_female", "sum"),
        )
        .reset_index()
    )
    total_counts = evidence.groupby("clean_tracklet_id")["tracklet_is_female"].size()
    grouped["nonfemale_observation_count"] = (
        grouped["clean_tracklet_id"].map(total_counts)
        - grouped["female_observation_count"]
    )
    assignment_sets = (
        evidence.groupby("clean_tracklet_id")["consensus_geometry_id"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
    )
    grouped["female_geometry_ids_observed_in_span"] = grouped[
        "clean_tracklet_id"
    ].map(assignment_sets)
    grouped["distinct_female_geometry_id_count"] = grouped[
        "female_geometry_ids_observed_in_span"
    ].map(lambda value: len(value.split(";")) if value else 0)
    grouped["ever_labeled_female"] = grouped["female_observation_count"].gt(0)
    grouped["mixed_female_nonfemale"] = grouped["female_observation_count"].gt(
        0
    ) & grouped["nonfemale_observation_count"].gt(0)
    grouped["more_than_two_female_id_assignments"] = grouped[
        "distinct_female_geometry_id_count"
    ].gt(2)
    grouped["subset"] = subset_name
    grouped = grouped.merge(
        spans,
        on=["clean_tracklet_id", "geometry_source_id"],
        how="left",
        validate="one_to_one",
    )
    grouped["tracklet_length_frames"] = (
        grouped["tracklet_end_frame"] - grouped["tracklet_start_frame"] + 1
    )

    female_tracklets = grouped.loc[grouped["ever_labeled_female"]].copy()
    mixed = female_tracklets.loc[female_tracklets["mixed_female_nonfemale"]]
    more_than_two = female_tracklets.loc[
        female_tracklets["more_than_two_female_id_assignments"]
    ]
    summary = {
        "subset": subset_name,
        "annotation_count": int(len(annotations)),
        "active_tracklet_evidence_rows": int(len(evidence)),
        "observed_tracklet_count": int(len(grouped)),
        "female_labeled_tracklet_count": int(len(female_tracklets)),
        "female_tracklets_with_one_annotation_frame": int(
            female_tracklets["observed_annotation_frames"].eq(1).sum()
        ),
        "female_tracklets_with_multiple_annotation_frames": int(
            female_tracklets["observed_annotation_frames"].gt(1).sum()
        ),
        "mixed_female_nonfemale_tracklet_count": int(len(mixed)),
        "mixed_female_nonfemale_tracklet_share": float(
            len(mixed) / max(len(female_tracklets), 1)
        ),
        "female_tracklets_with_more_than_two_female_id_assignments": int(
            len(more_than_two)
        ),
        "maximum_distinct_female_geometry_ids_in_one_female_tracklet": int(
            female_tracklets["distinct_female_geometry_id_count"].max()
            if len(female_tracklets)
            else 0
        ),
    }
    evidence["subset"] = subset_name
    return grouped, evidence, summary


def main() -> int:
    annotations = pd.read_csv(ANNOTATION_CSV, keep_default_na=False)
    annotations["frame"] = pd.to_numeric(
        annotations["frame"], errors="raise"
    ).astype(np.int64)
    mapped = annotations.loc[
        annotations["gt_status"].eq("confirmed")
        & annotations["sex_label"].eq("F")
        & annotations["consensus_geometry_id"].astype(str).str.len().gt(0)
    ].copy()

    details = pd.read_csv(REPORT_DIR / "annotation_comparison_details.csv")
    strict_ids = set(
        details.loc[
            details["gt_status"].eq("confirmed")
            & details["sex_label"].eq("F")
            & details["geometry_id_consistency"].eq("agree")
            & details["side1_reprojection_px"].le(8)
            & details["side2_reprojection_px"].le(8)
            & details["side1_click_to_pose_segment_px"].le(8)
            & details["side2_click_to_pose_segment_px"].le(8),
            "annotation_id",
        ]
    )
    strict = mapped.loc[mapped["annotation_id"].isin(strict_ids)].copy()

    geometry = pd.read_csv(
        GEOMETRY_CSV,
        usecols=["frame", "fish_id", "X", "Y", "Z", "is_interpolated"],
        low_memory=False,
    )
    segmented, diagnostics = build_refractive_geometry_tracklets(
        geometry,
        max_local_jump_mm=TRACKLET_JUMP_GATE_MM,
        id_column="fish_id",
    )
    active = segmented.loc[
        segmented["label_status"].eq("clean"),
        ["frame", "geometry_source_id", "clean_tracklet_id"],
    ].copy()
    spans = (
        active.groupby(["clean_tracklet_id", "geometry_source_id"], sort=True)
        .agg(
            tracklet_start_frame=("frame", "min"),
            tracklet_end_frame=("frame", "max"),
        )
        .reset_index()
    )

    all_table, all_evidence, all_summary = _purity_table(
        mapped, active, spans, subset_name="all_mapped"
    )
    strict_table, strict_evidence, strict_summary = _purity_table(
        strict, active, spans, subset_name="strict_two_view_8px"
    )
    combined = pd.concat([all_table, strict_table], ignore_index=True)
    female = combined.loc[combined["ever_labeled_female"]].copy()
    mixed = female.loc[female["mixed_female_nonfemale"]].copy()
    more_than_two = female.loc[
        female["more_than_two_female_id_assignments"]
    ].copy()
    mixed_ids_by_subset = {
        subset: set(group["clean_tracklet_id"])
        for subset, group in mixed.groupby("subset")
    }
    evidence = pd.concat([all_evidence, strict_evidence], ignore_index=True)
    mixed_evidence = pd.concat(
        [
            group.loc[
                group["clean_tracklet_id"].isin(
                    mixed_ids_by_subset.get(subset, set())
                )
            ]
            for subset, group in evidence.groupby("subset", sort=False)
        ],
        ignore_index=True,
    ).sort_values(["subset", "clean_tracklet_id", "frame"], kind="stable")

    output_columns = [
        "subset",
        "clean_tracklet_id",
        "geometry_source_id",
        "tracklet_start_frame",
        "tracklet_end_frame",
        "tracklet_length_frames",
        "observed_annotation_frames",
        "female_observation_count",
        "nonfemale_observation_count",
        "female_geometry_ids_observed_in_span",
        "distinct_female_geometry_id_count",
        "mixed_female_nonfemale",
        "more_than_two_female_id_assignments",
    ]
    female[output_columns].to_csv(
        REPORT_DIR / "female_tracklet_purity.csv", index=False
    )
    mixed[output_columns].to_csv(
        REPORT_DIR / "mixed_female_tracklets.csv", index=False
    )
    more_than_two[output_columns].to_csv(
        REPORT_DIR / "female_tracklets_with_more_than_two_id_assignments.csv",
        index=False,
    )
    mixed_evidence[
        [
            "subset",
            "clean_tracklet_id",
            "geometry_source_id",
            "annotation_id",
            "frame",
            "consensus_geometry_id",
            "tracklet_is_female",
        ]
    ].to_csv(REPORT_DIR / "mixed_female_tracklet_evidence.csv", index=False)
    payload = {
        "definition": (
            "A female-labeled tracklet is mixed when it is the selected female at "
            "one annotation frame but a different geometry ID is selected as female "
            "at another annotation frame while the same clean tracklet is active."
        ),
        "all_mapped": all_summary,
        "strict_two_view_8px": strict_summary,
        "tracklet_segmentation_diagnostics": diagnostics,
        "limitations": [
            "Female GT distinguishes F from not-F but does not distinguish the two males.",
            "A single positive annotation cannot prove purity over the full tracklet.",
            "Sparse annotations can miss identity changes between sampled frames.",
        ],
        "mixed_tracklets": _records(mixed[output_columns]),
        "mixed_tracklet_frame_evidence": _records(
            mixed_evidence[
                [
                    "subset",
                    "clean_tracklet_id",
                    "geometry_source_id",
                    "frame",
                    "consensus_geometry_id",
                    "tracklet_is_female",
                ]
            ]
        ),
        "tracklets_with_more_than_two_id_assignments": _records(
            more_than_two[output_columns]
        ),
    }
    (REPORT_DIR / "female_tracklet_purity_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
