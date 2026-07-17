"""Build the canonical portable-report artifact for the consistency audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "analysis_summary.json").read_text(encoding="utf-8"))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = frame.replace({np.nan: None})
    return frame.to_dict(orient="records")


identity_counts = pd.read_csv(HERE / "identity_counts.csv")
temporal_counts = pd.read_csv(HERE / "temporal_identity_counts.csv")
temporal_counts = temporal_counts.loc[
    temporal_counts["temporal_bin"].isin(["early", "middle", "late"])
].copy()
mapping_status = pd.read_csv(HERE / "geometry_mapping_status_counts.csv")
runs = pd.read_csv(HERE / "female_geometry_id_runs.csv")
transitions = pd.read_csv(HERE / "observed_female_id_transitions.csv")
details = pd.read_csv(HERE / "annotation_comparison_details.csv")
tracklet = pd.read_csv(HERE / "exact_frame_tracklet_comparison.csv")

mapping_issues = details.loc[
    details["geometry_id_consistency"].eq("disagree")
    | details["side1_reprojection_px"].gt(8)
    | details["side2_reprojection_px"].gt(8)
].copy()
mapping_issues["issue"] = mapping_issues.apply(
    lambda row: "; ".join(
        item
        for item in (
            "cross-view IDs disagree"
            if row["geometry_id_consistency"] == "disagree"
            else "",
            "selected-ID reprojection exceeds 8 px"
            if (
                (pd.notna(row["side1_reprojection_px"]) and row["side1_reprojection_px"] > 8)
                or (
                    pd.notna(row["side2_reprojection_px"])
                    and row["side2_reprojection_px"] > 8
                )
            )
            else "",
        )
        if item
    ),
    axis=1,
)
mapping_issues = mapping_issues[
    [
        "frame",
        "issue",
        "geometry_id_consistency",
        "consensus_geometry_id",
        "side1_reprojection_px",
        "side2_reprojection_px",
        "candidate_max_reprojection_distance_px",
    ]
].sort_values("frame", kind="stable")

missing_tracklet = tracklet.loc[
    tracklet["consensus_tracklet_id"].isna()
    | tracklet["consensus_tracklet_id"].astype(str).str.strip().eq("")
].copy()
missing_tracklet = missing_tracklet[
    [
        "frame",
        "consensus_geometry_id",
        "clean_tracklet_id",
        "label_status",
        "candidate_frame",
    ]
]

identity = SUMMARY["identity_consistency"]
mapping = SUMMARY["mapping"]
sensitivity = SUMMARY["sensitivity"]
track = SUMMARY["track"]
annotation = SUMMARY["annotation"]
tracklet_summary = SUMMARY["tracklet_consistency"]

headline = pd.DataFrame(
    [
        {
            "mapping_coverage": mapping["mapped_consensus_share_of_confirmed"],
            "two_view_agreement": mapping["two_view_agree_share"],
            "best_fixed_accuracy": identity["best_fixed_geometry_id_accuracy"],
            "strict_fixed_accuracy": sensitivity[
                "strict_two_view_best_fixed_accuracy"
            ],
            "observed_switches": identity["observed_id_switch_count"],
            "observed_runs": identity["observed_run_count"],
        }
    ]
)

sensitivity_rows = pd.DataFrame(
    [
        {
            "subset": "All mapped",
            "annotation_count": mapping["mapped_consensus_count"],
            "best_fixed_accuracy": identity["best_fixed_geometry_id_accuracy"],
            "observed_switches": identity["observed_id_switch_count"],
        },
        {
            "subset": "Strict two-view <=8 px",
            "annotation_count": sensitivity["strict_two_view_count"],
            "best_fixed_accuracy": sensitivity[
                "strict_two_view_best_fixed_accuracy"
            ],
            "observed_switches": sensitivity[
                "strict_two_view_observed_switch_count"
            ],
        },
        {
            "subset": "Very strict candidate <=8 px",
            "annotation_count": sensitivity["very_strict_count"],
            "best_fixed_accuracy": sensitivity[
                "very_strict_best_fixed_accuracy"
            ],
            "observed_switches": sensitivity[
                "very_strict_observed_switch_count"
            ],
        },
    ]
)

audit_source = {
    "id": "female_refractive_consistency_audit",
    "label": "Female GT and round7 refractive-track audit",
    "path": "report/20260716/female_sex_gt_consistency/female_track_consistency_analysis.py",
    "query": {
        "language": "python",
        "description": (
            "Deterministic frame/ID join, exact refractive reprojection, 8 mm "
            "tracklet reconstruction, sensitivity filtering, and sparse transition audit."
        ),
        "executed_at": SUMMARY["generated_at_utc"],
        "tables_used": [
            "annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv",
            "OIST bucket/StephensU/medaka/reconstruction_3d/20251225_K_1_2_FCMC/round7_260704_refractive/round7_260704_refractive_3d.csv",
            "annotations/fish_sex_gt/20251225_K_1_2_FCMC/clean_three_view_candidates_summary.json",
        ],
        "filters": [
            {
                "field": "gt_status/sex_label",
                "value": "confirmed/F for identity metrics",
            },
            {
                "field": "strict sensitivity",
                "value": "two-view agreement and click/reprojection <=8 px",
            },
        ],
        "metric_definitions": {
            "best_fixed_geometry_id_accuracy": (
                "Maximum annotation count for any one consensus geometry ID divided "
                "by confirmed annotations with a non-empty consensus geometry ID."
            ),
            "observed_id_switch_count": (
                "Number of changes in consensus geometry ID between consecutive mapped "
                "annotations sorted by source frame; a lower bound because samples are sparse."
            ),
            "two_view_agreement": (
                "Confirmed annotations where independently selected side-camera poses map "
                "to the same geometry ID divided by all confirmed annotations."
            ),
        },
    },
}

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "Female GT vs. Round7 Refractive Track",
        "description": "Technical consistency audit for sparse female identity labels.",
        "generatedAt": SUMMARY["generated_at_utc"],
        "filters": [],
        "cards": [
            {
                "id": "mapping_coverage_card",
                "description": "Confirmed female annotations with a consensus geometry ID.",
                "dataset": "headline",
                "sourceId": audit_source["id"],
                "metrics": [
                    {
                        "label": "Mapped female labels",
                        "field": "mapping_coverage",
                        "format": "percent",
                    },
                    {
                        "label": "Two-view agreement",
                        "field": "two_view_agreement",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "fixed_id_card",
                "description": "Accuracy from treating one geometry ID as the female for the full recording.",
                "dataset": "headline",
                "sourceId": audit_source["id"],
                "metrics": [
                    {
                        "label": "Best fixed-ID accuracy",
                        "field": "best_fixed_accuracy",
                        "format": "percent",
                    },
                    {
                        "label": "Strict subset",
                        "field": "strict_fixed_accuracy",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "switch_card",
                "description": "Lower-bound changes between consecutive sparse female annotations.",
                "dataset": "headline",
                "sourceId": audit_source["id"],
                "metrics": [
                    {
                        "label": "Observed ID changes",
                        "field": "observed_switches",
                        "format": "number",
                    },
                    {
                        "label": "Observed runs",
                        "field": "observed_runs",
                        "format": "number",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "identity_counts_chart",
                "title": "Confirmed female labels by geometry ID",
                "subtitle": "298 mapped annotations; no fixed geometry ID covers half of the labels.",
                "type": "bar",
                "dataset": "identity_counts",
                "sourceId": audit_source["id"],
                "encodings": {
                    "x": {"field": "geometry_id", "type": "nominal", "label": "Geometry ID"},
                    "y": {
                        "field": "annotation_count",
                        "type": "quantitative",
                        "label": "Confirmed female annotations",
                        "format": "number",
                    },
                    "tooltip": [
                        {
                            "field": "share",
                            "type": "quantitative",
                            "label": "Share",
                            "format": "percent",
                        }
                    ],
                },
                "valueFormat": "number",
                "options": {"palette": {"kind": "categorical", "roots": ["blue"]}},
            },
            {
                "id": "temporal_identity_chart",
                "title": "Female geometry-ID share by candidate time bin",
                "subtitle": "Early, middle, and late bins contain 114, 94, and 88 mapped candidate annotations.",
                "type": "bar",
                "dataset": "temporal_identity_counts",
                "sourceId": audit_source["id"],
                "encodings": {
                    "x": {"field": "temporal_bin", "type": "nominal", "label": "Recording bin"},
                    "y": {
                        "field": "share_within_bin",
                        "type": "quantitative",
                        "label": "Share of mapped female labels",
                        "format": "percent",
                    },
                    "color": {"field": "geometry_id", "type": "nominal", "label": "Geometry ID"},
                    "tooltip": [
                        {
                            "field": "annotation_count",
                            "type": "quantitative",
                            "label": "Annotations",
                            "format": "number",
                        }
                    ],
                },
                "valueFormat": "percent",
                "options": {
                    "grouping": "grouped",
                    "palette": {"kind": "categorical", "roots": ["blue", "gold", "pink"]},
                    "legend": {"show": True, "position": "top"},
                },
            },
            {
                "id": "sensitivity_chart",
                "title": "Best fixed-ID accuracy under stricter evidence filters",
                "subtitle": "The maximum remains below 45% after excluding wide-gate and one-view rows.",
                "type": "bar",
                "dataset": "sensitivity",
                "sourceId": audit_source["id"],
                "encodings": {
                    "x": {"field": "subset", "type": "nominal", "label": "Evidence subset"},
                    "y": {
                        "field": "best_fixed_accuracy",
                        "type": "quantitative",
                        "label": "Best fixed-ID accuracy",
                        "format": "percent",
                    },
                    "tooltip": [
                        {
                            "field": "annotation_count",
                            "type": "quantitative",
                            "label": "Annotations",
                            "format": "number",
                        },
                        {
                            "field": "observed_switches",
                            "type": "quantitative",
                            "label": "Observed changes",
                            "format": "number",
                        },
                    ],
                },
                "valueFormat": "percent",
                "options": {"palette": {"kind": "categorical", "roots": ["blue"]}},
            },
            {
                "id": "mapping_status_chart",
                "title": "Cross-view geometry mapping status",
                "subtitle": "300 confirmed annotations; 283 agree in both side cameras.",
                "type": "bar",
                "dataset": "mapping_status",
                "sourceId": audit_source["id"],
                "encodings": {
                    "x": {"field": "mapping_status", "type": "nominal", "label": "Mapping status"},
                    "y": {
                        "field": "share",
                        "type": "quantitative",
                        "label": "Share of confirmed annotations",
                        "format": "percent",
                    },
                    "tooltip": [
                        {
                            "field": "annotation_count",
                            "type": "quantitative",
                            "label": "Annotations",
                            "format": "number",
                        }
                    ],
                },
                "valueFormat": "percent",
                "options": {"palette": {"kind": "categorical", "roots": ["blue"]}},
            },
        ],
        "tables": [
            {
                "id": "transition_table",
                "title": "Observed female-ID transition brackets",
                "subtitle": "Consecutive mapped annotations that carry different geometry IDs; exact swap frames are not identified.",
                "dataset": "transitions",
                "sourceId": audit_source["id"],
                "defaultSort": {"field": "right_annotation_frame", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "left_annotation_frame", "label": "Left frame", "format": "number"},
                    {"field": "right_annotation_frame", "label": "Right frame", "format": "number"},
                    {"field": "from_geometry_id", "label": "From", "type": "text"},
                    {"field": "to_geometry_id", "label": "To", "type": "text"},
                    {"field": "gap_frames", "label": "Bracket width, frames", "format": "number"},
                    {"field": "candidate_max_reprojection_distance_px", "label": "Right candidate max reproj., px", "format": "number"},
                ],
            },
            {
                "id": "mapping_issue_table",
                "title": "Rows requiring visual re-review",
                "subtitle": "Cross-view disagreements or selected-ID reprojection above the supplied 8 px gate.",
                "dataset": "mapping_issues",
                "sourceId": audit_source["id"],
                "defaultSort": {"field": "frame", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "frame", "label": "Frame", "format": "number"},
                    {"field": "issue", "label": "Issue", "type": "text"},
                    {"field": "geometry_id_consistency", "label": "View status", "type": "text"},
                    {"field": "consensus_geometry_id", "label": "Consensus ID", "type": "text"},
                    {"field": "side1_reprojection_px", "label": "Side1 reproj., px", "format": "number"},
                    {"field": "side2_reprojection_px", "label": "Side2 reproj., px", "format": "number"},
                ],
            },
            {
                "id": "missing_tracklet_table",
                "title": "Mapped rows without a saved provisional tracklet",
                "subtitle": "Two manual-jump records; recomputed exact-frame segments are clean.",
                "dataset": "missing_tracklet",
                "sourceId": audit_source["id"],
                "defaultSort": {"field": "frame", "direction": "asc"},
                "columns": [
                    {"field": "frame", "label": "Frame", "format": "number"},
                    {"field": "consensus_geometry_id", "label": "Geometry ID", "type": "text"},
                    {"field": "clean_tracklet_id", "label": "Recomputed tracklet", "type": "text"},
                    {"field": "label_status", "label": "Exact-frame status", "type": "text"},
                ],
            },
        ],
        "sources": [
            {
                "id": audit_source["id"],
                "label": audit_source["label"],
                "path": audit_source["path"],
            }
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Female GT vs. Round7 Refractive Track",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "**The human female labels are internally strong enough to show that the "
                    "round7 refractive geometry IDs are not stable biological identities.** "
                    f"Of {mapping['mapped_consensus_count']} mapped confirmed labels, the best "
                    f"single ID (`{identity['best_fixed_geometry_id']}`) covers only "
                    f"{identity['best_fixed_geometry_id_accuracy']:.1%}; the annotation sequence "
                    f"contains at least {identity['observed_id_switch_count']} changes across "
                    f"{identity['observed_run_count']} observed runs. A 'change' means adjacent "
                    "mapped GT samples carry different geometry IDs, so it is a lower bound rather "
                    "than an exact count of swap events.\n\n"
                    "**Overall assessment: share the identity-inconsistency conclusion with caveats; "
                    "revise the GT provenance before treating this ledger as a frozen gold release.** "
                    "The ledger uses the placeholder annotator `YOUR_NAME`, and the saved candidate "
                    "queue records an 80 px association gate although the supplied command states 8 px."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": ["mapping_coverage_card", "fixed_id_card", "switch_card"],
            },
            {
                "id": "identity_finding",
                "type": "markdown",
                "body": (
                    "## All three geometry IDs carry the female label\n\n"
                    f"The mapped female labels split across `ID_1` ({int(identity_counts.loc[identity_counts.geometry_id.eq('ID_1'), 'annotation_count'].iloc[0])}), "
                    f"`ID_2` ({int(identity_counts.loc[identity_counts.geometry_id.eq('ID_2'), 'annotation_count'].iloc[0])}), and "
                    f"`ID_3` ({int(identity_counts.loc[identity_counts.geometry_id.eq('ID_3'), 'annotation_count'].iloc[0])}). "
                    "A global `fish_id -> sex` lookup would therefore mislabel 56.7% of these mapped "
                    "female anchors even under the best fixed choice. The implication is operational: "
                    "sex must be attached after identity stitching, not directly to the current geometry ID."
                ),
            },
            {"id": "identity_chart_block", "type": "chart", "chartId": "identity_counts_chart", "layout": "full"},
            {
                "id": "temporal_finding",
                "type": "markdown",
                "body": (
                    "## The dominant label shifts over recording time\n\n"
                    "`ID_3` carries 68.4% of mapped female labels in the early candidate bin, "
                    "whereas `ID_1` carries 55.3% in the middle and 64.8% in the late bin. "
                    "This is temporal reassignment, not a small number of isolated row errors. "
                    "The two manual-jump rows have no temporal-bin metadata and are excluded from this chart."
                ),
            },
            {"id": "temporal_chart_block", "type": "chart", "chartId": "temporal_identity_chart", "layout": "full"},
            {
                "id": "sensitivity_finding",
                "type": "markdown",
                "body": (
                    "## Strict 8 px filtering leaves the conclusion unchanged\n\n"
                    f"The strict two-view subset retains {sensitivity['strict_two_view_count']} rows, "
                    f"{sensitivity['strict_two_view_observed_switch_count']} observed changes, and a "
                    f"best fixed-ID accuracy of only {sensitivity['strict_two_view_best_fixed_accuracy']:.1%}. "
                    f"Adding the candidate-wide <=8 px rule retains {sensitivity['very_strict_count']} rows "
                    f"and {sensitivity['very_strict_observed_switch_count']} changes. This robustness check "
                    "rules out the saved 80 px queue gate as the main explanation for the ID reassignment."
                ),
            },
            {"id": "sensitivity_chart_block", "type": "chart", "chartId": "sensitivity_chart", "layout": "full"},
            {
                "id": "mapping_finding",
                "type": "markdown",
                "body": (
                    "## Cross-view annotations are mostly self-consistent\n\n"
                    f"The two side cameras agree on geometry ID for {mapping['two_view_agree_count']} "
                    f"of {annotation['confirmed_female_count']} confirmed rows ({mapping['two_view_agree_share']:.1%}); "
                    f"{mapping['one_view_count']} rows use one view and {mapping['disagree_count']} disagree. "
                    "All 298 selected consensus coordinates are finite and non-interpolated in the current "
                    "track. This supports using the ledger as sparse identity evidence while keeping the "
                    "disagreement and wide-gate rows out of a locked training set until re-reviewed."
                ),
            },
            {"id": "mapping_chart_block", "type": "chart", "chartId": "mapping_status_chart", "layout": "full"},
            {
                "id": "scope_definitions",
                "type": "markdown",
                "body": (
                    "## Scope, data, and metric definitions\n\n"
                    f"The ledger contains {annotation['row_count']} unique source frames from "
                    f"{annotation['frame_min']:,} to {annotation['frame_max']:,} at 166 fps: "
                    f"{annotation['confirmed_female_count']} confirmed female rows and "
                    f"{annotation['ambiguous_count']} ambiguous row. The compared track is the current "
                    f"{track['backend']} output with {track['row_count']:,} rows, frames "
                    f"{track['frame_min']:,}-{track['frame_max']:,}, IDs `ID_1`-`ID_3`, and score unit "
                    f"`{track['score_unit']}`. 'Best fixed-ID accuracy' is the largest female-label count "
                    "for any one geometry ID divided by all mapped confirmed labels."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Exact joins and reprojection reproduce the GUI evidence\n\n"
                    "The audit validates the canonical `(experiment, frame, annotator)` key and deterministic "
                    "annotation IDs, reconciles the append-only audit ledger, joins every selected geometry ID "
                    "to the exact 3D frame, and reprojects that 3D point with the checked-in fixed-interface "
                    "refractive calibration. It also rebuilds provisional tracklets with the production "
                    "finite, non-interpolated, <=8 mm local-jump rule. The current provenance JSON hash and "
                    "candidate CSV hash match the saved candidate-build records, and the track size matches "
                    "the recorded source."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and rows that need re-review\n\n"
                    "**The 26 observed changes are a lower bound.** Sparse candidate spacing cannot identify "
                    "the exact swap frame and can miss an even number of swaps that return to the same label. "
                    "The reconstruction has no >8 mm consecutive non-interpolated jump, so smooth trajectory "
                    "continuity alone does not protect biological identity at crossings.\n\n"
                    "**The saved evidence is incompatible with an 8 px gate on every row.** The queue summary "
                    "records 80 px, 32/300 candidate rows exceed 8 px at the candidate-wide maximum, and the "
                    "selected-ID reprojection has a 74.4 px side2 outlier at frame 246665. These rows do not "
                    "drive the conclusion, but they should not enter a frozen GT set without visual review."
                ),
            },
            {"id": "mapping_issue_table_block", "type": "table", "tableId": "mapping_issue_table", "layout": "full"},
            {
                "id": "transition_note",
                "type": "markdown",
                "body": (
                    "## Transition brackets localize identity-stitching work\n\n"
                    "The table lists every consecutive mapped GT pair whose geometry ID changes. Use each "
                    "left/right bracket to inspect the full-resolution trajectory and side-camera evidence, "
                    "then place a biological identity boundary or stitching constraint inside the interval."
                ),
            },
            {"id": "transition_table_block", "type": "table", "tableId": "transition_table", "layout": "full"},
            {
                "id": "tracklet_note",
                "type": "markdown",
                "body": (
                    "## Provisional tracklet joins are otherwise exact\n\n"
                    f"All {tracklet_summary['exact_tracklet_match_count']} non-empty saved tracklet IDs match "
                    "the clean segment recomputed at the exact annotation frame. Two mapped manual-jump rows "
                    "have no saved provisional tracklet; their recomputed exact-frame segments are clean."
                ),
            },
            {"id": "missing_tracklet_table_block", "type": "table", "tableId": "missing_tracklet_table"},
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Correct the placeholder annotator value through a controlled ledger migration that "
                    "preserves the old audit records and regenerates deterministic annotation IDs.\n"
                    "2. Re-open the two cross-view disagreement frames (290906 and 338780), the 74.4 px "
                    "outlier at frame 246665, and the remaining rows in the re-review table with an explicit "
                    "8 px gate.\n"
                    "3. Use the 26 transition brackets as sparse identity-stitching supervision; do not assign "
                    "female sex from one global geometry ID.\n"
                    "4. After stitching, rerun this audit and require one stable final female ID across the "
                    "high-confidence subset."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- Which exact frames inside the 26 brackets contain the identity exchanges?\n"
                    "- Do top-camera 2D identity anchors or segmentation identities collapse these runs to "
                    "one biological female?\n"
                    "- Should the two manual-jump records be re-saved from the candidate queue so they carry "
                    "explicit provisional tracklet IDs?"
                ),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": SUMMARY["generated_at_utc"],
        "status": "ready",
        "datasets": {
            "headline": _records(headline),
            "identity_counts": _records(identity_counts),
            "temporal_identity_counts": _records(temporal_counts),
            "sensitivity": _records(sensitivity_rows),
            "mapping_status": _records(mapping_status),
            "transitions": _records(transitions),
            "mapping_issues": _records(mapping_issues),
            "missing_tracklet": _records(missing_tracklet),
            "runs": _records(runs),
        },
        "accessIssues": [],
    },
    "sources": [audit_source],
}

(HERE / "artifact.json").write_text(
    json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print(HERE / "artifact.json")
