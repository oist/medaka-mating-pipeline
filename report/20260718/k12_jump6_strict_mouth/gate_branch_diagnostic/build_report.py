"""Build the portable technical report for the 6 mm gate-branch diagnostic."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_ID = "jump_gate_branch_diagnostic"


def _records(table: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(table.to_json(orient="records"))


def main() -> None:
    transitions = pd.read_csv(HERE / "gate_branch_transitions.csv")
    details = pd.read_csv(HERE / "gate_branch_trigger_details.csv")
    windows = pd.read_csv(HERE / "gate_branch_review_windows.csv")
    summary = json.loads((HERE / "gate_branch_summary.json").read_text(encoding="utf-8"))

    transitions["trigger_time_minutes"] = transitions["trigger_time_seconds"] / 60.0
    transitions["annotation_context"] = transitions[
        "existing_confirmed_female_annotations"
    ].map(lambda value: "Existing female annotation" if value else "No annotation in +/-1 s")
    transitions["new_state_span_seconds"] = transitions["new_state_span_frames"] / 166.0
    transitions["trigger_distance_summary"] = transitions[
        "first_trigger_distances_mm"
    ].astype(str)
    transitions["branch_number"] = transitions["branch_id"].str[1:].astype(int)

    windows["annotation_context"] = windows[
        "existing_confirmed_female_annotations"
    ].map(lambda value: "Existing female annotation" if value else "No annotation in window")
    windows["duration_seconds"] = (
        windows["end_frame"] - windows["start_frame"] + 1
    ) / 166.0

    single_frame_details = int(details["gap_frames"].eq(1).sum())
    single_frame_onsets = 0
    for row in transitions.itertuples(index=False):
        first = details[
            details["branch_id"].eq(row.branch_id)
            & details["frame"].eq(row.trigger_frame)
        ]
        if len(first) and first["gap_frames"].eq(1).any():
            single_frame_onsets += 1
    unannotated_windows = windows[
        windows["existing_confirmed_female_annotations"].eq(0)
    ]
    priority_labels = ", ".join(unannotated_windows["window_id"].astype(str))
    generated_at = datetime.now(timezone.utc).isoformat()

    source = {
        "id": SOURCE_ID,
        "label": "Controlled 6 mm versus 8 mm gate-branch diagnostic",
        "path": "report/20260718/k12_jump6_strict_mouth/gate_branch_diagnostic/find_gate_branches.py",
        "query": {
            "language": "python",
            "description": (
                "Exact-frame comparison of paired 6 mm and 8 mm identity permutations, "
                "verification of effective-gate exclusions, and reconciliation to sparse "
                "confirmed female annotations."
            ),
            "executed_at": generated_at,
            "tables_used": [
                "report/20260718/k12_jump6_strict_mouth/mouth_only_jump6_3d_aligned_to_jump8.csv",
                "report/20260718/k12_jump8_strict_mouth_rerun/mouth_only_jump8_rerun_3d_aligned_to_historical_frame.csv",
                "annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv",
            ],
            "filters": [
                {
                    "field": "direct 3D observation",
                    "value": "finite XYZ and is_interpolated=False in both paired outputs",
                },
                {
                    "field": "verified 6 mm exclusion",
                    "value": "distance > min(6*gap,80) and <= min(8*gap,80) mm",
                },
                {
                    "field": "female annotation context",
                    "value": "gt_status=confirmed and sex_label=F",
                },
            ],
            "metric_definitions": {
                "branch_transition": (
                    "A change in the exact permutation mapping 6 mm IDs to the same "
                    "unlabeled 3D observations assigned by the 8 mm run."
                ),
                "verified_gate_trigger": (
                    "The expected observation under the prior permutation was outside "
                    "the effective 6 mm gate but inside the effective 8 mm gate."
                ),
                "new_state_span_frames": (
                    "Frames from the first common all-three direct frame in the new "
                    "permutation through its last common frame before the next change."
                ),
            },
        },
    }

    cards = [
        {
            "id": "branches_card",
            "description": "Permutation changes with a verified 6-only gate exclusion.",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "metrics": [{"label": "Verified branch transitions", "field": "branches", "format": "number"}],
        },
        {
            "id": "windows_card",
            "description": "Nearby triggers consolidated for efficient visual review.",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "metrics": [{"label": "Review windows", "field": "windows", "format": "number"}],
        },
        {
            "id": "unannotated_card",
            "description": "Review windows with no confirmed female annotation inside them.",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "metrics": [{"label": "Unannotated windows", "field": "unannotated_windows", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "trigger_timeline_chart",
            "title": "Verified gate-trigger moments across the recording",
            "subtitle": (
                "26 branch onsets; vertical position is progress from the 6 mm boundary "
                "(0) to the 8 mm boundary (1), normalized for the observation gap."
            ),
            "type": "scatter",
            "dataset": "branches",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {
                    "field": "trigger_time_minutes",
                    "type": "quantitative",
                    "label": "Source time, minutes",
                    "format": "number",
                },
                "y": {
                    "field": "maximum_gate_interval_fraction",
                    "type": "quantitative",
                    "label": "Position inside 6-to-8 mm gate interval",
                    "format": "number",
                },
                "color": {
                    "field": "annotation_context",
                    "type": "nominal",
                    "label": "Existing female evidence",
                },
                "tooltip": [
                    {"field": "branch_id", "type": "nominal", "label": "Branch"},
                    {"field": "trigger_frame", "type": "quantitative", "label": "Trigger frame", "format": "number"},
                    {"field": "first_trigger_six_ids", "type": "nominal", "label": "6 mm ID"},
                    {"field": "trigger_distance_summary", "type": "nominal", "label": "Trigger distance(s), mm"},
                    {"field": "new_state_span_frames", "type": "quantitative", "label": "New-state span, frames", "format": "number"},
                ],
            },
            "options": {
                "palette": {"kind": "categorical", "roots": ["blue", "orange"]},
                "legend": {"show": True, "position": "top"},
            },
        }
    ]

    tables = [
        {
            "id": "review_windows_table",
            "title": "Female-ID review windows",
            "subtitle": "Twelve source-frame ranges; zero in the annotation column marks the four highest-priority windows.",
            "dataset": "windows",
            "sourceId": SOURCE_ID,
            "defaultSort": {"field": "start_frame", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "window_id", "label": "Window", "type": "text"},
                {"field": "start_frame", "label": "Start frame", "format": "number"},
                {"field": "end_frame", "label": "End frame", "format": "number"},
                {"field": "trigger_count", "label": "Triggers", "format": "number"},
                {"field": "trigger_frames", "label": "Exact trigger frames", "type": "text"},
                {"field": "existing_confirmed_female_annotations", "label": "Existing female labels", "format": "number"},
            ],
        },
        {
            "id": "branch_transitions_table",
            "title": "Exact gate-induced branch transitions",
            "subtitle": "One row per change in the 6 mm-to-8 mm identity permutation, sorted by source frame.",
            "dataset": "branches",
            "sourceId": SOURCE_ID,
            "defaultSort": {"field": "trigger_frame", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "branch_id", "label": "Branch", "type": "text"},
                {"field": "trigger_frame", "label": "Trigger frame", "format": "number"},
                {"field": "first_trigger_six_ids", "label": "Rejected 6 mm ID", "type": "text"},
                {"field": "trigger_distance_summary", "label": "Distance(s), mm", "type": "text"},
                {"field": "first_common_new_state_frame", "label": "First new-state frame", "format": "number"},
                {"field": "new_state_last_common_frame", "label": "Last frame in state", "format": "number"},
                {"field": "old_mapping_6_to_8", "label": "Old 6-to-8 mapping", "type": "text"},
                {"field": "new_mapping_6_to_8", "label": "New 6-to-8 mapping", "type": "text"},
                {"field": "existing_confirmed_female_annotations", "label": "Female labels +/-1 s", "format": "number"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Where the 6 mm gate creates new identity branches"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## Technical summary\n\n"
                "**The 6 mm gate creates 26 verified identity-branch transitions that the paired 8 mm run does not create.** "
                f"They begin at frame {int(transitions.trigger_frame.min()):,} and end at frame {int(transitions.trigger_frame.max()):,}; "
                "each transition is backed by an expected observation that the 8 mm linker accepted but the 6 mm linker excluded.\n\n"
                f"**Start female review with {priority_labels}.** These four consolidated windows contain no existing confirmed female annotation. "
                "The prepared GUI queue contains all 26 exact trigger frames, while the window table reduces navigation to 12 ranges.\n\n"
                "**This proves an algorithmic branch, not that the accepted 8 mm displacement is biologically correct.** Visual side-camera review is still required to distinguish true fast motion from reflection, triangulation error, or the wrong fish."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["branches_card", "windows_card", "unannotated_card"]},
        {
            "id": "timing_finding",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## Branch creation is clustered, not uniform\n\n"
                f"The 26 onsets collapse into 12 review windows. Nineteen onsets include a one-frame 6-to-8 mm rejection; across all affected IDs, {single_frame_details} of {len(details)} verified exclusions occur after a one-frame gap. "
                "The densest region is W10 (frames 133,641-134,654), with seven branch changes and no existing confirmed female label. "
                "In the chart, orange points are triggers lacking an existing label within the +/-166-frame context."
            ),
        },
        {"id": "timeline_chart_block", "type": "chart", "chartId": "trigger_timeline_chart", "layout": "full"},
        {
            "id": "window_finding",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## Four windows should be reviewed first\n\n"
                "W03 (30,646-30,978), W08 (111,681-112,013), W10 (133,641-134,654), and W12 (217,376-217,749) have no confirmed female annotation inside the proposed range. "
                "W12 is especially consequential: its second trigger, B026 at frame 217,583, enters a relative ID permutation that remains through the final common frame 348,599 unless a later biological correction is supplied."
            ),
        },
        {"id": "window_table_block", "type": "table", "tableId": "review_windows_table", "layout": "full"},
        {
            "id": "exact_trigger_finding",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## The exact trigger table separates onset from downstream persistence\n\n"
                "`Trigger frame` is the first verified 6-only gate exclusion. `First new-state frame` is the first later frame where both runs have all three direct observations and their ID permutation is demonstrably different. "
                "The mapping columns show which 6 mm geometry ID lands on each 8 mm physical observation before and after the branch."
            ),
        },
        {"id": "transition_table_block", "type": "table", "tableId": "branch_transitions_table", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## Scope, data, and branch definition\n\n"
                f"The population is the full 348,600-frame K_1_2 Round-7 strict-mouth reconstruction. The paired tracks have {summary['common_all_three_direct_frames']:,} frames where all three observations are finite, direct, and exactly match as an unlabeled set. "
                "A branch is a change in the exact permutation mapping the 6 mm IDs to those same 8 mm observations. The effective gate for a gap of N frames is `min(jump_mm x N, 80 mm)`."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## Exact coordinate matching verifies the gate mechanism\n\n"
                "For every common direct frame, points are matched at 1e-6 mm tolerance and required to form a one-to-one permutation. At each permutation change, the prior mapping is propagated through the short intervening gap. "
                "The first mismatched expected observation is measured from the last direct 6 mm state. A trigger is retained only when its distance is above the 6 mm effective gate and at or below the 8 mm effective gate. All 26 permutation changes pass this test; there are 41 affected-ID details because some transitions reject more than one ID."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": SOURCE_ID,
            "body": (
                "## What this diagnostic does not establish\n\n"
                "The paired comparison establishes the linker's decision boundary and downstream ID permutation; it does not establish which geometry ID is the biological female. The existing female ledger is sparse and cannot detect male-to-male swaps. "
                "`New-state span` ends at the next permutation change and should not be interpreted as a confidence interval. Interpolated rows are excluded from branch verification."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Review the prepared queue in the female annotator\n\n"
                "1. Launch `launch_female_review_gui.ps1 -Annotator <NAME>` from this diagnostic directory.\n"
                "2. Review priority-3 queue rows first; these have no confirmed female label within +/-166 source frames.\n"
                "3. At each trigger, record the female immediately before and after the gate event, using both side views.\n"
                "4. Save into the separate `jump6_gate_branch_female_review.csv` ledger so the existing 300-row female GT remains unchanged."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- At how many of the 26 triggers does the 8 mm continuation follow the true female rather than a reflection or another fish?\n"
                "- Are the long-lived B022 and B026 permutations corrected by top-camera identity binding?\n"
                "- Would a velocity-aware or image-supported gate preserve true 6-to-8 mm motions without accepting wrong-fish reacquisition?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Where the 6 mm gate creates new identity branches",
            "description": "Exact trigger moments and GUI-ready female-ID review windows for K_1_2.",
            "generatedAt": generated_at,
            "filters": [],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": SOURCE_ID, "label": source["label"], "path": source["path"]}],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [{
                    "branches": len(transitions),
                    "windows": len(windows),
                    "unannotated_windows": len(unannotated_windows),
                    "single_frame_onsets": single_frame_onsets,
                }],
                "branches": _records(transitions),
                "windows": _records(windows),
            },
            "accessIssues": [],
        },
        "sources": [source],
    }
    (HERE / "artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {HERE / 'artifact.json'}")


if __name__ == "__main__":
    main()
