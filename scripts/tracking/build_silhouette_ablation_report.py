from __future__ import annotations

"""Build a portable-report artifact for the silhouette tracking ablation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    sql: str,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        # Portable reports reject machine-specific absolute paths. The source
        # files live beside artifact.json/report.html, so their basenames are
        # both sufficient for audit and safe for self-contained packaging.
        "path": path.name,
        "query": {
            "engine": "DuckDB",
            "language": "SQL",
            "description": description,
            "sql": sql.strip(),
        },
    }


def build_artifact(comparison_dir: Path, c_dir: Path) -> dict:
    metrics_path = comparison_dir / "ablation_metrics.csv"
    windows_path = comparison_dir / "worst_qc_windows.csv"
    metrics = pd.read_csv(metrics_path)
    windows = pd.read_csv(windows_path)
    c_edges_path = next(c_dir.glob("*cross_view_edges.csv"))
    edges = pd.read_csv(c_edges_path)
    mask_edges = edges[edges["edge_source"].eq("silhouette_continuity")].copy()
    edge_outcomes = (
        mask_edges.groupby(
            ["eligible", "accepted", "reason", "silhouette_3d_validation"],
            dropna=False,
        )
        .size()
        .rename("edge_count")
        .reset_index()
    )
    edge_outcomes.to_csv(comparison_dir / "silhouette_edge_outcomes.csv", index=False)

    metric_rows = metrics.to_dict("records")
    for row in metric_rows:
        row["coverage_rate"] = row["all_three_coverage_pct"] / 100.0
        row["resolved_2d_rate"] = row["resolved_2d_pct"] / 100.0
    by_variant = metrics.set_index("variant")
    headline = [{
        "a_coverage_rate": float(by_variant.loc["A", "all_three_coverage_pct"] / 100),
        "c_coverage_rate": float(by_variant.loc["C", "all_three_coverage_pct"] / 100),
        "coverage_delta": float(
            (by_variant.loc["C", "all_three_coverage_pct"] - by_variant.loc["A", "all_three_coverage_pct"])
            / 100
        ),
        "a_resolved_2d_rate": float(by_variant.loc["A", "resolved_2d_pct"] / 100),
        "c_resolved_2d_rate": float(by_variant.loc["C", "resolved_2d_pct"] / 100),
        "resolved_2d_delta": float(
            (by_variant.loc["C", "resolved_2d_pct"] - by_variant.loc["A", "resolved_2d_pct"])
            / 100
        ),
        "a_fallback_frames": int(by_variant.loc["A", "geometry_fallback_frames"]),
        "c_fallback_frames": int(by_variant.loc["C", "geometry_fallback_frames"]),
        "fallback_delta": int(
            by_variant.loc["C", "geometry_fallback_frames"]
            - by_variant.loc["A", "geometry_fallback_frames"]
        ),
        "accepted_mask_edges": int(by_variant.loc["C", "accepted_silhouette_edges"]),
        "new_mask_edges": int(by_variant.loc["C", "new_silhouette_edges"]),
    }]
    window_rows = windows.head(20).to_dict("records")
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    metrics_source = source(
        "ablation_metrics_source",
        "Full-window ablation metrics",
        metrics_path,
        "Metrics computed identically for the A, B, and C tracking variants over frames 10000–14999.",
        """
        SELECT
            *,
            all_three_coverage_pct / 100.0 AS coverage_rate,
            resolved_2d_pct / 100.0 AS resolved_2d_rate
        FROM read_csv_auto('ablation_metrics.csv')
        ORDER BY variant
        """,
    )
    headline_metrics_source = source(
        "headline_metrics_source",
        "Headline ablation comparisons",
        metrics_path,
        "Variant-C headline values and deltas versus the SLEAP-only variant A.",
        """
        WITH metrics AS (
            SELECT * FROM read_csv_auto('ablation_metrics.csv')
        )
        SELECT
            MAX(CASE WHEN variant = 'A' THEN all_three_coverage_pct END) / 100.0 AS a_coverage_rate,
            MAX(CASE WHEN variant = 'C' THEN all_three_coverage_pct END) / 100.0 AS c_coverage_rate,
            (MAX(CASE WHEN variant = 'C' THEN all_three_coverage_pct END)
             - MAX(CASE WHEN variant = 'A' THEN all_three_coverage_pct END)) / 100.0 AS coverage_delta,
            MAX(CASE WHEN variant = 'A' THEN resolved_2d_pct END) / 100.0 AS a_resolved_2d_rate,
            MAX(CASE WHEN variant = 'C' THEN resolved_2d_pct END) / 100.0 AS c_resolved_2d_rate,
            (MAX(CASE WHEN variant = 'C' THEN resolved_2d_pct END)
             - MAX(CASE WHEN variant = 'A' THEN resolved_2d_pct END)) / 100.0 AS resolved_2d_delta,
            CAST(MAX(CASE WHEN variant = 'A' THEN geometry_fallback_frames END) AS INTEGER) AS a_fallback_frames,
            CAST(MAX(CASE WHEN variant = 'C' THEN geometry_fallback_frames END) AS INTEGER) AS c_fallback_frames,
            CAST(MAX(CASE WHEN variant = 'C' THEN geometry_fallback_frames END)
                 - MAX(CASE WHEN variant = 'A' THEN geometry_fallback_frames END) AS INTEGER) AS fallback_delta
        FROM metrics
        """,
    )
    windows_source = source(
        "qc_windows_source",
        "Ranked QC windows",
        windows_path,
        "Non-overlapping 71-frame windows ranked from mask conflicts, identity conflicts, motion, turn angle, and A/B coordinate disagreement.",
        """
        SELECT *
        FROM read_csv_auto('worst_qc_windows.csv')
        ORDER BY rank
        LIMIT 20
        """,
    )
    edge_outcomes_path = comparison_dir / "silhouette_edge_outcomes.csv"
    edges_source = source(
        "silhouette_edges_source",
        "Silhouette graph edge audit",
        edge_outcomes_path,
        "Graph decisions for every candidate cross-view and silhouette-continuity edge in variant C.",
        """
        SELECT
            eligible,
            accepted,
            reason,
            silhouette_3d_validation,
            edge_count
        FROM read_csv_auto('silhouette_edge_outcomes.csv')
        ORDER BY edge_count DESC
        """,
    )
    headline_edges_source = source(
        "headline_edges_source",
        "Headline silhouette-link counts",
        edge_outcomes_path,
        "Accepted silhouette-continuity candidates and the subset that created new graph connections.",
        """
        SELECT
            CAST(SUM(CASE WHEN accepted THEN edge_count ELSE 0 END) AS INTEGER) AS accepted_mask_edges,
            CAST(SUM(CASE WHEN accepted AND reason = 'accepted_new_soft_connection'
                          THEN edge_count ELSE 0 END) AS INTEGER) AS new_mask_edges
        FROM read_csv_auto('silhouette_edge_outcomes.csv')
        """,
    )

    title = "Silhouette evidence in three-view fish identity tracking"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Technical ablation of SLEAP-only, segmentation-veto, and segmentation-link identity tracking.",
        "generatedAt": generated,
        "cards": [
            {
                "id": "coverage_card",
                "description": "Frames with finite XYZ for all three fish in variant C.",
                "dataset": "headline",
                "sourceId": "headline_metrics_source",
                "metrics": [
                    {"label": "C all-three coverage", "field": "c_coverage_rate", "format": "percent"},
                    {"label": "versus A", "field": "coverage_delta", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "identity_card",
                "description": "Retained 2D tracklet rows that inherit an identity in variant C.",
                "dataset": "headline",
                "sourceId": "headline_metrics_source",
                "metrics": [
                    {"label": "C resolved 2D", "field": "c_resolved_2d_rate", "format": "percent"},
                    {"label": "versus A", "field": "resolved_2d_delta", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "fallback_card",
                "description": "Frames forced to the geometry-coverage fallback selector.",
                "dataset": "headline",
                "sourceId": "headline_metrics_source",
                "metrics": [
                    {"label": "C fallback frames", "field": "c_fallback_frames", "format": "number"},
                    {"label": "versus A", "field": "fallback_delta", "format": "number", "signed": True},
                ],
            },
            {
                "id": "edges_card",
                "description": "Compatible silhouette-continuity links accepted by the constrained graph.",
                "dataset": "headline",
                "sourceId": "headline_edges_source",
                "metrics": [
                    {"label": "Accepted mask edges", "field": "accepted_mask_edges", "format": "number"},
                    {"label": "New connections", "field": "new_mask_edges", "format": "number"},
                ],
            },
        ],
        "charts": [
            {
                "id": "coverage_chart",
                "title": "All-three-fish 3D coverage",
                "subtitle": "Frames 10000–14999; no spline gap filling",
                "type": "bar",
                "dataset": "metrics",
                "sourceId": "ablation_metrics_source",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "Variant"},
                    "y": {"field": "coverage_rate", "type": "quantitative", "label": "Coverage", "format": "percent"},
                },
                "yAxisTitle": "Coverage",
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "identity_chart",
                "title": "Retained 2D observations with an assigned identity",
                "subtitle": "Resolved rows divided by all retained concrete-tracklet rows",
                "type": "bar",
                "dataset": "metrics",
                "sourceId": "ablation_metrics_source",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "Variant"},
                    "y": {"field": "resolved_2d_rate", "type": "quantitative", "label": "Resolved share", "format": "percent"},
                },
                "yAxisTitle": "Resolved share",
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "risk_chart",
                "title": "Highest-risk QC window centers",
                "subtitle": "Heuristic ranking for review prioritization, not a tracking-accuracy score",
                "type": "bar",
                "dataset": "windows",
                "sourceId": "qc_windows_source",
                "encodings": {
                    "x": {"field": "center_frame", "type": "ordinal", "label": "Center frame"},
                    "y": {"field": "risk_score", "type": "quantitative", "label": "Ranking score"},
                },
                "yAxisTitle": "Ranking score",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "metrics_table",
                "title": "Exact ablation metrics",
                "subtitle": "Identical frame window, calibration, triangulation, and motion gates",
                "dataset": "metrics",
                "sourceId": "ablation_metrics_source",
                "defaultSort": {"field": "variant", "direction": "asc"},
                "columns": [
                    {"field": "variant", "label": "Variant", "type": "text"},
                    {"field": "all_three_coverage_pct", "label": "All-three coverage (%)", "format": "number"},
                    {"field": "geometry_fallback_frames", "label": "Fallback frames", "format": "number"},
                    {"field": "resolved_2d_pct", "label": "Resolved 2D (%)", "format": "number"},
                    {"field": "tracklet_nodes", "label": "Tracklet nodes", "format": "number"},
                    {"field": "unlabeled_tracklet_nodes", "label": "Unlabeled nodes", "format": "number"},
                    {"field": "identity_conflict_rows", "label": "Identity conflicts", "format": "number"},
                    {"field": "accepted_silhouette_edges", "label": "Accepted mask edges", "format": "number"},
                ],
            },
            {
                "id": "windows_table",
                "title": "Ranked non-overlapping QC windows",
                "subtitle": "Top 20 windows, each up to 71 frames",
                "dataset": "windows",
                "sourceId": "qc_windows_source",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "rank", "label": "Rank", "format": "number"},
                    {"field": "start_frame", "label": "Start", "format": "number"},
                    {"field": "end_frame", "label": "End", "format": "number"},
                    {"field": "risk_score", "label": "Risk score", "format": "number"},
                    {"field": "hard_mask_conflicts", "label": "Mask conflicts", "format": "number"},
                    {"field": "identity_conflicts", "label": "Identity conflicts", "format": "number"},
                    {"field": "ab_coordinate_change_mm", "label": "A/B change (mm)", "format": "number"},
                ],
            },
            {
                "id": "edges_table",
                "title": "Silhouette edge outcomes",
                "subtitle": "All 89 locally trusted candidates submitted to variant C",
                "dataset": "edge_outcomes",
                "sourceId": "silhouette_edges_source",
                "defaultSort": {"field": "edge_count", "direction": "desc"},
                "columns": [
                    {"field": "eligible", "label": "3D eligible", "type": "text"},
                    {"field": "accepted", "label": "Accepted", "type": "text"},
                    {"field": "reason", "label": "Graph outcome", "type": "text"},
                    {"field": "silhouette_3d_validation", "label": "3D validation", "type": "text"},
                    {"field": "edge_count", "label": "Edges", "format": "number"},
                ],
            },
        ],
        "sources": [
            metrics_source,
            headline_metrics_source,
            windows_source,
            edges_source,
            headline_edges_source,
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "The current segmentation integration is **not ready for expansion to held-out windows**. "
                    "Hard mask contradictions reduce geometry fallback frames from 12 to 4, but splitting/removing every contradicted observation fragments the identity graph: "
                    "tracklet nodes rise from 330 to 836 and the resolved 2D share falls from 98.88% to 28.06%. "
                    "Trusted mask links accept 65 edges and create 30 new graph connections, yet label only one additional node and do not change any B-versus-C 3D coordinate. "
                    "The next iteration should retain segmentation as confidence and targeted veto evidence while avoiding unconditional hard breaks."
                ),
            },
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["coverage_card", "identity_card", "fallback_card", "edges_card"]},
            {
                "id": "coverage_finding",
                "type": "markdown",
                "body": (
                    "## Geometry remains nearly complete, but segmentation does not improve coverage\n\n"
                    "All-three-fish coverage decreases from **99.72% in A to 99.42% in B/C**. "
                    "B changes 221 coordinate rows relative to A, while C is coordinate-for-coordinate identical to B. "
                    "Thus accepted silhouette links affect component connectivity, not the reconstructed geometry in this experiment."
                ),
            },
            {"id": "coverage_visual", "type": "chart", "chartId": "coverage_chart", "layout": "full"},
            {
                "id": "fragmentation_finding",
                "type": "markdown",
                "body": (
                    "## Hard conflicts overfragment the identity graph\n\n"
                    "The segmentation-veto variants retain high XYZ coverage because triangulation can continue without a labeled 2D component. "
                    "Identity inheritance does not: B leaves 702 of 836 tracklet nodes unlabeled, compared with 50 of 330 in A. "
                    "This is the main driver of the apparent discrepancy between good 3D coverage and poor resolved-identity coverage."
                ),
            },
            {"id": "identity_visual", "type": "chart", "chartId": "identity_chart", "layout": "full"},
            {
                "id": "link_finding",
                "type": "markdown",
                "body": (
                    "## Most accepted mask links are redundant or remain disconnected from seeded identity\n\n"
                    "Of 89 submitted candidates, 65 are accepted: 35 were already connected by geometry and 30 create a new graph connection. "
                    "Twenty-two fail 3D validation and two are blocked by simultaneous-fish cannot-link evidence. "
                    "Despite 30 new connections, C labels only one additional node, showing that local mask continuity alone does not bridge the larger identity-component gaps."
                ),
            },
            {"id": "edge_audit", "type": "table", "tableId": "edges_table", "layout": "full"},
            {
                "id": "qc_finding",
                "type": "markdown",
                "body": (
                    "## Automated QC recovers the reported crossing as the highest-risk scene\n\n"
                    "The frame-11267 window ranks first without being manually forced into the list. "
                    "The ranking combines independent mask contradictions, global identity conflicts, motion, turn angle, and A/B coordinate disagreement. "
                    "It is a review-priority heuristic, not an estimate of swap probability."
                ),
            },
            {"id": "risk_visual", "type": "chart", "chartId": "risk_chart", "layout": "full"},
            {"id": "qc_table", "type": "table", "tableId": "windows_table", "layout": "full"},
            {
                "id": "definitions",
                "type": "markdown",
                "body": (
                    "## Scope and metric definitions\n\n"
                    "The cohort is experiment `20251225_K_1_2_FCMC`, round7 SLEAP predictions, frames 10000–14999, three fish, and three synchronized views. "
                    "A uses node-aware SLEAP concrete tracklets. B removes hard silhouette-conflict rows and splits affected tracklets. C adds trusted mask-continuity edges to B. "
                    "All-three coverage is the share of 5,000 frames with three finite XYZ rows before spline filling. Resolved 2D share is assigned retained tracklet rows divided by all retained tracklet rows."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Methodology held constant across variants\n\n"
                    "All variants use six SLEAP candidates, visible-node centroids, identical calibration, XYZ bounds, residual limits, and motion gates. "
                    "Mask links are one-to-one local Hungarian tracks and must pass endpoint displacement, vector-turn, and global cannot-link constraints. "
                    "Kinematic speed is normalized by the true frame gap; no variant has a speed above 8 mm/frame in this window."
                ),
            },
            {"id": "metric_audit", "type": "table", "tableId": "metrics_table", "layout": "full"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and robustness\n\n"
                    "There is no frame-level biological identity ground truth, so these results measure internal consistency and review burden rather than true swap accuracy. "
                    "A lower fallback count does not prove better identity. The mask-conflict thresholds were developed on this window, and Detectron2 masks include reflections and merges. "
                    "The A/B/C comparison does establish the negative result that C adds no 3D-coordinate change over B and that hard splitting causes a large identity-resolution regression."
                ),
            },
            {
                "id": "recommendation",
                "type": "markdown",
                "body": (
                    "## Recommended next step\n\n"
                    "Do **not** run the current hard-veto policy on the remaining five windows. Replace unconditional row removal with a three-state policy: "
                    "strong mask contradictions stop local linking only when two-view/3D evidence agrees; ambiguous contradictions lower edge weight without splitting; missing masks remain neutral. "
                    "Then rerun A/B/C on frames 10000–14999 and require identity resolution to stay near baseline before freezing thresholds for held-out windows."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Which of the 30 new mask connections join two unlabeled components versus connect to a seeded component? "
                    "Can a windowed global optimizer use mask evidence without turning every single-frame contradiction into a new graph node? "
                    "Manual review of the ranked windows should label whether each hard conflict is a true SLEAP tangent/merge error, a segmentation failure, or a real occlusion."
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "metrics": metric_rows,
                "windows": window_rows,
                "edge_outcomes": edge_outcomes.to_dict("records"),
            },
        },
        "sources": [
            metrics_source,
            headline_metrics_source,
            windows_source,
            edges_source,
            headline_edges_source,
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--variant-c-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = build_artifact(args.comparison_dir, args.variant_c_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
