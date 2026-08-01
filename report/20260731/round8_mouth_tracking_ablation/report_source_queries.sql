-- DuckDB-compatible source queries for the portable Round8 report.
-- The canonical reviewed evidence remains the bound CSV/JSON/Markdown files.
-- full_metrics
SELECT
    CASE method
        WHEN 'A0' THEN 'A0 centroid reference'
        WHEN 'A1' THEN 'A1 strict mouth'
        WHEN 'A2' THEN 'A2 selected motion'
        WHEN 'A3' THEN 'A3 deferred'
    END AS method,
    female_full_accuracy AS female_best_fixed_rate,
    CAST(round(female_full_accuracy * female_eligible) AS INTEGER) AS female_best_fixed_n,
    female_mapped AS mapped_n,
    female_identity_changes AS identity_changes,
    all_three_direct_fraction AS all_three_direct,
    unavailable_or_unresolved_fraction AS unavailable_fraction,
    identity_count AS identity_segments,
    direct_jump_p95_mm AS jump_p95_mm,
    direct_jump_p99_mm AS jump_p99_mm,
    jump_ge_7_5mm_count AS jumps_ge_7_5_mm,
    jump_gt_8mm_count AS jumps_gt_8_mm,
    residual_p95_mm2,
    residual_p99_mm2
FROM read_csv_auto(
    'quality_review/full_length_independent_method_summary.csv',
    header = true
)
WHERE method IN ('A0', 'A1', 'A2', 'A3')
ORDER BY method;

-- heldout_context
-- Values are the independently reviewed annotation partition reported in
-- QUALITY_REVIEW.md and full_length_annotation_partition_summary.json.
SELECT *
FROM (
    VALUES
        ('A0 centroid reference', 9, 9, 0),
        ('A1 strict mouth',        9, 6, 1),
        ('A2 selected motion',     9, 6, 1),
        ('A3 deferred',            9, 6, 1)
) AS heldout_context(
    method,
    bounded_correct,
    global_correct,
    global_changes
)
ORDER BY method;
