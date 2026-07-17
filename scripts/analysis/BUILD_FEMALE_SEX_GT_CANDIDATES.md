# Clean three-view female-sex GT candidates

`scripts.analysis.build_female_sex_gt_candidates` proposes exact source frames
for `scripts.analysis.female_sex_gt_annotator`. It measures whether a scene is
clean and easy to annotate. It does **not** decide which fish is female, assign
sex, propagate identity, modify tracking, or use persistent raw SLEAP instance
identity.

The top camera is used only to confirm a clean three-view scene and association.
Only the two side cameras contribute to `view_hint`.

## Geometry source modes

The default mode uses a verified refractive reconstruction directly. Omit
`--clean-tracklets-csv` and provide the reconstructed `_3d.csv`. The generator
automatically resolves the adjacent `_reconstruction_provenance.json`, or it
can be supplied with `--reconstruction-provenance-json`. It refuses direct mode
unless:

- triangulation and projection provenance both say `refractive`;
- the active calibration also resolves to `refractive`; and
- the reconstruction and active calibration name the same three model files.

Direct mode requires `frame`, a geometry ID such as `fish_id`, `X`, `Y`, `Z`,
and `is_interpolated`. It constructs conservative provisional tracklets for
each geometry ID. Missing and interpolated rows break a segment. When the
displacement between consecutive measured frames exceeds
`--max-local-jump-mm`, both boundary rows are marked unresolved and excluded,
and the geometry fragment is split. IDs such as `R3D:ID_1:S000004` identify
only a contiguous geometry fragment; they are not biological identities.

The optional reviewed mode remains available by passing the authoritative
`clean_3d_tracklets.csv` written by
`scripts.tracking.export_geometry_swap_tracklets`. It requires these review
semantics and provenance columns:

```text
frame, clean_tracklet_id, geometry_source_id, geometry_slot,
X, Y, Z, label_status, mapping_resolved, is_interpolated
```

Reviewed mode accepts the repository meanings of `label_status`: `clean`, `interpolated`,
`missing`, `unresolved`, `ambiguous`, and `false_positive`. Unknown statuses,
unknown boolean values, and missing status fields are errors. This prevents an
unreviewed legacy table from being silently treated as clean. It reconciles
every reviewed row with the immutable geometry CSV and verifies that finite XYZ
values agree.

All inputs are opened read-only. Output paths are checked so they cannot
overwrite a clean-tracklet, provenance, geometry, SLP, or video input.

## Eligibility

A proposed frame has exactly three distinct provisional or reviewed tracklets
and three distinct geometry slots. Every row must be resolved, measured rather
than interpolated, finite in XYZ, and free of ambiguous, unresolved, or
false-positive status.

For each camera, the calibration projects the three geometry points into the
image. Raw SLEAP detections are read at that exact source frame through
`IndexedSlpPredictions`. A pose is usable only when both `mouth` and `tail` are
finite. Their mean is the production visible-node centroid. A gated global
Hungarian assignment then selects three different poses for the three geometry
points. Extra detections, including reflections, are allowed and remain
unselected.

The selected poses must also:

- be within `--match-gate-px` of their geometry reprojections in every view;
- have pairwise centroid separation of at least
  `--min-view-separation-px` in every view;
- keep every selected mouth and tail node at least `--border-margin-px` from
  every image border; and
- lie within the common frame extent of all three videos and indexed SLP data.

## Score and temporal selection

The quality score is in `[0, 1]` and contains only annotation-quality signals:

```text
0.15 * cross_view_pose_completeness
+ 0.35 * reprojection
+ 0.25 * separation
+ 0.15 * border_clearance
+ 0.10 * clean_boundary_clearance
```

- `cross_view_pose_completeness` is `1` after eligibility filtering.
- `reprojection` is `clip(1 - max_reprojection / match_gate, 0, 1)`.
- `separation` rises linearly from the configured minimum to saturation at
  three times that minimum.
- `border_clearance` rises linearly from the configured margin to saturation at
  four times that margin.
- `clean_boundary_clearance` rises with distance from the closest clean-interval
  boundary and saturates at `--min-frame-spacing`.

Image sharpness is intentionally not included: computing it would require
random decoding of every eligible frame from three large remote videos. The
score never uses body size, appearance, a sex label, a historical female ID, or
an appearance model. Per-candidate component values are preserved in
`quality_components_json`.

Temporal non-maximum suppression enforces `--min-frame-spacing`. Selection first
offers one middle anchor per clean interval (`priority=3`), then early and late
anchors where spacing permits (`priority=2`). Remaining capacity is filled
round-robin across intervals (`priority=1`). Interval traversal is spread across
the recording, and seeded hashes resolve otherwise equivalent choices. The
final CSV is sorted by descending priority, descending score, and ascending
source frame.

`view_hint=both_sides` when the two side-view quality strengths differ by no
more than `0.05`; otherwise it names the stronger side camera. It is never
`top_cam`.

## Output and audit summary

The candidate CSV contains the annotator's standard fields plus clean-tracklet,
geometry, pose-count, separation, reprojection, border, interval, and quality
provenance. `frame` is always an exact integer source-frame number and
`source` is always `clean_tracklet_three_view`.

`geometry_tracklet_map_json` explicitly maps each frame-local geometry ID to
its reviewed or provisional segment ID. The annotator already records the
selected `side1_geometry_id`, `side2_geometry_id`, and consensus geometry ID,
and preserves candidate extras in `candidate_metadata_json`. A downstream
training-table builder can therefore join a human female selection to the
correct segment at that frame. The selector itself never copies that label to
another frame or across a segment boundary.

The summary JSON records normalized paths, practical SHA-256 hashes (large
video/SLP hashes are skipped with an explicit note), calibration backend and
model provenance, video and SLP extents, first-rejection counts for every rule,
eligible and selected counts, score distributions, temporal-bin coverage,
clean-interval coverage, resolved thresholds, Git revision, and UTC creation
time.

## K_1_2 / round7 PowerShell example

The production directory was inspected on 2026-07-16. Its reconstruction
provenance explicitly records refractive triangulation and projection, score
unit `mm^2`, and the three bundles in
`calibration_results\fixed_refractive_192x192x186`. The 2x2 rendered video in
the same directory is useful for human QC; automatic selection uses the source
videos, SLP files, geometry CSV, and provenance JSON.

```powershell
.venv\Scripts\python.exe -m scripts.analysis.build_female_sex_gt_candidates `
  20251225_K_1_2_FCMC `
  --paths-config configs\paths.local.toml `
  --geometry-csv 'Z:\StephensU\medaka\reconstruction_3d\20251225_K_1_2_FCMC\round7_260704_refractive\round7_260704_refractive_3d.csv' `
  --reconstruction-provenance-json 'Z:\StephensU\medaka\reconstruction_3d\20251225_K_1_2_FCMC\round7_260704_refractive\round7_260704_refractive_reconstruction_provenance.json' `
  --slp-dir 'Z:\StephensU\medaka\sleap_output\20251225_K_1_2_FCMC\round7_260704' `
  --video-dir 'Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC' `
  --calibration-model-dir 'calibration_results\fixed_refractive_192x192x186' `
  --calibration-backend refractive `
  --output-csv 'annotations\fish_sex_gt\20251225_K_1_2_FCMC\clean_three_view_candidates.csv' `
  --summary-json 'annotations\fish_sex_gt\20251225_K_1_2_FCMC\clean_three_view_candidates_summary.json' `
  --target-count 300 `
  --min-frame-spacing 166 `
  --match-gate-px 80 `
  --min-view-separation-px 40 `
  --border-margin-px 20 `
  --max-local-jump-mm 8 `
  --seed 0
```

Launch the annotator with the generated queue:

```powershell
.venv\Scripts\python.exe -m scripts.analysis.female_sex_gt_annotator `
  20251225_K_1_2_FCMC `
  --candidate-csv 'annotations\fish_sex_gt\20251225_K_1_2_FCMC\clean_three_view_candidates.csv' `
  --slp-dir 'Z:\StephensU\medaka\sleap_output\20251225_K_1_2_FCMC\round7_260704' `
  --video-dir 'Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC' `
  --geometry-csv 'Z:\StephensU\medaka\reconstruction_3d\20251225_K_1_2_FCMC\round7_260704_refractive\round7_260704_refractive_3d.csv' `
  --calibration-model-dir 'calibration_results\fixed_refractive_192x192x186' `
  --calibration-backend refractive `
  --annotator YOUR_NAME
```
