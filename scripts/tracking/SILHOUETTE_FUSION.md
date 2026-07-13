# Silhouette–SLEAP fusion

`silhouette_sleap_fusion.py` treats instance segmentation as an independent
observation channel for concrete SLEAP tracklets. It does not assume that a
Detectron2 mask ID is a biological identity.

## Evidence generated

For each SLEAP mouth–tail skeleton and mask candidate in a frame, the fusion
stage measures:

- mouth and tail containment;
- centerline coverage by the mask;
- width-sampled body-capsule coverage;
- skeleton/mask centroid distance normalized by body length and mask radius;
- mouth–tail versus mask-major-axis agreement;
- one-to-one Hungarian assignment cost and alternative margin;
- whether one mask covers multiple skeletons;
- whether the two endpoints fall in different masks;
- consecutive-mask IoU, centroid displacement, and area change.

Masks are also linked independently over time using Hungarian assignment on
predicted centroid, cropped-mask IoU, area, and major-axis continuity. These
short mask tracklets may bridge SLEAP fragment gaps, but ambiguous mask links
are downgraded and never override geometry.

## Status semantics

Every observation receives one of three top-level evidence states:

- `support`: `strong_match` or `supported_match`; the independent silhouette
  agrees with the SLEAP skeleton.
- `unknown`: missing masks, ambiguous assignment, a mask covering multiple
  skeletons, mouth/tail falling in different masks, modest/low overlap, or
  temporal mask discontinuity. These rows remain in their SLEAP tracklet but
  cannot extend a trusted mask segment.
- `contradiction`: currently limited to `strong_contradiction_low_overlap`.
  This requires a complete frame-level one-to-one assignment, unique support
  for every other fish-mask pair, no plausible mask candidate for the affected
  skeleton, high mask score and assignment margin, very low skeleton coverage,
  no endpoint containment, and a sufficiently separated mask centroid.

Only `contradiction` rows are removed and split a concrete SLEAP tracklet.
Unknown evidence is deliberately soft: it records uncertainty without turning
a segmentation failure into an identity discontinuity. Trusted mask segments
contain only strong/supported, temporally consistent rows. When one trusted
mask segment touches multiple SLEAP fragments, the script writes a soft
fragment-link candidate for later 3D validation.

## Example

```powershell
.venv\Scripts\python.exe -m scripts.tracking.silhouette_sleap_fusion `
  --input side1_cam=<side1.tracklets.csv>,<side1.detectron2_masks.h5> `
  --input side2_cam=<side2.tracklets.csv>,<side2.detectron2_masks.h5> `
  --output-dir <fusion_output>
```

Outputs per view:

- `silhouette_annotated.csv`: observation-level evidence and provenance;
- `silhouette_fused_tracklets.csv`: contradiction-split 2D fragments, with
  `silhouette_evidence_state`, `silhouette_soft_reason`, and the stricter
  `silhouette_break_reason` retained for audit;
- `silhouette_concrete_tracklets.csv`: reconstruction-ready accepted rows whose
  `tracklet_id` is the post-split fused ID; `source_tracklet_id` preserves the
  original SLEAP-tracklet provenance;
- `silhouette_tracklet_summary.csv`: aggregate weights for a 3D graph node;
- `silhouette_link_candidates.csv`: trusted mask-supported fragment edges;
- `silhouette_summary.json`: status, evidence-state, soft-reason, and true-break
  counts.

`silhouette_link_candidates.csv` is evidence, not a final link decision. A link
must also pass 3D triangulation, reprojection, motion, and global exclusivity.
The reconstruction CLI loads these soft edges with repeatable
`--silhouette-links VIEW=CSV` arguments. For example:

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.reconstruct_from_slp `
  <experiment> --num-candidates 6 --use-visible-node-centroid `
  --tracklets top_cam=<top.tracklets.csv> `
  --tracklets side1_cam=<fusion>/side1_cam.silhouette_concrete_tracklets.csv `
  --tracklets side2_cam=<fusion>/side2_cam.silhouette_concrete_tracklets.csv `
  --silhouette-links side1_cam=<fusion>/side1_cam.silhouette_link_candidates.csv `
  --silhouette-links side2_cam=<fusion>/side2_cam.silhouette_link_candidates.csv
```

The graph accepts a silhouette edge only when the two 3D endpoints are
temporally ordered, their displacement is within the gap-scaled motion gate,
their velocity turn is compatible, and no simultaneous-fish cannot-link rule
would be violated. Cross-view geometric edges are processed first.

Do not treat reconstruction-local geometry IDs as biological identity labels.
They can permute when detections are missing or candidate assignments are
equivalent. The reconstruction option `--temporal-geometry-annotations` exists
only for controlled experiments and is disabled by default; held-out testing
showed that it creates many within-frame identity conflicts. Use 3D motion and
reprojection as edge costs for global fragment stitching instead.
