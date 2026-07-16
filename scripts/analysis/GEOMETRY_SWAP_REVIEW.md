# Geometry-only swap review

This workflow is for finding hidden biological-ID swaps inside otherwise smooth
anonymous 3D geometry trajectories. It never edits the reconstruction CSV.
Human review is stored as a small event ledger and replayed later to create
discontinuous, conservative training tracklets.

ID mappings are independent for `top_cam`, `side1_cam`, `side2_cam`, and the
`3d` track panel. This lets a switch that appears in only one camera be recorded
without changing the other cameras or the geometry track.

## Current K_1_2 inputs

- Geometry source: `round7_260704_3d.csv` (`fish_id=ID_1/ID_2/ID_3`).
- Display labels: `G1/G2/G3`; these are anonymous geometry slots, not female or
  male identities.
- Human-reviewed identities: `H1/H2/H3`. Mapping these to `F/M1/M2` is a
  separate optional step.
- Interpolated geometry is visible in the video/GUI but is never exported as a
  clean training observation.

The prepared local review directory is:

```text
runs/round7_full_qc/20251225_K_1_2_FCMC/geometry_swap_review/
```

It contains the 451-event candidate queue and an initially empty event ledger.

## Launch the GUI

Install the optional GUI dependencies if necessary:

```powershell
uv sync --extra gui
```

Then run:

```powershell
.venv\Scripts\python.exe -m scripts.analysis.geometry_swap_review `
  20251225_K_1_2_FCMC `
  --geometry-csv "Z:\StephensU\medaka\reconstruction_3d\round7_260704\20251225_K_1_2_FCMC\round7_260704_3d.csv" `
  --video-dir "Z:\StephensU\medaka\medaka_202512\20251225_K_1_2_FCMC" `
  --slp-dir "Z:\StephensU\medaka\sleap_output\20251225_K_1_2_FCMC\round7_260704" `
  --calibration-model-dir calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models `
  --events-csv runs\round7_full_qc\20251225_K_1_2_FCMC\geometry_swap_review\geometry_review_events.csv `
  --candidate-csv runs\round7_full_qc\20251225_K_1_2_FCMC\geometry_swap_review\geometry_review_candidates.csv `
  --audit-jsonl runs\round7_full_qc\20251225_K_1_2_FCMC\geometry_swap_review\geometry_review_audit.jsonl
```

The GUI reads original source frames. The separate stride-10 MP4 is for fast
scanning only; do not estimate an exact swap boundary from its encoded frame
number. Camera annotations are rendered live in the GUI with compact one-pixel
SLEAP arrows, small mouth markers, and small identity text.

Every raw SLEAP mouth-tail prediction is drawn in gray. Geometry slots are
matched one-to-one to SLEAP instances using the visible-node centroid, because
the production 3D geometry is centroid-based. The colored `G/H` identity marker
is then drawn at that matched instance's **mouth**, not at the centroid. If no
instance passes the 80 px matching gate, no colored 2D identity point is shown.
The `.slp` files are indexed on startup and individual frames are loaded on
demand, so the full-length reviewer opens without first materializing every
prediction in memory.

The matching centroid itself is never displayed. In raw-overlay videos, the
complete SLEAP pose remains gray and only the small identity dot and label are
colored at the mouth.

## Controls

| Action | Button / key | Meaning |
| --- | --- | --- |
| Select target | Click a panel or use the target menu | Select top, side 1, side 2, or 3D; the yellow border is the active target |
| Swap | `G1 <-> G2` / `Q` | Exchange only this pair in the selected target starting at the current source frame |
| Swap | `G1 <-> G3` / `W` | Exchange only this pair in the selected target starting at the current source frame |
| Swap | `G2 <-> G3` / `E` | Exchange only this pair in the selected target starting at the current source frame |
| Ambiguous start/end | `I`, then `O` | Leave the inclusive interval unresolved only in the selected target |
| Single-G false positive | Choose G1/G2/G3, then press `F` at the first and last FP frame | Mark only that matched camera pose as FP; other IDs and later mapping remain valid |
| Node/link QC | Choose a camera, G slot, and `bad mouth`, `bad tail`, or `wrong mouth-tail link`; press `K` at the first and last affected frame | Record a node-specific defect without discarding the whole pose or changing an ID |
| Break | `B` | End selected-scope training fragments at this frame |
| Direct identity assignment | `A` | Choose H1/H2/H3 for every G slot using dropdowns after uncertainty |
| Reviewed, no swap | `R` | Mark the selected target checked without changing identities |
| Review checkpoint | `C` | Record that the selected camera has been reviewed sequentially through the current source frame |
| Undo | `Ctrl+Z` | Remove the last event added in this session |
| Source-frame step | Left/right | Move exactly one source frame |
| Larger step | Shift or Ctrl + left/right | Move 10 or 100 source frames |
| Candidate navigation | `P` / `N` | Previous or next risk event |
| Raw camera preview | `U` or `Camera overlays` checkbox | Instantly hide/show all camera arrows, markers, labels, and panel text; annotations are not edited |
| Zoom | `Zoom -`, `Fit`, `Zoom +` | Scale the complete four-panel image |
| Mouse zoom | `Ctrl` + mouse wheel | Zoom around the current scroll position |
| Keyboard zoom | `Ctrl+-`, `Ctrl+=`, `Ctrl+0` | Zoom out, zoom in, or fit to the window |

When zoomed in, use the horizontal and vertical scroll bars (or the normal
mouse-wheel scrolling) to inspect a specific camera panel.

A swap at source frame `f` means that the selected target's mapping changes
between `f-1` and `f`. Each saved event shows its scope in brackets. The four
mapping states are listed separately beside the panels. The event list
autosaves after every action. The append-only audit JSONL records adds,
deletes, and undo operations.

For an interval where the exact crossing cannot be seen:

1. Press `I` on the first uncertain frame.
2. Move to the last uncertain frame and press `O`.
3. Move to the first clearly interpretable frame and press `A` to re-anchor the
   complete mapping.

The ambiguous interval remains unlabeled even when an anchor is supplied. All
frames after it also remain unresolved until that anchor is recorded.

For a false SLEAP prediction affecting only one assigned geometry slot, do not
use `I/O`. Select the camera, choose the affected G slot, press `F` on the first
FP frame, move to the last FP frame, and press `F` again. Only that G marker is
replaced by a magenta `G/FP` cross during the interval. The other two identities
stay colored, and no mapping anchor is needed afterward.

If the mouth remains correct but the tail is duplicated, misplaced, or linked
to the wrong mouth, do not mark the complete G pose as false positive. Select
that camera and G slot, choose `bad tail` or `wrong mouth-tail link`, press `K`
on the first affected frame, and press `K` again on the last. Use `bad mouth`
only when the mouth coordinate itself is invalid. These magenta node/link marks
are diagnostic: they do not change the G-to-H mapping or delete otherwise valid
mouth coordinates.

At the end of a sequential review session, press `C` in each camera at the last
frame actually checked. The three explicit checkpoints are shown beside the
mapping states and avoid inferring review coverage from the last swap event.

## Candidate queue

The queue combines:

- each remaining motion-QC jump frame;
- close approaches below 18 mm;
- geometry steps above 8 mm/source-frame; and
- starts/ends of interpolation gaps at least 30 frames long.

Signals within 166 source frames are merged into one review event. Candidate
rows show scoped progress (`0/4` through `4/4`); a check mark means all four
targets have an annotation, including `reviewed_no_swap`.

Rebuild the queue with `--build-candidates-only`. Extra CSVs may contain
`frame`, `center_frame`, `center`, or `jump_frame`; motion-QC JSON reports are
also supported directly through `--extra-candidates`.

## Export clean tracklets

After reviewing, run:

```powershell
.venv\Scripts\python.exe -m scripts.tracking.export_geometry_swap_tracklets `
  --geometry-csv "Z:\StephensU\medaka\reconstruction_3d\round7_260704\20251225_K_1_2_FCMC\round7_260704_3d.csv" `
  --events-csv runs\round7_full_qc\20251225_K_1_2_FCMC\geometry_swap_review\geometry_review_events.csv `
  --output-dir runs\round7_full_qc\20251225_K_1_2_FCMC\geometry_swap_review\clean_tracklets `
  --calibration-model-dir calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models `
  --write-constraints
```

Outputs:

- `corrected_geometry_tracks.csv`: complete audit table with mapping and label
  status on every source row;
- `clean_3d_tracklets.csv`: only finite, measured, non-ambiguous observations;
- `clean_2d_tracklets.csv`: calibration reprojections of those clean 3D rows,
  explicitly marked `coordinate_source=calibration_reprojection`;
- `tracklet_manifest.csv`: one row per clean fragment;
- `cannot_link_tracklets.csv`: unique fragment pairs that coexist in a frame;
- `clean_tracklet_export_summary.json`: counts and provenance.

Clean fragments always split at pair swaps, mapping anchors, explicit breaks,
ambiguous intervals, interpolated regions, and missing-data gaps. No identity is
silently propagated through uncertainty. Geometry exports use only `3d` events
plus legacy events created before scopes were introduced. Camera-specific swaps
stay in the ledger for per-view diagnosis and do not change 3D tracklets.
