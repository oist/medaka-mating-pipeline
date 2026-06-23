# Tracklet Identity Pilot

## Purpose

This pilot compares reproducible methods for assigning short reliable 2D/3D
tracklets to three biological fish identities, while keeping crossing, occlusion,
false-positive, and uncertain moments inspectable.

It is deliberately a small-window experiment layer. It consumes existing
CSV/parquet artifacts from the repository and writes another inspectable result
directory. It does not launch SLEAP, SAM2, Detectron2, idtracker.ai, or full-video
3D reconstruction by default.

## Existing Inputs

- Conservative stIFD-gated tracklets:
  `scripts.tracking.concrete_tracklets`
  writes `frame,tracklet_id,x,y,score`.
- 3D-bound per-view tracklets:
  `scripts.tracking.bind_tracklets_3d`
  writes `<view>.tracklet_bound.csv` with `global_id`, bind residuals, confidence,
  and `in_hardcore`.
- 3D tracks:
  `scripts.reconstruction_3d.reconstruct_from_slp`
  writes `<tag>_3d.csv` and `<tag>_3d_identified.csv`.
- Hard-event tables:
  `scripts.tracking.bind_tracklets_3d` writes `hardcore_events.csv`, and
  `scripts.tracking.resolve_hardcore_events` can classify those events.
- Segmentation/appearance evidence:
  `sam2_id_experiment` scripts can export tracklet identity predictions from
  SAM2/Detectron2 crops or masks.
- UDMT-inspired utilities:
  the local `UDMT/` package provides crossing detection, motion anomaly detection,
  backward consensus repair, and quality metrics.

## Pilot CLI

```bash
python -m scripts.tracking.tracklet_identity_pilot \
    --tracklets top_cam=<tracklet_bind_dir>/top_cam.tracklet_bound.csv \
    --tracklets side1_cam=<tracklet_bind_dir>/side1_cam.tracklet_bound.csv \
    --tracklets side2_cam=<tracklet_bind_dir>/side2_cam.tracklet_bound.csv \
    --csv-3d <reconstruction_dir>/<tag>_3d_identified.csv \
    --events-csv <tracklet_bind_dir>/hardcore_events.csv \
    --windows 176300-181300 \
    --out-dir runs/tracklet_identity_pilot/<experiment>_f176300_181300
```

Use `--dry-run --auto-windows N` to inspect available inputs and write suggested
windows before running the graph experiment:

```bash
python -m scripts.tracking.tracklet_identity_pilot \
    --tracklets top_cam=<tracklet_bind_dir>/top_cam.tracklet_bound.csv \
    --csv-3d <reconstruction_dir>/<tag>_3d_identified.csv \
    --auto-windows 4 \
    --dry-run \
    --out-dir runs/tracklet_identity_pilot/<experiment>_plan
```

The script refuses an implicit full run unless `--allow-full-run` is passed.

## Pilot Windows

Start with explicit small windows that include:

- easy separated-fish periods,
- hard 3D close-contact crossings or occlusions,
- high-speed spike windows,
- false-positive or ambiguous detection windows,
- the early single-fish to three-fish transition, if present.

`--auto-windows` suggests windows from 3D close-contact events, easy-separated
3D intervals, speed spikes, and the earliest available tracklet frames.

## Graph Definition

Nodes are reliable tracklets or tracklet fragments. Edges are possible same-fish
connections within `--max-gap-frames`.

Edge features include:

- time gap,
- endpoint distance and speed,
- simple heading continuity,
- 3D-bound geometry label agreement and confidence,
- appearance label agreement and confidence,
- crossing/occlusion event overlap,
- hybrid and UDMT-inspired edge scores.

## Compared Methods

- `geometry_only`: reconnects fragments using 3D-bound slot labels and bind
  confidence.
- `appearance_only`: uses optional SAM2/Detectron2 crop or mask identity
  predictions.
- `temporal_dynamics`: links by endpoint motion plausibility outside hard-event
  spans.
- `hybrid`: combines geometry, appearance, and temporal evidence.
- `udmt_inspired`: uses temporal evidence but gates uncertain crossing intervals
  to `crossing` instead of forcing a biological ID.

`idtracker.ai v2` is not launched by this pilot. Direct integration is a separate
raw-video job with its own environment and runtime. Once an idtracker export
exists, add it as an external baseline artifact and score it against the same
windows.

## Output Directory

Every non-dry run writes:

```text
config_used.json
pilot.log
pilot_windows.csv
event_windows.csv
tracklet_nodes.csv
tracklet_edges.csv
selected_edges.csv
method_assignments.csv
per_frame_assignments.csv
metrics_summary.csv
event_reconnections.csv
reference_metrics.csv
summary.json
recommendation.md
plots/
videos/video_manifest.csv
```

Use `--render-videos --video top_cam=<video.mp4>` to write short annotated event
clips. Without video arguments, the manifest records that video rendering was
skipped.

## Metrics

The pilot reports:

- grouping into three IDs,
- assigned/unknown/mix/crossing fractions,
- expected per-frame three-ID coverage,
- 2D speed-spike rate after stitching by assigned ID,
- 3D speed-spike metrics when a 3D CSV is provided,
- event reconnection outcome before/after hard events,
- agreement of assigned identity sets across views,
- literal and best-remapped accuracy when reference labels exist.

## First Minimal Experiment

1. Produce or locate `<view>.tracklet_bound.csv` for a short event-rich window.
2. Run `tracklet_identity_pilot --dry-run --auto-windows 4` and inspect
   `pilot_windows.csv`.
3. Run one manual or suggested window with the three bound tracklet CSVs and the
   3D CSV.
4. Inspect `timeline_hybrid.png`, `edge_scores.png`,
   `event_reconnections.csv`, and `recommendation.md`.
5. If the best method leaves hard events unresolved, annotate those event rows
   first.

## Annotation Guidance

If the conclusion is that more annotation is needed, prioritize:

- frames listed in `event_windows.csv` that are `unresolved` or `conflict` in
  `event_reconnections.csv`,
- tracklet-pair same/different labels before and after each hard event,
- crossing/mix labels inside the event span,
- segmentation masks or crop-level fish-ID labels for appearance failures.

A practical first annotation batch is 20-50 clean crops per fish per view plus
10-20 hard-event examples with tracklet-pair same/different labels. This targets
the weakest part of the graph: deciding whether before/after fragments separated
by a hard crossing are the same biological fish.

For the two-male classifier specifically, do not spend the next labeling pass only
on easy separated fish. The current segmentation classifier can already learn those
frames, and the male result is confounded by position. Prioritize close-contact and
crossing-adjacent frames, then train/evaluate with
`sam2_id_experiment.scripts.39_train_identity_classifier` using a whole-window holdout
and its position-only baseline.
