# Calibration

This document defines the calibration procedure and conventions.

## Conventions

- Camera coordinate origin and axis directions should be consistent across views.
- World/tank axes should be defined once and used everywhere.

## Notes

If recording through water/glass, consider refraction-aware calibration/triangulation.

## Annotation Workflow

Recommended flow for the colored-bead rigs:

1. Generate draft bead annotations from the raw images:

```powershell
python -m scripts.calibration.preannotate_colored_beads `
  --folder "<calibration_raw_root>" `
  --way_name way1 `
  --output "<draft_output_dir>"
```

2. Open the interactive GUI annotator and correct the draft points:

```powershell
python -m scripts.calibration.annotate_calibration_gui `
  --folder "<calibration_raw_root>" `
  --way_name way1 `
  --draft_root "<draft_output_dir>" `
  --include_bottom_corners `
  --output "<final_output_dir>"
```

In calibration mode the GUI now also writes a per-camera remap preview after
each save under `<final_output_dir>/remap_preview/`, using color + geometry to
suggest more consistent bead IDs across cameras.

After the third camera is saved, the GUI also runs the post-annotation
pipeline automatically:

1. remap bead IDs across the three cameras
2. train a calibration model from the manual annotations by default
3. write point-mapping and reprojection QC outputs
4. print the resulting mm/px error summary in the terminal

The remap files are preview suggestions unless you pass
`--use_remapped_for_post_pipeline`. Automatic click-to-ID snapping is also off
by default; pass `--auto_assign` or press `a` in the GUI if you want to enable
it.

Focused review mode for suspicious beads:

```powershell
python -m scripts.calibration.annotate_calibration_gui `
  --folder "<calibration_raw_root>" `
  --way_name way2 `
  --draft_root "<manual_output_dir>" `
  --output "<manual_output_dir>" `
  --review_csv "<qc_dir>\\way2_worst_beads_to_review.csv"
```

When `--review_csv` is provided, the GUI:

- preloads the existing annotation file from `--output`
- highlights the suspicious beads with red rings
- starts on the first suspicious bead
- uses left/right to move through the review list quickly

Auxiliary 2D-only correspondence mode for pillar-line points:

```powershell
python -m scripts.calibration.annotate_calibration_gui `
  --folder "<calibration_raw_root>" `
  --way_name way1 `
  --mode auxiliary `
  --aux_preset pillar_lines `
  --output "<aux_output_dir>"
```

This writes separate files such as `cali_way1_cam0_aux_pillar_lines.txt`.
Those points are intended for cross-camera correspondence or later geometric
refinement and should not be mixed into the main world-coordinate calibration
fit unless you have trustworthy 3D coordinates for them.

GUI controls:

- click: exact point placement
- drag a box: refine the center inside the selected region
- faint `+` markers: predicted locations for still-missing calibration points when enough draft/manual anchors exist
- right click: clear current point
- left/right arrows: previous or next point
- `n`: skip current point
- `m`: jump to the first missing point
- `a`: toggle auto-assignment to the nearest reasonable missing calibration point; off by default
- `s`: save and close
- `q`: quit without saving
