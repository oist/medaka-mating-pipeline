# SAM2-assisted body/fin annotation

`47_sam_assisted_fin_annotator.py` is a local OpenCV GUI for refining an existing
Detectron2 instance (or a SLEAP point supplied in the manifest) with SAM2 image
prompts. It writes new, timestamped masks and provenance records. It never edits
the source mask HDF5, COCO data, or model.

## 1. Make bounded comparison candidates

The comparison script matches Detectron2 and SAM2 masks on shared frames, measures
several dilation radii, renders a contact sheet, and writes an annotation manifest.
SAM2 is treated only as a pseudo-reference; these metrics are not fin accuracy.

```powershell
.venv\Scripts\python.exe sam2_id_experiment\scripts\46_compare_mask_expansion.py `
  --detectron-masks outputs\20251225_K_1_2_FCMC\side1_cam_f176300_181300_sam2_fulltest_firstprompt\detectron2_masks.h5 `
  --sam-masks outputs\20251225_K_1_2_FCMC\side1_cam_f176300_181300_sam2_fulltest_firstprompt\sam2_masks.h5 `
  --image-dir outputs\20251225_K_1_2_FCMC\side1_cam_f176300_181300_sam2_fulltest_firstprompt\coco_sam2_reviewed\images `
  --experiment 20251225_K_1_2_FCMC --view side1_cam --radii 0,2,4,8,12 `
  --output-dir runs\fin_segmentation_audit\20251225_K_1_2_FCMC\side1_cam
```

Repeat `--manifest` to move across views in one GUI session. Every row may point
at a different image and mask HDF5.

## 2. Validate, then launch on Windows

```powershell
.venv\Scripts\python.exe sam2_id_experiment\scripts\47_sam_assisted_fin_annotator.py `
  --manifest runs\fin_segmentation_audit\20251225_K_1_2_FCMC\side1_cam\annotation_candidates.csv `
  --manifest runs\fin_segmentation_audit\20251225_K_1_2_FCMC\side2_cam\annotation_candidates.csv `
  --output-dir runs\fin_annotations\20251225_K_1_2_FCMC\session_01 `
  --sam-model-cfg sam2_hiera_l.yaml `
  --checkpoint checkpoints\sam2_hiera_large.pt --device cuda --dry-run

.venv\Scripts\python.exe sam2_id_experiment\scripts\47_sam_assisted_fin_annotator.py `
  --manifest runs\fin_segmentation_audit\20251225_K_1_2_FCMC\side1_cam\annotation_candidates.csv `
  --manifest runs\fin_segmentation_audit\20251225_K_1_2_FCMC\side2_cam\annotation_candidates.csv `
  --output-dir runs\fin_annotations\20251225_K_1_2_FCMC\session_01 `
  --sam-model-cfg sam2_hiera_l.yaml `
  --checkpoint checkpoints\sam2_hiera_large.pt --device cuda
```

## Controls

- `I`, then left-click: select an existing Detectron2 instance; `X` cycles instances.
- Mouse wheel or `+` / `-`: zoom around the cursor or view center; `0` resets to fit.
- Middle-button drag: pan while zoomed.
- Each candidate automatically centers and zooms on its selected Detectron2 fish. Use
  `--no-auto-zoom` to disable this or `--auto-zoom-padding` to change the context margin.
- `H`: toggle a raw-image preview by hiding every mask and prompt marker. This changes
  only display visibility; all annotation state remains untouched. New candidates
  restore overlays automatically.
- Left/right click: positive/negative SAM2 point; `U` undoes; `R` clears the active region.
- Prompt locations are shown as small, fixed-size crosshairs with an open center so
  the exact source pixel remains visible while zoomed.
- `B` / `F`: edit body or fins separately; `P` previews SAM2; `[` / `]` selects another SAM proposal.
- `1` / `2` / `3`: save as clean / ambiguous / reject, then advance.
- `N` or Page Down: next candidate; Page Up: previous; `Q` or Escape: quit.

Each save creates timestamped body, fin, and combined PNG masks, a full JSON record
(source paths, source instance, prompts, SAM2 model, scores, image SHA-256), and an
append-only `annotation_records.csv`. A reject may intentionally contain no masks.

## Manifest contract

Required columns are `sample_id`, `view`, `frame`, and `image_path`. Optional useful
columns are `experiment`, `video_path`, `masks_h5`, `mask_instance_id`, `sleap_x`,
`sleap_y`, and any additional provenance fields. Extra columns are copied verbatim
into each JSON record.
