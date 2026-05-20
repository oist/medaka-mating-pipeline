# Implementation Plan: SAM2 segmentation for medaka, with pose-splatter pseudo-labels

## Context (grounded in current repo state + pose-splatter inspection)

**Current pipeline:** 3 cameras (top + 2 sides), polynomial calibration, SLEAP keypoints (mouth + mid-fin + tail, `round2_trt_n6` mouth+tail variant), 3D reconstruction with 99% 3-fish coverage, 2D ID per top-cam + 3D ID via reprojection-NN binding. **No segmentation today.**

**Why segmentation might help:**
- Body-shape features (orientation, body curvature, area) for finer behavior than 3-keypoint pose.
- Better appearance-based ID re-identification (color, fin extent) when 2D tracking drops.
- Cleaner overlays for figures/videos and for downstream 3D mesh / volume work.

**What pose-splatter actually does** (verified from `jackgoffinet/pose-splatter@master` README + sources, NeurIPS 2025 paper [arXiv 2505.18342](https://arxiv.org/abs/2505.18342)):
- Pipeline: `estimate_up_direction` → `calculate_center_rotation` → `calculate_crop_indices` (3D **shape carving** from per-view binary masks) → `copy_to_zarr` → `train_script` → `evaluate_model` → `render_image` → visual features/embeddings.
- **Inputs masks, does not produce them** — `calculate_crop_indices.py` reads pre-computed binary masks from the input HDF5 and threshold-carves a voxel grid; `train_script.py` consumes `mask` + `img` + `p_3d` per view and supervises with **IoU + SSIM + L1** against the input alpha masks.
- Loss confirms the splatter **renders refined per-view alpha masks** that match the rough input masks across views, multi-view-consistent. → those refined alphas are the artifact we want to exploit as SAM2 pseudo-labels.
- Designed for **single-instance** scenes (mouse, zebra finch); their published dataset is 5.6 GB total. Scene assumption: one animal, static background, calibrated multi-view.

**Implication for medaka:** pose-splatter cannot run directly on the live 3-fish footage. Two ways to make it applicable:
1. **Single-fish recordings** (cleanest): record a few minutes per fish with only one fish in the tank.
2. **Crop per-instance** windows around each SLEAP-tracked fish per view, treat each crop as a single-instance mini-scene. Loses 3D consistency unless we lift the crop carefully.

Cheaper bootstrap exists upstream of pose-splatter: the tank background is static, the fish is small and dark on a uniform bottom → **median-frame background subtraction** alone likely gives 80–90% IoU silhouettes for single-fish. Pose-splatter is the *refinement* layer on top of that. Plan tests the cheap baseline first before committing to pose-splatter infrastructure.

**Independent fast track:** SLEAP keypoints already exist for every fish, every frame, every view. SAM2 accepts point prompts. → an immediate test of default-SAM2 quality on multi-fish scenes is one script away, no training.

## Task Type

- [x] Backend / ML
- [ ] Frontend
- [ ] Fullstack

## Two parallel tracks

```
Track A (fast, this week):  Default SAM2 + SLEAP keypoint prompts on existing 3-fish footage
Track B (deeper, weeks):    Background-sub OR pose-splatter pseudo-labels on single-fish footage
                            → fine-tune SAM2 (LoRA on mask decoder) → re-test on 3-fish footage
```

Track A is the quick "is SAM2 even close?" probe. Track B is the route to a robust fine-tuned model. We do A first because its result decides how much B is needed.

---

## Phase 0 — Default SAM2 sanity test (Track A, ~1 day)

**Goal:** measure how good vanilla SAM2 is on medaka frames before we invest in pseudo-label generation.

1. Add `sam2` and `huggingface_hub` to `pyproject.toml` under a new `[project.optional-dependencies] segmentation` extra. Confirm CUDA wheel availability (the project already pulls torch from `pytorch-cu130`).
2. Download SAM2.1 hiera-large checkpoint (or hiera-base for speed) into `Z:\StephensU\medaka\sam2_models\` and reference via `configs/paths.local.toml`.
3. New script `scripts/segmentation/sam2_probe.py`:
   - Loads one experiment (e.g. `20251225_K_1_2_FCMC`), one view (`top_cam`), 30 evenly-spaced frames.
   - For each frame: read existing SLEAP `mouth` keypoint per fish (from `round2_trt_n6`) → use as **positive point prompt** to `SAM2ImagePredictor`. Optionally add a **negative point** at another fish's mouth to discourage merging.
   - Save `(mask, score)` per fish per frame to a `runs/sam2_probe_<exp>/` folder, plus an overlay PNG.
   - Repeat for `side1_cam` and `side2_cam`.
4. Same on the `SAM2VideoPredictor` (mask propagation) over a 1000-frame window with one initial-frame prompt per fish — measures temporal coherence of vanilla SAM2.
5. Manual eval (eyeball + 5 frames hand-labeled in napari) → IoU table per view.

**Decision gate:**
- If IoU ≥ ~0.85 on all 3 views → we're done; ship Track A as-is, skip Track B.
- If 0.6 ≤ IoU < 0.85 → fine-tune (Phase 2+).
- If IoU < 0.6 → vanilla is unusable; fine-tune is mandatory and prompts may need to change (e.g. box from SLEAP bbox).

## Phase 1 — Pseudo-label source for fine-tuning (Track B start, ~3–5 days)

**Goal:** build per-frame ground-truth-quality silhouettes for ≥ 5 k single-fish frames per view, with no human labeling.

**Step 1A — Cheap baseline: background subtraction on single-fish footage.**
- New script `scripts/segmentation/bg_subtract_masks.py`:
  - Compute per-view median frame from a 30 s no-fish clip (or a temporal-median over a long stretch of footage; medaka are sparse pixels in the frame).
  - Per frame: `mask = (|frame − median| > τ)` → morphological open/close → keep largest connected component inside the tank ROI.
  - Tank ROI from `calibration_results/.../tank_corners`.
- Hand-check 50 random frames per view in napari → IoU vs eyeball label.
- **Gate:** if median IoU ≥ 0.9 → skip pose-splatter for now, use these as labels. If 0.7–0.9 → use these as **rough masks fed into pose-splatter**. If < 0.7 → revisit thresholds before adding pose-splatter complexity.

**Step 1B (only if needed) — Pose-splatter refinement.**
- New folder `external/pose-splatter/` (git submodule from `jackgoffinet/pose-splatter@master`).
- Adapter script `scripts/segmentation/posesplatter_adapt.py`:
  - Convert one single-fish recording to the HDF5 layout pose-splatter expects (`copy_to_zarr.py` consumes HDF5 → zarr; we need to produce that HDF5 first with `images` + `masks` per view + camera params).
  - Reuse our calibration model: pose-splatter wants per-camera intrinsics + extrinsics. Our model is a 6D-pixel→3D polynomial — we'll need a small wrapper that exposes a `project(XYZ)` and an approximate intrinsic/extrinsic for the splatter's voxel renderer. Risk: this may not be a clean pinhole. **De-risk first** with a 1-frame projection round-trip test before investing in the full conversion.
  - Run the 10-step pipeline (`estimate_up_direction` … `render_image`) on one short single-fish window.
  - Extract refined per-view alpha masks from the renderer output → write back as PNGs aligned to original frame indices.
- Hand-check 50 random frames → IoU vs eyeball label.
- **If pose-splatter doesn't add ≥ 5 IoU points over background subtraction**, drop it. The infrastructure cost (HDF5 conversion + calibration adapter + zarr + train loop) is high relative to a marginal mask quality gain.

**Recording requirement:** acquire ~5 min single-fish footage per view (3 fish × 5 min × 166 fps = ~150 k frames per view). Subsample to ~5 k uniform frames for fine-tuning to avoid overfitting redundant near-still frames.

## Phase 2 — Fine-tune SAM2 (Track B core, ~1 week)

**Goal:** raise SAM2 multi-fish IoU on the 3-fish recordings.

1. Convert pseudo-labels to **SAM2 fine-tuning format** (image + binary mask + a few positive/negative point prompts per mask, sampled from the mask centroid + SLEAP keypoints).
2. Use Meta's reference fine-tuning script (or `sam2-adapter` / LoRA on mask decoder + prompt encoder; freeze the image encoder for the first run — much faster, much less overfit risk on 5 k images).
3. Train per-view? Or one fine-tuned model across all 3 views? Recommend **one model**, with view as an implicit conditioning via training-set diversity. Re-evaluate if cross-view IoU is uneven.
4. Eval on the same 30 frames + 1000-frame video used in Phase 0. Report IoU lift per view, latency per frame.
5. Hard-case mining: take frames where Phase-0 vanilla SAM2 failed and the fine-tune still fails → these become the **manual-label** queue for round 2.

## Phase 3 — Manual-label round (~3 days, optional)

Only if Phase 2 leaves visible failure modes (e.g. fish-on-fish overlap, reflections eaten by mask, fin loss). Hand-label ~200 hard frames in napari (load the SAM2 mask, click-correct), retrain. This is the only stage that should require human time.

## Phase 4 — Integration with the existing pipeline (~2 days)

1. New script `scripts/segmentation/sam2_inference.py` mirroring `scripts/sleap/batch_track.py`: takes an experiment, runs the fine-tuned SAM2 over all 3 views with SLEAP keypoint prompts, writes per-fish, per-view, per-frame masks to **parquet of RLE-encoded masks** (CSV is too large; PNG-per-frame too many files — RLE in parquet is the right shape per `[I/O format]` memory).
2. Add a 4th panel to `scripts/visualization/rerun_viewer.py`: per-view mask overlay.
3. Add metrics to `scripts/analysis/`: body area, principal-axis orientation, body-curvature-from-mask. These extend the SLEAP-based metrics without replacing them.
4. **Do not** rip out SLEAP — masks complement keypoints; SLEAP gives part identities and is faster.

## Risks & open questions

| Risk | Mitigation |
|---|---|
| pose-splatter calibration adapter is fragile (our model is not pinhole) | De-risk in Phase 1B step 1 with a single-frame projection round-trip; abandon early if it doesn't fit. |
| Background subtraction is good enough → pose-splatter is wasted work | Phase 1A gate is explicit; we only proceed to pose-splatter if BG-sub falls in 0.7–0.9 IoU. |
| Vanilla SAM2 already adequate → fine-tune is wasted work | Phase 0 gate is explicit. |
| Fine-tune overfits to single-fish appearance, fails on 3-fish reflections | Mix in Phase-3 hard-case manual labels; hold out a 3-fish eval set from start. |
| Mask output volume is huge (3 views × 166 fps × 30 min × 3 fish) | RLE-in-parquet; only persist masks for the ~20 % of frames in named courtship clips by default. |
| Single-fish footage requires re-acquisition | Confirm with user: does any existing recording have a stretch with only one fish? If yes, that's the training data and Phase 1 starts immediately. |

## What I'm asking the user before starting

1. Do you have any single-fish footage already, or do we need to record fresh 1-fish clips per fish?
2. Is "body shape / area / orientation per fish per frame" the headline downstream use, or is this primarily for figure/video aesthetics? (Answers determine how much we invest in fine-tune precision vs. just running Track A.)
3. OK to add `sam2` to the `[segmentation]` extra in `pyproject.toml`?

## Out of scope (explicitly)

- 3D mesh reconstruction of the fish body (pose-splatter's actual headline output) — useful but a separate later effort.
- Replacing SLEAP. Masks complement keypoints; we keep both.
- Cross-experiment generalization test. Fine-tune on this rig's footage; re-validate per-rig if the rig changes.
