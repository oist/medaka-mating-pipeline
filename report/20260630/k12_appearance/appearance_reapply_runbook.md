# A4 — Re-apply the validated appearance pipeline onto the NEW backbone

**Experiment:** `20251225_K_1_2_FCMC` (348,600 frames @166 fps, 1F+2M same size)
**NEW backbone:** `C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv`
(schema `frame,fish_id(ID_1/2/3),X,Y,Z,is_interpolated,source`; full coverage 344–348k frames/slot, geometrically correct, swap-prone at crossings).

**Goal:** produce a NEW-backbone `*_individual_id.csv` (per-frame `individual_id ∈ {F,M1,M2}` + confidence) by running the clean K_1_2 female/male appearance classifiers per frame, reusing the existing Detectron2 masks (no GPU detector re-run).

---

## TL;DR / recommendation

**Recommended = Path A (re-run script 45 against the NEW backbone).** It is correct-by-construction and cheap because the expensive stage (Detectron2 inference) is already done — only ResNet18 classifier forward passes are needed (~**60–90 min** total, GPU, main `.venv`).

**Do NOT use the naive cheap re-bind (Path B).** Measured: the OLD per-frame probability tables are keyed to a **different slot labeling** than the NEW backbone. OLD slot-k and NEW slot-k are the *same physical fish* in only **22.9%** of (frame,slot) rows; **70.3%** differ by >10 mm (median 42 mm). Even a per-frame Hungarian position remap only achieves a clean 3-of-3 match (max-residual <5 mm) in **68.5%** of frames, with **8.2%** unmatchable (>10 mm) — and those unmatchable frames are concentrated in exactly the dense-contact crossings that the appearance pass exists to resolve. A re-bind would therefore transfer stale/wrong probabilities precisely where it matters most. (A position-remap re-bind is documented below only as a fast *approximate* preview, not for production.)

---

## What was verified (assets present)

| Asset | Path | Status |
|---|---|---|
| NEW backbone | `C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv` | ✅ 85 MB, 348,600 frames, ID_1/2/3 → slots 0/1/2 |
| Female model | `C:/udmt_validation/segid/K_1_2/model_female_clean/model.pt` | ✅ 45 MB |
| Male model | `C:/udmt_validation/segid/K_1_2/model_male_clean/model.pt` | ✅ 45 MB |
| Top male model (optional fuse) | `C:/udmt_validation/segid/K_1_2/model_top_male/model.pt` | ✅ 45 MB |
| Detectron2 masks (saion fan-out) | `Z:/StephensU/medaka/detectron2_output/20251225_K_1_2_FCMC/{side1_cam,side2_cam,top_cam}.masks_<S>-<E>.h5` | ✅ 8 chunks/view, full 0–348599 |
| Calibration | `calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models` | ✅ |
| Videos | `\\bucket.oist.jp\...\medaka_202512\20251225_K_1_2_FCMC\<view>\cam*.mp4` | ✅ (bucket SMB; stage locally for speed) |

**Cost answer:** No path needs Detectron2 re-inference (masks exist on Z:). The recommended path is **pure classifier forward passes** in the main `.venv` (no `.venv-detectron2`).

---

## PATH A (RECOMMENDED) — re-run script 45 on the NEW backbone

Two steps: (0) reproject the NEW backbone's 3 slots into each view to build fresh `tracklet_bound.csv` slot points; (1) run script 45 with the existing masks + models.

`reproject_slot_points.slot_from_fish_id` maps `ID_1/2/3 → 0/1/2`, and script 45 asserts `fish_id` maps to exactly slots 0..2 (it does). So the NEW slot points are guaranteed consistent with the NEW backbone — no remap, no staleness.

### A.0 — build NEW slot points (reproject; fast, CPU, main .venv)

Run the female-anchor orchestrator and **stop after stage 0** (it calls `Calibration` + `build_slot_points` and writes `<view>.tracklet_bound.csv`):

```powershell
# from repo root C:\Users\machi\Documents\GitHub\medaka-mating-pipeline
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.40_female_anchor_pipeline 20251225_K_1_2_FCMC `
  --paths-config configs/paths.local.toml `
  --round-tag round5_260611 `
  --identified-3d-csv C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv `
  --work-dir runs/female_anchor/20251225_K_1_2_FCMC/round5_recovered `
  --calibration-model-dir calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models `
  --views side1_cam side2_cam top_cam `
  --stop-after slot-points
```

Output (the slot-points dir consumed by stage 5):
`runs/female_anchor/20251225_K_1_2_FCMC/round5_recovered/slot_points/{side1_cam,side2_cam,top_cam}.tracklet_bound.csv`
(columns `frame,x,y,score,global_id,tracklet_id`; `global_id == geometry slot`).

> Wall-clock: minutes (one batched reprojection per view over ~1.04 M finite fish-rows).

### A.1 — (recommended, optional) stage the 3 videos locally

Bucket SMB HEVC decode is slow; the OLD run staged videos to a local scratchpad and got ~840 s/view. Stage `side1_cam/cam1_side.mp4`, `side2_cam/cam2_side.mp4`, `top_cam/cam0_top.mp4` to a fast local disk and pass them with `--video VIEW=PATH`. (Basenames come from `VIEW_VIDEO_BASENAMES`; pass explicit `--video` overrides if you staged them.)

### A.2 — run the per-frame resolver (script 45)

GPU, main `.venv`. Masks already exist on Z: → **no Detectron2 run**.

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.45_per_frame_resolve 20251225_K_1_2_FCMC `
  --paths-config configs/paths.local.toml `
  --identified-3d-csv C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d.csv `
  --masks-dir Z:/StephensU/medaka/detectron2_output/20251225_K_1_2_FCMC `
  --slot-points-dir runs/female_anchor/20251225_K_1_2_FCMC/round5_recovered/slot_points `
  --views side1_cam side2_cam `
  --female-model C:/udmt_validation/segid/K_1_2/model_female_clean/model.pt `
  --male-model   C:/udmt_validation/segid/K_1_2/model_male_clean/model.pt `
  --top-model    C:/udmt_validation/segid/K_1_2/model_top_male/model.pt --top-view top_cam `
  --stride 10 `
  --out-csv C:/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d_individual_id.csv
```

If you staged videos locally (A.1), append e.g.
`--video side1_cam=D:/stage/side1_cam/cam1_side.mp4 --video side2_cam=D:/stage/side2_cam/cam2_side.mp4 --video top_cam=D:/stage/top_cam/cam0_top.mp4`.

**Outputs:**
- `…_recovered_3d_individual_id.csv` — backbone rows + `individual_id ∈ {F,M1,M2}`, `app_female_slot`, `app_conf`, `app_male_beta_slot`, `male_conf`, `indiv_conf`.
- `…_recovered_3d_individual_id.report.json` — `frames_one_of_each`, high-confidence fraction per individual.

Drop `--top-model … --top-view top_cam` (and `top_cam` from the A.0 `--views`) for a sides-only run (~15 min faster, slightly larger male-low core).

### A.3 — recommended post-processing (motion-continuity debounce)

The OLD pipeline applied `per_frame_resolve.debounce_track_teleports` after resolution to revert physically-impossible brief swaps. Apply the same to the NEW output before analysis (it reads `frame, geometry_slot, individual_id/id_final, X, Y, Z`).

---

## Cost / feasibility estimate (Path A)

Measured from the OLD run logs (same masks, same models, stride 10, same machine — RTX 5070 Ti):

| Stage | Views | Per-view | Subtotal |
|---|---|---|---|
| A.0 reproject slot points | 3 | ~1–2 min | ~5 min (CPU) |
| Female P(female) | side1+side2 | ~840–884 s | ~29 min |
| Male P(M2) | side1+side2 | ~840–884 s | ~29 min |
| Top male fuse (optional) | top | ~840 s | ~14 min |
| **Total (with top fuse)** | | | **~75–80 min** |
| **Total (sides only)** | | | **~60 min** |

- Crop volume per pass ≈ 101k crops (34,814 sampled frames × ~2.9 slots) through ResNet18 at batch 512 → I/O-bound on video decode, not GPU.
- **Bottleneck = video decode**, not classifier. Staging videos locally (A.1) is the single biggest speedup; on bucket SMB expect 2–4× slower.
- **No Detectron2 / GPU detector run** on any path — masks are pre-computed on Z:.

---

## PATH B (FALLBACK / preview only) — re-bind existing per-frame probabilities

Use only for a quick approximate preview, never for the final analysis (see TL;DR for why). The OLD tables are:
- `runs/dual_line/lineB_fulltimeline/step2/per_frame_female_0_348599.csv` — `frame,female_slot,p_slot0/1/2,margin,n_slots` (P(female) per OLD slot).
- `…/per_frame_male_0_348599.csv` — `frame,pb_slot0/1/2,n_slots` (P(M2) per OLD slot).
- `…/per_frame_top_male_0_348599.csv` — `frame,tm_slot0/1/2,n_slots` (P(M2) top per OLD slot).

These store the appearance probability **per geometry SLOT per sampled frame** — but the slots are the OLD `tracklet_bind_extend_smooth` `global_id`, which is a **different labeling** from the NEW backbone (measured: same-fish only 22.9% by index).

**Re-bind = remap OLD slot → NEW slot per frame by nearest 3D position (Hungarian), then carry P columns into the NEW slot order and feed the relabeled tables to the pure resolver functions** (`per_frame_resolve.resolve_female_track` / `resolve_male_track` / `top_male_anchors` / `fuse_side_top` / `assemble_individual_labels`). No video, no masks, no GPU → seconds.

Pseudocode for the remap (per sampled frame f with finite 3D in both backbones):
```
A = NEW slot xyz (3×3), B = OLD slot xyz (3×3)   # OLD xyz from round5_260611_3d_individual_id.csv geometry_slot
C[i,j] = ||A_i - B_j||;  r,c = linear_sum_assignment(C)   # newslot r[k] <- oldslot c[k]
if C[r,c].max() < TOL_MM (e.g. 5):   accept; p_newslot[r] = p_oldslot[c]
else:                                 mark frame UNRESOLVED (do NOT transfer)
```
Then rename `p_slot{oldj}` → `p_slot{newi}` and run the resolver.

**Cost:** seconds (pure pandas/scipy, ~3.4k–35k sampled frames). **But** ~8–31% of sampled frames are unmatchable/ambiguous and must be dropped to `UNRESOLVED`, leaving holes exactly at crossings. Treat Path B output as indicative coverage only.

---

## Single most likely failure mode + detection

**Failure mode: the per-frame probabilities are bound to the WRONG slot** — i.e. the slot-point `global_id` fed to the classifier does not correspond to the physical fish the NEW backbone places there. This is *guaranteed* if Path B is used naively (slot labelings differ, 22.9% agreement) and is *possible* in Path A if the reprojected slot points land in the wrong Detectron2 mask at a crossing (two fish reproject into one mask — `link_gid` drops merged masks, which protects against this but also creates coverage gaps).

**How to detect (cheap, quantitative):**
1. **Sanity-from-the-report:** `…_individual_id.report.json → frames_one_of_each` should be a large fraction of `n_frames`. A low value means slots collide / mislabel. (The OLD K_1_2 run is the reference baseline.)
2. **Female-pin continuity:** load the output, build `app_female_slot` per frame; the female 3D track assembled from `individual_id=='F'` must be **spatially continuous** (no >60 mm single-frame jumps — the same `debounce_track_teleports` jump test). Count teleports; a correct binding has very few.
3. **BORIS spawn cross-check (ground truth):** at frame 135888 (818.6 s) the female must be in contact with a male (`compare_boris_trajectory.py`). If `individual_id=='F'` at the spawn is the fish NOT in the male bunch, the female bind is wrong.
4. **Path-B-specific:** report the fraction of sampled frames with Hungarian max-residual >5 mm (the unmatchable rate, measured ~8% at stride 100, higher near crossings). If that fraction is non-trivial in the windows you care about, prefer Path A.

---

## Measured numbers behind this runbook

- NEW backbone: 348,600 frames; ID_1/2/3 → slots 0/1/2; ~1.04 M finite (frame,slot) rows.
- OLD-vs-NEW slot index agreement: **same fish (dist<2 mm) 22.9%**, **different fish (dist>10 mm) 70.3%**, median slot-k displacement **42.1 mm** (1.04 M inner rows).
- Hungarian position remap (stride-100, 3,441 frames): clean 3-of-3 match <2 mm **48.9%**, <5 mm **68.5%**, unmatchable >10 mm **8.2%**.
- OLD per-pass timing (stride 10, same masks/models): **~840–884 s/view**; female crops ≈101k (34,814 frames × 2.91 slots).
- Detectron2 masks: 8 chunks/view × 3 views, full 0–348599, present on Z: → **0 detector re-runs needed**.
