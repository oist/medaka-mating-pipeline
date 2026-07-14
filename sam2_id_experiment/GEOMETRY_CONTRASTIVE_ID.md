# Geometry-supervised appearance identity

This workflow implements appearance retraining without K-means and without
treating reconstruction-local geometry slots as biological identities. Geometry
selects clean observations and supplies local pair constraints:

- time-separated crops from one continuous 2D fragment are **same-fish** pairs;
- synchronized cross-camera crops supported by one reconstruction source are
  **same-fish** pairs;
- co-visible, different fish in the same camera frame are **different-fish**
  pairs.

The network learns a metric embedding, not a three-class name. Biological names
enter later only through independently reviewed trusted anchors. Missing or
uncertain appearance remains unknown evidence in the global solver.

## 1. Build the audited dataset

Choose complete temporal blocks in which current geometry has three finite,
low-residual, well-separated fish. Put an embargo between train, calibration,
and test blocks. The builder also requires strict silhouette-fused
one-mask-to-one-SLEAP associations for every included camera.

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.48_build_geometry_contrastive_dataset `
  --experiment 20251225_K_1_2_FCMC `
  --identified-2d <identified_2d.csv> `
  --dispositions <detection_dispositions.csv> `
  --best-pre-link-json <best_pre_link.json> `
  --view-input top_cam <top-video-or-images> <top-masks.h5> <top-silhouette-fused.csv> `
  --view-input side1_cam <side1-video-or-images> <side1-masks.h5> <side1-silhouette-fused.csv> `
  --view-input side2_cam <side2-video-or-images> <side2-masks.h5> <side2-silhouette-fused.csv> `
  --window development:<start>:<end>:train `
  --window calibration:<start>:<end>:calibration `
  --window heldout:<start>:<end>:test `
  --output-dir runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\dataset_v1
```

The output directory must be new or empty. Key outputs are `crops.h5`
(black-background RGB), `samples.parquet`, explicit positive/negative pair
tables, a contact sheet, and `summary.json`. `geometry_source_idx_local` is
frame-local provenance and is never a classifier target.

Before training, review `contact_sheet.png`, split counts, rejection reasons,
camera balance, and mask/SLEAP provenance. Do not reuse the historical 925
duplicate mask assignments identified by the appearance audit.

## 2. Train the non-clustering embedding

Start with a frozen ResNet18 and 32 dimensions. The 8-D row is the controlled
idtracker.ai-inspired dimension ablation; 128-D matches the scale of the current
embedder. No row uses K-means, a class head, or global identity labels.

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.49_train_geometry_contrastive_embedding `
  --dataset runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\dataset_v1 `
  --output-root runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\models `
  --run-name frozen_d32 `
  --embedding-dim 32 `
  --mode frozen
```

Then repeat as separate runs for `frozen_d8`, `frozen_d128`, and only if the
frozen baseline is useful, `finetune_d32 --mode fine-tune`. Each run is
exclusive/resumable and records best/last checkpoints, the exact train-only fit
membership, a split-manifest hash, pair predictions, crop embeddings, metrics,
and source provenance.

No production training has been launched. For roughly 20,000 128×128 crops, a
frozen projection run is expected to take about 2–5 minutes on the local RTX
5070 Ti and a full fine-tune about 3–10 minutes, using roughly 3–8 GB VRAM. The
first pretrained run may need ResNet18 weights already cached or network access.
Write every ablation to a new run name. These are planning estimates; first use
`--smoke` and extrapolate the measured time on the final dataset.

The bounded CPU integration smoke is:

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.49_train_geometry_contrastive_embedding `
  --synthetic-smoke `
  --output-root .tmp\geometry_embedding_smoke `
  --embedding-dim 8
```

## 3. Evaluate without leakage

Use the emitted `fit_membership.parquet`; it contains only samples that entered
training pairs. Calibration fits the probability model and decision thresholds.
Test labels are read only after those choices are frozen.

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.50_evaluate_geometry_contrastive_embedding `
  --dataset runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\dataset_v1 `
  --embeddings runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\models\frozen_d32\embeddings.parquet `
  --fit-membership runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\models\frozen_d32\fit_membership.parquet `
  --split-col split `
  --out-dir runs\appearance_geometry_contrastive\20251225_K_1_2_FCMC\evaluation\frozen_d32
```

The evaluator reports pair AUROC/AP/EER, calibration (Brier/NLL/ECE), frozen-
threshold confusion and TAR/FAR, risk-coverage, time-block and camera strata,
cross-camera retrieval when independent identities exist, train-fitted PCA, and
trusted-prototype similarities. It never clusters or emits an automatic
biological assignment.

## 4. Feed appearance into global stitching

Pool crop embeddings by the graph component provenance key, balancing camera
views so a heavily sampled view cannot dominate:

```powershell
.venv\Scripts\python.exe -m sam2_id_experiment.scripts.51_prepare_fragment_appearance_evidence `
  --embeddings <model-run>\embeddings.parquet `
  --samples <dataset>\samples.parquet `
  --fragment-column component_id_provenance `
  --min-samples 3 `
  --output-dir <new-evidence-output>
```

Pass `<new-evidence-output>\fragment_embeddings.npz` to
`scripts.tracking.benchmark_global_fragment_stitching` with
`--fragment-embeddings`. The benchmark attaches label-free cosine distance to
candidate edges. If reviewed biological anchors exist, supply a CSV with exactly
`component_id,trusted_identity,anchor_status`; only `anchor_status=trusted` rows
are accepted by `--trusted-anchors`. A geometry-ID-only table is rejected.

## Acceptance and rollback

Keep appearance advisory until the development-selected threshold reaches at
least 0.95 precision on independently reviewed hard events and beats a
position/background-only control under temporal and camera transfer. Promote it
only if the locked held-out benchmark reduces biological ID errors/switches,
does not increase simultaneous conflicts, and leaves XYZ and speed-spike counts
unchanged. Otherwise retain geometry-only global stitching and treat appearance
as unknown.

